"""分位数 conformal 校准。

输入：
  val_predictions.csv  校准折，含 y 与 q0.05...q0.95
  test_predictions.csv 测试预测，同列
输出：
  test_predictions_calibrated.csv
  calibration_summary.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def calibrate(val, test, clip=None):
    test = test.copy()
    deltas = {}
    for t in TAUS:
        col = f"q{t:g}"
        s = val["y"].to_numpy() - val[col].to_numpy()
        delta = float(np.quantile(s, t))
        deltas[col] = delta
        test[col] = test[col] + delta
    cols = [f"q{t:g}" for t in TAUS]
    test[cols] = np.sort(test[cols].to_numpy(), axis=1)
    if clip is not None:
        test[cols] = np.clip(test[cols], clip[0], clip[1])
    return test, deltas


def metrics(y, qmat):
    out = {}
    for i, t in enumerate(TAUS):
        e = y - qmat[:, i]
        out[f"pinball_{t:g}"] = float(np.mean(np.maximum(t * e, (t - 1) * e)))
    y_arr = np.asarray(y, float)
    q_arr = np.asarray(qmat, float)
    out["mean_pinball"] = float(np.mean([out[f"pinball_{t:g}"] for t in TAUS]))
    w = np.zeros(len(TAUS))
    w[0] = (TAUS[1] - TAUS[0]) / 2
    w[-1] = (TAUS[-1] - TAUS[-2]) / 2
    w[1:-1] = (np.array(TAUS[2:]) - np.array(TAUS[:-2])) / 2
    rho = np.array([np.mean(np.maximum(t * (y_arr - q_arr[:, i]), (t - 1) * (y_arr - q_arr[:, i])))
                    for i, t in enumerate(TAUS)])
    out["crps_q7_trunc"] = float(2.0 * np.sum(w * rho))
    for nom, lo, hi in ((0.5, 0.25, 0.75), (0.8, 0.10, 0.90), (0.9, 0.05, 0.95)):
        out[f"coverage_{int(nom*100)}"] = float(np.mean(
            (y >= qmat[:, TAUS.index(lo)]) & (y <= qmat[:, TAUS.index(hi)])))
        out[f"width_{int(nom*100)}"] = float(np.mean(
            qmat[:, TAUS.index(hi)] - qmat[:, TAUS.index(lo)]))
    q50 = qmat[:, TAUS.index(0.5)]
    out["mae"] = float(np.mean(np.abs(y - q50)))
    out["rmse"] = float(np.sqrt(np.mean((y - q50) ** 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", default=None)
    args = ap.parse_args()
    clip = tuple(float(x) for x in args.clip.split(",")) if args.clip else None
    val = pd.read_csv(args.val)
    test = pd.read_csv(args.test)
    cal, deltas = calibrate(val, test, clip)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cal.to_csv(out / "test_predictions_calibrated.csv", index=False)
    # 校准后的分组评估（与训练器 results.csv 同 schema，供出图直接读取）
    cols = [f"q{t:g}" for t in TAUS]
    g_rows = []
    groups = [("all", [None]),
              ("lead_time", sorted(cal.lead_time.unique())),
              ("station_id", sorted(cal.station_id.unique()))]
    if "season" in cal.columns:
        groups.append(("season", sorted(cal.season.unique())))
    for group, keys in groups:
        for k in keys:
            g = cal if k is None else cal[cal[group] == k]
            if not len(g):
                continue
            qm = g[cols].to_numpy()
            rec = dict(group=group, key="all" if k is None else k, n=len(g),
                       crossing_rate=0.0, **metrics(g.y.to_numpy(), qm))
            g_rows.append(rec)
    pd.DataFrame(g_rows).to_csv(out / "results_calibrated.csv", index=False)
    before = metrics(test.y.to_numpy(), test[cols].to_numpy())
    after = metrics(cal.y.to_numpy(), cal[cols].to_numpy())
    lines = ["# Quantile calibration", "", "| metric | before | after |", "|---|---|---|"]
    for k in before:
        lines.append(f"| {k} | {before[k]:.4f} | {after[k]:.4f} |")
    (out / "calibration_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("calibrated ->", out)


if __name__ == "__main__":
    main()
