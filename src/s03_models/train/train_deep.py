"""v2–v15 深度分位数模型统一训练/评估。

协议（见 project_manifest.yaml；结果在正式重训前为 provisional）：
  nested purged rolling-origin 用于模型选择；最终顺序为
  fit -> purge -> early-stop -> purge -> calibration -> final test。
标准化：数值特征与目标的均值/标准差只用本次运行的训练子集拟合，
        验证折与测试年不参与（--subsample 只影响训练子集大小，不影响测试）。
输入：每（station, lead）按时间排序的 33 特征 + 20 站/4 季 one-hot 静态通道，
      窗口长度 24 h（长序列模型 168 h，--seq-len）。
模型（--model）：mlp/lstm/cnn/tcn/transformer/dlinear/tsmixer/itransformer/patchtst/
                 autoformer/informer/fedformer/timesnet/pinn
输出：reports/03_modeling/vX_{model}/{lite|full}/quantile/{ghi,cloud}/
"""
import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(CODE_ROOT / "src" / "s03_models" / "train"))
import train_v1 as tv  # noqa: E402
sys.path.insert(0, str(CODE_ROOT / "src" / "s02_experiment"))  # split_protocol
from split_protocol import default_split, split_early_stop  # noqa: E402
from cloud_impute import impute_cloud_forecast  # noqa: E402
sys.path.insert(0, str(CODE_ROOT / "src" / "s03_models"))
from model_registry import deep_model_vnum  # noqa: E402

import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_deep")

SEQ_LEN = 24
SEED = 0
TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
MODEL_VNUM = deep_model_vnum()

def season_of(ts):
    m = ts.month
    return ("spring" if m in (3, 4, 5) else "summer" if m in (6, 7, 8)
            else "autumn" if m in (9, 10, 11) else "winter")


def load_groups(target):
    sites = yaml.safe_load((CODE_ROOT / "config" / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    code = {s["id"]: i for i, s in enumerate(sites)}
    obs = "ghi_obs_sat" if target == "ghi" else "cloud_cover_obs"
    cols = tv.FEATURES_NUM + ["station_id", "target_time_utc", obs]
    frames = []
    for s in sites:
        p = CODE_ROOT / "data" / "03_featured" / f"{s['id']}_featured_2024-02_2026-09.parquet"
        df = pd.read_parquet(p, columns=cols)
        df["target_time_utc"] = pd.to_datetime(df["target_time_utc"], utc=True)
        df[tv.FEATURES_NUM] = df[tv.FEATURES_NUM].astype("float32")
        df[obs] = df[obs].astype("float32")
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    del frames
    if target == "ghi":
        d = d.query(tv.DAY_FILTER)
    d = d.dropna(subset=[obs]).copy()
    if target == "cloud":
        d = impute_cloud_forecast(d, feature_cols=tv.FEATURES_NUM)
    d = d.sort_values(["station_id", "lead_time", "target_time_utc"]).reset_index(drop=True)
    d["season"] = d["target_time_utc"].apply(season_of)
    d["st_code"] = d["station_id"].map(code).astype(float)
    d["lead_norm"] = d["lead_time"].astype(float) / 72.0

    # 深度模型类别编码：station 与 season 使用 one-hot 静态通道。
    st = pd.get_dummies(d["station_id"].astype(str), prefix="st").astype("float32")
    se = pd.get_dummies(d["season"].astype(str), prefix="season").astype("float32")
    d = pd.concat([d, st, se], axis=1)
    num_feats = tv.FEATURES_NUM + ["lead_norm"]
    cat_feats = list(st.columns) + list(se.columns)
    feats = num_feats + cat_feats

    groups, indices, y, meta = [], [], [], []
    for (sid, lead), g in d.groupby(["station_id", "lead_time"], sort=False):
        arr = g[feats].to_numpy(np.float32)
        groups.append(arr)
        gid = len(groups) - 1
        t = g["target_time_utc"].to_numpy()
        for i in range(SEQ_LEN - 1, len(g)):
            indices.append((gid, i))
            y.append(float(g[obs].iloc[i]))
            meta.append((sid, int(lead), t[i], str(g["season"].iloc[i])))
    y = np.asarray(y, np.float32)
    meta = pd.DataFrame(meta, columns=["station_id", "lead_time", "target_time_utc", "season"])
    idx = np.asarray(indices, dtype=np.int64)
    return groups, idx, y, meta


class SeqDS(torch.utils.data.Dataset):
    def __init__(self, groups, idx, y, mask):
        self.groups = groups
        self.idx = idx[mask]
        self.y = y[mask]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        gid, end = self.idx[i]
        x = self.groups[gid][end - SEQ_LEN + 1:end + 1]
        return torch.from_numpy(x), torch.tensor(self.y[i])


class MLP(nn.Module):
    def __init__(self, d, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(SEQ_LEN * d, 256), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, out_dim))

    def forward(self, x):
        return self.net(x)


class LSTMNet(nn.Module):
    def __init__(self, d, out_dim=1):
        super().__init__()
        self.rnn = nn.LSTM(d, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1])


class CNNNet(nn.Module):
    def __init__(self, d, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(128, out_dim)

    def forward(self, x):
        z = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.head(z)


class TCNNet(nn.Module):
    def __init__(self, d, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, 64, 2, padding=1, dilation=1), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 2, padding=2, dilation=2), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 2, padding=4, dilation=4), nn.ReLU())
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):
        z = self.net(x.transpose(1, 2))
        return self.head(z[:, :, -1])


class TransEnc(nn.Module):
    def __init__(self, d, nhead=4, layers=2, out_dim=1):
        super().__init__()
        self.proj = nn.Linear(d, 64)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, 64))
        enc = nn.TransformerEncoderLayer(64, nhead, 128, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):
        z = self.proj(x) + self.pos
        z = self.enc(z)
        return self.head(z.mean(1))


class DLinearNet(nn.Module):
    def __init__(self, d, out_dim=1):
        super().__init__()
        self.trend = nn.Linear(SEQ_LEN, 1)
        self.season = nn.Linear(SEQ_LEN, 1)
        self.mix = nn.Linear(d, out_dim)

    def forward(self, x):
        z = x.transpose(1, 2)
        trend = z - torch.nn.functional.avg_pool1d(z, 5, 1, 2)
        t = self.trend(trend).squeeze(-1)
        s = self.season((z - trend)).squeeze(-1)
        return self.mix(t + s)


class TSMixerNet(nn.Module):
    def __init__(self, d, out_dim=1):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(SEQ_LEN, 32), nn.ReLU(), nn.Linear(32, SEQ_LEN))
        self.feat = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, d))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(SEQ_LEN * d, 128), nn.ReLU(),
                                  nn.Linear(128, out_dim))

    def forward(self, x):
        x = x + self.time(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.feat(x)
        return self.head(x)


class ITransformerNet(nn.Module):
    def __init__(self, d, out_dim=1):
        super().__init__()
        self.proj = nn.Linear(SEQ_LEN, 64)
        enc = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, 2)
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):
        z = self.enc(self.proj(x.transpose(1, 2)))       # B × d × 64
        return self.head(z.mean(1))


class PatchTSTNet(nn.Module):
    def __init__(self, d, patch=12, out_dim=1):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(patch, 32)
        enc = nn.TransformerEncoderLayer(32, 2, 64, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, 1)
        self.head = nn.Linear(32, out_dim)

    def forward(self, x):
        b, t, d = x.shape
        p = t // self.patch
        z = x[:, :p * self.patch].reshape(b, p, self.patch, d).permute(0, 3, 1, 2)
        z = self.proj(z).reshape(b * d, p, 32)
        z = self.enc(z).mean(1).reshape(b, d, 32).mean(1)
        return self.head(z)


class AutoformerNet(TransEnc):
    def forward(self, x):
        trend = torch.nn.functional.avg_pool1d(x.transpose(1, 2), 5, 1, 2).transpose(1, 2)
        z = super().forward(x - trend)
        return z


class InformerNet(TransEnc):
    pass


class FEDformerNet(TransEnc):
    def forward(self, x):
        z = self.proj(x) + self.pos
        f = torch.fft.rfft(z, dim=1)
        f = f * torch.sigmoid(torch.fft.rfft(z, dim=1).abs().mean(1, keepdim=True))
        z = torch.fft.irfft(f, n=SEQ_LEN, dim=1)
        return self.head(self.enc(z).mean(1)).squeeze(-1)


class TimesNetNet(nn.Module):
    def __init__(self, d, out_dim=1):
        super().__init__()
        self.feat_pool = nn.AdaptiveAvgPool1d(8)
        self.conv = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(8 * SEQ_LEN * 8, 64), nn.ReLU(),
                                  nn.Linear(64, out_dim))

    def forward(self, x):
        z = self.feat_pool(x).unsqueeze(1)
        z = self.conv(z)
        return self.head(z)


class PINNNet(nn.Module):
    def __init__(self, d, out_dim=1):
        super().__init__()
        self.net = MLP(d, out_dim)
    def forward(self, x):
        pred = self.net(x)
        # The previous clear-sky ceiling compared target-scaled predictions with
        # feature-scaled clear-sky values and treated an uncertain reference as
        # a hard physical upper bound. It is disabled pending site adaptation.
        pen = torch.relu(-pred).mean()
        return pred, pen


def make_model(name, d, out_dim=1):
    cls = {"mlp": MLP, "lstm": LSTMNet, "cnn": CNNNet, "tcn": TCNNet,
           "transformer": TransEnc, "dlinear": DLinearNet, "tsmixer": TSMixerNet,
           "itransformer": ITransformerNet, "patchtst": PatchTSTNet,
           "autoformer": AutoformerNet, "informer": InformerNet,
           "fedformer": FEDformerNet, "timesnet": TimesNetNet, "pinn": PINNNet}[name]
    if name in ("transformer", "autoformer", "informer", "fedformer", "patchtst"):
        return cls(d, out_dim=out_dim)
    return cls(d, out_dim)


def pinball_loss(pred, y, taus=TAUS):
    tau = torch.tensor(taus, dtype=pred.dtype, device=pred.device).view(1, -1)
    e = y.view(-1, 1) - pred
    base = torch.maximum(tau * e, (tau - 1) * e).mean()
    # 分位单调惩罚：相邻分位逆序按 0.1 权重进入损失（仅深度模型可实现）。
    viol = torch.relu(pred[:, :-1] - pred[:, 1:]).mean()
    return base + 0.1 * viol


def run_epoch(model, loader, opt, loss_fn, train=True, pinn=False, qmode=False,
              device="cpu"):
    model.train(train)
    total, errs = 0, 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = model(xb)
            if pinn:
                pred, pen = out
                loss = (pinball_loss(pred, yb) if qmode else loss_fn(pred, yb)) + 0.1 * pen
            else:
                pred = out
                loss = pinball_loss(pred, yb) if qmode else loss_fn(pred, yb)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        errs += float(loss.detach()) * len(yb)
        total += len(yb)
    return errs / max(total, 1)


def predict(model, loader, pinn=False, qmode=False, device="cpu"):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb)
            if pinn:
                pred = pred[0]
            out.append(pred.detach().cpu().numpy())
    return np.concatenate(out)


def metrics(y, yhat):
    e = yhat - y
    ss = np.sum((y - y.mean()) ** 2)
    return dict(n=len(y), mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                bias=float(np.mean(e)), r2=float(1 - np.sum(e ** 2) / ss) if ss else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["ghi", "cloud"], default="ghi")
    ap.add_argument("--model", required=True, choices=list(MODEL_VNUM))
    ap.add_argument("--loss", choices=["mae", "mse"], default="mae")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--subsample", type=int, default=1,
                    help="训练样本按该步长抽样（仅训练集，验证/测试全量）")
    ap.add_argument("--val-subsample", type=int, default=1,
                    help="早停验证样本抽样步长，仅影响 early stopping；测试全量")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budget", choices=["lite", "full"], default="lite")
    ap.add_argument("--quantile", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--impl", choices=["lite", "formal"], default="lite")
    ap.add_argument("--seq-len", type=int, default=168)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    global SEQ_LEN
    SEQ_LEN = int(args.seq_len)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else args.device if args.device != "auto" else "cpu")
    groups, idx, y, meta = load_groups(args.target)
    t = pd.to_datetime(meta.target_time_utc, utc=True)
    proto = meta.assign(target_time_utc=t)
    tr_df, va_df, te_df = default_split(proto)
    tr_keys = set(zip(tr_df.station_id, tr_df.target_time_utc))
    va_keys = set(zip(va_df.station_id, va_df.target_time_utc))
    te_keys = set(zip(te_df.station_id, te_df.target_time_utc))
    keys = list(zip(meta.station_id, t))
    m_tr = np.array([k in tr_keys for k in keys])
    m_va = np.array([k in va_keys for k in keys])
    m_te = np.array([k in te_keys for k in keys])
    # Causal final split. Calibration is strictly later than early stopping.
    tr_fit_df, tr_es_df = split_early_stop(tr_df)
    fit_keys = set(zip(tr_fit_df.station_id, tr_fit_df.target_time_utc))
    es_keys = set(zip(tr_es_df.station_id, tr_es_df.target_time_utc))
    m_fit = np.array([k in fit_keys for k in keys])
    m_es = np.array([k in es_keys for k in keys])
    if args.subsample > 1:
        keep = np.zeros(len(m_fit), dtype=bool)
        keep[::args.subsample] = True
        m_fit = m_fit & keep
    if args.val_subsample > 1:
        keep = np.zeros(len(m_es), dtype=bool)
        keep[::args.val_subsample] = True
        m_es = m_es & keep
    logger.info("train pool=%d  fit=%d  early-stop=%d  calibration=%d  test=%d",
                m_tr.sum(), m_fit.sum(), m_es.sum(), m_va.sum(), m_te.sum())
    # 标准化统计量只用拟合子集（m_fit）计算：不含早停尾部、校准折与测试年。
    n_num = len(tv.FEATURES_NUM) + 1  # 数值块含 lead_norm
    tr_rows = np.nonzero(m_fit)[0]
    vals = np.empty((len(tr_rows), n_num), dtype=np.float32)
    for k, j in enumerate(tr_rows):
        gid, pos = idx[j]
        vals[k] = groups[gid][pos, :n_num]
    mu = np.nanmean(vals, axis=0)
    sd = np.nanstd(vals, axis=0)
    sd[sd == 0] = 1.0
    for g in groups:
        # 缺失值按训练均值填补（标准化后为 0），与旧实现口径一致。
        g[:, :n_num] = np.nan_to_num((g[:, :n_num] - mu) / sd, nan=0.0)
    logger.info("scaler fit on %d training rows (mean/std of %d numeric features)",
                len(tr_rows), n_num)
    y_mean, y_std = float(y[m_fit].mean()), float(y[m_fit].std())
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)
    ds_tr, ds_es, ds_va, ds_te = (SeqDS(groups, idx, y_scaled, m)
                                  for m in (m_fit, m_es, m_va, m_te))
    if args.smoke:
        ds_tr = torch.utils.data.Subset(ds_tr, range(min(4000, len(ds_tr))))
        ds_es = torch.utils.data.Subset(ds_es, range(min(2000, len(ds_es))))
    ld_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                                        num_workers=args.workers, pin_memory=device.type == "cuda")
    ld_es = torch.utils.data.DataLoader(ds_es, batch_size=args.batch,
                                        num_workers=args.workers, pin_memory=device.type == "cuda")
    ld_va = torch.utils.data.DataLoader(ds_va, batch_size=args.batch,
                                        num_workers=args.workers, pin_memory=device.type == "cuda")
    ld_te = torch.utils.data.DataLoader(ds_te, batch_size=args.batch,
                                        num_workers=args.workers, pin_memory=device.type == "cuda")
    logger.info("%s/%s: n=%d feat=%d", args.model, args.target, len(y), groups[0].shape[1])

    d = groups[0].shape[1]
    qmode = bool(args.quantile)
    if args.impl == "formal" and args.model in (
            "autoformer", "informer", "fedformer", "patchtst", "timesnet", "pinn"):
        from formal_architectures import build_formal_model
        model = build_formal_model(args.model, d, SEQ_LEN, len(TAUS) if qmode else 1)
    else:
        model = make_model(args.model, d, len(TAUS) if qmode else 1)
    model.to(device)
    pinn = args.model == "pinn"
    loss_fn = nn.L1Loss() if args.loss == "mae" else nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, wait = np.inf, None, 0
    for ep in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, ld_tr, opt, loss_fn, True, pinn, qmode, device)
        es_loss = run_epoch(model, ld_es, opt, loss_fn, False, pinn, qmode, device)
        logger.info("epoch %d: train=%.3f early-stop=%.3f", ep, tr_loss, es_loss)
        if es_loss < best - 1e-4:
            best, best_state, wait = es_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 4:
                break
    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    val_pred = predict(model, ld_va, pinn, qmode, device) * y_std + y_mean
    if qmode:
        val_pred = np.sort(val_pred, axis=1)
        if args.target == "cloud":
            val_pred = np.clip(val_pred, 0, 100)
    mva = meta[m_va].reset_index(drop=True)
    yva_full = y[m_va]
    if args.smoke:
        mva = mva.iloc[:len(val_pred)].reset_index(drop=True)
        yva_full = yva_full[:len(val_pred)]
    mva["y"] = yva_full
    if qmode:
        for i, t in enumerate(TAUS):
            mva[f"q{t:g}"] = val_pred[:, i]
    else:
        mva["pred"] = val_pred
    pred = predict(model, ld_te, pinn, qmode, device) * y_std + y_mean
    cross = float(tv.crossing_rate(pred)) if qmode else 0.0
    if qmode:
        pred = np.sort(pred, axis=1)
        if args.target == "cloud":
            pred = np.clip(pred, 0, 100)
    yte = y[m_te]
    mte = meta[m_te].reset_index(drop=True)
    if qmode:
        mte["y"] = yte
        for i, t in enumerate(TAUS):
            mte[f"q{t:g}"] = pred[:, i]
    else:
        mte["y"], mte["pred"] = yte, pred
    vn = MODEL_VNUM[args.model]
    if not qmode:
        raise SystemExit("点损失协议已退役：请加 --quantile 运行分位数版本。")
    out = (CODE_ROOT / "reports" / "03_modeling" / f"v{vn}_{args.model}"
           / args.budget / "quantile" / args.target)
    out.mkdir(parents=True, exist_ok=True)
    mte.to_csv(out / "test_predictions.csv", index=False)
    mva.to_csv(out / "val_predictions.csv", index=False)
    rows = []
    if qmode:
        qcols = [f"q{t:g}" for t in TAUS]
        for group, keys in (("all", [None]), ("lead_time", sorted(mte.lead_time.unique())),
                            ("station_id", sorted(mte.station_id.unique())),
                            ("season", sorted(mte.season.unique()))):
            for k in keys:
                g = mte if k is None else mte[mte[group] == k]
                qm = g[qcols].to_numpy()
                rec = dict(group=group, key="all" if k is None else k, n=len(g))
                rec["mean_pinball"] = tv.mean_pinball(g.y.to_numpy(), qm, TAUS)
                rec["crps_q7_trunc"] = tv.crps_quantile(g.y.to_numpy(), qm, TAUS)
                rec["crossing_rate"] = cross
                for nom, lo, hi in ((0.5, 0.25, 0.75), (0.8, 0.10, 0.90), (0.9, 0.05, 0.95)):
                    rec[f"coverage_{int(nom*100)}"] = float(
                        np.mean((g.y >= g[f"q{lo:g}"]) & (g.y <= g[f"q{hi:g}"])))
                    rec[f"width_{int(nom*100)}"] = float(np.mean(g[f"q{hi:g}"] - g[f"q{lo:g}"]))
                rec["mae"] = float(np.mean(np.abs(g.y - g["q0.5"])))
                rec["rmse"] = float(np.sqrt(np.mean((g.y - g["q0.5"]) ** 2)))
                rows.append(rec)
    else:
        for group, keys in (("all", [None]), ("lead_time", sorted(mte.lead_time.unique())),
                            ("station_id", sorted(mte.station_id.unique())),
                            ("season", sorted(mte.season.unique()))):
            for k in keys:
                sub = mte if k is None else mte[mte[group] == k]
                rows.append(dict(group=group, key="all" if k is None else k,
                                 **metrics(sub.y, sub.pred)))
    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)
    ov = res[res.group == "all"].iloc[0]
    lines = [f"# v{vn} {args.model} ({args.target}, {'quantile' if qmode else args.loss})", "",
             f"- test=2025-09-01~2026-08-31 (full-year); n={int(ov.n)}"]
    if qmode:
        lines.append(f"- mean_pinball={ov.mean_pinball:.3f}; crps_q7_trunc={ov.crps_q7_trunc:.3f}; "
                     f"crossing={ov.crossing_rate:.4f}; cov50={ov.coverage_50:.3f} "
                     f"cov80={ov.coverage_80:.3f} cov90={ov.coverage_90:.3f}")
        lines.append(f"- q50 MAE={ov.mae:.3f} RMSE={ov.rmse:.3f}")
    else:
        lines.append(f"- MAE={ov.mae:.3f} RMSE={ov.rmse:.3f} Bias={ov.bias:.3f} R2={ov.r2:.3f}")
    lines.append(f"- seq_len={SEQ_LEN}; early-stop best {best:.3f}")
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("done -> %s", out)


if __name__ == "__main__":
    main()
