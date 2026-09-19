#!/bin/bash
set -u
echo "BLOCKED: retired v1 server queue. Do not generate formal results with month-balanced CV or cloud as a co-equal target."
echo "Resume only after the v2 feature refetch, equal-budget deep nested tuner, and hourly Himawari truth gate pass; see scripts/05_server/README.md."
exit 2
cd /root/autodl-tmp/meteo_prediction || exit 1
PY=.venv/bin/python
LOGD=reports/03_modeling/00_logs/server
mkdir -p "$LOGD"

# 旧产物清场：克隆实例可能带有旧编码/旧协议结果，未带 v3 协议戳时直接删除，
# 避免下面的“已存在则跳过”逻辑把过期结果留在正式目录。
if [ ! -f "$LOGD/.v3_cleared" ]; then
  for d in reports/03_modeling/v*_*; do
    [ -d "$d" ] || continue
    rm -rf "$d"
    echo "REMOVED stale results: $d"
  done
  : > "$LOGD/.v3_cleared"
fi

# 0) 月分层五折调参：正式超参来源（selection.json）
for t in ghi cloud; do
  sel="reports/02_experiment/cv/month_balanced/$t/selection.json"
  ok=$( [ -f "$sel" ] && "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(1 if (int(d.get('n_folds',0))==5 and not d.get('smoke')) else 0)" "$sel" 2>/dev/null || echo 0 )
  if [ "$ok" != "1" ]; then
    echo "CV START $t $(date)" >> "$LOGD/cv_${t}.log"
    "$PY" src/s02_experiment/cv_month_balanced_quantile.py --target "$t" \
      >> "$LOGD/cv_${t}.log" 2>&1
    echo "CV EXIT $? $t $(date)" >> "$LOGD/cv_${t}.log"
  fi
done

# 1) v1_ml quantile: GHI + cloud (two background jobs)
for t in ghi cloud; do
  if [ ! -f "$LOGD/v1_${t}.done" ]; then
    nohup "$PY" src/s03_models/train/train_quantile_v1.py --target "$t" \
      > "$LOGD/v1_${t}.log" 2>&1 &
    echo $! > "$LOGD/v1_${t}.pid"
  fi
done
wait
for t in ghi cloud; do
  if [ -f "reports/03_modeling/v1_ml/full/quantile/$t/per_lead_results.csv" ]; then
    : > "$LOGD/v1_${t}.done"
  fi
done

# 1.5) 预检（幂等）：6 个正式结构在 GPU 上各跑 1 个 smoke；任一失败立即停止，避免半路翻车
PRE="$LOGD/preflight"
mkdir -p "$PRE"
for m in autoformer informer fedformer patchtst timesnet pinn; do
  [ -f "$PRE/$m.ok" ] && continue
  b=1024
  case "$m" in patchtst|timesnet) b=512;; esac
  "$PY" src/s03_models/train/train_deep.py --target ghi --model "$m" --quantile \
    --impl formal --seq-len 168 --budget lite --epochs 1 --smoke --batch "$b" \
    --workers 2 --device cuda --seed 0 > "$PRE/$m.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "PREFLIGHT FAIL $m $(date)" >> "$PRE/fail.log"
    echo "PREFLIGHT FAILED: $m（详见 $PRE/$m.log），已停止，未开始正式训练" > "$LOGD/all.done"
    exit 1
  fi
  echo "PREFLIGHT OK $m $(date)" > "$PRE/$m.ok"
done
# 只清理预检产生的 lite 目录，保留已完成的 full 结果（可断点续跑）
for d in reports/03_modeling/v*_*; do
  [ -d "$d/lite" ] && rm -rf "$d/lite"
done

# 2) v2-v15 deep quantile, full budget, GPU, sequential
models=(mlp lstm cnn tcn transformer autoformer informer fedformer itransformer patchtst dlinear timesnet tsmixer pinn)
declare -A VNUM=( [mlp]=2 [lstm]=3 [cnn]=4 [tcn]=5 [transformer]=6 [autoformer]=7 [informer]=8 [fedformer]=9 [itransformer]=10 [patchtst]=11 [dlinear]=12 [timesnet]=13 [tsmixer]=14 [pinn]=15 )
formal=(autoformer informer fedformer patchtst timesnet pinn)
for t in ghi cloud; do
  for m in "${models[@]}"; do
    v=${VNUM[$m]}
    sum="reports/03_modeling/v${v}_${m}/full/quantile/${t}/summary.md"
    if [ -f "$sum" ] && [ -f "$LOGD/deep_${t}_${m}.done" ]; then
      echo "SKIP $t $m" >> "$LOGD/deep_${t}_${m}.log"
      continue
    fi
    impl=lite
    seq=168
    b=1024
    for f in "${formal[@]}"; do [ "$m" = "$f" ] && impl=formal; done
    [ "$impl" = "formal" ] && seq=168
    case "$m" in patchtst|timesnet) b=512;; esac
    echo "START $t $m impl=$impl seq=$seq batch=$b $(date)" >> "$LOGD/deep_${t}_${m}.log"
    "$PY" src/s03_models/train/train_deep.py --target "$t" --model "$m" --quantile \
      --impl "$impl" --seq-len "$seq" --batch "$b" --budget full --epochs 50 \
      --subsample 1 --val-subsample 1 \
      --workers 4 --device cuda --seed 0 >> "$LOGD/deep_${t}_${m}.log" 2>&1
    echo "EXIT $? $t $m $(date)" >> "$LOGD/deep_${t}_${m}.log"
    [ -f "$sum" ] && : > "$LOGD/deep_${t}_${m}.done"
    # 校准：conformal 分位修正（val=校准折预测，test=整年测试预测）
    if [ "$t" = "cloud" ]; then clip=0,100; else clip=0,2000; fi
    if [ -f "reports/03_modeling/v${v}_${m}/full/quantile/${t}/val_predictions.csv" ] && \
       [ -f "reports/03_modeling/v${v}_${m}/full/quantile/${t}/test_predictions.csv" ]; then
      "$PY" src/s04_evaluation/calibration/calibrate_quantiles.py \
        --val "reports/03_modeling/v${v}_${m}/full/quantile/${t}/val_predictions.csv" \
        --test "reports/03_modeling/v${v}_${m}/full/quantile/${t}/test_predictions.csv" \
        --out "reports/03_modeling/v${v}_${m}/full/quantile/${t}/calibrated" \
        --clip "$clip" >> "$LOGD/deep_${t}_${m}.log" 2>&1 || true
    fi
  done
done

# 3) figures
"$PY" src/s04_evaluation/analysis/plot_quantile_results.py > "$LOGD/fig_v1.log" 2>&1 || true
"$PY" src/s04_evaluation/analysis/plot_deep_quantile.py > "$LOGD/fig_deep.log" 2>&1 || true
echo "ALL DONE $(date)" > "$LOGD/all.done"
