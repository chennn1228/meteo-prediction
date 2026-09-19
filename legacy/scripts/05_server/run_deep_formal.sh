#!/bin/bash
# 说明：正式结构子队列（Autoformer/Informer/FEDformer/PatchTST/TimesNet/PINN + 其余 v2–v15）。
# 主流程请用 run_server_full.sh（含五折 CV 与 v1）；本脚本用于断点续跑。
echo "BLOCKED: legacy two-target deep queue; see scripts/05_server/README.md"
exit 2
set -u
cd /root/autodl-tmp/meteo_prediction || exit 1
PY=.venv/bin/python
LOGD=reports/03_modeling/00_logs/server
mkdir -p "$LOGD"
models=(mlp lstm cnn tcn transformer autoformer informer fedformer itransformer patchtst dlinear timesnet tsmixer pinn)
declare -A VNUM=( [mlp]=2 [lstm]=3 [cnn]=4 [tcn]=5 [transformer]=6 [autoformer]=7 [informer]=8 [fedformer]=9 [itransformer]=10 [patchtst]=11 [dlinear]=12 [timesnet]=13 [tsmixer]=14 [pinn]=15 )
formal=(autoformer informer fedformer patchtst timesnet pinn)
for t in ghi cloud; do
  for m in "${models[@]}"; do
    v=${VNUM[$m]}
    out="reports/03_modeling/v${v}_${m}/full/quantile/${t}"
    log="$LOGD/formal_${t}_${m}.log"
    if [ -f "$out/summary.md" ] && [ -f "$LOGD/${t}_${m}.done" ]; then echo "SKIP" >> "$log"; continue; fi
    impl=lite
    seq=168
    for f in "${formal[@]}"; do [ "$m" = "$f" ] && impl=formal; done
    [ "$impl" = "formal" ] && seq=168
    echo "START $t $m impl=$impl seq=$seq $(date)" >> "$log"
    "$PY" src/s03_models/train/train_deep.py --target "$t" --model "$m" --quantile \
      --impl "$impl" --seq-len "$seq" --budget full --epochs 50 --subsample 1 \
      --val-subsample 1 --workers 4 --device cuda --seed 0 >> "$log" 2>&1
    echo "EXIT $? $t $m $(date)" >> "$log"
    [ -f "$out/summary.md" ] && : > "$LOGD/${t}_${m}.done"
    if [ "$t" = "cloud" ]; then clip=0,100; else clip=0,2000; fi
    if [ -f "$out/val_predictions.csv" ] && [ -f "$out/test_predictions.csv" ]; then
      "$PY" src/s04_evaluation/calibration/calibrate_quantiles.py --val "$out/val_predictions.csv" \
        --test "$out/test_predictions.csv" --out "$out/calibrated" --clip "$clip" \
        >> "$log" 2>&1 || true
    fi
  done
done
echo "FORMAL DEEP DONE $(date)" > "$LOGD/all.done"
