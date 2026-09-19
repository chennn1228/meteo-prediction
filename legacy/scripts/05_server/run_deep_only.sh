#!/bin/bash
echo "BLOCKED: legacy two-target deep queue; see scripts/05_server/README.md"
exit 2
set -u
cd /root/autodl-tmp/meteo_prediction || exit 1
PY=.venv/bin/python
LOGD=reports/03_modeling/00_logs/server
mkdir -p "$LOGD"
models=(mlp lstm cnn tcn transformer autoformer informer fedformer itransformer patchtst dlinear timesnet tsmixer pinn)
declare -A VNUM=( [mlp]=2 [lstm]=3 [cnn]=4 [tcn]=5 [transformer]=6 [autoformer]=7 [informer]=8 [fedformer]=9 [itransformer]=10 [patchtst]=11 [dlinear]=12 [timesnet]=13 [tsmixer]=14 [pinn]=15 )
for t in ghi cloud; do
  for m in "${models[@]}"; do
    v=${VNUM[$m]}
    sum="reports/03_modeling/v${v}_${m}/full/quantile/${t}/summary.md"
    log="$LOGD/deep_${t}_${m}.log"
    if [ -f "$sum" ]; then echo "SKIP" >> "$log"; continue; fi
    echo "START $t $m $(date)" >> "$log"
    "$PY" src/s03_models/train/train_deep.py --target "$t" --model "$m" --quantile \
      --budget full --epochs 50 --subsample 1 --val-subsample 1 \
      --workers 4 --device cuda --seed 0 >> "$log" 2>&1
    echo "EXIT $? $t $m $(date)" >> "$log"
  done
done
while pgrep -f train_quantile_v1.py > /dev/null; do sleep 60; done
"$PY" src/s04_evaluation/analysis/plot_quantile_results.py > "$LOGD/fig_v1.log" 2>&1 || true
"$PY" src/s04_evaluation/analysis/plot_deep_quantile.py > "$LOGD/fig_deep.log" 2>&1 || true
echo "DEEP ALL DONE $(date)" > "$LOGD/all.done"
