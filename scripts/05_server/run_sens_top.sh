#!/bin/bash
# Sensitivity/full-rerun of TOP models on FIXED-kt featured (new instance).
# Sequential queue: v1_ml (both targets) then top deep models (ghi then cloud).
echo "BLOCKED: legacy two-target sensitivity queue; see scripts/05_server/README.md"
exit 2
cd /root/autodl-tmp/meteo_prediction || exit 1
export PATH=/root/miniconda3/bin:$PATH
LOG=reports/03_modeling/00_logs/server/sens_top_fixedkt.log
mkdir -p reports/03_modeling/00_logs/server
echo "$(date '+%F %T') queue start" >> "$LOG"
for tgt in ghi cloud; do
  echo "$(date '+%F %T') == v1_ml $tgt ==" >> "$LOG"
  python src/s03_models/train/train_quantile_v1.py --target "$tgt" >> "$LOG" 2>&1
  for m in tcn transformer informer autoformer; do
    echo "$(date '+%F %T') == deep $m $tgt ==" >> "$LOG"
    python src/s03_models/train/train_deep.py --target "$tgt" --model "$m" \
      --quantile --budget full --device cuda >> "$LOG" 2>&1
  done
done
echo "$(date '+%F %T') ALL_TOP_DONE" >> "$LOG"
