#!/bin/bash
# Deep-only fixed-kt rerun queue on new instance (v1 runs LOCALLY, not here).
echo "BLOCKED: legacy two-target sensitivity queue; see scripts/05_server/README.md"
exit 2
cd /root/autodl-tmp/meteo_prediction || exit 1
export PATH=/root/miniconda3/bin:$PATH
LOG=reports/03_modeling/00_logs/server/sens_top_fixedkt.log
mkdir -p reports/03_modeling/00_logs/server
echo "$(date '+%F %T') deep-only queue start" >> "$LOG"
for tgt in ghi cloud; do
  for m in tcn transformer informer autoformer; do
    echo "$(date '+%F %T') == deep $m $tgt ==" >> "$LOG"
    python src/s03_models/train/train_deep.py --target "$tgt" --model "$m" \
      --quantile --budget full --device cuda >> "$LOG" 2>&1
  done
done
echo "$(date '+%F %T') ALL_TOP_DONE" >> "$LOG"
