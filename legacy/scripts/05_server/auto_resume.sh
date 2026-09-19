#!/bin/bash
# 自动续跑 → 跑完打包摘要 → 等本地下载确认 → 自动关机。
# 开机自启：/etc/autodl.sh 调用本脚本；取消：pkill -f auto_resume.sh
# 参数：$1 = 等待本地下载确认的最长小时数（默认 6）
echo "BLOCKED: auto-resume targets a retired/missing v1 queue; see scripts/05_server/README.md"
exit 2
cd /root/autodl-tmp/meteo_prediction 2>/dev/null || exit 1
LOGD=reports/03_modeling/00_logs/server
LOG=$LOGD/auto_resume.log
mkdir -p "$LOGD"
WAITH=${1:-6}
echo "$(date '+%F %T') auto_resume start" >> "$LOG"

while true; do
  if [ -f "$LOGD/all.done" ] && grep -q "ALL DONE" "$LOGD/all.done"; then
    break
  fi
  # 队列已在跑（例如本脚本被重启）→ 只等待，不再起第二个队列
  if pgrep -f 'run_server_ful[l]_v2' >/dev/null; then
    sleep 60
    continue
  fi
  bash scripts/05_server/run_server_full_v2.sh >> "$LOG" 2>&1
  if [ -f "$LOGD/all.done" ] && grep -q "ALL DONE" "$LOGD/all.done"; then
    break
  fi
  echo "$(date '+%F %T') queue exited without ALL DONE -> retry in 60s" >> "$LOG"
  sleep 60
done

echo "$(date '+%F %T') ALL DONE -> packing summary results" >> "$LOG"
tar czf /root/autodl-tmp/meteo_results_summary.tar.gz \
  reports/03_modeling/*/full/quantile/*/summary.md \
  reports/03_modeling/*/full/quantile/*/results.csv \
  reports/03_modeling/*/full/quantile/*/calibrated/results_calibrated.csv \
  reports/03_modeling/v1_ml/full/quantile/*/*_summary.md \
  reports/02_experiment/cv/month_balanced/*/selection.json \
  figs reports/03_modeling/00_logs/server/preflight 2>/dev/null
echo "$(date '+%F %T') packed -> /root/autodl-tmp/meteo_results_summary.tar.gz" >> "$LOG"
sync

# 等本地下载确认（本地脚本下载完成后会 touch /root/autodl-tmp/DOWNLOAD_OK）
deadline=$(( $(date +%s) + WAITH * 3600 ))
echo "$(date '+%F %T') waiting for DOWNLOAD_OK (max ${WAITH}h)" >> "$LOG"
while [ ! -f /root/autodl-tmp/DOWNLOAD_OK ]; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "$(date '+%F %T') download confirmation timeout -> shutdown anyway" >> "$LOG"
    break
  fi
  sleep 60
done

echo "$(date '+%F %T') download confirmed -> shutdown now" >> "$LOG"
shutdown now 2>>"$LOG" || shutdown -h now 2>>"$LOG" || poweroff 2>>"$LOG"
echo "$(date '+%F %T') shutdown command failed; retry in 30min" >> "$LOG"
sleep 1800
exit 1
