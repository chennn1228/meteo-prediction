#!/bin/bash
# 队列跑完后自动关机（独立监工进程，不影响正在跑的队列）
#   启动：setsid nohup bash scripts/05_server/auto_shutdown.sh 600 >/dev/null 2>&1 &
#   取消：pkill -f auto_shutdown.sh
# 参数：$1 = 完成后等待秒数（默认 600，留时间给你看结果）；$2 = 最长等待小时（默认 96）
echo "BLOCKED: shutdown watcher is disabled while the formal v2 queue is unavailable; see scripts/05_server/README.md"
exit 2
cd /root/autodl-tmp/meteo_prediction 2>/dev/null || exit 1
LOGD=reports/03_modeling/00_logs/server
LOG=$LOGD/auto_shutdown.log
DELAY=${1:-600}
MAXH=${2:-96}
echo "$(date '+%F %T') watcher start  delay=${DELAY}s  max=${MAXH}h" >> "$LOG"
deadline=$(( $(date +%s) + MAXH * 3600 ))

while true; do
  # 1) 成功完成：打包摘要结果并关机
  if [ -f "$LOGD/all.done" ] && grep -q "ALL DONE" "$LOGD/all.done"; then
    echo "$(date '+%F %T') ALL DONE detected -> waiting ${DELAY}s" >> "$LOG"
    sleep "$DELAY"
    tar czf /root/autodl-tmp/meteo_results_summary.tar.gz \
      reports/03_modeling/*/full/quantile/*/summary.md \
      reports/03_modeling/*/full/quantile/*/results.csv \
      reports/03_modeling/*/full/quantile/*/calibrated/results_calibrated.csv \
      reports/03_modeling/v1_ml/full/quantile/*/*_summary.md \
      reports/02_experiment/cv/month_balanced/*/selection.json \
      figs reports/03_modeling/00_logs/server/preflight 2>/dev/null
    echo "$(date '+%F %T') results packed -> /root/autodl-tmp/meteo_results_summary.tar.gz" >> "$LOG"
    sync
    echo "$(date '+%F %T') shutdown now" >> "$LOG"
    shutdown now 2>>"$LOG" || shutdown -h now 2>>"$LOG" || poweroff 2>>"$LOG"
    exit 0
  fi
  # 2) 队列挂了但没写成功标记：不关机，保留现场
  if ! pgrep -f 'run_server_ful[l]_v2' >/dev/null; then
    sleep 60
    if ! pgrep -f 'run_server_ful[l]_v2' >/dev/null; then
      echo "$(date '+%F %T') queue stopped without ALL DONE -> NO shutdown（请查日志）" >> "$LOG"
      exit 1
    fi
  fi
  # 3) 超时保护
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "$(date '+%F %T') max wait reached -> NO shutdown" >> "$LOG"
    exit 1
  fi
  sleep 60
done
