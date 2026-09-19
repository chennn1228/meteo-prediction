#!/bin/bash
# 训练看板：bash scripts/05_server/watch.sh          # 看一次
#           bash scripts/05_server/watch.sh 30       # 每 30 秒刷新
cd /root/autodl-tmp/meteo_prediction 2>/dev/null || { echo "项目目录不存在"; exit 1; }
LOGD=reports/03_modeling/00_logs/server
INT=${1:-0}

render() {
  echo "===== $(date '+%F %T') ====="
  echo "-- 队列进程 --"
  pgrep -af 'run_server_full|train_deep|train_quantile_v1|cv_month_balanced' \
    | grep -v pgrep | head -3 || echo "(未在运行)"
  echo "-- 完成度 --"
  fin=$(ls reports/03_modeling/v*_*/full/quantile/*/summary.md 2>/dev/null | wc -l)
  cal=$(ls reports/03_modeling/v*_*/full/quantile/*/calibrated/results_calibrated.csv 2>/dev/null | wc -l)
  v1=$(ls reports/03_modeling/v1_ml/full/quantile/*/per_lead_results.csv 2>/dev/null | wc -l)
  echo "v1_ml 目标完成：$v1 / 2   |   深度模型完成：$fin / 28   |   已校准：$cal / 28"
  for t in ghi cloud; do
    s="reports/02_experiment/cv/month_balanced/$t/selection.json"
    [ -f "$s" ] && echo "CV($t)：完成"
  done
  [ -f "$LOGD/all.done" ] && echo "完成标记：$(cat "$LOGD/all.done")"
  echo "-- 最近日志（倒数 3 行）--"
  cur=$(ls -t "$LOGD"/cv_*.log "$LOGD"/v1_*.log "$LOGD"/deep_*.log "$LOGD"/preflight/*.log 2>/dev/null | head -1)
  if [ -n "$cur" ]; then
    ep=$(grep -c 'epoch' "$cur" 2>/dev/null | tr -d ' ')
    echo "[$cur]"
    echo "epoch 已完成：${ep:-0}   最后写入：$(date -r "$cur" '+%H:%M:%S' 2>/dev/null)"
    grep 'epoch' "$cur" 2>/dev/null | tail -2
    tail -2 "$cur"
  else
    echo "(暂无日志)"
  fi
  echo "-- GPU --"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
  echo "-- 磁盘 --"
  df -h /root/autodl-tmp | tail -1
}

if [ "$INT" -gt 0 ] 2>/dev/null; then
  while true; do clear; render; sleep "$INT"; done
else
  render
fi
