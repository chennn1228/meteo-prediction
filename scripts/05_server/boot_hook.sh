#!/bin/bash
# AutoDL 开机自动执行（由 /etc/autodl.sh 调用）：
# 若训练队列尚未完成，则自动续跑；完成后由 auto_resume.sh 打包摘要并关机。
P=/root/autodl-tmp/meteo_prediction/scripts/05_server/auto_resume.sh
cd /root/autodl-tmp/meteo_prediction 2>/dev/null || exit 0
[ -f "$P" ] || exit 0
if ! pgrep -f 'auto_resume.s[h]' >/dev/null; then
  mkdir -p reports/03_modeling/00_logs/server
  setsid nohup bash "$P" >> reports/03_modeling/00_logs/server/auto_resume_boot.log 2>&1 &
fi
