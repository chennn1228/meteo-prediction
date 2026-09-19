#!/bin/bash
# 克隆机自检：确认环境、代码与训练所需数据是否齐备。
# 用法（项目根目录）：bash scripts/05_server/check_server.sh
set -u
cd /root/autodl-tmp/meteo_prediction 2>/dev/null || { echo "[FAIL] 项目目录不存在"; exit 1; }

echo "== 主机 =="
whoami; hostname; date
echo
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "[WARN] nvidia-smi 不可用"
echo
echo "== 虚拟环境 =="
if [ -x .venv/bin/python ]; then
  .venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  .venv/bin/python -c "import lightgbm, xgboost, pandas, sklearn; print('lightgbm', lightgbm.__version__, '| xgboost', xgboost.__version__, '| pandas', pandas.__version__)"
else
  echo "[FAIL] .venv/bin/python 不存在"
fi
echo
echo "== 代码版本 =="
for f in src/s02_experiment/split_protocol.py src/s03_models/train/train_deep.py \
         src/s04_evaluation/verification/verification_robustness.py \
         scripts/05_server/run_server_full.sh; do
  [ -f "$f" ] && echo "[OK]   $f" || echo "[MISS] $f"
done
[ -d src/s05_legacy ] && echo "[WARN] 仍存在旧目录 src/s05_legacy（应删除）"
[ -d reports/99_archive ] && echo "[WARN] 仍存在旧目录 reports/99_archive（应删除）"
echo
echo "== 训练数据（featured 2024-02_2026-09） =="
n=$(ls data/03_featured/*_featured_2024-02_2026-09.parquet 2>/dev/null | wc -l)
echo "文件数：$n / 20"
if [ "$n" -eq 20 ]; then
  du -sh data/03_featured | awk '{print "总大小：" $1}'
  md5sum data/03_featured/*_featured_2024-02_2026-09.parquet | sort -k2
else
  echo "缺失清单："
  for s in changzhou_1 huai_an_1 huai_an_2 huai_an_3 lianyungang_1 lianyungang_2 \
           nanjing_1 nanjing_2 nantong_1 nantong_2 suqian_1 suzhou_1 taizhou_1 taizhou_2 \
           wuxi_1 xuzhou_1 xuzhou_2 yancheng_1 yancheng_2 yancheng_3; do
    p="data/03_featured/${s}_featured_2024-02_2026-09.parquet"
    [ -f "$p" ] || echo "  [MISS] $p"
  done
fi
echo
echo "== 审计证据（可选） =="
ls data/03_featured/*_featured_2019-02_2026-01.parquet 2>/dev/null | wc -l | awk '{print "2019 特征表文件数：" $1 "/20"}'
echo
echo "== 旧目录命名（仅当需要在服务器上重跑 clean/features 时才重要） =="
for d in 01_gfs 02_era5 03_satellite previous_runs era5 satellite; do
  [ -d "data/01_raw/$d" ] && echo "  存在 data/01_raw/$d"
done
echo
echo "自检完成。"
