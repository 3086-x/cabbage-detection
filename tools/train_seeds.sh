#!/usr/bin/env bash
# 种子 run 的串行训练驱动。可重入：已完成的 run 会被跳过，崩溃的 run 会重试一次。
#
# 为什么不用一条 && 命令链
# ------------------------
# 之前用 `run_one A && run_one B && ...` 跑，结果第 3 个 run 因内存耗尽崩溃，
# 整条链随之中断，剩下 9 个 run 一个都没开始，而且无法知道中断点在哪。
# 现在每个 run 单独判断状态，中断后重跑本脚本即可从断点继续。
#
# 用法
#    bash tools/train_seeds.sh            # 跑完剩下没做的
#    bash tools/train_seeds.sh --dry-run  # 只看会跑哪些
set -u
cd "$(dirname "$0")/.."

PY=".venv/Scripts/python.exe"
STATUS="logs/seed_train_status.txt"
DRY=${1:-}
mkdir -p logs

# 训练完成的判据：results.csv 比 50 个 epoch 多一行表头
is_done() {
  local n
  n=$( (wc -l < "runs/detect/$1/results.csv") 2>/dev/null || echo 0 )
  [ "$n" -ge 51 ]
}

run_one() {
  local nm=$1; shift
  if is_done "$nm"; then
    echo "[skip] $nm 已完成"
    return 0
  fi
  if [ "$DRY" = "--dry-run" ]; then
    echo "[dry ] $PY $* --name $nm"
    return 0
  fi
  local attempt
  for attempt in 1 2; do
    echo "=== $nm start $(date +%H:%M:%S) attempt=$attempt ===" | tee -a "$STATUS"
    if "$PY" "$@" --name "$nm" > "logs/$nm.log" 2>&1; then
      echo "=== $nm done  $(date +%H:%M:%S) ===" | tee -a "$STATUS"
      return 0
    fi
    echo "!!! $nm failed attempt=$attempt $(date +%H:%M:%S)" | tee -a "$STATUS"
    # 崩溃留下的半成品必须挪走：best.pt 在第 1 轮就会写，留在原地会污染后续评估
    if [ -d "runs/detect/$nm" ]; then
      mv "runs/detect/$nm" "runs/detect/${nm}_crashed_$(date +%H%M%S)" 2>/dev/null
    fi
    sleep 5
  done
  echo "XXX $nm GAVE UP" | tee -a "$STATUS"
  return 1
}

# ---- 6 个配置 × seed 1/2 ----------------------------------------------------
# 顺序按「结论价值」排：YOLOv8n 是唯一可能过线的，CBAM@P3 次之
run_one train_yolov8n_s1       tools/train_baseline.py --model yolov8n.pt --seed 1
run_one train_yolov8n_s2       tools/train_baseline.py --model yolov8n.pt --seed 2
run_one train_cbam_p3_s1       tools/train_cbam_fixed.py --variant plain --insert-index 16 --seed 1 --workers 0
run_one train_cbam_p3_s2       tools/train_cbam_fixed.py --variant plain --insert-index 16 --seed 2 --workers 0
run_one train_base_mixup_s1    tools/train_baseline.py --mixup 0.15 --seed 1
run_one train_base_mixup_s2    tools/train_baseline.py --mixup 0.15 --seed 2
run_one train_cbam_res_s1      tools/train_cbam_fixed.py --variant residual --seed 1 --workers 0
run_one train_cbam_res_s2      tools/train_cbam_fixed.py --variant residual --seed 2 --workers 0
run_one train_cbam_nomosaic_s1 tools/train_cbam_fixed.py --variant plain --mosaic 0.0 --seed 1 --workers 0
run_one train_cbam_nomosaic_s2 tools/train_cbam_fixed.py --variant plain --mosaic 0.0 --seed 2 --workers 0
run_one train_ca_nomosaic_s1   tools/train_cbam_fixed.py --variant ca --mosaic 0.0 --seed 1 --workers 0
run_one train_ca_nomosaic_s2   tools/train_cbam_fixed.py --variant ca --mosaic 0.0 --seed 2 --workers 0

echo "ALL_DONE $(date +%H:%M:%S)" | tee -a "$STATUS"
