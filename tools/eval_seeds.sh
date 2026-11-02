#!/usr/bin/env bash
# 对补跑的种子 run 逐个做 test 评估。可重入：已有 json 的跳过。
#
# 重要：**不要在训练还在跑的时候执行本脚本**。
# 两边同时跑会让 16G 内存被打爆 —— 实测过一次，训练在第 46 epoch 因
# OpenCV `Insufficient memory` 崩溃，白丢约 10 分钟的算力。
# 训练 vs 评估必须串行。
set -u
cd "$(dirname "$0")/.."

PY=".venv/Scripts/python.exe"
mkdir -p eval_reports

RUNS=(
  train_yolov8n_s1 train_yolov8n_s2
  train_cbam_p3_s1 train_cbam_p3_s2
  train_base_mixup_s1 train_base_mixup_s2
  train_cbam_res_s1 train_cbam_res_s2
  train_cbam_nomosaic_s1 train_cbam_nomosaic_s2
  train_ca_nomosaic_s1 train_ca_nomosaic_s2
)

fail=0
for d in "${RUNS[@]}"; do
  if [ -f "eval_reports/$d.json" ]; then
    echo "[skip] $d 已有评估"
    continue
  fi
  w="runs/detect/$d/weights/best.pt"
  if [ ! -f "$w" ]; then
    echo "!!! $d 缺少 best.pt，跳过"
    fail=1
    continue
  fi
  echo "=== $d eval start $(date +%H:%M:%S) ==="
  if "$PY" tools/eval_full.py --weights "$w" --split test --name "$d" > "logs/eval_$d.log" 2>&1; then
    echo "=== $d eval done  $(date +%H:%M:%S) ==="
  else
    echo "!!! $d eval FAILED，见 logs/eval_$d.log"
    tail -3 "logs/eval_$d.log"
    fail=1
  fi
done

echo "EVAL_DONE $(date +%H:%M:%S) fail=$fail"
