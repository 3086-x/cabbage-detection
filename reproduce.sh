#!/usr/bin/env bash
# ==========================================================
# 一键复现入口
#
# 设计前提：数据集不随仓库分发（见 DATA.md）。
#   因此本脚本分两类用途：
#     A. 只有代码，想看结论 → stats 阶段即可，不需要数据/GPU
#     B. 有同构数据，想完整重跑 → env → train → eval → stats
#
# 用法：
#   bash reproduce.sh stats     # 从仓库内产物复算全部统计结论（无需数据）
#   bash reproduce.sh env       # 创建虚拟环境并装依赖
#   bash reproduce.sh train     # 训练 33 个 run（需数据 + GPU，数小时）
#   bash reproduce.sh eval      # 逐 run 评估 test 集（需 runs/ + 数据）
#   bash reproduce.sh all       # 全流程
# ==========================================================
set -euo pipefail

PY="${PY:-.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="python"

stage="${1:-stats}"

need_data() {
  if [ ! -d dataset/train/images ]; then
    echo "[!] 缺少 dataset/train/images。"
    echo "    数据不随仓库分发，请按 DATA.md 第 3 节自备同构数据。"
    exit 1
  fi
}

case "$stage" in
  env)
    echo "== 创建虚拟环境 =="
    python -m venv .venv
    .venv/Scripts/python.exe -m pip install -U pip
    .venv/Scripts/python.exe -m pip install -r requirements.txt
    echo "== 完成。PyTorch 请按本机 CUDA 单独安装，见 docs/ENVIRONMENT.md =="
    ;;

  train)
    need_data
    echo "== 训练 33 个 run（11 组 × 3 种子，串行）=="
    echo "   注意：训练与评估严禁并行，8G 显存会 OOM，见 docs/ENVIRONMENT.md"
    bash tools/train_seeds.sh
    ;;

  eval)
    need_data
    echo "== 逐 run 评估 test 集 =="
    bash tools/eval_seeds.sh
    ;;

  stats)
    echo "== 第 1 层：逐 run 的组内 σ 与配对差值 =="
    "$PY" tools/seed_variance.py --all

    echo "== 第 2 层：评估端配对 bootstrap（需数据+权重，缺失则跳过）=="
    if [ -d runs/detect ] && [ -d dataset/test/images ]; then
      "$PY" tools/paired_bootstrap.py --all || echo "[!] paired_bootstrap 失败，沿用仓库内已有结果"
    else
      echo "   跳过：无 runs/ 或 dataset/。仓库内 eval_reports/paired_bootstrap.json 为原始结果。"
    fi

    echo "== 第 3 层：噪声预算 / 判定门槛 =="
    "$PY" tools/noise_budget.py

    echo "== 配对一致性校验（需 runs/ 内的 args.yaml）=="
    "$PY" tools/check_pairing.py || echo "   跳过：无 runs/"

    echo "== 完成，结果见 eval_reports/ =="
    ;;

  all)
    need_data
    bash reproduce.sh env
    bash reproduce.sh train
    bash reproduce.sh eval
    bash reproduce.sh stats
    ;;

  *)
    echo "未知阶段: $stage"
    echo "可用: stats | env | train | eval | all"
    exit 1
    ;;
esac
