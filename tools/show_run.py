"""汇总一次训练的结果轨迹（从 results.csv 读，不重训）。

跑法：
    python tools/show_run.py runs/detect/train_cbam
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def col(row: dict, name: str) -> float:
    for k, v in row.items():
        if k.strip() == name:
            return float(v)
    return 0.0


def summarize(run_dir: str) -> None:
    path = Path(run_dir) / "results.csv"
    if not path.exists():
        print(f"找不到 {path}")
        return
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    best = max(rows, key=lambda r: col(r, "metrics/mAP50-95(B)"))
    print(f"== {run_dir} ==")
    print(f"轮数 {len(rows)}")
    print(
        "最好轮 ep%d: mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f"
        % (
            col(best, "epoch"),
            col(best, "metrics/mAP50(B)"),
            col(best, "metrics/mAP50-95(B)"),
            col(best, "metrics/precision(B)"),
            col(best, "metrics/recall(B)"),
        )
    )
    last = rows[-1]
    print(
        "最后一轮: mAP50=%.4f  mAP50-95=%.4f"
        % (col(last, "metrics/mAP50(B)"), col(last, "metrics/mAP50-95(B)"))
    )
    picks = [0, 4, 9, 19, 29, 39, len(rows) - 1]
    for i in dict.fromkeys(picks):
        if 0 <= i < len(rows):
            r = rows[i]
            print("  ep%-3d mAP50=%.4f  mAP50-95=%.4f" % (col(r, "epoch"), col(r, "metrics/mAP50(B)"), col(r, "metrics/mAP50-95(B)")))


if __name__ == "__main__":
    for d in sys.argv[1:] or ["runs/detect/train_cbam"]:
        summarize(d)
        print()
