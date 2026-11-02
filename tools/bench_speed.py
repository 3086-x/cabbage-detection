"""推理速度与参数量基准。

为什么需要单独测
------------
ultralytics 训练日志里的 "ms inference" 是验证阶段的单张耗时，受显存、
batch、预处理干扰，不同 run 之间不可比。要论证「加注意力的代价」，必须
用同一段代码、同一块卡、同样的 batch 和迭代次数去测。

做法：预热 10 次后用 CUDA event 计时 100 次前向，取平均值（不含后处理）。

跑法：
    python tools/bench_speed.py --models base_v2:runs/detect/train_base_v2/weights/best.pt

输出：
    eval_reports/speed.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from tools.cbam_module import register  # noqa: E402


def bench(model, imgsz: int = 640, iters: int = 100, warmup: int = 10, device: str = "cuda") -> tuple[float, int]:
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    model.model.to(device).eval()
    with torch.no_grad():
        for _ in range(warmup):
            model.model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            model.model(x)
        end.record()
        torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    n_params = sum(p.numel() for p in model.model.parameters())
    return ms, n_params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="标签:权重")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    register(verbose=False)
    from ultralytics import YOLO

    rows = []
    for spec in args.models:
        label, w = spec.split(":", 1)
        m = YOLO(w)
        ms, n_params = bench(m, args.imgsz, args.iters, device=args.device)
        # 融合后（conv+BN 合并）的真实部署速度，通常比训练态更快
        try:
            m.model.fuse()
            ms_fused, _ = bench(m, args.imgsz, args.iters, device=args.device)
        except Exception:  # noqa: BLE001
            ms_fused = float("nan")
        rows.append({"model": label, "params_M": round(n_params / 1e6, 4), "ms": round(ms, 3), "ms_fused": round(ms_fused, 3)})
        print(f"{label:14s} params={n_params/1e6:.4f}M  {ms:.2f} ms/图  (fuse 后 {ms_fused:.2f} ms)")

    out = Path("eval_reports/speed.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "params_M", "ms", "ms_fused"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
