"""在同一环境里重训一次纯 YOLOv11n 基线。

为什么需要它
----------
历史基线（runs/detect/train-7）是 ultralytics 8.4.37 + torch 2.5.1 + GTX 1650 跑的，
而新训的 CBAM 模型是 8.4.194 + torch 2.11 + RTX 5060。环境不同直接比数字不严谨
（新版本的默认增强、优化器细节都可能有变化）。

所以要比较「加 CBAM 到底有没有用」，必须让基线用**同一份代码、同一套超参、
同一台机器**再跑一次。

跑法：
    python tools/train_baseline.py --epochs 50 --batch 8 --device 0 --name train_base_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--name", default="train_base_v2")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--mosaic", type=float, default=None, help="覆盖 mosaic 概率，0.0 = 无 Mosaic（实验 B）")
    ap.add_argument("--mixup", type=float, default=None, help="覆盖 mixup 概率，0.0 = 无 MixUp（实验 B）")
    ap.add_argument(
        "--optimizer",
        default="SGD",
        help="优化器。YOLO 系列用 SGD 与原实验对齐；RT-DETR 需传 auto（内部会切 AdamW + 更小 lr）",
    )
    ap.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="全局随机种子（实验 E：种子方差）。trainer 内部实际用的是 seed+1+RANK，单卡即 seed+1",
    )
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="确定性算子开关。None = 沿用 ultralytics 默认（本版本 8.4.194 默认 True）",
    )
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    params = sum(p.numel() for p in model.model.parameters())
    print(f"[ok] 基线模型参数量: {params / 1e6:.4f} M")

    aug = {}
    if args.mosaic is not None:
        aug["mosaic"] = args.mosaic
    if args.mixup is not None:
        aug["mixup"] = args.mixup
    if aug:
        print(f"[ok] 增强覆盖: {aug}")

    # deterministic 传 None 时不写进 train()，沿用 ultralytics 默认（8.4.194 = True）
    extra = {}
    if args.deterministic is not None:
        extra["deterministic"] = args.deterministic
    print(f"[ok] seed={args.seed}（trainer 内部用 {args.seed + 1}）deterministic={args.deterministic or '默认'}")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=0.937,
        weight_decay=0.0005,
        seed=args.seed,
        name=args.name,
        exist_ok=True,
        plots=True,
        **aug,
        **extra,
    )

    save_dir = Path(getattr(model.trainer, "save_dir", f"runs/detect/{args.name}"))
    print(f"\n[done] 权重: {(save_dir / 'weights/best.pt').resolve()}")


if __name__ == "__main__":
    main()
