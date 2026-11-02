"""全量评估：不只看 mAP，还算农业计数真正关心的指标。

为什么需要这个
------------
mAP50 已经在 96.4% 饱和了 —— 不管加 SE、CBAM、CA、开不开 Mosaic，都在 96.4~96.5 之间，
差异完全被噪声吃掉。这不是模型没改进，是**指标失效了**：mAP50 只问"框到了没"，
而这个任务真正要的是"数对没"和"框准没"。

所以补三类指标：
  1) 定位精度：mAP50-95 / AP_small（小目标更严格）
  2) 计数精度：MAE / RMSE / MAPE（农业巡检的直接业务指标）
  3) 分密度表现：按每图甘蓝数量分稀疏/中等/密集三档，看模型在哪档掉链子

跑法
----
    python tools/eval_full.py --weights runs/detect/train-7/weights/best.pt --data data.yaml
    python tools/eval_full.py --weights runs/detect/train_cbam/weights/best.pt --data data.yaml

产物
----
    eval_report.json        机器可读
    eval_report.md          可直接贴进 README / 简历附件的表格
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _load_data_yaml(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg


def _count_gt(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    lines = [ln for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return len(lines)


def _label_dir_for(img_path: Path) -> Path:
    """把 .../images/xxx.jpg 换成 .../labels/xxx.txt"""
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--conf", type=float, default=0.25, help="计数时使用的置信度阈值")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--rect",
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "验证时的矩形推理开关。auto=沿用 ultralytics 默认（rect=True，会把 640 补齐 stride 变成 672）；"
            "false=强制 640x640。评估固定输入尺寸的 ONNX 时必须用 false，否则 ONNX 会被二次缩放、mAP 假性下跌。"
        ),
    )
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    import sys

    from ultralytics import YOLO

    # 加载带 CBAM 的权重前必须把 CBAM 注册进 ultralytics 命名空间，
    # 否则反序列化时找不到 tools.cbam_module.CBAM
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.cbam_module import register

    register(verbose=False)

    data = _load_data_yaml(args.data)
    img_dir = Path(data[args.split])
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not images:
        raise SystemExit(f"没找到图片: {img_dir}")

    model = YOLO(args.weights)

    # ---- 1) 标准检测指标（交给 ultralytics）----
    val_kwargs = {"data": args.data, "split": args.split, "imgsz": args.imgsz, "verbose": False}
    if args.rect != "auto":
        val_kwargs["rect"] = args.rect == "true"
    val = model.val(**val_kwargs)
    det = {
        "mAP50": float(val.box.map50),
        "mAP50_95": float(val.box.map),
        "precision": float(val.box.mp),
        "recall": float(val.box.mr),
    }
    try:  # AP_small 需要 ultralytics 提供 per-scale 指标
        maps = val.box.maps  # 每类的 mAP50-95
        det["AP_per_class"] = [float(x) for x in maps]
    except Exception:  # noqa: BLE001
        pass

    # ---- 2) 计数指标 ----
    rows = []
    for img in images:
        gt = _count_gt(_label_dir_for(img))
        pred = model.predict(str(img), conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
        n_pred = int(len(pred[0].boxes)) if pred and pred[0].boxes is not None else 0
        rows.append({"image": img.name, "gt": gt, "pred": n_pred, "diff": n_pred - gt})

    gts = np.array([r["gt"] for r in rows], dtype=float)
    preds = np.array([r["pred"] for r in rows], dtype=float)
    diff = preds - gts

    abs_err = np.abs(diff)
    counting = {
        "images": len(rows),
        "gt_total": int(gts.sum()),
        "pred_total": int(preds.sum()),
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt((diff ** 2).mean())),
        "MAPE": float((abs_err / np.maximum(gts, 1)).mean() * 100),
        "bias": float(diff.mean()),
        "exact_match_rate": float((diff == 0).mean() * 100),
        "within_3_rate": float((abs_err <= 3).mean() * 100),
    }

    # ---- 3) 分密度 ----
    if len(gts) > 0:
        q1, q2 = np.percentile(gts, [33, 66])
    else:
        q1 = q2 = 0
    groups = {
        "sparse": gts <= q1,
        "medium": (gts > q1) & (gts <= q2),
        "dense": gts > q2,
    }
    by_density = {}
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        by_density[name] = {
            "images": int(mask.sum()),
            "avg_gt": float(gts[mask].mean()),
            "MAE": float(abs_err[mask].mean()),
            "RMSE": float(np.sqrt((diff[mask] ** 2).mean())),
            "bias": float(diff[mask].mean()),
        }

    report = {
        "weights": str(Path(args.weights).resolve()),
        "split": args.split,
        "conf": args.conf,
        "rect": args.rect,
        "detection": det,
        "counting": counting,
        "by_density": by_density,
    }

    tag = args.name or Path(args.weights).parent.parent.name
    # 按 name 分开存，否则多模型评估会互相覆盖（tools/make_report.py 按 {tag}.json 读取）
    out_dir = Path("eval_reports")
    out_dir.mkdir(exist_ok=True)
    report = {"name": tag, **report}
    (out_dir / f"{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [f"## 评估报告：{tag}", "", "### 检测指标", "", "| 指标 | 数值 |", "|---|---|"]
    md += [
        f"| mAP50 | {det['mAP50']:.4f} |",
        f"| mAP50-95 | {det['mAP50_95']:.4f} |",
        f"| Precision | {det['precision']:.4f} |",
        f"| Recall | {det['recall']:.4f} |",
        "",
        "### 计数指标（农业巡检核心）",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 图像数 | {counting['images']} |",
        f"| 真实总数 / 预测总数 | {counting['gt_total']} / {counting['pred_total']} |",
        f"| MAE（平均绝对误差） | {counting['MAE']:.2f} 个/图 |",
        f"| RMSE | {counting['RMSE']:.2f} |",
        f"| MAPE | {counting['MAPE']:.2f}% |",
        f"| 系统偏差 bias | {counting['bias']:+.2f}（正=多检，负=漏检） |",
        f"| 完全数对比例 | {counting['exact_match_rate']:.1f}% |",
        f"| 误差≤3个比例 | {counting['within_3_rate']:.1f}% |",
        "",
        "### 分密度表现",
        "",
        "| 密度档 | 图像数 | 平均真实数 | MAE | RMSE | 偏差 |",
        "|---|---|---|---|---|---|",
    ]
    for name, v in by_density.items():
        md.append(
            f"| {name} | {v['images']} | {v['avg_gt']:.0f} | {v['MAE']:.2f} | {v['RMSE']:.2f} | {v['bias']:+.2f} |"
        )

    (out_dir / f"{tag}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[done] 已写入 {out_dir / (tag + '.json')} / {out_dir / (tag + '.md')}")


if __name__ == "__main__":
    main()
