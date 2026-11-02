"""计数指标的阈值/ NMS 敏感性分析。

为什么需要这个脚本
----------------
tools/eval_full.py 用「conf=0.25 + NMS iou=0.7」数框，然后算 MAE。
但甘蓝是密集小目标（单图 31~89 株），相邻植株的检测框大量重叠在
IoU 0.5~0.7 之间 —— 这些"近重复框"不会被 iou=0.7 的 NMS 抑制，
于是同一株被数了两次，MAE 被系统性地抬高。结果是**同一个模型，只
要换一个阈值，MAE 就能差一倍**，用固定阈值下的 MAE 排名并不可靠。

做法
----
1. 每张图只做一次前向（conf=0.01, iou=0.9，几乎不过滤），把框和分数存下来；
2. 事后网格扫描 (conf, iou)，每次只做「按分数过滤 + NMS + 计数」；
   这样 6×3 = 18 组配置只需要 1 次推理；
3. 在 **val** 上选出让各模型平均 MAE 最小的那一组配置（不在 test 上选，避免信息泄漏）；
4. 用这组配置在 **test** 上报告所有模型的计数指标。

跑法：
    python tools/count_sweep.py \
        --models base_v2:runs/detect/train_base_v2/weights/best.pt \
                cbam_v2:runs/detect/train_cbam_v2/weights/best.pt

输出：
    eval_reports/count_sweep_val.csv      val 上 18 组配置的 MAE
    eval_reports/count_sweep_test.md      test 上选定配置的对比表
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torchvision.ops import nms  # noqa: E402

from tools.cbam_module import register  # noqa: E402

REPORT_DIR = Path("eval_reports")
CONFS = [0.25, 0.35, 0.45, 0.55, 0.65]
IOUS = [0.5, 0.6, 0.7]


def load_split(data_yaml: str, split: str):
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    img_dir = Path(cfg[split])
    label_dir = Path(str(img_dir).replace("images", "labels"))
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    gts = []
    for img in images:
        lab = label_dir / (img.stem + ".txt")
        n = 0
        if lab.exists():
            n = len([ln for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()])
        gts.append(n)
    return images, np.array(gts, dtype=float)


def collect_boxes(model, images, imgsz: int = 640, batch: int = 8):
    """一次前向，保存每张图的所有候选框（几乎不过滤）。"""
    per_image = []
    for i in range(0, len(images), batch):
        chunk = [str(p) for p in images[i : i + batch]]
        results = model.predict(chunk, conf=0.01, iou=0.9, imgsz=imgsz, verbose=False)
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                per_image.append((torch.zeros((0, 4)), torch.zeros((0,))))
            else:
                per_image.append((b.xyxy.detach().cpu().float(), b.conf.detach().cpu().float()))
    return per_image


def count_with(per_image, conf: float, iou: float) -> np.ndarray:
    counts = []
    for boxes, scores in per_image:
        keep = scores >= conf
        bx, sc = boxes[keep], scores[keep]
        if len(bx) == 0:
            counts.append(0)
            continue
        idx = nms(bx, sc, iou)
        counts.append(int(len(idx)))
    return np.array(counts, dtype=float)


def mae(counts: np.ndarray, gts: np.ndarray) -> float:
    return float(np.abs(counts - gts).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="标签:权重 形式，例如 base_v2:runs/detect/train_base_v2/weights/best.pt",
    )
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    register(verbose=False)
    from ultralytics import YOLO

    models = []
    for spec in args.models:
        label, w = spec.split(":", 1)
        models.append((label, w))

    REPORT_DIR.mkdir(exist_ok=True)

    # ---------- 第一步：val 上选配置 ----------
    val_imgs, val_gts = load_split(args.data, "val")
    print(f"[val] {len(val_imgs)} 张，真实框 {int(val_gts.sum())}")

    cache_val = {}
    for label, w in models:
        m = YOLO(w)
        cache_val[label] = collect_boxes(m, val_imgs, args.imgsz)
        print(f"[val] {label} 推理完成")

    grid_rows = []
    for conf in CONFS:
        for iou in IOUS:
            row = {"conf": conf, "iou": iou}
            maes = []
            for label, _ in models:
                v = mae(count_with(cache_val[label], conf, iou), val_gts)
                row[label] = round(v, 3)
                maes.append(v)
            row["avg_MAE"] = round(float(np.mean(maes)), 3)
            grid_rows.append(row)

    with open(REPORT_DIR / "count_sweep_val.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_rows)

    best = min(grid_rows, key=lambda r: r["avg_MAE"])
    conf_best, iou_best = best["conf"], best["iou"]
    print(f"\n[val] 平均 MAE 最小的配置: conf={conf_best} iou={iou_best} (avg MAE={best['avg_MAE']})")
    print("      说明：配置在 val 上选，下面的 test 数字才是可报告的。")

    # ---------- 第二步：test 上用选定配置报告 ----------
    test_imgs, test_gts = load_split(args.data, "test")
    print(f"\n[test] {len(test_imgs)} 张，真实框 {int(test_gts.sum())}")

    lines = [
        f"| 模型 | MAE(个/图) | RMSE | MAPE | 偏差 | 预测总数 | 真实总数 |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = []
    for label, w in models:
        m = YOLO(w)
        boxes = collect_boxes(m, test_imgs, args.imgsz)
        c = count_with(boxes, conf_best, iou_best)
        diff = c - test_gts
        rec = {
            "model": label,
            "MAE": round(float(np.abs(diff).mean()), 3),
            "RMSE": round(float(np.sqrt((diff**2).mean())), 3),
            "MAPE": round(float((np.abs(diff) / np.maximum(test_gts, 1)).mean() * 100), 2),
            "bias": round(float(diff.mean()), 3),
            "pred": int(c.sum()),
            "gt": int(test_gts.sum()),
        }
        rows.append(rec)
        lines.append(
            f"| {label} | {rec['MAE']} | {rec['RMSE']} | {rec['MAPE']}% | {rec['bias']:+.2f} "
            f"| {rec['pred']} | {rec['gt']} |"
        )
        print(f"[test] {label}: MAE={rec['MAE']} MAPE={rec['MAPE']}% pred={rec['pred']}")

    table = "\n".join(lines)
    header = (
        f"# 计数指标（conf={conf_best}, NMS iou={iou_best}，配置在 val 上选定）\n\n"
        f"测试集 {len(test_imgs)} 张，真实框 {int(test_gts.sum())} 个。\n\n"
    )
    (REPORT_DIR / "count_sweep_test.md").write_text(header + table + "\n", encoding="utf-8")

    # 给 make_report.py 用的结构化结果
    import json

    (REPORT_DIR / "count_sweep_test.json").write_text(
        json.dumps(
            {
                "config": {"conf": conf_best, "iou": iou_best, "selected_on": "val"},
                "models": {r["model"]: r for r in rows},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + table)
    print(f"\n[done] {REPORT_DIR / 'count_sweep_val.csv'} / {REPORT_DIR / 'count_sweep_test.md'}")


if __name__ == "__main__":
    main()
