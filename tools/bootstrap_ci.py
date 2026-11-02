"""对 test 集做 bootstrap 重采样，量化「75 张图」本身的抽样误差。

为什么需要
----------
test 只有 75 张图，mAP 是在这个样本上算出来的。换一批同样大小的图，结果会变多少？
不量化这个，就无法判断「A 比 B 高 1.2pp」到底是模型差异还是这批图没抽好。
这是与「训练随机性（种子方差）」相互独立的另一个噪声来源。

做法
----
1. 用 DetectionValidator 在 test 上跑一遍，**batch=1**，这样 self.stats 里每个元素
   恰好对应一张图的 (tp, conf, pred_cls, target_cls)，共 75 份
2. 从这 75 份里有放回抽 75 份，拼接后交给 ultralytics 官方的 ap_per_class 重算
3. 重复 N 次，取 2.5% / 97.5% 分位数即 95% 置信区间

注意
----
重采样单位是「图」，不是「框」。所以每次抽到的图数恒为 75，但总框数会波动
（单图 31~89 个目标，框数本来就不均匀）。这正是我们要模拟的真实抽样过程。

用法
----
    python tools/bootstrap_ci.py --weights runs/detect/train_base_v2/weights/best.pt --name base_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPORT_DIR = Path("eval_reports")


def collect_per_image_stats(
    weights: str, data: str, split: str, device: str, imgsz: int, conf: float, iou: float
) -> tuple[dict[str, list[np.ndarray]], int]:
    """跑一次 test 验证，返回按图分开的统计量。"""
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect import DetectionValidator

    args = get_cfg(
        overrides={
            "model": weights,
            "data": data,
            "split": split,
            "imgsz": imgsz,
            "batch": 1,  # 必须为 1，否则一个 batch 里多张图混在一起
            "device": device,
            "workers": 0,
            "plots": False,
            "verbose": False,
            "conf": conf,
            "iou": iou,
            "save_json": False,
            # 必须与 model.val() 的默认一致，否则结果对不上：
            # rect=True → mAP50-95 0.7910；rect=False → 0.7774（差 1.36pp）
            "rect": True,
            "mode": "val",
        }
    )
    model = YOLO(weights)
    validator = DetectionValidator(args=args)

    # per-image 统计量累积在 validator.metrics.stats；但 get_stats() 结尾会 clear_stats()，
    # 所以必须在它被调用前先拷一份。这里包一层 get_stats 做截获。
    captured: dict[str, dict] = {}
    orig_get_stats = validator.get_stats

    def _spy_get_stats():
        captured["stats"] = {k: [np.asarray(x) for x in v] for k, v in validator.metrics.stats.items()}
        return orig_get_stats()

    validator.get_stats = _spy_get_stats
    validator(model=model.model)

    stats = captured.get("stats")
    if not stats:
        raise RuntimeError("没有截获到 per-image 统计量，无法做 bootstrap")
    n = len(stats["tp"])
    if n == 0:
        raise RuntimeError("统计量为空，无法做 bootstrap")

    out = stats
    # 一致性检查：每张图都应有 conf 与 pred_cls
    for k in ("tp", "conf", "pred_cls", "target_cls"):
        if len(out[k]) != n:
            raise RuntimeError(f"stats['{k}'] 长度 {len(out[k])} != {n}")
    return out, n


def metrics_on_subset(stats: dict[str, list[np.ndarray]], idx: np.ndarray) -> tuple[float, float] | None:
    """在给定的图索引子集上重算 mAP50 与 mAP50-95。"""
    from ultralytics.utils.metrics import ap_per_class

    tp = np.concatenate([stats["tp"][i] for i in idx], 0)
    conf = np.concatenate([stats["conf"][i] for i in idx], 0)
    pred_cls = np.concatenate([stats["pred_cls"][i] for i in idx], 0)
    target_cls = np.concatenate([stats["target_cls"][i] for i in idx], 0)
    if len(conf) == 0 or len(target_cls) == 0:
        return None

    out = ap_per_class(tp, conf, pred_cls, target_cls, plot=False, names={0: "cabbage"})
    ap = out[5]  # (n_classes, 10)，10 = IoU 0.50:0.95
    return float(ap[:, 0].mean()), float(ap.mean())


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--weights", required=True)
    ap_.add_argument("--name", required=True, help="输出文件名，写 eval_reports/bootstrap_<name>.json")
    ap_.add_argument("--data", default="data.yaml")
    ap_.add_argument("--split", default="test")
    ap_.add_argument("--imgsz", type=int, default=640)
    ap_.add_argument("--device", default="0")
    ap_.add_argument("--conf", type=float, default=0.001, help="mAP 计算用低阈值，与 ultralytics val 默认一致")
    ap_.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap_.add_argument("--n", type=int, default=1000, help="bootstrap 次数")
    ap_.add_argument("--seed", type=int, default=0)
    args = ap_.parse_args()

    stats, n_img = collect_per_image_stats(
        args.weights, args.data, args.split, args.device, args.imgsz, args.conf, args.iou
    )
    n_box = int(sum(len(x) for x in stats["target_cls"]))
    print(f"[ok] {args.name}: {n_img} 张图 / {n_box} 个真实框")

    point = metrics_on_subset(stats, np.arange(n_img))
    if point is None:
        raise RuntimeError("全量重算失败：预测或标注为空")

    rng = np.random.default_rng(args.seed)
    boot = []
    for _ in range(args.n):
        idx = rng.integers(0, n_img, n_img)
        r = metrics_on_subset(stats, idx)
        if r is not None:
            boot.append(r)

    arr = np.array(boot)  # (n_boot, 2)：mAP50, mAP50-95
    lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
    mid = np.percentile(arr, 50, axis=0)
    std = arr.std(axis=0, ddof=1)

    report = {
        "name": args.name,
        "weights": str(Path(args.weights).resolve()),
        "split": args.split,
        "n_images": n_img,
        "n_boxes": n_box,
        "n_boot": len(boot),
        "conf_nms_iou": [args.conf, args.iou],
        "point": {"mAP50": point[0], "mAP50_95": point[1]},
        "bootstrap": {
            "mAP50": {"mean": float(arr[:, 0].mean()), "median": float(mid[0]),
                      "std": float(std[0]), "ci95": [float(lo[0]), float(hi[0])]},
            "mAP50_95": {"mean": float(arr[:, 1].mean()), "median": float(mid[1]),
                         "std": float(std[1]), "ci95": [float(lo[1]), float(hi[1])]},
        },
    }
    REPORT_DIR.mkdir(exist_ok=True)
    out_json = REPORT_DIR / f"bootstrap_{args.name}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    half = (hi[1] - lo[1]) / 2
    md = [
        f"# {args.name}：test 集 bootstrap 置信区间",
        "",
        f"- 样本：{n_img} 张图 / {n_box} 个真实框",
        f"- 重采样：{len(boot)} 次，单位是「图」（有放回抽 {n_img} 张）",
        f"- 计算口径：conf={args.conf}、NMS iou={args.iou}，与 ultralytics val 一致",
        "",
        "| 指标 | 点估计 | bootstrap 均值 | 标准差 | 95% CI | 半宽 |",
        "|---|---|---|---|---|---|",
        f"| mAP50 | {point[0]:.4f} | {arr[:, 0].mean():.4f} | {std[0]:.4f} | "
        f"[{lo[0]:.4f}, {hi[0]:.4f}] | ±{(hi[0] - lo[0]) / 2:.4f} |",
        f"| mAP50-95 | {point[1]:.4f} | {arr[:, 1].mean():.4f} | {std[1]:.4f} | "
        f"[{lo[1]:.4f}, {hi[1]:.4f}] | ±{half:.4f} |",
        "",
        "**怎么用**：两个模型在这批 test 上的 mAP50-95 差异，如果小于上表的半宽（"
        f"{half * 100:.2f}pp），就不能认为是模型差异，可能是这 75 张图没抽好。",
    ]
    (REPORT_DIR / f"bootstrap_{args.name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (REPORT_DIR / f"bootstrap_{args.name}_samples.npy").write_bytes(arr.tobytes())
    print("\n".join(md))
    print(f"\n[done] {out_json}")


if __name__ == "__main__":
    main()
