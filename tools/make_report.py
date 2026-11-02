"""把评估结果汇总成对比表和图表（最终版）。

输入：
    eval_reports/{base_v2,cbam_v2,cbam_res_v2,cbam_p3}.json   detection 指标（tools/eval_full.py）
    eval_reports/count_sweep_test.json                        计数指标（tools/count_sweep.py，阈值在 val 上选）
    eval_reports/count_sweep_val.csv                          val 上的阈值敏感性网格
    eval_reports/speed.csv                                    推理耗时（tools/bench_speed.py）

输出：
    eval_reports/compare.md              主对比表，可直接贴进 README
    eval_reports/compare_real.png        四联图：mAP50 / mAP50-95 / 计数MAE / 推理耗时
    eval_reports/count_sensitivity.png   计数误差随 conf 阈值变化的曲线

跑法：
    python tools/make_report.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "eval_reports"

# (文件名, 图例, 颜色) —— 顺序决定表格与图表顺序
# 前 4 个 = 复现原论文的「三种注意力模块」消融（基线 / SE / CBAM / CA，均插 SPPF 后）
# 后 2 个 = 本工作新增的「注入形式」消融（残差化、改插 P3）
# 最后 4 个 = 复现原论文的「数据增强消融」B 组与「增强 × 注意力」交叉实验 D 组
ORDER = [
    ("base_v2", "YOLOv11n 基线", "#9aa5b1"),
    ("se", "+SE", "#d4a017"),
    ("cbam_v2", "+CBAM", "#e8833a"),
    ("ca", "+CA", "#2e86c1"),
    ("cbam_res_v2", "残差CBAM@SPPF", "#4c9f70"),
    ("cbam_p3", "CBAM@P3", "#9b59b6"),
    ("base_nomosaic", "基线·无Mosaic", "#6b7280"),
    ("base_mixup", "基线·加MixUp(0.15)", "#8d6e63"),
    ("cbam_nomosaic", "CBAM·无Mosaic", "#f0a05a"),
    ("ca_nomosaic", "CA·无Mosaic", "#5dade2"),
    ("yolov8n", "YOLOv8n", "#34495e"),
    ("rtdetr_legacy", "RT-DETR-L（20ep／跨环境）", "#7f8c8d"),
]

# 表格分组：标题 -> 参与的行（都含基线，便于看相对变化）
GROUPS = [
    ("C 组｜注意力模块消融（SPPF 后，完整增强）", ["base_v2", "se", "cbam_v2", "ca"]),
    ("注入形式扩展（本工作新增）", ["base_v2", "cbam_res_v2", "cbam_p3"]),
    ("B 组｜数据增强消融（基线架构）", ["base_v2", "base_nomosaic", "base_mixup"]),
    ("D 组｜增强 × 注意力 交叉实验", ["base_v2", "cbam_v2", "ca", "base_nomosaic", "cbam_nomosaic", "ca_nomosaic"]),
    ("A 组｜模型对比（RT-DETR 为跨环境 20ep，不可与 YOLO 直接比较）", ["base_v2", "yolov8n", "rtdetr_legacy"]),
]

# 四联图只画注意力相关的 6 组，避免 10 根柱子过密
CHART_KEYS = ["base_v2", "se", "cbam_v2", "ca", "cbam_res_v2", "cbam_p3"]


def load_det(name: str) -> dict | None:
    p = REPORT_DIR / f"{name}.json"
    if not p.exists():
        print(f"[skip] 缺少 {p.name}")
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "mAP50": d["detection"]["mAP50"],
        "mAP50_95": d["detection"]["mAP50_95"],
        "P": d["detection"]["precision"],
        "R": d["detection"]["recall"],
    }


def load_count() -> tuple[dict, dict]:
    p = REPORT_DIR / "count_sweep_test.json"
    if not p.exists():
        return {}, {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["models"], d["config"]


def load_speed() -> dict:
    p = REPORT_DIR / "speed.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    return {r["model"]: {"params_M": float(r["params_M"]), "ms": float(r["ms_fused"])} for r in rows}


def main() -> None:
    dets = {k: load_det(k) for k, _, _ in ORDER}
    dets = {k: v for k, v in dets.items() if v}
    counts, cfg = load_count()
    speeds = load_speed()
    if not dets:
        raise SystemExit("没有可用的评估结果")

    rows = []
    for key, label, _ in ORDER:
        if key not in dets:
            continue
        d = dets[key]
        c = counts.get(key, {})
        s = speeds.get(key, {})
        rows.append(
            {
                "key": key,
                "label": label,
                "mAP50": d["mAP50"],
                "mAP50_95": d["mAP50_95"],
                "P": d["P"],
                "R": d["R"],
                "MAE": c.get("MAE"),
                "MAPE": c.get("MAPE"),
                "bias": c.get("bias"),
                "ms": s.get("ms"),
                "params": s.get("params_M"),
            }
        )

    by_key = {r["key"]: r for r in rows}
    base = by_key.get("base_v2", rows[0])

    # ---------------- 主表（按实验分组） ----------------
    header = [
        "| 模型 | mAP50 | mAP50-95 | P | R | 计数MAE(个/图) | 计数MAPE | 偏差 | 推理(ms/图) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def _row(r: dict) -> str:
        mae = f"{r['MAE']:.2f}" if r["MAE"] is not None else "—"
        mape = f"{r['MAPE']:.2f}%" if r["MAPE"] is not None else "—"
        bias = f"{r['bias']:+.2f}" if r["bias"] is not None else "—"
        ms = f"{r['ms']:.1f}" if r["ms"] is not None else "—"
        return (
            f"| {r['label']} | {r['mAP50']:.4f} | {r['mAP50_95']:.4f} | {r['P']:.4f} | {r['R']:.4f} "
            f"| {mae} | {mape} | {bias} | {ms} |"
        )

    lines: list[str] = []
    for title, keys in GROUPS:
        picked = [by_key[k] for k in keys if k in by_key]
        if not picked:
            continue
        lines += [f"**{title}**", "", *header, *[_row(r) for r in picked], ""]

    lines.append(
        f"- 测试集 75 张图、3765 个真实框（单类 cabbage，单图 31~89 个目标）\n"
        f"- 计数指标阈值 conf={cfg.get('conf')} / NMS iou={cfg.get('iou')}，**在 val 上选定**，避免在 test 上调参\n"
        f"- 推理耗时：RTX 5060 Laptop 8GB，batch=1、640×640、模型 fuse 后，100 次平均；**只能同批次内比较**\n"
        f"- 注意：笔记本 GPU 频率漂移会让绝对时延在 7.8~12.0ms 间波动（同一模型跨批次实测差 35%），\n"
        f"  因此「谁比谁快几毫秒」不构成结论，只能看**同一次测量内的相对关系**\n"
        f"- **本表全部是 seed 0 单次结果**。差异是否可信见 `noise_budget.md`：\n"
        f"  10 个对比里只有 **YOLOv8n（−2.30pp，余量 4.3×）** 与 **加 MixUp（−1.06pp，余量 1.5×）**\n"
        f"  过线，其余全部不可区分；`+CBAM` 的 0.7695 是它三个种子里最差的一次\n"
        f"  （三种子 0.7826 ± 0.0116）"
    )
    table = "\n".join(lines)
    (REPORT_DIR / "compare.md").write_text(table + "\n", encoding="utf-8")
    print(table)

    # ---------------- 四联图（只看注意力相关的 6 组） ----------------
    chart_rows = [r for r in rows if r["key"] in CHART_KEYS]
    label_map = {k: lab for k, lab, _ in ORDER}
    color_map = {k: c for k, _, c in ORDER}
    labels = [label_map[r["key"]].replace("@", "\n@") for r in chart_rows]
    colors = [color_map[r["key"]] for r in chart_rows]

    panels = [
        ("mAP50", "mAP50（y 轴已放大：SE/CBAM/CA 与基线极差仅 0.19pp，75 张上不可区分）", "%.4f", None),
        ("mAP50_95", "mAP50-95（定位精度，区分度更大）", "%.4f", None),
        ("MAE", f"计数 MAE 个/图（conf={cfg.get('conf')}, iou={cfg.get('iou')}）", "%.2f", True),
        ("ms", "推理耗时 ms/图（fuse 后 batch=1；6 组均在 9.7~10.8ms，±1ms 内属测量噪声）", "%.1f", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=150, constrained_layout=True)
    for ax, (field, title, fmt, lower_better) in zip(axes.ravel(), panels):
        vals = [r[field] for r in chart_rows]
        if any(v is None for v in vals):
            ax.axis("off")
            continue
        bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.6 + hi * 0.02
        ax.set_ylim(max(0, lo - pad), hi + pad)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt % v, ha="center", va="bottom", fontsize=10, fontweight="bold")
        # 标出相对基线的变化（用轴比例坐标，避免落在 ylim 之外把画布撑开）
        for b, v in zip(bars, vals):
            if v is base[field] or base[field] in (None, 0):
                continue
            delta = (v - base[field]) / base[field] * 100
            better = (delta < 0) if lower_better else (delta > 0)
            ax.text(
                b.get_x() + b.get_width() / 2,
                0.12,
                f"{delta:+.2f}%",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                color="#0a7d33" if better else "#c0392b",
                fontweight="bold",
            )
        ax.tick_params(labelsize=9)

    fig.suptitle("甘蓝检测：注意力模块与注入方式消融（测试集 75 张 / 3765 个真实框）", fontsize=14, fontweight="bold")
    out = REPORT_DIR / "compare_real.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"\n[done] 图表: {out}")
    print(f"[done] 表格: {REPORT_DIR / 'compare.md'}")

    # ---------------- 增强 × 注意力 双因素交互图（B 组 + D 组） ----------------
    aug_cats = ["完整增强\nmosaic=1, mixup=0", "无 Mosaic\nmosaic=0", "加 MixUp\nmixup=0.15"]
    aug_series = [
        ("基线", ["base_v2", "base_nomosaic", "base_mixup"], "#9aa5b1"),
        ("+CBAM", ["cbam_v2", "cbam_nomosaic", None], "#e8833a"),
        ("+CA", ["ca", "ca_nomosaic", None], "#2e86c1"),
    ]
    aug_panels = [
        ("mAP50_95", "mAP50-95（越高越好）", "%.4f"),
        ("MAE", "计数 MAE（个/图，越低越好）", "%.2f"),
        ("ms", "推理耗时 ms/图（fuse 后）", "%.1f"),
    ]
    if all(k in by_key for k in ["base_nomosaic", "cbam_nomosaic", "ca_nomosaic"]):
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), dpi=170, constrained_layout=True)
        import numpy as np

        n_ser = len(aug_series)
        width = 0.8 / n_ser
        xpos = np.arange(len(aug_cats))
        for ax, (field, title, fmt) in zip(axes, aug_panels):
            all_vals = []
            for i, (sname, keys, color) in enumerate(aug_series):
                vals = [by_key[k][field] if k in by_key and by_key[k].get(field) is not None else None for k in keys]
                all_vals += [v for v in vals if v is not None]
                bars = ax.bar(
                    xpos + (i - (n_ser - 1) / 2) * width,
                    [v if v is not None else 0 for v in vals],
                    width=width * 0.9,
                    label=sname,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                for b, v in zip(bars, vals):
                    if v is None:
                        # 用轴相对坐标：该面板 y 轴不从 0 开始，画在数据 y=0 会把画布撑高
                        ax.text(
                            b.get_x() + b.get_width() / 2,
                            0.02,
                            "未跑",
                            transform=ax.get_xaxis_transform(),
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color="#aaa",
                        )
                        continue
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        v,
                        fmt % v,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )
            if all_vals:
                lo, hi = min(all_vals), max(all_vals)
                pad = (hi - lo) * 0.9 + hi * 0.004
                ax.set_ylim(max(0, lo - pad * 0.8), hi + pad * 1.6)
            ax.set_xticks(xpos)
            ax.set_xticklabels(aug_cats, fontsize=9)
            ax.set_title(title, fontsize=11.5, fontweight="bold")
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)
        handles, tags = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            tags,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            fontsize=10.5,
            frameon=False,
        )
        fig.suptitle(
            "数据增强 × 注意力模块 双因素实验（测试集 75 张 / 3765 个真实框；MixUp 组只跑了基线）",
            fontsize=13,
            fontweight="bold",
        )
        out3 = REPORT_DIR / "aug_cross.png"
        fig.savefig(out3, bbox_inches="tight")
        print(f"[done] 增强交互图: {out3}")

    # ---------------- 阈值敏感性曲线 ----------------
    grid_path = REPORT_DIR / "count_sweep_val.csv"
    if grid_path.exists():
        grid = list(csv.DictReader(grid_path.open(encoding="utf-8")))
        confs = sorted({float(r["conf"]) for r in grid})
        labels_map = {k: lab for k, lab, _ in ORDER}
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        for key, lab, color in ORDER:
            ys = []
            for c in confs:
                hit = [r for r in grid if float(r["conf"]) == c and float(r["iou"]) == 0.5]
                ys.append(float(hit[0][key]) if hit and key in hit[0] else None)
            if any(y is None for y in ys):
                continue
            ax.plot(confs, ys, marker="o", label=lab, color=color, linewidth=2)
        ax.set_xlabel("置信度阈值 conf", fontsize=11)
        ax.set_ylabel("val 集计数 MAE（个/图）", fontsize=11)
        ax.set_title("计数误差对置信度阈值高度敏感（NMS iou=0.5）", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out2 = REPORT_DIR / "count_sensitivity.png"
        fig.savefig(out2, bbox_inches="tight")
        print(f"[done] 敏感性曲线: {out2}")


if __name__ == "__main__":
    main()
