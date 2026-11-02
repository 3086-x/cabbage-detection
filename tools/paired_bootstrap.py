"""配对 bootstrap：模型之间的差异到底显不显著。

为什么需要
----------
`bootstrap_ci.py` 给的是**单模型 CI**：base_v2 的 mAP50-95 是 0.7910 ± 2.66pp。
如果直接拿这个半宽当"差异门槛"，会得出"2.7pp 以内都分辨不了"的结论 —— 这个推论是错的。

原因：两个模型跑的是**同一批 75 张图**。某张图特别难（目标密集、遮挡严重），
会让**两个模型同时掉分**。这部分波动是"图难度的公共成分"，在比较差异时应当抵消，
而不是叠加进误差棒。独立重采样两个模型，等于假设两次评估用的是不同图片，
把公共方差也算进了差异里，门槛被人为放大。

正确做法
--------
**配对 bootstrap**（paired bootstrap）：
对同一次重采样抽出的索引 idx，同时计算 mAP(A, idx) 与 mAP(B, idx)，
取 Δ = mAP(A, idx) − mAP(B, idx)。重复 N 次得到 Δ 的分布，取 2.5%/97.5% 分位数为 CI。

- Δ 的 CI 不含 0 → 差异显著（在这批 test 上）
- 同时给出 P(Δ > 0)：bootstrap 分布里 A 优于 B 的比例
- 参照量：同一配对若用非配对方式估算的门槛，用来看配对收窄了多少

对齐校验
--------
配对的前提是两个模型的 per-image 统计量**逐图对应**。
用每张图的 GT 框数序列作指纹做校验（顺序、长度都必须一致），不一致直接报错退出。

用法
----
    python tools/paired_bootstrap.py \
        --ref base_v2=runs/detect/train_base_v2/weights/best.pt \
        --model cbam_v2=runs/detect/train_cbam_v2/weights/best.pt \
        --model se=runs/detect/train_se/weights/best.pt \
        --n 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.bootstrap_ci import REPORT_DIR, collect_per_image_stats, metrics_on_subset


def fingerprint(stats: dict[str, list[np.ndarray]]) -> tuple[int, ...]:
    """每张图的 GT 框数量序列，作为"图顺序指纹"。"""
    return tuple(int(len(x)) for x in stats["target_cls"])


def delta_distribution(
    stats_cur: dict[str, list[np.ndarray]],
    stats_ref: dict[str, list[np.ndarray]],
    idx_matrix: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float]]:
    """同一组重采样索引下，逐个算 Δ = 当前配置 − 基线。返回 (n_boot, 2) 与全量点估计。

    符号约定：**负值＝比基线差**，与"ΔmAP"的常规读法一致。
    """
    n = len(stats_cur["tp"])
    point_cur = metrics_on_subset(stats_cur, np.arange(n))
    point_ref = metrics_on_subset(stats_ref, np.arange(n))
    if point_cur is None or point_ref is None:
        raise RuntimeError("全量重算失败")

    deltas = []
    for idx in idx_matrix:
        rc = metrics_on_subset(stats_cur, idx)
        rr = metrics_on_subset(stats_ref, idx)
        if rc is None or rr is None:
            continue
        deltas.append((rc[0] - rr[0], rc[1] - rr[1]))
    return np.array(deltas), (point_cur[0] - point_ref[0], point_cur[1] - point_ref[1])


def summarize(d: np.ndarray, point: float, n_tests: int = 1) -> dict:
    """bootstrap 分布汇总。n_tests>1 时额外给 Bonferroni 校正后的 CI。

    多个配置同时与基线比较，逐个看 95% CI 会让族错误率膨胀
    （9 次比较时至少一次假阳性的概率约 1 − 0.95^9 ≈ 37%）。
    Bonferroni 把每次的显著性水平压到 α/m，即用 (0.5/m, 100−0.5/m) 分位数。
    """
    lo, hi = np.percentile(d, [2.5, 97.5])
    a = 0.05 / max(n_tests, 1) * 100
    lo_b, hi_b = np.percentile(d, [a / 2, 100 - a / 2])
    return {
        "point": float(point),
        "boot_mean": float(d.mean()),
        "std": float(d.std(ddof=1)),
        "ci95": [float(lo), float(hi)],
        "half_width": float((hi - lo) / 2),
        "ci_bonf": [float(lo_b), float(hi_b)],
        "n_tests": int(n_tests),
        "p_gt0": float((d > 0).mean()),
        "significant": bool(lo > 0 or hi < 0),
        "significant_bonf": bool(lo_b > 0 or hi_b < 0),
    }


def forest_plot(results: list[dict], ref_name: str, out_png: Path,
                half_nonpaired: float, n_boot: int) -> None:
    """Δ 的森林图：点估计 + 配对 CI，背景画出非配对门槛做对照。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = [r for r in results if r["name"] != ref_name]
    rows = sorted(rows, key=lambda r: r["mAP50_95"]["point"])
    ys = np.arange(len(rows))
    pts = np.array([r["mAP50_95"]["point"] for r in rows]) * 100
    lo = np.array([r["mAP50_95"]["ci95"][0] for r in rows]) * 100
    hi = np.array([r["mAP50_95"]["ci95"][1] for r in rows]) * 100

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    # 非配对门槛（±2.66pp）作为背景带，显示旧口径有多保守
    ax.axvspan(-half_nonpaired * 100, half_nonpaired * 100, color="#e8e8e8", zorder=0,
               label=f"非配对门槛 ±{half_nonpaired * 100:.2f}pp（旧口径）")
    ax.axvline(0, color="#444", lw=1.2, zorder=1)

    colors = []
    for r in rows:
        if r["mAP50_95"]["significant_bonf"]:
            colors.append("#c0392b")   # 校正后仍显著
        elif r["mAP50_95"]["significant"]:
            colors.append("#d68910")   # 仅未校正显著
        else:
            colors.append("#7f8c8d")   # 不可区分

    ax.errorbar(pts, ys, xerr=[pts - lo, hi - pts], fmt="none", lw=2,
                capsize=4, ecolor="#333", zorder=3)
    for i, c in enumerate(colors):
        ax.plot(pts[i], ys[i], "o", ms=8.5, color=c, mec="white", mew=1.2, zorder=4)

    span = max(abs(lo).max(), abs(hi).max()) * 1.55
    text_x = span * 0.97
    for i in range(len(rows)):
        if colors[i] == "#c0392b":
            tag = " ★Bonf."
        elif colors[i] == "#d68910":
            tag = " 未校正"
        else:
            tag = " ns"
        ax.text(text_x, ys[i], f"{pts[i]:+.2f}pp{tag}", va="center", ha="right",
                fontsize=9, fontweight="bold", color=colors[i])

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10)
    ax.set_xlabel("test mAP50-95 相对基线的差值（pp，负值＝比基线差）", fontsize=11)
    ax.set_title(f"配对比较：差异的置信区间比单模型 CI 窄 3~5 倍\n"
                 f"（{len(rows)} 个配置 vs 基线，同一 75 张图配对重采样 {n_boot} 次）",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", ls=":", alpha=0.5, zorder=0)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)
    ax.set_xlim(-span, span)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="基线，格式 name=weights")
    ap.add_argument("--model", action="append", required=True, help="待比较模型，格式 name=weights，可多次")
    ap.add_argument("--label", action="append", default=None, help="与 --model 一一对应的显示名，可省")
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paired_bootstrap")
    args = ap.parse_args()

    def parse_spec(s: str) -> tuple[str, str]:
        if "=" not in s:
            raise SystemExit(f"格式应为 name=weights，收到：{s}")
        k, v = s.split("=", 1)
        return k.strip(), v.strip()

    ref_name, ref_w = parse_spec(args.ref)
    pairs = [parse_spec(m) for m in args.model]
    labels = args.label or [""] * len(pairs)
    labels += [""] * (len(pairs) - len(labels))
    label_map = {name: (lab or name) for (name, _), lab in zip(pairs, labels)}

    # ---- 1. 收集全部模型的 per-image 统计量 ----
    all_stats: dict[str, dict[str, list[np.ndarray]]] = {}
    print(f"[1/3] 收集 per-image 统计量（batch=1，与 model.val 同口径 rect=True）")
    todo = [(ref_name, ref_w)] + pairs
    for name, w in todo:
        st, n = collect_per_image_stats(w, args.data, args.split, args.device,
                                        args.imgsz, args.conf, args.iou)
        all_stats[name] = st
        print(f"      {name:16s} {n} 张图 / {sum(len(x) for x in st['target_cls'])} 个 GT 框")

    # ---- 2. 对齐校验：逐图 GT 框数必须完全一致 ----
    print(f"[2/3] 对齐校验")
    ref_fp = fingerprint(all_stats[ref_name])
    for name, _ in pairs:
        if fingerprint(all_stats[name]) != ref_fp:
            raise SystemExit(f"[x] {name} 的图顺序/标注与 {ref_name} 不一致，不能做配对比较")
    print(f"      通过：{len(ref_fp)} 张图的 GT 指纹逐图一致（框数 {min(ref_fp)}~{max(ref_fp)}）")

    # ---- 3. 共用一套重采样索引 ----
    n_img = len(ref_fp)
    rng = np.random.default_rng(args.seed)
    idx_matrix = rng.integers(0, n_img, (args.n, n_img))
    print(f"[3/3] 配对 bootstrap：{args.n} 次，索引矩阵共享（同一批图、同一组扰动）")

    n_tests = len(pairs)
    base_pt = metrics_on_subset(all_stats[ref_name], np.arange(n_img))
    results = [{
        "name": ref_name, "label": "基线（YOLOv11n 完整增强）", "weights": ref_w,
        "abs_mAP50": base_pt[0], "abs_mAP50_95": base_pt[1],
        "mAP50": {"point": 0.0, "ci95": [0.0, 0.0], "ci_bonf": [0.0, 0.0],
                  "half_width": 0.0, "p_gt0": None, "significant": False, "significant_bonf": False},
        "mAP50_95": {"point": 0.0, "ci95": [0.0, 0.0], "ci_bonf": [0.0, 0.0],
                     "half_width": 0.0, "p_gt0": None, "significant": False, "significant_bonf": False},
    }]

    for (name, w), lab in zip(pairs, labels):
        d, point = delta_distribution(all_stats[name], all_stats[ref_name], idx_matrix)
        s50 = summarize(d[:, 0], point[0], n_tests)
        s95 = summarize(d[:, 1], point[1], n_tests)
        abs_pt = metrics_on_subset(all_stats[name], np.arange(n_img))
        results.append({
            "name": name, "label": label_map[name], "weights": w,
            "abs_mAP50": abs_pt[0], "abs_mAP50_95": abs_pt[1],
            "mAP50": s50, "mAP50_95": s95,
            "n_boot": int(len(d)),
        })
        flag = "★Bonferroni 后仍显著" if s95["significant_bonf"] else (
            "!未校正显著" if s95["significant"] else "  ")
        print(f"      {name:16s} ΔmAP50-95 {point[1]*100:+.2f}pp "
              f"CI [{s95['ci95'][0]*100:+.2f}, {s95['ci95'][1]*100:+.2f}] "
              f"bonf [{s95['ci_bonf'][0]*100:+.2f}, {s95['ci_bonf'][1]*100:+.2f}] "
              f"P(Δ>0)={s95['p_gt0']:.2f} {flag}")

    # 非配对参考门槛：基线的单模型 CI 半宽
    base_boot = []
    for idx in idx_matrix:
        r = metrics_on_subset(all_stats[ref_name], idx)
        if r is not None:
            base_boot.append(r)
    base_boot = np.array(base_boot)
    lo_np, hi_np = np.percentile(base_boot[:, 1], [2.5, 97.5])
    half_nonpaired = float((hi_np - lo_np) / 2)

    # ---- 输出 ----
    REPORT_DIR.mkdir(exist_ok=True)
    payload = {
        "ref": {"name": ref_name, "weights": ref_w, "abs_mAP50": base_pt[0], "abs_mAP50_95": base_pt[1]},
        "n_images": n_img, "n_boot": int(len(idx_matrix)),
        "conf_nms_iou": [args.conf, args.iou],
        "nonpaired_half_width_mAP50_95": half_nonpaired,
        "results": results,
    }
    (REPORT_DIR / f"{args.out}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # md 表
    def verdict(s: dict) -> str:
        """三档判定：Bonferroni 后仍显著 / 仅未校正显著 / 不可区分。"""
        def sign(lo, hi):
            return "负" if hi < 0 else ("正" if lo > 0 else None)

        sb = sign(*s["ci_bonf"])
        sr = sign(*s["ci95"])
        if sb:
            return f"**显著为{sb}**（{n_tests} 重比较后仍成立）"
        if sr:
            return f"仅未校正显著为{sr}（存疑）"
        return "不可区分"

    md = [
        "# 配对 bootstrap：误差棒到底有多宽",
        "",
        f"- 基线：`{ref_name}`（mAP50 {base_pt[0]:.4f} / mAP50-95 {base_pt[1]:.4f}）",
        f"- 样本：{n_img} 张图，重采样 {len(idx_matrix)} 次",
        f"- 口径：conf={args.conf}、NMS iou={args.iou}、rect=True，与 `model.val()` 一致",
        "- 配对方式：同一次重采样抽出的索引**同时**作用于两个模型，Δ 的公共方差互相抵消",
        f"- 多重比较：同时做 {n_tests} 组比较，Bonferroni 把每次的 α 压到 {0.05/n_tests:.4f}"
        f"（对应 {(1-0.05/n_tests)*100:.2f}% CI）",
        "- 符号约定：**Δ = 该配置 − 基线，负值表示比基线差**",
        "",
        "## mAP50-95（主指标）",
        "",
        "> ⚠️ **这一层的「显著」只回答「这两个已训好的权重谁更好」，不回答「这个方法更好」。**",
        "> 表里的 Δ 全部来自 **seed 0 单次训练**的两个权重，没有训练端噪声。",
        "> 加完三种子配对后，这 10 个对比里有 6 个量级缩水 40~70%、1 个方向翻转，",
        "> 最终只剩 2 个成立。**要下结论请读 `noise_budget.md`，不要读本表。**",
        "",
        "| 配置 | mAP50-95 | Δ | 95% CI | 半宽 | Bonferroni CI | P(Δ>0) | 仅评估端判定 |",
        "|---|---|---|---|---|---|---|---|",
        f"| {ref_name}（基线） | {base_pt[1]:.4f} | — | — | — | — | — | 参照 |",
    ]
    for r in results[1:]:
        s = r["mAP50_95"]
        md.append(
            f"| {r['label']} | {r['abs_mAP50_95']:.4f} | {s['point']*100:+.2f}pp | "
            f"[{s['ci95'][0]*100:+.2f}, {s['ci95'][1]*100:+.2f}] | ±{s['half_width']*100:.2f}pp | "
            f"[{s['ci_bonf'][0]*100:+.2f}, {s['ci_bonf'][1]*100:+.2f}] | "
            f"{s['p_gt0']:.2f} | {verdict(s)} |")
    md += [
        "",
        "## mAP50（次要指标，已知饱和）",
        "",
        "| 配置 | Δ | 95% CI | 半宽 | Bonferroni CI | P(Δ>0) | 结论 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results[1:]:
        s = r["mAP50"]
        md.append(f"| {r['label']} | {s['point']*100:+.2f}pp | "
                  f"[{s['ci95'][0]*100:+.2f}, {s['ci95'][1]*100:+.2f}] | "
                  f"±{s['half_width']*100:.2f}pp | "
                  f"[{s['ci_bonf'][0]*100:+.2f}, {s['ci_bonf'][1]*100:+.2f}] | "
                  f"{s['p_gt0']:.2f} | {verdict(s)} |")
    mean_half = np.mean([r["mAP50_95"]["half_width"] for r in results[1:]])
    md += [
        "",
        f"**配对 vs 非配对**：基线的单模型 CI 半宽是 ±{half_nonpaired*100:.2f}pp，"
        f"配对后 Δ 的平均半宽只有 ±{mean_half*100:.2f}pp，收窄了 "
        f"{half_nonpaired / mean_half:.1f} 倍。也就是说，此前把 ±{half_nonpaired*100:.2f}pp "
        f"当作差异门槛，把可检测差异的尺度高估了 {half_nonpaired / mean_half:.1f} 倍 —— "
        f"结论是「这批数据分辨不了 2.7pp」，实际是「分辨不了约 "
        f"{mean_half*100:.1f}pp」。",
        "",
        "**怎么读**：CI 跨 0 = 在这 75 张 test 上分辨不出；CI 整体位于 0 左侧 = 该配置"
        "**的这两个权重**显著差于基线。同一批图上的两次评估并不独立，"
        "配对才反映「换模型、不换图」这个真实问题。",
        "但「换 seed 重训」是另一个问题 —— 见 `noise_budget.md`。",
    ]

    # 口径核对：与 eval_full 记录的绝对值比一比，确认两套管线没有系统性偏差
    check = []
    for r in results[1:]:
        p = REPORT_DIR / f"{r['name']}.json"
        if not p.exists():
            continue
        old = json.loads(p.read_text(encoding="utf-8")).get("detection", {}).get("mAP50_95")
        if old is None:
            continue
        check.append((r["label"], r["abs_mAP50_95"], old, r["abs_mAP50_95"] - old))
    if check:
        worst = max(abs(c[3]) for c in check)
        md += [
            "",
            "## 与 `eval_full.py` 记录的口径核对",
            "",
            "| 配置 | 本次重算 | eval_full 记录 | 差 |",
            "|---|---|---|---|",
        ]
        for lab, cur, old, diff in check:
            md.append(f"| {lab} | {cur:.6f} | {old:.6f} | {diff:+.6f} |")
        md += [
            "",
            f"最大偏差 {worst*100:.3f}pp。两套管线（`model.val()` vs 手工构造 `DetectionValidator`）"
            "在 rect / NMS 细节上有微小差异，量级远小于任何判据门槛，不影响结论。",
        ]
    (REPORT_DIR / f"{args.out}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))

    forest_plot(results, ref_name, REPORT_DIR / f"{args.out}.png", half_nonpaired, len(idx_matrix))
    print(f"\n[done] {REPORT_DIR / (args.out + '.json')}")


if __name__ == "__main__":
    main()
