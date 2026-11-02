"""噪声预算合成：把「评估端」和「训练端」两个独立噪声源合起来，给每个结论一个最终判定。

为什么必须合成
--------------
`paired_bootstrap.py` 测的是**评估端**误差：给定两个已训好的权重，换 75 张同分布的图，
Δ 会漂多少。它把"图难度"这个公共成分消掉了，所以 CI 很窄（半宽 ~0.6pp）。

`seed_variance.py` 测的是**训练端**误差：换一个随机种子重训，同一个配置的 mAP 会漂多少。

这两者回答的是不同的问题，不能互相替代：
- 配对 CI 回答"这两个权重谁更好" → 真实但不可复现（换 seed 就变）
- 种子 σ  回答"这个配置换一次训练还能不能更好" → 才是"方法有效"的含义

所以一个诚实的结论需要把两者合成：

    SE_total = sqrt(SE_eval² + (σ_paired / sqrt(n))²)
    CI_total = Δ ± z · SE_total

三个口径细节（都是踩过坑才定下来的）
------------------------------------
1) **效应量取 n 个种子的配对均值**，不是 seed 0 单次值。
   同一 seed 下「配置 − 基线」相减，种子对两个模型的共同影响会抵消，剩下的才是配置差异。
   `+CBAM` 就是被单次值坑到的例子：seed 0 是 −2.15pp，另两个 seed 只有 −0.09 / −0.06。

2) **训练端标准误要除以 √n**。效应量是 3 次训练的平均，其标准误是 σ/√3，不是 σ。
   此前直接拿 σ 当 SE，等于把 CI 放大了 √3 ≈ 1.7 倍 —— 虽然偏保守、不会造假阳性，
   但会掩盖真实效应。保守口径仍然保留一列，用来检查结论在两种口径下是否一致。

3) **评估端标准误不除以 √n**。75 张 test 是固定的一份样本，"这份测试集偏难/偏易"
   是三个模型共有的偏差，多训几个种子消不掉它。这是这套数据的分辨率下限。

已知局限（必须写进文档）
------------------------
- n=3 时标准差本身很不可靠（σ 的 90% CI 约为 [0.58s, 4.4s]），所以合成只到量级，
  不应当被读成精确的 p 值。
- 三种子均值的评估端 SE 仍取 seed 0 那一对的配对半宽，作为该类配置的代表值。

用法
----
    python -m tools.noise_budget
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

# 允许 `python -m tools.noise_budget` 与 `python tools/noise_budget.py` 两种跑法
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.experiment_registry import BASELINE, GROUPS, PB_KEY_TO_GROUP  # noqa: E402

REPORT_DIR = Path("eval_reports")
Z95 = 1.959964
# 10 组同时比较的 Bonferroni 分位：α = 0.05/10 → 双侧 99.5% → z = 2.807
Z_BONF = {"n": len(GROUPS) - 1, "alpha": 0.05 / (len(GROUPS) - 1), "z": 2.807034}

# paired_bootstrap 里的模型名 → seed_variance 里的种子组名。
# 由 experiment_registry 派生，避免两份手写清单漂移（这个 bug 已经踩过一次）。
SEED_GROUP_MAP = PB_KEY_TO_GROUP

# 万一某组缺种子（例如只跑了一半），按"它属于哪一类改动"借用同类池的 σ。
ATTN_GROUPS = ["+SE", "+CBAM", "+CA", "CBAM@P3", "残差CBAM", "无Mosaic+CBAM", "无Mosaic+CA"]
AUG_GROUPS = [BASELINE, "无Mosaic", "加MixUp"]
BORROW_POOL = {
    "cbam_res_v2": ("attn", "注意力类"),
    "cbam_p3": ("attn", "注意力类"),
    "cbam_nomosaic": ("attn", "注意力类"),
    "ca_nomosaic": ("attn", "注意力类"),
    "base_mixup": ("aug", "增强类"),
    "yolov8n": ("all", "全体（架构不同，无同类可借）"),
}


def pool_sigma(seed: dict) -> dict[str, float]:
    """按类别把实测的组内 σ 合成几个"借用池"（仅作为缺种子时的兜底）。

    σ_pool = sqrt(mean(σ_i²))，与 seed_variance.py 的 MDD 口径一致。
    """
    sig = {k: v["mAP50_95"]["std"] for k, v in seed["groups"].items()}

    def _pool(names: list[str]) -> float:
        vals = [sig[n] for n in names if n in sig]
        return float(np.sqrt(np.mean([s ** 2 for s in vals]))) if vals else float("nan")

    return {
        "attn": _pool(ATTN_GROUPS),
        "aug": _pool(AUG_GROUPS),
        "all": _pool(list(sig)),
    }


def _pp(v: float) -> str:
    """把小数差值格式化成带符号的 pp。接近 0 时输出不带符号的 0.00，
    避免 ``-0.00`` / ``+0.00`` 这种会让读者误判方向的写法。"""
    x = v * 100
    if abs(x) < 0.005:
        return "0.00"
    return f"{x:+.2f}"


def _verdict(ci: list[float]) -> str | None:
    if ci[1] < 0:
        return "负"
    if ci[0] > 0:
        return "正"
    return None


def main() -> None:
    pb = json.loads((REPORT_DIR / "paired_bootstrap.json").read_text(encoding="utf-8"))
    seed = json.loads((REPORT_DIR / "seed_variance.json").read_text(encoding="utf-8"))

    sigma_pool = pool_sigma(seed)
    sig_each = {k: v["mAP50_95"]["std"] for k, v in seed["groups"].items()}
    n_each = {k: len(v["runs_mAP50_95"]) for k, v in seed["groups"].items()}

    ref = pb["ref"]
    base_runs = np.asarray(seed["groups"][BASELINE]["runs_mAP50_95"], dtype=float)

    print("[i] 训练端 σ（实测）：")
    for k in seed["groups"]:
        print(f"    ├ {k:10s} n={n_each[k]} σ = {sig_each[k]*100:.2f}pp")

    rows = []
    missing: list[str] = []
    for r in pb["results"]:
        if r["name"] == ref["name"]:
            continue
        s = r["mAP50_95"]
        se_eval = s["half_width"] / Z95

        gname = SEED_GROUP_MAP.get(r["name"])
        seed_mean = seed_std = None
        n_seed = 1
        if gname and gname in seed["groups"]:
            arr = np.asarray(seed["groups"][gname]["runs_mAP50_95"], dtype=float)
            if arr.size == base_runs.size and arr.size >= 2:
                d_seed = arr - base_runs          # 配对：同一 seed 下配置 − 基线
                seed_mean = float(d_seed.mean())
                seed_std = float(d_seed.std(ddof=1))
                n_seed = int(arr.size)

        if seed_std is not None:
            se_train_single = seed_std            # 单次训练的不确定度
            se_train = seed_std / math.sqrt(n_seed)  # 均值的不确定度（主口径）
            train_src = "实测"
        else:
            missing.append(r["name"])
            pool_key, pool_label = BORROW_POOL.get(r["name"], ("all", "全体"))
            se_train_single = sigma_pool[pool_key]
            se_train = se_train_single
            train_src = f"借用{pool_label}池"

        abs95 = (float(seed["groups"][gname]["runs_mAP50_95"][0])
                 if seed_std is not None else r["abs_mAP50_95"])
        delta = seed_mean if seed_mean is not None else s["point"]

        def _ci(se: float, z: float) -> list[float]:
            return [delta - z * se, delta + z * se]

        se_tot = float(np.hypot(se_eval, se_train))
        se_tot_cons = float(np.hypot(se_eval, se_train_single))
        ci_t = _ci(se_tot, Z95)
        ci_b = _ci(se_tot, Z_BONF["z"])
        ci_cons = _ci(se_tot_cons, Z_BONF["z"])

        vb = _verdict(ci_b)
        rows.append({
            "name": r["name"], "label": r["label"],
            "delta": delta, "seed0_delta": s["point"], "n_seed": n_seed,
            "abs_mAP50_95": abs95,
            "eval_half": s["half_width"], "eval_only_ci": s["ci95"],
            "se_eval": se_eval, "se_train_single": se_train_single, "se_train": se_train,
            "train_sigma_source": train_src,
            "seed_paired_mean": seed_mean, "seed_paired_std": seed_std,
            "se_total": se_tot, "ci95_total": ci_t, "ci_bonf_total": ci_b,
            "verdict_95": _verdict(ci_t), "verdict_bonf": vb,
            "significant_final": vb is not None,
            "ci_bonf_conservative": ci_cons, "verdict_conservative": _verdict(ci_cons),
        })

    if missing:
        print(f"[warn] 以下配置缺少 3 种子实测，σ 按同类归池借用：{missing}")

    rows.sort(key=lambda x: x["delta"])

    all_measured = not missing
    sig_names = [r["label"] for r in rows if r["significant_final"]]
    ns_names = [r["label"] for r in rows if not r["significant_final"]]
    d_abs = sorted(abs(r["delta"]) for r in rows if r["significant_final"])
    n_cfg = len(rows) + 1

    # 对显著项算「临界 σ」：3 种子的配对 σ 涨到多少，这个结论就翻转
    critic = []
    for r in rows:
        if not r["significant_final"]:
            continue
        need = (abs(r["delta"]) / Z_BONF["z"]) ** 2 - r["se_eval"] ** 2
        if need > 0:
            critic.append((r, math.sqrt(need) * math.sqrt(r["n_seed"])))

    # ---------- md ----------
    md = [
        "# 噪声预算合成：一个结论到底要多大的差异才站得住",
        "",
        "两个**独立**的噪声源，回答的是两个不同的问题：",
        "",
        "| 噪声源 | 测法 | 量级 | 回答的问题 |",
        "|---|---|---|---|",
        f"| 评估端 | 配对 bootstrap（75 图 × {pb['n_boot']} 次） | 半宽 ±{np.mean([r['eval_half'] for r in rows])*100:.2f}pp | 给定这两个权重，换一批同分布的图还成立吗 |",
        f"| 训练端 | 每配置 3 个种子重训 | 配对 σ = {min(sig_each.values())*100:.2f}~{max(sig_each.values())*100:.2f}pp | 换一个随机种子重训，这个配置还能更好吗 |",
        "",
        "```",
        "SE_total = sqrt( SE_eval² + (σ_paired / √n)² )      CI = Δ ± z · SE_total",
        "```",
        "",
        f"- `Δ` = **{min(n_each.values())} 个种子的配对均值**（同一 seed 下配置 − 基线，再平均）",
        f"- `SE_train = σ_paired / √n`：均值的不确定度，不是单次训练的不确定度",
        "- `SE_eval` **不除以 √n**：75 张 test 是固定的一份样本，多训几个种子消不掉"
        "「这份测试集偏难/偏易」这个共同偏差",
        f"- 基线 mAP50-95 = {ref['abs_mAP50_95']:.4f}（{ref['name']}）",
        "",
        "## 最终判定（mAP50-95）",
        "",
        "| 配置 | n | mAP50-95 | Δ（效应量） | 配对 σ（单次训练） | 合成 95% CI | 合成 Bonferroni CI | 最终判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mark = {"负": "**显著为负**", "正": "**显著为正**", None: "不可区分"}[r["verdict_bonf"]]
        star = "" if r["train_sigma_source"] == "实测" else "*"
        md.append(
            f"| {r['label']} | {r['n_seed']} | {r['abs_mAP50_95']:.4f} | {_pp(r['delta'])}pp | "
            f"{r['se_train_single']*100:.2f}pp{star} | "
            f"[{_pp(r['ci95_total'][0])}, {_pp(r['ci95_total'][1])}] | "
            f"[{_pp(r['ci_bonf_total'][0])}, {_pp(r['ci_bonf_total'][1])}] | {mark} |")
    md += [
        "",
        ("全部 σ 均为实测。" if all_measured else
         "\\* 该配置缺 3 种子实测，σ 为同类归池的借用值。"),
        "",
        "## 保守口径：如果把 3 个种子当成 1 次训练",
        "",
        "主表用 `σ/√n` 是因为 Δ 是 3 次训练的平均。但 n=3 的 σ 本身估不准，"
        "所以这里再给一列**不除 √n** 的版本 —— 它把 3 个种子当成 1 次训练处理，"
        "是在「σ 被低估」这个方向上的防线。两种口径都过线，结论才算硬。",
        "",
        "| 配置 | Δ | 主口径 Bonferroni CI | 保守口径 Bonferroni CI | 主口径 | 保守口径 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        m1 = {"负": "显著为负", "正": "显著为正", None: "不可区分"}[r["verdict_bonf"]]
        m2 = {"负": "显著为负", "正": "显著为正", None: "不可区分"}[r["verdict_conservative"]]
        md.append(
            f"| {r['label']} | {_pp(r['delta'])}pp | "
            f"[{_pp(r['ci_bonf_total'][0])}, {_pp(r['ci_bonf_total'][1])}] | "
            f"[{_pp(r['ci_bonf_conservative'][0])}, {_pp(r['ci_bonf_conservative'][1])}] | "
            f"{m1} | {m2} |")

    seed_rows = [r for r in rows if r["seed_paired_std"] is not None]
    if seed_rows:
        md += [
            "",
            "## 单次点估计错得有多离谱",
            "",
            "把 seed 0 的单次 Δ 和三种子均值并排放，能直接看出「一次训练」这个样本量有多不靠谱：",
            "",
            "| 配置 | seed 0 单次 Δ | 三种子 Δ 均值 | 偏差 | 配对 σ | 逐 seed Δ |",
            "|---|---|---|---|---|---|",
        ]
        for r in seed_rows:
            arr = np.asarray(seed["groups"][SEED_GROUP_MAP[r["name"]]]["runs_mAP50_95"], dtype=float)
            each = " / ".join(f"{x*100:+.2f}" for x in (arr - base_runs))
            md.append(
                f"| {r['label']} | {_pp(r['seed0_delta'])}pp | {_pp(r['seed_paired_mean'])}pp | "
                f"{_pp(r['seed0_delta'] - r['seed_paired_mean'])}pp | "
                f"{r['seed_paired_std']*100:.2f}pp | {each} |")
        worst = max(seed_rows, key=lambda r: abs(r["seed0_delta"] - r["seed_paired_mean"]))
        md += [
            "",
            f"最夸张的是 **{worst['label']}**：seed 0 单次给出 {_pp(worst['seed0_delta'])}pp，"
            f"三种子均值只有 {_pp(worst['seed_paired_mean'])}pp，"
            f"差了 {abs(worst['seed0_delta'] - worst['seed_paired_mean'])*100:.2f}pp —— "
            f"而它的配对 σ 是 {worst['seed_paired_std']*100:.2f}pp，"
            "单次结果整个落在自己的噪声里。**任何用单种子排序的消融表，都可能把这种抖动当成结论。**",
        ]

    md += [
        "",
        "**同一份数据、两种口径的差别**：如果只看评估端，"
        f"{sum(1 for r in rows if r['verdict_95'])} 个配置显著；"
        f"把训练端噪声算进来后，只剩 {sum(1 for r in rows if r['verdict_bonf'])} 个能过 Bonferroni。",
        "这就是为什么「配对 CI 很窄」不能直接当成「效应显著」—— 它只证明了这两个权重之间有差别，"
        "没证明这个配置换一次训练还会有差别。",
        "",
        "## 训练端 σ 不是一个常数",
        "",
        "| 配置 | n | 组内 σ | 相对基线的倍数 |",
        "|---|---|---|---|",
    ]
    base_sig = sig_each[BASELINE]
    for k, v in sorted(sig_each.items(), key=lambda kv: -kv[1]):
        md.append(f"| {k} | {n_each[k]} | {v*100:.2f}pp | {v / base_sig:.1f}× |")
    attn_vals = [sig_each[g] for g in ATTN_GROUPS if g in sig_each]
    md += [
        "",
        "**这一列数字否掉了一个假设**：最早只测了基线两个档位（σ = 0.23 / 0.68pp），"
        "然后把它当成所有配置的通用训练噪声。实测下来注意力类配置的 σ 落在 "
        f"{min(attn_vals)*100:.2f}~{max(attn_vals)*100:.2f}pp，"
        f"最高到基线的 {max(attn_vals) / base_sig:.1f} 倍 —— "
        "参数更多、收敛更不稳，随机性不是同一个量级。"
        "拿基线的 σ 去判定注意力类配置，方向上**偏乐观**，"
        "会把「其实只是抖了一下」的差异误判成「显著」。",
        "",
        "## 交叉验证：两种提问方式",
        "",
        "每一种配置现在都有「配对 bootstrap（seed 0 两个权重）」和「三种子配对（换 3 次训练）」两组数字。"
        "前者问「这两个权重谁更好」，后者问「这个方法是不是更好」：",
        "",
        "| 配置 | 配对 bootstrap Δ | 三种子配对 Δ | 差异 | 两种口径方向一致？ |",
        "|---|---|---|---|---|",
    ]
    for r in seed_rows:
        same = (r["seed0_delta"] > 0) == (r["seed_paired_mean"] > 0)
        md.append(
            f"| {r['label']} | {_pp(r['seed0_delta'])}pp | {_pp(r['seed_paired_mean'])}pp | "
            f"{_pp(r['seed0_delta'] - r['seed_paired_mean'])}pp | {'是' if same else '**否**'} |")
    n_flip = sum(1 for r in seed_rows if (r["seed0_delta"] > 0) != (r["seed_paired_mean"] > 0))
    md += [
        "",
        f"{len(seed_rows)} 个配置里有 **{n_flip} 个方向翻转过**。这不是哪一个数字错了，"
        "而是两个问题本来就可以有不同答案：某个权重确实更好，不代表产出它的那个方法更好。",
        "",
        "## 局限",
        "",
        f"1. n=3 时标准差极不可靠（σ 的 90% CI 约 [0.58s, 4.4s]），"
        "所以合成后的 CI 只到量级，**不报 p 值**。",
        "2. 合成假设两个噪声源独立、且近似正态。前者成立（一个来自数据抽样，一个来自训练随机性）。",
        "3. 三种子均值的评估端 SE 仍取 seed 0 那一对的配对半宽作为代表值。"
        "严格做法是对 3 个权重各做一次配对 bootstrap 再合并，但评估端 SE 主要由 75 张图决定，"
        "三个权重之间差别很小。",
    ]
    if not all_measured:
        md.append(f"4. **仍有 {len(missing)} 个配置的 σ 是借用值**：{missing}。")

    md += [
        "",
        "## 结论",
        "",
        f"- **{len(sig_names)} 个配置的差异能扛住双重噪声 + 多重比较校正**"
        + (f"：{'、'.join(sig_names)}，效应量 {d_abs[0]*100:.1f}pp"
           + (f"~{d_abs[-1]*100:.1f}pp。" if len(d_abs) > 1 else "。") if sig_names else "。"),
    ]
    if ns_names:
        md.append(f"- 其余 {len(ns_names)} 个配置（{'、'.join(ns_names)}）的差异落在不可区分区，"
                  "不应作为结论使用。")
    md += [
        "- 本表只判 mAP50-95。mAP50 的判定见 `seed_variance.md`：33 个 run 里最大的组间差只有 "
        "0.06pp，比它自己的种子 σ 还小 —— 它在这批数据上已经彻底饱和。",
        "",
    ]
    if critic:
        md += [
            "**过线结论有多稳**：把 3 种子的配对 σ 往上推到多少，它就会翻转成不可区分？",
            "",
            "| 配置 | 当前配对 σ | 临界 σ | 余量 |",
            "|---|---|---|---|",
        ]
        for r, sc in critic:
            md.append(
                f"| {r['label']} | {r['se_train_single']*100:.2f}pp | {sc*100:.2f}pp | "
                f"{sc / r['se_train_single']:.1f}× |")
        md += [
            "",
            "余量 1.x 倍的意思是：只要该配置的真实 σ 是当前估计的 1.x 倍，这个结论就站不住。"
            "所以过线不等于铁板钉钉 —— 它是「在现有证据下最值得说的一句」，而不是「已经证明」。",
            "",
        ]
    md += [
        "**这张表怎么用**：看到任何一个组间差异，先找它落在哪一行 —— "
        "能过 Bonferroni 的才写进结论，其余只能说「观测到，但不足以区分」。",
    ]
    (REPORT_DIR / "noise_budget.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (REPORT_DIR / "noise_budget.json").write_text(
        json.dumps({
            "ref": ref,
            "n_config": n_cfg,
            "all_sigma_measured": all_measured,
            "missing_sigma_groups": missing,
            "sigma_by_group": sig_each,
            "n_by_group": n_each,
            "z": {"ci95": Z95, "bonferroni": Z_BONF},
            "formula": "SE_total = hypot(SE_eval, sigma_paired / sqrt(n)); SE_eval = boot_half / 1.96",
            "rows": rows,
            "note": "效应量为 n 种子配对均值；训练端 SE = σ/√n；评估端 SE 不随 n 缩小；n=3 时 σ 自身不可靠，结论只到量级",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "\n".join(md))

    # ---------- 图 ----------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    ys = np.arange(len(rows))
    d = np.array([r["delta"] for r in rows]) * 100
    d0 = np.array([r["seed0_delta"] for r in rows]) * 100
    e_half = np.array([r["eval_half"] for r in rows]) * 100
    t_lo = np.array([r["ci95_total"][0] for r in rows]) * 100
    t_hi = np.array([r["ci95_total"][1] for r in rows]) * 100

    ax.axvline(0, color="#333", lw=1.3, zorder=2)
    # 上排：seed 0 单次 Δ 及其配对 bootstrap 区间（中心必须是 d0，才和区间同源）
    ax.errorbar(d0, ys + 0.16, xerr=e_half, fmt="o", ms=5, color="#5d8aa8",
                elinewidth=1.7, capsize=3, alpha=0.85,
                label="seed 0 单次 Δ ± 配对 bootstrap 95%", zorder=3)
    # 下排：三种子配对均值 ± 合成 95% CI
    ax.errorbar(d, ys - 0.16, xerr=[d - t_lo, t_hi - d], fmt="o", ms=6, color="#c0392b",
                elinewidth=2.2, capsize=3,
                label=f"三种子均值 ± 合成 95% CI（评估端 + 训练端/√{int(np.median([r['n_seed'] for r in rows]))}）",
                zorder=4)

    for i, r in enumerate(rows):
        if r["significant_final"]:
            ax.text(t_hi[i] + 0.12, ys[i] - 0.16, "★", color="#c0392b", fontsize=14,
                    va="center", ha="left", fontweight="bold", zorder=5)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10.5)
    ax.set_xlabel("test mAP50-95 相对基线的差值（pp）", fontsize=11.5)
    ax.set_title("单次训练 vs 三种子均值：上排的窄区间会骗人\n"
                 "（★ = 合成 CI + Bonferroni 校正后仍不跨 0）",
                 fontsize=12.5, fontweight="bold")
    ax.grid(axis="x", ls=":", alpha=0.45, zorder=0)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax.text(0.015, 0.975, "← 比基线差", transform=ax.transAxes, fontsize=9.5,
            color="#666", va="top")
    ax.text(0.985, 0.975, "比基线好 →", transform=ax.transAxes, fontsize=9.5,
            color="#666", va="top", ha="right")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "noise_budget.png", dpi=200)
    plt.close(fig)
    print(f"\n[done] {REPORT_DIR / 'noise_budget.md'} / noise_budget.png")


if __name__ == "__main__":
    main()
