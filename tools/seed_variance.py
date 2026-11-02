"""量化「单次训练」的随机波动（种子方差），并据此给出最小可检测差异 MDD。

为什么需要
----------
此前所有结论都是单种子点估计：一个配置只训练一次，两组一比就下结论。
但训练本身带随机性（数据打乱顺序、Mosaic 抽哪四张、随机翻转/缩放/HSV 的采样、
注意力分支的随机初始化），换一个种子结果就会变。不量化这个波动，
「A 比 B 高 1.2pp」这种话就没有判据。

两个统计量
----------
- σ（组内标准差）：同一配置换 seed 重复跑，指标本身的波动
- MDD（最小可检测差异）= 2 × σ_pooled，σ_pooled = sqrt((s_A² + s_B²) / 2)
  → 组间差 Δ 小于 MDD 时，判为不可区分

口径说明
--------
- n=3 时 s 的 90% 置信区间约为 [0.58s, 4.4s]，很宽。所以这套结果只能定「量级」，
  不足以支撑 0.2pp 级别的判断，也**不报 p 值**。
- σ 按配置分别测。早期版本假设「所有配置共用基线的 σ」，后来实测发现注意力类配置的 σ
  可以是基线的数倍，那个假设会把不确定性低估 —— 所以现在 10 个配置各测 3 个种子。

用法
----
    python tools/seed_variance.py \
        --group "完整增强=base_v2,base_v2_s1,base_v2_s2" \
        --group "无Mosaic=base_nomosaic,base_nomosaic_s1,base_nomosaic_s2" \
        ...
    # 第一组被当作基线，其余组与它逐 seed 配对
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# 允许 `python tools/seed_variance.py` 直接跑（此时 sys.path[0] 是 tools/）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.experiment_registry import group_specs  # noqa: E402

REPORT_DIR = Path("eval_reports")
RUNS_DIR = Path("runs/detect")
# 曲线的离线快照：runs/ 不进仓库，这份进仓库，供无数据者复算
ARTIFACT_DIR = Path("artifacts/results_csv")


def load_test_metrics(name: str) -> dict[str, float]:
    """读 tools/eval_full.py 写的 test 指标。"""
    p = REPORT_DIR / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"缺少 {p}，先跑 tools/eval_full.py --name {name}")
    d = json.loads(p.read_text(encoding="utf-8"))
    det = d.get("detection", {})
    return {
        "mAP50": float(det.get("mAP50", float("nan"))),
        "mAP50_95": float(det.get("mAP50_95", float("nan"))),
        "precision": float(det.get("precision", float("nan"))),
        "recall": float(det.get("recall", float("nan"))),
    }


def _find_run_dir(name: str) -> Path | None:
    """兼容历史命名：早期 run 带 train_ 前缀（train_base_v2），新跑的直接用 name。"""
    for cand in (name, f"train_{name}"):
        p = RUNS_DIR / cand
        if (p / "results.csv").exists():
            return p
    return None


def load_best_val(name: str) -> dict[str, float]:
    """从 results.csv 取 val 上最优那一轮的指标。

    输入优先级：
      1. `runs/detect/<run>/results.csv` —— 本机训练产物（训练完直接可读）
      2. `artifacts/results_csv/<name>.csv` —— 随仓库分发的曲线快照

    第 2 条是为了让没有数据集、没有 runs/ 的人也能离线复算 σ 与配对差值：
    runs/ 不进仓库，但每轮 50 行的曲线进了仓库（268 KB）。
    """
    d = _find_run_dir(name)
    p = (d / "results.csv") if d is not None else (ARTIFACT_DIR / f"{name}.csv")
    if not p.exists():
        return {}
    best = None
    with p.open(encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}
            key = "metrics/mAP50-95(B)"
            if key not in row:
                continue
            try:
                v = float(row[key])
            except (TypeError, ValueError):
                continue
            if best is None or v > best["val_mAP50_95"]:
                best = {
                    "val_mAP50": float(row.get("metrics/mAP50(B)", "nan")),
                    "val_mAP50_95": v,
                    "val_recall": float(row.get("metrics/recall(B)", "nan")),
                }
    return best or {}


def agg(vals: list[float]) -> dict[str, float]:
    a = np.asarray([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return {"n": 0}
    out = {"n": int(a.size), "mean": float(a.mean())}
    if a.size > 1:
        out["std"] = float(a.std(ddof=1))
        out["min"] = float(a.min())
        out["max"] = float(a.max())
        out["range"] = float(a.max() - a.min())
    else:
        out["std"] = 0.0
        out["min"] = out["max"] = float(a[0])
        out["range"] = 0.0
    return out


def plot(groups: dict[str, dict], pair: dict | None, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    names = list(groups)
    colors = ["#2e86c1", "#c0392b", "#27ae60", "#8e44ad", "#d68910", "#117a65"]
    ncol = 2 if pair else 1
    # 组数多时把画布拉高，否则 5 组的散点会挤在一起
    height = max(3.6, 1.05 * len(names) + 1.1)
    fig, axes = plt.subplots(1, ncol, figsize=(7.6 * ncol, height), dpi=170, constrained_layout=True)
    if ncol == 1:
        axes = [axes]

    ax = axes[0]
    for i, nm in enumerate(names):
        g = groups[nm]
        pts = np.asarray(g["runs_mAP50_95"], dtype=float)
        y = (len(names) - 1 - i)
        ax.scatter(pts, np.full_like(pts, y) + np.linspace(-0.06, 0.06, pts.size), s=70,
                   color=colors[i % len(colors)], zorder=3, edgecolor="white", linewidth=1.2)
        ax.errorbar(g["mAP50_95"]["mean"], y,
                    xerr=g["mAP50_95"]["std"], fmt="o", color=colors[i % len(colors)],
                    capsize=6, elinewidth=2, markersize=9, zorder=4)
        ax.text(g["mAP50_95"]["mean"], y - 0.22,
                f"{g['mAP50_95']['mean']:.4f} ± {g['mAP50_95']['std']:.4f}",
                ha="center", fontsize=9.5, fontweight="bold", color=colors[i % len(colors)])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=10.5)
    ax.set_ylim(-0.6, len(names) - 0.3)
    ax.set_xlabel("test mAP50-95（每个点 = 一个随机种子）", fontsize=10.5)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_title("同一配置换种子后的波动", fontsize=12, fontweight="bold")

    if pair and ncol == 2:
        ax2 = axes[1]
        base_nm = pair["a"]
        base_arr = np.asarray(groups[base_nm]["runs_mAP50_95"], dtype=float)
        base_std = groups[base_nm]["mAP50_95"]["std"]
        others = [n for n in names if n != base_nm]
        for i, nm in enumerate(others):
            arr = np.asarray(groups[nm]["runs_mAP50_95"], dtype=float)
            dd = arr - base_arr
            mdd = 2 * float(np.sqrt((groups[nm]["mAP50_95"]["std"] ** 2 + base_std ** 2) / 2))
            y = len(others) - 1 - i
            ok = abs(dd.mean()) >= mdd
            c = "#1e8449" if ok else "#b9770e"
            ax2.errorbar(dd.mean() * 100, y, xerr=mdd * 100 / 2, fmt="o", ms=8,
                         color=c, elinewidth=2.4, capsize=6, zorder=3)
            ax2.text(dd.mean() * 100, y + 0.26,
                     f"Δ{dd.mean()*100:+.2f}pp  MDD {mdd*100:.2f}pp",
                     ha="center", fontsize=9, color=c, fontweight="bold")
        ax2.axvline(0, color="#333", lw=1.2, zorder=1)
        ax2.set_yticks(range(len(others)))
        ax2.set_yticklabels(list(reversed(others)), fontsize=10.5)
        ax2.set_ylim(-0.6, len(others) - 0.25)
        ax2.set_xlabel("相对基线的配对 Δ（pp），误差棒 = ±MDD/2", fontsize=10.5)
        ax2.grid(axis="x", alpha=0.3, linestyle="--")
        ax2.set_axisbelow(True)
        n_ok = sum(1 for nm in others
                   if abs((np.asarray(groups[nm]["runs_mAP50_95"]) - base_arr).mean())
                   >= 2 * float(np.sqrt((groups[nm]["mAP50_95"]["std"] ** 2 + base_std ** 2) / 2)))
        ax2.set_title(f"误差棒跨 0 = 与基线不可区分（{len(others)-n_ok}/{len(others)} 个不可区分）",
                      fontsize=11.5, fontweight="bold",
                      color="#b9770e" if n_ok < len(others) else "#1e8449")

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--group", action="append", default=None,
                     help="格式 name=run1,run2,run3，可重复传多组；不传则等同 --all")
    ap_.add_argument("--all", action="store_true",
                     help="用 tools/experiment_registry.py 里登记的全部 11 组（推荐）")
    ap_.add_argument("--out", default="seed_variance")
    args = ap_.parse_args()
    if args.all or not args.group:
        args.group = group_specs()
        print(f"[i] 使用登记表中的 {len(args.group)} 组配置")

    groups: dict[str, dict] = {}
    for spec in args.group:
        if "=" not in spec:
            raise SystemExit(f"--group 格式错误，应为 name=run1,run2：{spec}")
        nm, runs = spec.split("=", 1)
        names = [r.strip() for r in runs.split(",") if r.strip()]
        rows = []
        for r in names:
            t = load_test_metrics(r)
            t.update(load_best_val(r))
            t["run"] = r
            rows.append(t)
        groups[nm] = {
            "runs": rows,
            "runs_mAP50_95": [r["mAP50_95"] for r in rows],
            "mAP50_95": agg([r["mAP50_95"] for r in rows]),
            "mAP50": agg([r["mAP50"] for r in rows]),
        }

    pair = None
    if len(groups) >= 2:
        a, b = list(groups)[:2]
        s1, s2 = groups[a]["mAP50_95"]["std"], groups[b]["mAP50_95"]["std"]
        m1, m2 = groups[a]["mAP50_95"]["mean"], groups[b]["mAP50_95"]["mean"]
        mdd = 2 * float(np.sqrt((s1 ** 2 + s2 ** 2) / 2))
        pair = {"a": a, "b": b, "delta": abs(m1 - m2), "mdd": mdd,
                "verdict": "可区分" if abs(m1 - m2) >= mdd else "不可区分"}

    report = {"groups": groups, "pair": pair,
              "note": "n=3 时标准差自身的不确定性很大（90% CI 约 [0.58s, 4.4s]），结论只到量级，不报 p 值"}
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / f"{args.out}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 种子方差与最小可检测差异（MDD）", "",
             "同一配置只换 `--seed`，其余超参完全相同。命令见 `REPRO_STATUS.md` 第 8 节。", ""]
    for nm, g in groups.items():
        s = g["mAP50_95"]
        lines.append(f"## {nm}（n={s['n']}）")
        lines.append("")
        lines.append("| run | test mAP50 | test mAP50-95 | val 最优 mAP50-95 |")
        lines.append("|---|---|---|---|")
        for r in g["runs"]:
            v = r.get("val_mAP50_95")
            lines.append(f"| `{r['run']}` | {r['mAP50']:.4f} | {r['mAP50_95']:.4f} | "
                         f"{v:.4f} |" if v == v and v is not None else
                         f"| `{r['run']}` | {r['mAP50']:.4f} | {r['mAP50_95']:.4f} | — |")
        lines.append("")
        if s["n"] > 1:
            lines.append(f"- **test mAP50-95 = {s['mean']:.4f} ± {s['std']:.4f}**（极差 {s['range']:.4f}）")
            lines.append(f"- test mAP50 = {g['mAP50']['mean']:.4f} ± {g['mAP50']['std']:.4f}")
        lines.append("")

    if pair:
        base_nm = pair["a"]
        base_arr = np.asarray(groups[base_nm]["runs_mAP50_95"], dtype=float)
        base_std = groups[base_nm]["mAP50_95"]["std"]
        lines += [
            "## 各配置相对基线的配对差", "",
            "「配对」= 同一 seed 下（配置 − 基线），再跨 3 个 seed 求均值与标准差。"
            "这样种子对两个模型的共同影响会抵消，剩下的才是配置本身的效应与波动。", "",
            "| 配置 | mAP50-95 均值 | 组内 σ | seed 0 单次 Δ | 配对 Δ 均值 | 配对 σ | MDD |",
            "|---|---|---|---|---|---|---|",
        ]
        for nm, g in groups.items():
            if nm == base_nm:
                lines.append(f"| **{nm}**（基线） | {g['mAP50_95']['mean']:.4f} | "
                             f"{g['mAP50_95']['std']*100:.2f}pp | — | — | — | — |")
                continue
            arr = np.asarray(g["runs_mAP50_95"], dtype=float)
            dd = arr - base_arr
            mdd_i = 2 * float(np.sqrt((g["mAP50_95"]["std"] ** 2 + base_std ** 2) / 2))
            lines.append(
                f"| {nm} | {g['mAP50_95']['mean']:.4f} | {g['mAP50_95']['std']*100:.2f}pp | "
                f"{(arr[0] - base_arr[0])*100:+.2f}pp | {dd.mean()*100:+.2f}pp | "
                f"{dd.std(ddof=1)*100:.2f}pp | {mdd_i*100:.2f}pp |")
        # 动态找出「单次 vs 三种子」偏差最大的配置，不再硬编码某个配置名
        devs = []
        flips = []
        for nm, g in groups.items():
            if nm == base_nm:
                continue
            arr = np.asarray(g["runs_mAP50_95"], dtype=float)
            dd = arr - base_arr
            mdd_i = 2 * float(np.sqrt((g["mAP50_95"]["std"] ** 2 + base_std ** 2) / 2))
            # 同一个配置，用单次 Δ 判一次、用三种子均值再判一次，结论会不会翻转？
            once = abs(dd[0]) >= mdd_i
            pooled = abs(dd.mean()) >= mdd_i
            if once != pooled:
                flips.append(nm)
            devs.append((abs(dd[0] - dd.mean()), nm, dd[0], dd.mean(), dd.std(ddof=1), mdd_i))
        devs.sort(reverse=True)
        worst = devs[0]
        sig_max_nm = max(((g["mAP50_95"]["std"], n) for n, g in groups.items()))[1]
        sig_max = groups[sig_max_nm]["mAP50_95"]["std"]
        base_std_i = groups[base_nm]["mAP50_95"]["std"]
        n_cfg = len(groups) - 1

        lines += [
            "",
            "**两列要对着看**：「seed 0 单次 Δ」和「配对 Δ 均值」相差越远，说明那个配置的单次结果越不可信。"
            f"最夸张的是 `{worst[1]}` —— 单次 {worst[2]*100:+.2f}pp，三种子平均 {worst[3]*100:+.2f}pp，"
            f"差 {worst[0]*100:.2f}pp，而它的配对 σ 是 {worst[4]*100:.2f}pp。",
            "",
            "**能直接说明问题的是这个**：同一个配置，用单次 Δ 跟 MDD 判一次、用三种子均值再判一次 —— "
            + (f"{n_cfg} 个配置里有 **{len(flips)} 个结论翻转**（{'、'.join(flips)}），"
               "单次看「可区分」，补种子后变成「不可区分」。"
               if flips else
               f"{n_cfg} 个配置里**没有一个结论翻转**，单次与三种子在 MDD 判据下一致。") +
            " 换句话说，" + ("**单种子排序一开始就不可能可靠**。" if flips else
                          "这批配置的效应量相对 MDD 足够大，单次也能定方向。"),
            "",
            "## 判定", "",
            f"- {pair['a']} vs {pair['b']}：Δ = **{pair['delta'] * 100:.2f}pp**，"
            f"MDD = **{pair['mdd'] * 100:.2f}pp**",
            f"- 结论：**{pair['verdict']}**",
            "",
            f"> 含义：在这套设置下，小于 {pair['mdd'] * 100:.2f}pp 的 mAP50-95 差异不足以支持任何排序结论。",
            "",
            "> **σ 不是常数**：基线的组内 σ 是 "
            f"{base_std_i*100:.2f}pp，最大的是 `{sig_max_nm}` 的 {sig_max*100:.2f}pp，"
            f"相差 {sig_max/base_std_i:.1f} 倍。所以「拿基线的 σ 当所有配置的通用噪声」"
            "会系统性低估参数更多、收敛更不稳的那一类配置。",
            "",
            "> 限制：n=3，σ 自身 90% CI 约 [0.58s, 4.4s]，结论只到量级，不报 p 值。"
            "跨两类的正式判定见 `eval_reports/noise_budget.md`（那里把评估端与训练端噪声合成了）。",
        ]

    (REPORT_DIR / f"{args.out}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(groups, pair, REPORT_DIR / f"{args.out}.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
