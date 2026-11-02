"""实验配置登记表：所有「配置名 ↔ 三个种子的 run 名」的唯一来源。

为什么要有这个文件
------------------
`seed_variance.py` 和 `noise_budget.py` 都要知道「哪个配置对应哪几个 run」。
此前这份映射在三个地方各写了一遍（命令行参数、噪声预算里的 SEED_GROUP_MAP、
文档里的复现命令），组名稍不一致就会静默出错 —— 实际上已经踩过一次
（`train_se_s1` vs `se_s1` 命名不统一，导致脚本读不到数据）。

现在只在这里维护一份，其余脚本 import。

命名约定
--------
每组 3 个 run，**第一个是 seed 0（原 run）**，后两个是补的 seed 1 / 2。
名字是 `eval_reports/<name>.json` 里的 name，不是 `runs/detect/` 下的目录名
（两者历史上不完全一致，比如目录叫 `train_base_v2`、报告叫 `base_v2`）。
顺序不能乱：配对是按下标相减的。
"""
from __future__ import annotations

# (组名, [seed0, seed1, seed2])
# 组名同时是 paired_bootstrap 里 --model 的 key 对应的展示名
GROUPS: list[tuple[str, list[str]]] = [
    ("完整增强",        ["base_v2",        "base_v2_s1",        "base_v2_s2"]),
    ("无Mosaic",        ["base_nomosaic",  "base_nomosaic_s1",  "base_nomosaic_s2"]),
    ("+SE",             ["se",             "train_se_s1",       "train_se_s2"]),
    ("+CBAM",           ["cbam_v2",        "train_cbam_s1",     "train_cbam_s2"]),
    ("+CA",             ["ca",             "train_ca_s1",       "train_ca_s2"]),
    ("CBAM@P3",         ["cbam_p3",        "train_cbam_p3_s1",  "train_cbam_p3_s2"]),
    ("残差CBAM",        ["cbam_res_v2",    "train_cbam_res_s1", "train_cbam_res_s2"]),
    ("加MixUp",         ["base_mixup",     "train_base_mixup_s1", "train_base_mixup_s2"]),
    ("无Mosaic+CBAM",   ["cbam_nomosaic",  "train_cbam_nomosaic_s1", "train_cbam_nomosaic_s2"]),
    ("无Mosaic+CA",     ["ca_nomosaic",    "train_ca_nomosaic_s1", "train_ca_nomosaic_s2"]),
    ("YOLOv8n",         ["yolov8n",        "train_yolov8n_s1",  "train_yolov8n_s2"]),
]

# 基线组（配对相减的对象）
BASELINE = "完整增强"

# 「配对 bootstrap 的模型 key」→ 组名。seed 0 的 run 名就是那个 key。
PB_KEY_TO_GROUP: dict[str, str] = {runs[0]: name for name, runs in GROUPS if name != BASELINE}


def group_specs() -> list[str]:
    """转成 seed_variance.py 的 `--group name=run1,run2,run3` 形式。"""
    return [f"{name}={','.join(runs)}" for name, runs in GROUPS]


def all_runs() -> list[str]:
    return [r for _, runs in GROUPS for r in runs]


def find_run_dir(name: str, runs_root: str = "runs/detect"):
    """把评估报告名解析成 `runs/detect/` 下的目录。

    历史上两套命名不一致：目录可能是 `train_base_v2`，而报告是 `base_v2`；
    也可能两边同名（`base_v2_s1`）。这里统一处理，避免每个脚本各写一份。
    """
    from pathlib import Path

    root = Path(runs_root)
    for cand in (name, f"train_{name}"):
        p = root / cand
        if p.exists():
            return p
    return None


def describe() -> str:
    lines = ["| 组 | seed 0 | seed 1 | seed 2 |", "|---|---|---|---|"]
    for name, runs in GROUPS:
        lines.append(f"| {name} | " + " | ".join(f"`{r}`" for r in runs) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print(f"\n共 {len(GROUPS)} 组 / {len(all_runs())} 个 run")
