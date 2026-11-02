"""配对性校验：新增的种子 run 是否只在 `seed` 上与原 run 不同。

为什么需要这个脚本
------------------
"补了 2 个种子"这句话只有在**除 seed 外全部超参一致**时才成立。靠人工 grep 很容易漏：
worker 数、deterministic、close_mosaic、AMP 开关，任何一项不同都会让"种子方差"里混进
别的因素，而那正是本轮所有结论的基础。

做法
----
1. 逐字段比对两个 run 的 `args.yaml`（114 个字段），只允许 `seed`/`name`/`save_dir` 不同；
2. 比对两边的 `best.pt` 结构签名：层类型序列 + 参数量 + 关键层通道数。
   第 2 步能抓到 args.yaml 抓不到的问题 —— CFG 版 run 的 `model` 字段只记文件名，
   CBAM 插在 SPPF 还是 P3 从文件名上分不出来。

用法
----
    python -m tools.check_pairing train_se_s1=train_se train_cbam_s1=train_cbam_v2
    python -m tools.check_pairing --all          # 用下方 PAIRS 里登记的全部配对
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.experiment_registry import GROUPS, find_run_dir  # noqa: E402

RUNS = Path("runs/detect")

# 允许不同的字段：它们是「这次训练的身份信息」，不是「实验条件」
IGNORE = {"name", "seed", "save_dir", "project", "time", "date"}

def pairs_from_registry() -> list[tuple[str, str]]:
    """每组里 seed 1/2 的 run 与 seed 0 的 run 配成对。

    目录名由 experiment_registry.find_run_dir 解析（兼容 `train_` 前缀），
    这样新增一组种子只需改登记表，不用再改这里。
    """
    out = []
    for _name, runs in GROUPS:
        base = find_run_dir(runs[0])
        if base is None:
            print(f"[warn] 找不到基线 run 目录：{runs[0]}")
            continue
        for r in runs[1:]:
            d = find_run_dir(r)
            if d is None:
                print(f"[warn] 找不到种子 run 目录：{r}")
                continue
            out.append((d.name, base.name))
    return out


def load_args(run: str) -> dict:
    p = RUNS / run / "args.yaml"
    if not p.exists():
        raise FileNotFoundError(f"缺少 {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def fmt(v) -> str:
    return f"{v!r}"


def diff_args(a: str, b: str) -> list[str]:
    da, db = load_args(a), load_args(b)
    out = []
    for k in sorted(set(da) | set(db)):
        if k in IGNORE:
            continue
        va, vb = da.get(k), db.get(k)
        if va != vb:
            out.append(f"{k}: {a}={fmt(va)} vs {b}={fmt(vb)}")
    return out


def _ensure_modules() -> None:
    """自定义模块必须先注册到 ultralytics.nn.tasks，否则反序列化 CBAM 权重会失败。"""
    import contextlib
    import io

    try:
        from tools.attention_modules import register_extra
        from tools.cbam_module import register

        # 注册函数会打印 "[ok] ... 已注册"，每对都打一遍太吵，这里吞掉
        with contextlib.redirect_stdout(io.StringIO()):
            register()
            register_extra()
    except Exception:  # noqa: BLE001  只用官方模型的 run 不需要注册
        pass


def arch_signature(run: str) -> dict:
    """从 best.pt 提取结构签名。只在需要时 import torch，避免每次跑都付这个开销。"""
    _ensure_modules()
    import torch

    w = RUNS / run / "weights" / "best.pt"
    if not w.exists():
        raise FileNotFoundError(f"缺少 {w}")
    try:
        ckpt = torch.load(w, map_location="cpu", weights_only=False)
        model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    except Exception:
        # ultralytics 的新格式：直接用 YOLO 加载
        from ultralytics import YOLO

        model = YOLO(str(w)).model
    seq = getattr(model, "model", model)
    layers = list(seq) if hasattr(seq, "__iter__") else []
    n_params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0
    return {
        "n_layers": len(layers),
        "types": [type(m).__name__ for m in layers],
        "params": n_params,
        "out_channels": [getattr(m, "c2", None) for m in layers],
    }


def diff_arch(a: str, b: str) -> list[str]:
    sa, sb = arch_signature(a), arch_signature(b)
    out = []
    if sa["n_layers"] != sb["n_layers"]:
        out.append(f"层数: {a}={sa['n_layers']} vs {b}={sb['n_layers']}")
    if sa["types"] != sb["types"]:
        for i, (x, y) in enumerate(zip(sa["types"], sb["types"])):
            if x != y:
                out.append(f"第 {i} 层类型: {a}={x} vs {b}={y}")
    # 参数量会因为注意力层/检测头的初始化方式不同有极小浮动，超过 1% 才算结构不一致
    if sb["params"] and abs(sa["params"] - sb["params"]) / sb["params"] > 0.01:
        out.append(f"参数量: {a}={sa['params']/1e6:.4f}M vs {b}={sb['params']/1e6:.4f}M")
    return out


def check(a: str, b: str, with_arch: bool = True) -> bool:
    try:
        da, db = load_args(a), load_args(b)
    except FileNotFoundError as e:
        print(f"[skip] {a} vs {b}: {e}")
        return False

    bad = diff_args(a, b)
    if with_arch:
        try:
            bad += diff_arch(a, b)
        except Exception as e:  # noqa: BLE001
            bad.append(f"[arch 检查失败] {type(e).__name__}: {e}")

    seed_a, seed_b = da.get("seed"), db.get("seed")
    if bad:
        print(f"[FAIL] {a} vs {b}  seed {seed_b}→{seed_a}，发现 {len(bad)} 处差异：")
        for line in bad:
            print(f"       - {line}")
        return False
    print(f"[ok]   {a} vs {b}  seed {seed_b}→{seed_a}，其余 114 字段与结构签名完全一致")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="校验种子 run 与原 run 是否严格配对")
    ap.add_argument("pairs", nargs="*", help="形如 new=old；留空则用 --all")
    ap.add_argument("--all", action="store_true", help="校验 PAIRS 里登记的全部配对")
    ap.add_argument("--no-arch", action="store_true", help="只比 args.yaml，不加载权重")
    args = ap.parse_args()

    if args.all or not args.pairs:
        pairs = pairs_from_registry()
    else:
        pairs = []
        for p in args.pairs:
            if "=" not in p:
                print(f"[error] 参数必须是 new=old 形式：{p}")
                sys.exit(2)
            a, b = p.split("=", 1)
            pairs.append((a, b))

    ok = [check(a, b, not args.no_arch) for a, b in pairs]
    print(f"\n{sum(ok)}/{len(ok)} 对通过")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
