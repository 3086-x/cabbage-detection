"""把 CBAM 真正插进 YOLOv11n 并训练 —— 修正版。

原来的脚本为什么不行
------------------
1) `train_cbam.py` 里 `from ultralytics.nn.modules import CBAM`：
   ultralytics 官方没有 CBAM 这个类，这行直接 ImportError，脚本从没跑起来过。
2) `train_cbam.py` 里 `model.model.model = nn.Sequential(layers[:10] + [cbam] + layers[10:])`：
   YOLO 的 model.model 不是普通 Sequential —— 每层都有自己的 `f`（来源层索引），
   硬塞一层进去会让后面所有层的索引错位，前向直接崩。
3) `yolo11_cbam.py` 整个文件没有一句真正训练，只有注释。

正确做法
--------
用 yaml 描述结构，让 ultralytics 自己算索引：
  1. CBAM 定义在 tools/cbam_module.py（必须在可 import 的模块里，不能是 __main__）
  2. register() 把 CBAM 注入 ultralytics.nn.tasks，parse_model 才能按名字找到它
  3. 读官方 yolo11.yaml，在 SPPF 后插入一层 CBAM，并把所有 >= 插入点的绝对索引 +1
  4. 通道要写**缩放后**的值（yolo11n：1024 × 0.25 = 256），自定义模块不会被自动缩放
  5. 加载预训练权重时必须**按索引重映射**（插入点之后的层号整体 +1），否则只能
     迁移插入点之前的权重，对比就不公平了

跑法
----
    cd D:/cabbage_project
    python tools/train_cbam_fixed.py --epochs 50 --batch 8 --device 0

产物
----
    cfg/yolo11n-cbam.yaml          生成的模型结构
    runs/detect/train_cbam/        训练结果（含真的带 CBAM 的 best.pt）
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 让 `from tools.cbam_module import ...` 能找到包（无论从哪个目录启动）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from tools.attention_modules import CoordAtt, SE, register_extra  # noqa: E402
from tools.cbam_module import CBAM, register  # noqa: E402

# variant -> (yaml 里的模块名, 默认 reduction)
# reduction 与原论文脚本对齐：train_se.py 用 16，train_ca.py 用 32
VARIANTS = {
    "plain": ("CBAM", 16),
    "residual": ("CBAMResidual", 16),
    "se": ("SE", 16),
    "ca": ("CoordAtt", 32),
}


def make_divisible(x: float, divisor: int = 8) -> int:
    """和 ultralytics 内部一致的最小可整除取整。"""
    return int((x + divisor / 2) // divisor * divisor) if x else divisor


# ---------------------------------------------------------------------------
# 1. 生成带 CBAM 的 yaml
# ---------------------------------------------------------------------------


def _shift(layer: list, at: int) -> list:
    """把某一层 f 字段的绝对索引 >= at 的全部 +1（负数是相对索引，不动）。"""
    f = layer[0]
    if isinstance(f, int):
        f = f + 1 if f >= at else f
    elif isinstance(f, list):
        f = [x + 1 if x >= at else x for x in f]
    return [f] + list(layer[1:])


def build_cbam_yaml(
    insert_after: str = "SPPF",
    scale: str = "n",
    out_path: Path | None = None,
    module: str = "CBAM",
    insert_index: int | None = None,
    reduction: int = 16,
    kernel_size: int = 7,
) -> tuple[Path, int, int]:
    """读官方 yolo11.yaml，在指定层后插入注意力模块，写出新 yaml。

    Args:
        insert_after: backbone 里的层名（默认 SPPF），按名字找插入点
        insert_index: 绝对层号（backbone+head 拼接后的下标），优先级高于 insert_after。
                      例：16 = head 里输出 P3/8 的 C3k2，做"插到 neck 小目标层"的对比
        module: 写进 yaml 的模块名，CBAM / CBAMResidual / SE / CoordAtt
        reduction: 通道压缩比（SE=16，CoordAtt=32，与原论文脚本一致）
        out_path: 输出 yaml 路径，默认 cfg/yolo11{scale}-{module}.yaml

    返回 (yaml 路径, 插入位置索引, 注意力层通道数)
    """
    import ultralytics

    base = Path(ultralytics.__file__).parent / "cfg" / "models" / "11" / "yolo11.yaml"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    backbone, head = cfg["backbone"], cfg["head"]
    layers = backbone + head

    if insert_index is not None:
        idx = int(insert_index)
        if not 0 <= idx < len(layers):
            raise RuntimeError(f"insert_index={idx} 超出范围（共 {len(layers)} 层）")
        anchor = layers[idx][2]
    else:
        idx = next((i for i, layer in enumerate(backbone) if layer[2] == insert_after), None)
        if idx is None:
            raise RuntimeError(f"在 backbone 里没找到 {insert_after} 层")
        anchor = insert_after

    # 关键：自定义模块不会被 parse_model 自动缩放，这里必须自己算出缩放后的通道
    depth, width, max_channels = cfg["scales"][scale]
    raw_ch = layers[idx][3][0]
    ch_out = make_divisible(min(raw_ch, max_channels) * width, 8)

    cbam_layer = [-1, 1, module, [ch_out, int(reduction), int(kernel_size)]]

    new_layers = []
    for i, layer in enumerate(layers):
        if i == idx + 1:
            new_layers.append(cbam_layer)
        new_layers.append(_shift(layer, idx + 1))
    if idx + 1 >= len(layers):
        new_layers.append(cbam_layer)

    # 插在 skip-connection 的源头（如 P3）时，后面引用该层的层必须改指向注意力层，
    # 否则 Detect / Concat 仍然取用「没经过注意力」的原始特征，插进去等于白插。
    if insert_index is not None:
        redirect = []
        for pos, layer in enumerate(new_layers):
            if pos <= idx + 1:
                redirect.append(layer)
                continue
            f = layer[0]
            if isinstance(f, int):
                f = idx + 1 if f == idx else f
            elif isinstance(f, list):
                f = [idx + 1 if x == idx else x for x in f]
            redirect.append([f] + list(layer[1:]))
        new_layers = redirect

    cfg["backbone"] = new_layers[: len(backbone) + 1]
    cfg["head"] = new_layers[len(backbone) + 1 :]

    out_path = out_path or Path(f"cfg/yolo11{scale}-{module}.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# 由 tools/train_cbam_fixed.py 自动生成\n"
        f"# 在 {anchor}（第 {idx} 层）之后插入 {module}，通道 {raw_ch} x width {width} = {ch_out}\n"
    )
    out_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[ok] 模型结构: {out_path.resolve()}")
    print(f"     插入位置: {anchor}(idx={idx}) 之后 → 新层 idx={idx + 1}，通道 {ch_out}")
    return out_path, idx + 1, ch_out


# ---------------------------------------------------------------------------
# 2. 预训练权重按索引重映射加载
# ---------------------------------------------------------------------------


def _shift_key(key: str, at: int) -> str:
    """'model.11.cv1.conv.weight' 在 at=10 插入一层后变成 'model.12.cv1.conv.weight'。"""
    m = re.match(r"^model\.(\d+)\.(.*)$", key)
    if not m:
        return key
    i, rest = int(m.group(1)), m.group(2)
    return f"model.{i + 1 if i >= at else i}.{rest}"


def load_pretrained_shifted(model, ckpt_path: str, insert_at: int) -> tuple[int, int]:
    """加载预训练权重，把插入点之后的层号 +1，返回 (命中数, 跳过数)。"""
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    src_sd = src.float().state_dict() if hasattr(src, "state_dict") else src

    tgt = model.state_dict()
    new_sd, skipped = {}, []
    for k, v in src_sd.items():
        nk = _shift_key(k, insert_at)
        if nk in tgt and tgt[nk].shape == v.shape:
            new_sd[nk] = v
        else:
            skipped.append(k)
    model.load_state_dict(new_sd, strict=False)
    return len(new_sd), len(skipped)


# ---------------------------------------------------------------------------
# 3. 训练
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="0 = 第一块 GPU，cpu = 用 CPU")
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--name", default="train_cbam")
    ap.add_argument("--pretrained", default="yolo11n.pt")
    ap.add_argument("--scale", default="n", choices=list("nsmlx"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--variant",
        default="plain",
        choices=list(VARIANTS),
        help=(
            "plain = CBAM，x*mask；residual = CBAMResidual，x + gamma*CBAM(x)（gamma 初始 0）；"
            "se = SE；ca = CoordAtt。后两者用于复现原论文的「三种注意力模块」消融。"
        ),
    )
    ap.add_argument(
        "--reduction",
        type=int,
        default=None,
        help="通道压缩比，不填则按 variant 取默认值（SE=16，CA=32，CBAM=16）",
    )
    ap.add_argument(
        "--insert-after",
        default="SPPF",
        help="backbone 内的层名，默认 SPPF（实验 A 可改用 --insert-index 16 插到 P3）",
    )
    ap.add_argument("--insert-index", type=int, default=None, help="绝对层号，优先于 --insert-after")
    ap.add_argument(
        "--mosaic",
        type=float,
        default=None,
        help="覆盖 mosaic 增强概率（默认用 ultralytics 的 1.0）。实验 B/D 传 0.0 表示「无 Mosaic」",
    )
    ap.add_argument(
        "--mixup",
        type=float,
        default=None,
        help="覆盖 mixup 增强概率（默认用 ultralytics 的 0.0）。实验 B 传 0.0 表示「无 MixUp」",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="全局随机种子（实验 F：种子方差）。trainer 内部实际用的是 seed+1+RANK，单卡即 seed+1",
    )
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="确定性算子开关。None = 沿用 ultralytics 默认（本版本 8.4.194 默认 True）",
    )
    args = ap.parse_args()

    module, default_reduction = VARIANTS[args.variant]
    reduction = args.reduction if args.reduction is not None else default_reduction
    if args.name == "train_cbam":
        # 同名会覆盖上一组实验的产物，按 variant 自动区分
        args.name = "train_cbam" if args.variant == "plain" else f"train_{args.variant}"

    register()
    register_extra()
    yaml_path, insert_at, ch_cbam = build_cbam_yaml(
        insert_after=args.insert_after,
        scale=args.scale,
        module=module,
        insert_index=args.insert_index,
        reduction=reduction,
    )

    import torch

    from ultralytics import YOLO

    model = YOLO(str(yaml_path))
    # 只统计最外层注意力模块：CBAMResidual 内部还嵌着一个 CBAM，按类名数会重复计数
    blocks = [m for m in model.model.modules() if type(m).__name__ == module]
    print(f"[ok] 构建模型完成，{module} 实例数: {len(blocks)}")

    # 先做一次前向，确认索引没错位、通道对得上（这里报错就不用浪费时间训练了）
    probe = torch.zeros(1, 3, args.imgsz, args.imgsz)
    model.model.eval()
    with torch.no_grad():
        out = model.model(probe)
    y = out[0] if isinstance(out, (list, tuple)) else out
    print(f"[ok] 前向自检通过，输出形状 {tuple(y.shape)}")

    # 加载预训练（带索引重映射）
    if Path(args.pretrained).exists():
        hit, miss = load_pretrained_shifted(model.model, args.pretrained, insert_at)
        print(f"[ok] 预训练权重迁移: 命中 {hit} 个张量，跳过 {miss} 个（注意力层与 Detect 头随机初始化）")
    else:
        print(f"[warn] 找不到 {args.pretrained}，将从零训练")

    n_params = sum(p.numel() for p in model.model.parameters())
    attn_params = sum(p.numel() for m in blocks for p in m.parameters())
    print(f"[ok] 模型参数量: {n_params / 1e6:.4f} M（YOLOv11n 基线 2.6241 M，{module} 贡献 {attn_params} 个参数）")

    # 残差版的关键性质：训练开始前必须是恒等映射，否则预训练特征会被缩放打乱
    if args.variant == "residual":
        g = blocks[0].gamma.detach()
        print(f"[ok] gamma 初值: min={g.min().item():.4f} max={g.max().item():.4f}（应全为 0，模块此时等价于恒等映射）")

    # ------------------------------------------------------------------
    # 关键一步：让训练器复用这份「已加载预训练」的模型
    # ------------------------------------------------------------------
    # ultralytics 的 Model.train() 里是这样写的：
    #     if not args.get("resume") and self.ckpt:      # ← ckpt 为空就不进这个分支
    #         self.trainer.model = self.trainer.get_model(weights=self.model, ...)
    # 用 YOLO("xxx.yaml") 构建的模型 self.ckpt 是 None，于是训练器会**按 yaml 重建
    # 一个随机初始化的模型**，我们手工 load_pretrained_shifted 迁移过来的权重被
    # 静默丢弃 —— 表现是第 1 轮 box_loss ≈ 4.0（从零训练），而用了预训练是 ≈ 2.3。
    #
    # 修法：给 self.ckpt 塞一个最小可用的字典，让上面的分支成立。
    snapshot = {k: v.detach().clone() for k, v in model.model.state_dict().items()}
    model.ckpt = {"model": model.model, "epoch": -1, "train_args": {}, "best_fitness": None}

    # 检测头永远是 Sequential 的最后一层；插入点不同（SPPF→23，P3→24）层号会变，
    # 所以这里动态取，不能写死 "model.23."
    detect_idx = len(model.model.model) - 1

    def _assert_weights_kept(trainer) -> None:
        """on_train_start 回调：确认训练器里的模型确实是我们加载好预训练的那份。"""
        tgt = trainer.model.state_dict()
        pairs = [
            (k, v)
            for k, v in snapshot.items()
            # 只比对浮点参数/缓冲区；stride、anchors 这类整型 buffer 不参与
            if k in tgt and tgt[k].shape == v.shape and v.is_floating_point() and tgt[k].is_floating_point()
        ]
        same = [k for k, v in pairs if torch.allclose(tgt[k].float().cpu(), v.float())]
        total = len(pairs)
        print(f"[check] 训练器权重与加载的预训练一致: {len(same)}/{total} 个张量")
        # Detect 头因类别数变化会被重新初始化，属于预期；其余层不一致则说明权重丢了
        same_set = set(same)
        bad = [k for k, _ in pairs if k not in same_set and not k.startswith(f"model.{detect_idx}.")]
        if bad:
            print(f"[warn] 有 {len(bad)} 个非检测头的张量不一致，前 5 个: {bad[:5]}")
        if total and len(same) < total * 0.5:
            raise RuntimeError("训练器没有复用预训练权重（超过一半张量不一致），检查 model.ckpt 设置")

    model.add_callback("on_train_start", _assert_weights_kept)

    # 增强覆盖：只在显式传参时才写进 train()，否则沿用 ultralytics 默认（完整增强）
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
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        seed=args.seed,
        # 不传 project：ultralytics 会拼成 <runs_dir>/detect/<name>，即 runs/detect/train_cbam
        name=args.name,
        exist_ok=True,
        plots=True,
        **aug,
        **extra,
    )

    save_dir = Path(getattr(model.trainer, "save_dir", f"runs/detect/{args.name}"))
    best = save_dir / "weights/best.pt"
    print(f"\n[done] 权重: {best.resolve()}")
    if best.exists():
        blob = best.read_bytes().lower()
        print(f"验证权重里是否真的包含 {module}：")
        print(f"  '{module.lower()}' 出现次数 = {blob.count(module.lower().encode())} （>0 才算成功）")
        print(f"  'attention_modules' 出现次数 = {blob.count(b'attention_modules')}")
        print(f"  'cbam_module' 出现次数       = {blob.count(b'cbam_module')}")


if __name__ == "__main__":
    main()
