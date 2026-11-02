"""诊断脚本：确认带注意力层的模型在「训练前」的 val mAP 是否与基线一致。

用途
----
朴素 CBAM 第一轮 mAP 只有 0.0036，基线是 0.89。有两种可能：
  (a) 预训练权重没迁移好；
  (b) 预训练权重迁移好了，但 CBAM 的 x*mask 把特征缩放了 4 倍。
本脚本直接对比「加载完预训练、还没训练」时的 mAP，就能区分这两者。

跑法：
    python tools/probe_pretrained.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from tools.cbam_module import register  # noqa: E402
from tools.train_cbam_fixed import build_cbam_yaml, load_pretrained_shifted  # noqa: E402


def main() -> None:
    register(verbose=False)
    from ultralytics import YOLO

    for module in ("CBAM", "CBAMResidual"):
        p, at, ch = build_cbam_yaml(module=module, out_path=Path(f"cfg/_probe_{module}.yaml"))
        m = YOLO(str(p))
        hit, miss = load_pretrained_shifted(m.model, "yolo11n.pt", at)
        r = m.val(data="data.yaml", split="val", verbose=False, plots=False)
        print(f"[{module}] 预训练命中 {hit} 跳过 {miss} | epoch0 val mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f}", flush=True)

    b = YOLO("yolo11n.pt")
    r = b.val(data="data.yaml", split="val", verbose=False, plots=False)
    print(f"[baseline] epoch0 val mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f}", flush=True)

    # 逐层核对迁移是否真的正确：SPPF 在插入点之前（键不变），C2PSA 在插入点之后（键 +1）
    src = torch.load("yolo11n.pt", map_location="cpu", weights_only=False)["model"].float().state_dict()
    p, at, _ = build_cbam_yaml(module="CBAMResidual", out_path=Path("cfg/_probe_CBAMResidual.yaml"))
    m = YOLO(str(p))
    load_pretrained_shifted(m.model, "yolo11n.pt", at)
    tgt = m.model.state_dict()
    checks = [
        ("model.9.cv2.conv.weight", "model.9.cv2.conv.weight", "SPPF（插入点前，键不变）"),
        ("model.8.cv2.conv.weight", "model.8.cv2.conv.weight", "backbone C3k2（插入点前）"),
        ("model.10.cv1.conv.weight", "model.11.cv1.conv.weight", "C2PSA（插入点后，键 +1）"),
        ("model.13.cv1.conv.weight", "model.14.cv1.conv.weight", "head C3k2（插入点后，键 +1）"),
    ]
    for sk, tk, note in checks:
        if sk in src and tk in tgt:
            same = torch.allclose(src[sk], tgt[tk])
            print(f"[check] {note:26s} {sk} -> {tk}  一致={same}", flush=True)
        else:
            print(f"[check] {note:26s} 缺键: {sk} in src={sk in src}, {tk} in tgt={tk in tgt}", flush=True)


if __name__ == "__main__":
    main()
