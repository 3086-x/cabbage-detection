"""把训练好的 .pt 导出成 ONNX，并当场验证「导出的图和原模型算的是同一件事」。

为什么导出后必须自检
------------------
`model.export(format="onnx")` 成功返回只说明「图能被序列化」，不说明输出对。
常见翻车方式有三种，且都不会报错：

  1. 自定义模块（CBAM）被 trace 成常量，图里少了一层；
  2. opset 太低，某个算子被近似实现，输出轻微偏移；
  3. 输出张量的排列顺序变了（(1,4+nc,N) vs (1,N,4+nc)），后处理默默解出空框。

所以这里做两级检查：

  张量级：同一输入，PyTorch 与 ORT 的原始输出逐元素比，报 max|Δ| 与相对误差；
  框级  ：同一张真实图片，两条链路各自解码，报检出数与置信度的最大偏差。

任何一级超阈值就打印 FAIL 并以非零码退出——不允许「导出了但不知道对不对」。

跑法
----
    python tools/export_onnx.py --weights runs/detect/train_base_v2/weights/best.pt --tag base_v2
    python tools/export_onnx.py --weights runs/detect/train_cbam/weights/best.pt --tag cbam_v2
    python tools/export_onnx.py --weights x.pt --dynamic        # 动态 batch

产物
----
    deploy/models/<tag>.onnx
    deploy/onnx_manifest.json   （每个模型的输入输出签名、opset、自检结果）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMGSZ = 640
COORD_TOL = 0.05    # 坐标（0~640 像素量纲）容忍度，约 8e-5 相对误差
CONF_TOL = 5e-4     # 置信度（0~1 量纲）容忍度
BOX_TOL = 1.5       # 解码后框坐标最大偏差（像素）


def _pick_pred_tensor(outputs):
    """从 PyTorch 前向结果里挑出 (bs, 4+nc, N) 那个张量。"""
    import torch

    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (list, tuple)):
        for o in outputs:
            if isinstance(o, torch.Tensor) and o.ndim == 3:
                return o
        raise RuntimeError(f"前向输出里没有 3 维张量，实际: {[type(o) for o in outputs]}")
    raise RuntimeError(f"无法识别的输出类型: {type(outputs)}")


def _torch_forward(model, x_np: np.ndarray):
    import torch

    m = model.model
    m.eval()
    x = torch.from_numpy(x_np).float()
    with torch.no_grad():
        out = m(x)
    return _pick_pred_tensor(out).cpu().numpy()


def _ort_forward(onnx_path: Path, x_np: np.ndarray) -> np.ndarray:
    from tools.onnx_common import build_session

    sess = build_session(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: x_np})[0]
    if out.ndim != 3:
        raise RuntimeError(f"ONNX 输出不是 3 维: {out.shape}")
    return out


def check_tensor(pt_out: np.ndarray, ort_out: np.ndarray) -> dict:
    """逐元素比对原始输出，按「坐标 / 置信度」两段分开报。

    两段量纲不同：坐标是 0~640 的像素，置信度是 0~1 的概率。
    混在一起算一个 max|Δ|，会被坐标那一段主导，看不出分类头有没有坏。
    """
    if pt_out.shape != ort_out.shape:
        return {"ok": False, "reason": f"形状不一致 pt={pt_out.shape} ort={ort_out.shape}"}

    pt = pt_out.astype(np.float64)
    ort = ort_out.astype(np.float64)
    coord_diff = np.abs(pt[:, :4, :] - ort[:, :4, :])
    conf_diff = np.abs(pt[:, 4:, :] - ort[:, 4:, :])

    coord_max = float(coord_diff.max()) if coord_diff.size else 0.0
    conf_max = float(conf_diff.max()) if conf_diff.size else 0.0
    ok = coord_max <= COORD_TOL and conf_max <= CONF_TOL
    return {
        "ok": bool(ok),
        "shape": list(pt_out.shape),
        "coord_max_abs_diff_px": coord_max,
        "coord_max_rel_diff": coord_max / IMGSZ,
        "conf_max_abs_diff": conf_max,
        "tol_coord_px": COORD_TOL,
        "tol_conf": CONF_TOL,
    }


def check_boxes(pt_model_boxes: dict, ort_boxes: dict) -> dict:
    """框级比对：数量、置信度、以及按置信度排序后的坐标偏差。"""
    n_pt, n_ort = len(pt_model_boxes["scores"]), len(ort_boxes["scores"])
    res = {"ok": True, "n_pytorch": n_pt, "n_onnx": n_ort}

    if n_pt != n_ort:
        res["ok"] = False
        res["reason"] = f"检出数不一致 {n_pt} vs {n_ort}"
        return res

    if n_pt == 0:
        res["max_score_diff"] = 0.0
        res["max_box_diff_px"] = 0.0
        return res

    order_pt = np.argsort(-pt_model_boxes["scores"])
    order_ort = np.argsort(-ort_boxes["scores"])
    s_pt = pt_model_boxes["scores"][order_pt]
    s_ort = ort_boxes["scores"][order_ort]
    b_pt = pt_model_boxes["boxes"][order_pt]
    b_ort = ort_boxes["boxes"][order_ort]

    res["max_score_diff"] = float(np.abs(s_pt - s_ort).max())
    res["max_box_diff_px"] = float(np.abs(b_pt - b_ort).max())
    if res["max_box_diff_px"] > BOX_TOL:
        res["ok"] = False
        res["reason"] = f"框偏差 {res['max_box_diff_px']:.2f}px 超过 {BOX_TOL}px"
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic", action="store_true", help="导出动态 batch/尺寸")
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--check-image", default="", help="框级自检用的图片，默认取 test 第一张")
    ap.add_argument("--outdir", default="deploy/models")
    args = ap.parse_args()

    from ultralytics import YOLO

    from tools.cbam_module import register
    from tools.onnx_common import preprocess

    register(verbose=False)

    wpath = Path(args.weights)
    if not wpath.exists():
        raise SystemExit(f"权重不存在: {wpath}")
    tag = args.tag or wpath.parent.parent.name
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 自检图片：必须在 .pt 侧先做，因为 export() 会 inplace fuse 模型
    img = Path(args.check_image) if args.check_image else sorted((ROOT / "dataset/test/images").glob("*.jpg"))[0]
    print(f"[info] 权重 {wpath}")
    print(f"[info] 自检图片 {img.name}")

    # ---- 1) PyTorch 侧基准 ----
    model = YOLO(str(wpath))
    x_np, r, pad = preprocess(img, args.imgsz)
    pt_raw = _torch_forward(model, x_np)

    import cv2

    from tools.onnx_common import postprocess

    im = cv2.imread(str(img))
    pt_boxes = postprocess(pt_raw, conf=0.25, iou=0.7, r=r, pad=pad, orig_shape=im.shape[:2])
    print(f"[info] PyTorch 原始输出 {pt_raw.shape}，检出 {len(pt_boxes['scores'])} 个框")

    # ---- 2) 导出 ----
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
        batch=1,
        device="cpu",
    )
    exported = Path(str(exported))
    dst = outdir / f"{tag}.onnx"
    if exported.resolve() != dst.resolve():
        shutil.move(str(exported), dst)
    print(f"[ok] 已导出 {dst} ({dst.stat().st_size / 1e6:.1f} MB)")

    # ---- 3) 张量级自检 ----
    ort_raw = _ort_forward(dst, x_np)
    tcheck = check_tensor(pt_raw, ort_raw)
    print(
        f"[check] 张量级 坐标 max|Δ|={tcheck.get('coord_max_abs_diff_px', float('nan')):.2e}px "
        f"({tcheck.get('coord_max_rel_diff', 0) * 100:.4f}%) / "
        f"置信度 max|Δ|={tcheck.get('conf_max_abs_diff', float('nan')):.2e} "
        f"-> {'PASS' if tcheck['ok'] else 'FAIL'}"
    )
    if not tcheck["ok"]:
        print(f"        {tcheck.get('reason', '')}")

    # ---- 4) 框级自检（走 ultralytics 自己的 ONNX 后端，覆盖给 NMS 的完整链路）----
    onnx_model = YOLO(str(dst), task="detect")
    res = onnx_model.predict(str(img), conf=0.25, iou=0.7, imgsz=args.imgsz, verbose=False)[0]
    ort_boxes = {
        "boxes": res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4)),
        "scores": res.boxes.conf.cpu().numpy() if res.boxes is not None else np.empty((0,)),
    }
    bcheck = check_boxes(pt_boxes, ort_boxes)
    print(
        f"[check] 框级 检出 {bcheck['n_pytorch']} vs {bcheck['n_onnx']}，"
        f"max conf Δ={bcheck.get('max_score_diff', float('nan')):.2e}，"
        f"max 坐标 Δ={bcheck.get('max_box_diff_px', float('nan')):.3f}px -> {'PASS' if bcheck['ok'] else 'FAIL'}"
    )
    if not bcheck["ok"]:
        print(f"        {bcheck.get('reason', '')}")

    # ---- 5) 记录签名 ----
    import onnxruntime as ort

    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    sig = {
        "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in sess.get_inputs()],
        "outputs": [{"name": o.name, "shape": o.shape, "type": o.type} for o in sess.get_outputs()],
    }
    manifest_path = Path("deploy/onnx_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[tag] = {
        "source_weights": str(wpath).replace("\\", "/"),
        "onnx": str(dst).replace("\\", "/"),
        "size_mb": round(dst.stat().st_size / 1e6, 2),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "dynamic": args.dynamic,
        "simplify": not args.no_simplify,
        "signature": sig,
        "check_tensor": tcheck,
        "check_boxes": bcheck,
    }
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] 签名与自检结果写入 {manifest_path}")

    if not (tcheck["ok"] and bcheck["ok"]):
        raise SystemExit("[FAIL] 自检未通过，不要用这个 ONNX")
    print("[PASS] 导出链路可用")


if __name__ == "__main__":
    main()
