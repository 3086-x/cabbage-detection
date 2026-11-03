"""校验 ORT-GPU 与 PyTorch-GPU 的检测结果是否等价。

为什么必须单独做这件事
------------------
速度表只有在「两边算的是同一个模型」时才有意义。ONNX 导出错位、GPU EP 静默回落、
预处理路径不一致，都可能让 ORT 跑得又快又对不上。

本脚本不做 mAP 重算（那是 eval_full.py 的事），只做两件更硬的事：
  1. 全 test 集逐图比对株数：ORT-GPU vs PyTorch-GPU，报完全一致率与 |Δ| 分布
  2. 单图逐框比对：按中心点一对一匹配后报坐标/置信度最大偏差

顺带确认 CUDA EP 真的接管了全部节点——否则这一整轮比的是 CPU。

跑法
----
    python tools/verify_ort_gpu.py
    python tools/verify_ort_gpu.py --onnx deploy/models/base_v2.onnx --conf 0.65
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="runs/detect/train_base_v2/weights/best.pt")
    ap.add_argument("--onnx", default="deploy/models/base_v2.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.65)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0 表示全量")
    args = ap.parse_args()

    from tools.bench_deploy import build_gpu_session, node_assignment
    from tools.onnx_common import postprocess, preprocess

    imgs = sorted((ROOT / "dataset/test/images").glob("*.jpg"))
    if args.limit:
        imgs = imgs[: args.limit]
    labels = ROOT / "dataset/test/labels"

    # ---- EP 归属 ----
    sess = build_gpu_session(args.onnx, profile=True)
    x0, _, _ = preprocess(imgs[0], args.imgsz)
    for _ in range(3):
        sess.run(None, {sess.get_inputs()[0].name: x0})
    assign = node_assignment(sess)
    total = sum(assign.values())
    share = assign.get("CUDAExecutionProvider", 0) / total * 100 if total else 0.0
    print(f"[EP] {assign} → CUDA {share:.1f}%")
    if share < 99.9:
        print("[FAIL] CUDA EP 未接管全部节点，本次比对的是 CPU 结果")
        return

    # ---- PyTorch GPU ----
    import torch

    from ultralytics import YOLO
    from tools.cbam_module import register

    register(verbose=False)
    pt = YOLO(args.pt).model.to("cuda").eval()

    def pt_boxes(path: Path):
        x, r, pad = preprocess(path, args.imgsz)
        with torch.no_grad():
            out = pt(torch.from_numpy(x).float().cuda())
        arr = out[0].cpu().numpy() if isinstance(out, (list, tuple)) else out.cpu().numpy()
        return postprocess(arr, conf=args.conf, iou=args.iou, r=r, pad=pad)

    def ort_boxes(path: Path):
        x, r, pad = preprocess(path, args.imgsz)
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        return postprocess(out, conf=args.conf, iou=args.iou, r=r, pad=pad)

    # ---- 逐图株数 ----
    diffs, exact, rows = [], 0, []
    for p in imgs:
        a, b = pt_boxes(p), ort_boxes(p)
        na, nb = len(a["scores"]), len(b["scores"])
        gt = len([l for l in (labels / f"{p.stem}.txt").read_text().splitlines() if l.strip()])
        diffs.append(na - nb)
        exact += int(na == nb)
        rows.append((p.name, na, nb, gt))

    d = np.asarray(diffs)
    print(f"\n[株数] {len(imgs)} 张：完全一致 {exact}/{len(imgs)} ({exact / len(imgs) * 100:.1f}%)，"
          f"Δ 均值 {d.mean():+.2f}，|Δ| 最大 {np.abs(d).max()}")
    bad = [(n, a, b, g) for n, a, b, g in rows if a != b]
    for n, a, b, g in bad[:8]:
        print(f"   {n:16s} PyTorch={a:3d}  ORT-GPU={b:3d}  GT={g:3d}")

    # ---- 单图逐框 ----
    p0 = imgs[0]
    a, b = pt_boxes(p0), ort_boxes(p0)
    idx = _match(a["boxes"], b["boxes"])
    xdiff = float(np.abs(a["boxes"] - b["boxes"][idx]).max())
    sdiff = float(np.abs(a["scores"] - b["scores"][idx]).max())
    print(f"\n[逐框] {p0.name}  {len(a['scores'])} 个框："
          f"坐标 max|Δ|={xdiff:.3f}px，置信度 max|Δ|={sdiff:.2e}")

    out = {
        "onnx": str(args.onnx),
        "n_images": len(imgs),
        "count_exact_match": int(exact),
        "count_exact_rate": round(exact / len(imgs) * 100, 1),
        "count_diff_mean": round(float(d.mean()), 3),
        "count_abs_diff_max": int(np.abs(d).max()),
        "box_max_diff_px": round(xdiff, 3),
        "score_max_diff": round(sdiff, 6),
        "ep_assignment": assign,
        "cuda_share_pct": round(share, 1),
    }
    Path("eval_reports").mkdir(exist_ok=True)
    dst = Path("eval_reports/ort_gpu_parity.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 写入 {dst}")


def _match(b_ref: np.ndarray, b_other: np.ndarray) -> np.ndarray:
    """按中心点一对一匹配。不能按分数排序后逐行比：分数接近时顺序会互换，产生假偏差。"""
    n, m = len(b_ref), len(b_other)
    idx = -np.ones(n, dtype=int)
    if n == 0 or m == 0:
        return idx
    c1 = (b_ref[:, :2] + b_ref[:, 2:]) / 2
    c2 = (b_other[:, :2] + b_other[:, 2:]) / 2
    dist = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=-1)
    free_r = np.ones(n, bool)
    free_c = np.ones(m, bool)
    for _ in range(min(n, m)):
        dd = np.where(free_r[:, None] & free_c[None, :], dist, np.inf)
        i, j = np.unravel_index(np.argmin(dd), dd.shape)
        if not np.isfinite(dd[i, j]):
            break
        idx[i] = j
        free_r[i] = free_c[j] = False
    return idx


if __name__ == "__main__":
    main()
