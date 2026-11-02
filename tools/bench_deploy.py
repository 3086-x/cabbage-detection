"""部署后端测速：PyTorch vs ONNX Runtime，FP32 vs INT8。

一个必须先讲清楚的前提
------------------
「PyTorch GPU 8.7ms，ONNX CPU 40ms，所以 ONNX 更慢」——这种结论是错的，
因为它比的是两块硬件。加速比只有在**同一块硬件、同一段代码、同一次进程**里才有意义。

所以这个脚本分两组报：

  A 组（同硬件 CPU）：PyTorch-CPU / ORT-CPU-FP32 / ORT-CPU-INT8
      → 这里才是「ONNX 图优化带来多少」和「量化带来多少」的真实答案
  B 组（GPU 参考）：PyTorch-GPU / 本机无 ORT-GPU，仅作单点参考，不作加速比

每组同时报两个口径：
  纯前向   —— 只有 forward，用来看算子层面的差距
  端到端   —— 读图 + letterbox + 归一化 + forward + 解码 + NMS，才是服务真实延迟

笔记本 GPU 频率会漂移，所以每个后端都报 CV（变异系数）。CV > 10% 的时延
不要跨批次横向比，本脚本会把不稳定的行标出来。

跑法
----
    python tools/bench_deploy.py                       # 默认测 base_v2 三后端
    python tools/bench_deploy.py --iters 100 --batch 4

产物
----
    eval_reports/deploy_speed.csv
    eval_reports/deploy_speed.md
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def bench(fn, warmup: int, iters: int, cuda: bool = False) -> list[float]:
    """跑 warmup 次预热，再逐次计时 iters 次，返回每次的毫秒数。

    GPU 上每次前后都 synchronize：否则测到的是「内核入队时间」而不是执行时间，
    会得到好看但假的数字。
    """
    import torch

    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="runs/detect/train_base_v2/weights/best.pt")
    ap.add_argument("--onnx", default="deploy/models/base_v2.onnx")
    ap.add_argument("--int8", default="deploy/models/base_v2_int8.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()

    from tools.onnx_common import build_session, postprocess, preprocess, summarize

    img = sorted((ROOT / "dataset/test/images").glob("*.jpg"))[0]
    Path("eval_reports").mkdir(exist_ok=True)

    # 端到端要读图，所以预处理在计时外只算一次基准输入；端到端时每次重算
    x, r, pad = preprocess(img, args.imgsz)
    xb = np.repeat(x, args.batch, axis=0)

    rows: list[dict] = []

    # ---- A 组：PyTorch CPU ----
    import torch

    from ultralytics import YOLO
    from tools.cbam_module import register

    register(verbose=False)
    pt_model = YOLO(args.pt)
    torch_cpu = pt_model.model.to("cpu").eval()
    xt = torch.from_numpy(xb).float()
    with torch.no_grad():
        pure = bench(lambda: torch_cpu(xt), args.warmup, args.iters)

    def e2e_torch_cpu():
        xx, rr, pp = preprocess(img, args.imgsz)
        with torch.no_grad():
            out = torch_cpu(torch.from_numpy(np.repeat(xx, args.batch, axis=0)).float())
        arr = out[0].cpu().numpy() if isinstance(out, (list, tuple)) else out.cpu().numpy()
        postprocess(arr, conf=0.25, iou=0.7, r=rr, pad=pp)

    e2e = bench(e2e_torch_cpu, max(3, args.warmup // 2), args.iters)
    rows.append({"backend": "PyTorch", "device": "CPU", "precision": "FP32", "batch": args.batch,
                 "forward": summarize(pure), "end2end": summarize(e2e)})

    # ---- A 组：ORT CPU FP32 / INT8 ----
    for label, path in (("ONNX Runtime", args.onnx), ("ONNX Runtime", args.int8)):
        if not Path(path).exists():
            print(f"[skip] {path} 不存在")
            continue
        prec = "INT8" if "int8" in Path(path).stem else "FP32"
        sess = build_session(path, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        pure = bench(lambda: sess.run(None, {iname: xb}), args.warmup, args.iters)

        def e2e_ort(img=img, sess=sess, iname=iname):
            xx, rr, pp = preprocess(img, args.imgsz)
            out = sess.run(None, {iname: np.repeat(xx, args.batch, axis=0)})[0]
            postprocess(out, conf=0.25, iou=0.7, r=rr, pad=pp)

        e2e = bench(e2e_ort, max(3, args.warmup // 2), args.iters)
        rows.append({"backend": label, "device": "CPU", "precision": prec, "batch": args.batch,
                     "forward": summarize(pure), "end2end": summarize(e2e)})

    # ---- B 组：PyTorch GPU（单点参考，不做加速比）----
    if not args.skip_gpu and torch.cuda.is_available():
        torch_gpu = pt_model.model.to("cuda").eval()
        xg = torch.from_numpy(xb).float().cuda()
        with torch.no_grad():
            pure = bench(lambda: torch_gpu(xg), args.warmup, args.iters, cuda=True)

        def e2e_torch_gpu():
            xx, rr, pp = preprocess(img, args.imgsz)
            with torch.no_grad():
                out = torch_gpu(torch.from_numpy(np.repeat(xx, args.batch, axis=0)).float().cuda())
            arr = out[0].cpu().numpy() if isinstance(out, (list, tuple)) else out.cpu().numpy()
            postprocess(arr, conf=0.25, iou=0.7, r=rr, pad=pp)

        e2e = bench(e2e_torch_gpu, max(3, args.warmup // 2), args.iters, cuda=True)
        rows.append({"backend": "PyTorch", "device": "GPU", "precision": "FP32", "batch": args.batch,
                     "forward": summarize(pure), "end2end": summarize(e2e)})

    # ---- 汇总 ----
    cpu_rows = [r for r in rows if r["device"] == "CPU"]
    base = next((r for r in cpu_rows if r["precision"] == "FP32" and r["backend"] == "ONNX Runtime"), None)
    base_fwd = base["forward"]["median_ms"] if base else None

    for r in rows:
        r["speedup_vs_ort_fp32"] = (
            round(base_fwd / r["forward"]["median_ms"], 2) if base_fwd else None
        )
        r["stable"] = "yes" if r["forward"]["cv_pct"] < 10 else "NO"

    hdr = f"{'后端':<16}{'设备':<6}{'精度':<6}{'batch':<6}{'前向中位':>10}{'前向P95':>9}{'CV%':>7}{'端到端中位':>11}{'相对ORT-CPU-FP32':>18}"
    print("\n" + hdr)
    print("-" * len(hdr.encode("gbk", errors="ignore")))
    for r in rows:
        f, e = r["forward"], r["end2end"]
        sp = f"{r['speedup_vs_ort_fp32']:.2f}×" if r["speedup_vs_ort_fp32"] else "-"
        print(f"{r['backend']:<16}{r['device']:<6}{r['precision']:<6}{r['batch']:<6}"
              f"{f['median_ms']:>10.2f}{f['p95_ms']:>9.2f}{f['cv_pct']:>7.1f}"
              f"{e['median_ms']:>11.2f}{sp:>18}")

    # ---- 落盘 ----
    csv_path = Path("eval_reports/deploy_speed.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["backend", "device", "precision", "batch", "fwd_median_ms", "fwd_p95_ms", "fwd_cv_pct",
                    "e2e_median_ms", "e2e_p95_ms", "e2e_cv_pct", "speedup_vs_ort_cpu_fp32", "stable"])
        for r in rows:
            w.writerow([r["backend"], r["device"], r["precision"], r["batch"],
                        f"{r['forward']['median_ms']:.2f}", f"{r['forward']['p95_ms']:.2f}",
                        f"{r['forward']['cv_pct']:.1f}",
                        f"{r['end2end']['median_ms']:.2f}", f"{r['end2end']['p95_ms']:.2f}",
                        f"{r['end2end']['cv_pct']:.1f}",
                        r["speedup_vs_ort_fp32"] or "", r["stable"]])

    md = [
        "# 部署后端测速",
        "",
        f"输入 {args.imgsz}×{args.imgsz}，batch={args.batch}，预热 {args.warmup} 次，计时 {args.iters} 次取中位数。",
        "",
        "| 后端 | 设备 | 精度 | 前向中位(ms) | 前向P95(ms) | CV | 端到端中位(ms) | 相对 ORT-CPU-FP32 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        f, e = r["forward"], r["end2end"]
        sp = f"{r['speedup_vs_ort_fp32']:.2f}×" if r["speedup_vs_ort_fp32"] else "—"
        md.append(f"| {r['backend']} | {r['device']} | {r['precision']} | {f['median_ms']:.2f} | "
                  f"{f['p95_ms']:.2f} | {f['cv_pct']:.1f}% | {e['median_ms']:.2f} | {sp} |")
    md += [
        "",
        "> 相对加速比只在同设备（CPU）内成立。GPU 那一行是与 CPU 不同硬件的单点参考，不参与比较。",
        "> CV > 10% 表示计时不稳定（频率漂移/调度抖动），该行数字不要跨批次横比。",
    ]
    Path("eval_reports/deploy_speed.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[done] 写入 {csv_path} 与 eval_reports/deploy_speed.md")


if __name__ == "__main__":
    main()
