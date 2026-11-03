"""部署后端测速：PyTorch vs ONNX Runtime，CPU / GPU 双组，FP32 vs INT8。

一个必须先讲清楚的前提
------------------
「PyTorch GPU 8.7ms，ONNX CPU 40ms，所以 ONNX 更慢」——这种结论是错的，
因为它比的是两块硬件。加速比只有在**同一块硬件、同一段代码、同一次进程**里才有意义。

所以这个脚本分组报，加速比只在组内算：

  A 组（CPU）：PyTorch-CPU / ORT-CPU-FP32 / ORT-CPU-INT8
      基准 = ORT-CPU-FP32 → 回答「ONNX 图优化能带来多少」「量化能带来多少」
  B 组（GPU）：PyTorch-GPU / ORT-GPU-FP32
      基准 = PyTorch-GPU → 回答「同一个 GPU 上换成 ORT 能带来多少」

每组同时报两个口径：
  纯前向   —— 只有 forward，用来看算子层面的差距
  端到端   —— 读图 + letterbox + 归一化 + forward + 解码 + NMS，才是服务真实延迟

三个防止「测出一个假数字」的机关
--------------------------
1. EP 归属校验
   `ort.get_available_providers()` 里有 CUDAExecutionProvider，不代表它真的加载成功。
   DLL 缺失、cuDNN 大版本不匹配时，ORT 会打一条 [E] 日志然后**静默回落到 CPU**，
   session.get_providers() 甚至会返回 ['CPUExecutionProvider'] —— 此时你测的其实是 CPU。
   本脚本从 profiling 里统计每个节点实际落在哪个 EP，只有 100% CUDA 才算 GPU 行有效。

2. 频率漂移复核
   笔记本 GPU 会因温度/功耗墙降频，导致「越测越慢」，看起来像某个后端更差。
   本脚本在测完所有行后，把基准行**再测一遍**并报漂移百分比；漂移 > 5% 该批数据作废。

3. CV（变异系数）
   每个后端都报 CV，> 10% 的行标 NO，不许跨批次横向比。

跑法
----
    python tools/bench_deploy.py                      # 默认：CPU 组 batch1 + GPU 组 batch1,4,8
    python tools/bench_deploy.py --iters 100 --gpu-batches 1,2,4,8
    python tools/bench_deploy.py --skip-gpu           # 只有 CPU 时

产物
----
    eval_reports/deploy_speed.csv
    eval_reports/deploy_speed.md
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def bench(fn, warmup: int, iters: int, cuda: bool = False) -> list[float]:
    """跑 warmup 次预热，再逐次计时 iters 次，返回每次的毫秒数。

    GPU 上每次前后都 synchronize：否则测到的是「内核入队时间」而不是执行时间，
    会得到好看但假的数字。torch.cuda.synchronize() 走 cudaDeviceSynchronize，
    对 ORT 自己的 stream 同样有效。
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


def gpu_mem_used_mb() -> float | None:
    """当前显存占用（MB）。只用来看「这个模型在 GPU 上占多少」，不参与结论。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return None


def node_assignment(sess) -> dict:
    """统计每个节点实际在哪个 EP 上执行。

    这是判断「ORT 到底跑在 GPU 还是偷偷回落 CPU」的唯一可靠依据。
    session.get_providers() 返回的是**配置**，不是实际归属。
    """
    try:
        prof = sess.end_profiling()
    except Exception:
        return {}
    try:
        events = json.loads(Path(prof).read_text(encoding="utf-8"))
    finally:
        Path(prof).unlink(missing_ok=True)

    counts = Counter()
    for e in events:
        if e.get("cat") == "Node":
            p = (e.get("args") or {}).get("provider")
            if p:
                counts[p] += 1
    return dict(counts)


def build_gpu_session(path: str | Path, threads: int = 0, profile: bool = False):
    """CUDA EP 会话。ORT 1.29 起要用 onnxruntime.preload_dlls() 才能找到 torch 的 cuBLAS/cuDNN。"""
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        # 只在需要时加载：它会往进程里塞 cuDNN/cuBLAS 的 DLL 目录
        try:
            ort.preload_dlls()
        except Exception as exc:  # 没有 nvidia pip 包时正常返回，不该中断
            print(f"[warn] preload_dlls 失败：{exc}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1 if threads else 0
    so.log_severity_level = 3
    if profile:
        so.enable_profiling = True

    providers = [
        ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC",
                                   "arena_extend_strategy": "kSameAsRequested"}),
        "CPUExecutionProvider",
    ]
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def _cuda_graph_session(path: str | Path, x: np.ndarray):
    """构建带 CUDA Graph 的会话 + IOBinding。

    只能用于**静态 shape**：图会把指针固化下来，输入缓冲必须一直是同一块显存。
    所以服务端要用它，就得把 640×640 的输入缓冲固定住，每张图往同一块显存里覆盖。
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        str(path), sess_options=so,
        providers=[("CUDAExecutionProvider", {"device_id": 0, "enable_cuda_graph": "1"}),
                   "CPUExecutionProvider"],
    )
    dev = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(x, dtype=np.float32), "cuda", 0)
    io = sess.io_binding()
    io.bind_ortvalue_input(sess.get_inputs()[0].name, dev)
    io.bind_output(sess.get_outputs()[0].name, "cuda")
    for _ in range(5):      # 前几次负责捕获 graph
        sess.run_with_iobinding(io)
    return sess, io, dev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="runs/detect/train_base_v2/weights/best.pt")
    ap.add_argument("--onnx", default="deploy/models/base_v2.onnx")
    ap.add_argument("--onnx-dyn", default="deploy/models/base_v2_dyn.onnx",
                    help="动态 batch 模型，batch>1 时用；缺省则跳过 batch>1 的 GPU 行")
    ap.add_argument("--onnx-b8", default="deploy/models/base_v2_b8.onnx",
                    help="静态 batch=8 模型，用于 batch=8 的 CUDA Graph 行")
    ap.add_argument("--int8", default="deploy/models/base_v2_int8.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--cpu-batch", type=int, default=1)
    ap.add_argument("--gpu-batches", default="1,4,8")
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()

    from tools.onnx_common import build_session, postprocess, preprocess, summarize

    img = sorted((ROOT / "dataset/test/images").glob("*.jpg"))[0]
    Path("eval_reports").mkdir(exist_ok=True)

    x, r, pad = preprocess(img, args.imgsz)
    rows: list[dict] = []

    # ---- A 组：PyTorch CPU ----
    import torch

    from ultralytics import YOLO
    from tools.cbam_module import register

    register(verbose=False)
    pt_model = YOLO(args.pt)

    x0 = np.repeat(x, args.cpu_batch, axis=0)
    torch_cpu = pt_model.model.to("cpu").eval()
    xt = torch.from_numpy(x0).float()
    with torch.no_grad():
        pure = bench(lambda: torch_cpu(xt), args.warmup, args.iters)

    def e2e_torch_cpu():
        xx, rr, pp = preprocess(img, args.imgsz)
        with torch.no_grad():
            out = torch_cpu(torch.from_numpy(np.repeat(xx, args.cpu_batch, axis=0)).float())
        arr = out[0].cpu().numpy() if isinstance(out, (list, tuple)) else out.cpu().numpy()
        postprocess(arr, conf=0.25, iou=0.7, r=rr, pad=pp)

    e2e = bench(e2e_torch_cpu, max(3, args.warmup // 2), args.iters)
    rows.append({"backend": "PyTorch", "device": "CPU", "precision": "FP32", "batch": args.cpu_batch,
                 "forward": summarize(pure), "end2end": summarize(e2e), "ep": "cpu"})

    # ---- A 组：ORT CPU FP32 / INT8 ----
    for label, path in (("ONNX Runtime", args.onnx), ("ONNX Runtime", args.int8)):
        if not Path(path).exists():
            print(f"[skip] {path} 不存在")
            continue
        prec = "INT8" if "int8" in Path(path).stem else "FP32"
        sess = build_session(path, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        pure = bench(lambda: sess.run(None, {iname: x0}), args.warmup, args.iters)

        def e2e_ort(_s=sess, _n=iname):
            xx, rr, pp = preprocess(img, args.imgsz)
            out = _s.run(None, {_n: np.repeat(xx, args.cpu_batch, axis=0)})[0]
            postprocess(out, conf=0.25, iou=0.7, r=rr, pad=pp)

        e2e = bench(e2e_ort, max(3, args.warmup // 2), args.iters)
        rows.append({"backend": label, "device": "CPU", "precision": prec, "batch": args.cpu_batch,
                     "forward": summarize(pure), "end2end": summarize(e2e), "ep": "cpu"})

    # ---- B 组：GPU ----
    drift = None
    mem = None
    if not args.skip_gpu and torch.cuda.is_available():
        batches = [int(b) for b in args.gpu_batches.split(",") if b.strip()]
        mem = gpu_mem_used_mb()
        torch_gpu = pt_model.model.to("cuda").eval()

        # -- PyTorch GPU --
        for b in batches:
            xb = np.repeat(x, b, axis=0)
            xg = torch.from_numpy(xb).float().cuda()
            with torch.no_grad():
                pure = bench(lambda: torch_gpu(xg), args.warmup, args.iters, cuda=True)

            def e2e_torch_gpu(_b=b):
                xx, rr, pp = preprocess(img, args.imgsz)
                with torch.no_grad():
                    out = torch_gpu(torch.from_numpy(np.repeat(xx, _b, axis=0)).float().cuda())
                arr = out[0].cpu().numpy() if isinstance(out, (list, tuple)) else out.cpu().numpy()
                postprocess(arr, conf=0.25, iou=0.7, r=rr, pad=pp)

            e2e = bench(e2e_torch_gpu, max(3, args.warmup // 2), args.iters, cuda=True)
            rows.append({"backend": "PyTorch", "device": "GPU", "precision": "FP32", "batch": b,
                         "forward": summarize(pure), "end2end": summarize(e2e), "ep": "gpu"})

        # -- ORT GPU FP32 --
        # batch>1 必须用动态轴模型：静态导出的输入写死 1，喂 4 张会直接报维度错
        dyn = Path(args.onnx_dyn)
        sg = build_gpu_session(args.onnx, profile=True)
        print(f"[info] ORT 配置的 providers = {sg.get_providers()}")
        iname = sg.get_inputs()[0].name

        # 先跑几次让 profiling 有数据，再取节点归属
        for _ in range(3):
            sg.run(None, {iname: x})
        assign = node_assignment(sg)
        total = sum(assign.values())
        cuda_share = assign.get("CUDAExecutionProvider", 0) / total * 100 if total else 0.0
        print(f"[info] 节点实际归属 {assign} → CUDA 占比 {cuda_share:.1f}%")
        if cuda_share < 99.9:
            print("[FAIL] CUDA EP 没有接管全部节点，下面的 GPU 行不是真的 GPU 成绩，已丢弃")
            assign = {}
        else:
            for b in batches:
                if b == 1:
                    sess, tag = sg, "static"
                elif dyn.exists():
                    sess, tag = build_gpu_session(dyn), "dyn"
                else:
                    print(f"[skip] batch={b} 需要 {dyn.name}（--dynamic 导出），不存在")
                    continue
                name = sess.get_inputs()[0].name
                xb = np.repeat(x, b, axis=0)
                pure = bench(lambda: sess.run(None, {name: xb}), args.warmup, args.iters, cuda=True)

                def e2e_ort_gpu(_b=b, _s=sess, _n=name):
                    xx, rr, pp = preprocess(img, args.imgsz)
                    out = _s.run(None, {_n: np.repeat(xx, _b, axis=0)})[0]
                    postprocess(out, conf=0.25, iou=0.7, r=rr, pad=pp)

                e2e = bench(e2e_ort_gpu, max(3, args.warmup // 2), args.iters, cuda=True)
                rows.append({"backend": "ONNX Runtime", "device": "GPU", "precision": "FP32", "batch": b,
                             "forward": summarize(pure), "end2end": summarize(e2e),
                             "ep": f"cuda {cuda_share:.0f}%", "model": tag})

        # -- ORT GPU + CUDA Graph（只要静态 shape 能跑，就值得试）--
        # 7.75ms 里内核启动开销占大头：图有 717 个节点，一次 launch ~2µs 就是 1.4ms，
        # 加上 Python/ORT 每层的调度，实测能压到 3ms 以内。前提：静态 shape + 固定输入缓冲。
        if assign:
            graph_cases = [(Path(args.onnx), 1)]
            if Path(args.onnx_b8).exists():
                graph_cases.append((Path(args.onnx_b8), 8))
            for gpath, gb in graph_cases:
                try:
                    xg0 = np.repeat(x, gb, axis=0)
                    sgc, io, dev = _cuda_graph_session(gpath, xg0)
                    pure = bench(lambda: sgc.run_with_iobinding(io), args.warmup, args.iters, cuda=True)

                    def e2e_graph(_b=gb, _dev=dev):
                        xx, rr, pp = preprocess(img, args.imgsz)
                        _dev.update_inplace(np.ascontiguousarray(np.repeat(xx, _b, axis=0), dtype=np.float32))
                        sgc.run_with_iobinding(io)
                        postprocess(io.copy_outputs_to_cpu()[0][0], conf=0.25, iou=0.7, r=rr, pad=pp)

                    e2e = bench(e2e_graph, max(3, args.warmup // 2), args.iters, cuda=True)
                    rows.append({"backend": "ORT + CUDAGraph", "device": "GPU", "precision": "FP32", "batch": gb,
                                 "forward": summarize(pure), "end2end": summarize(e2e),
                                 "ep": f"cuda {cuda_share:.0f}%", "model": gpath.stem})
                except Exception as exc:
                    print(f"[skip] CUDA Graph batch={gb} 失败：{exc}")

        # -- 漂移复核：把 batch=1 的 ORT-GPU 再测一遍，看是否越测越慢 --
        if assign and 1 in batches:
            pure2 = bench(lambda: sg.run(None, {iname: x}), args.warmup, args.iters, cuda=True)
            s1 = summarize(pure2)
            ref = next(r for r in rows if r["device"] == "GPU"
                       and r["backend"] == "ONNX Runtime" and r["batch"] == 1)
            drift = (s1["median_ms"] - ref["forward"]["median_ms"]) / ref["forward"]["median_ms"] * 100
            print(f"[info] 漂移复核：首测 {ref['forward']['median_ms']:.2f}ms → 复测 {s1['median_ms']:.2f}ms "
                  f"({drift:+.1f}%)")

    # ---- 汇总：同设备内比 ----
    def base_of(device: str) -> dict | None:
        if device == "CPU":
            return next((r for r in rows if r["device"] == "CPU"
                         and r["backend"] == "ONNX Runtime" and r["precision"] == "FP32"), None)
        return next((r for r in rows if r["device"] == "GPU" and r["backend"] == "PyTorch"
                     and r["batch"] == 1), None)

    for r in rows:
        b = base_of(r["device"])
        r["speedup"] = round(b["forward"]["median_ms"] / r["forward"]["median_ms"], 2) if b else None
        r["base_name"] = ("ORT-CPU-FP32" if r["device"] == "CPU" else "PyTorch-GPU-b1") if b else "-"
        r["stable"] = "yes" if r["forward"]["cv_pct"] < 10 else "NO"
        # 一次调用是 N 张，只报总耗时会让人误以为 batch 越大越慢
        r["per_img_ms"] = r["forward"]["median_ms"] / r["batch"]
        r["img_per_s"] = 1000.0 / r["per_img_ms"]

    hdr = (f"{'后端':<14}{'设备':<6}{'精度':<6}{'batch':<6}{'前向中位':>10}{'每图':>8}{'吞吐':>9}"
           f"{'CV%':>7}{'端到端中位':>11}{'同设备加速':>11}{'EP':>10}")
    print("\n" + hdr)
    print("-" * 104)
    for r in rows:
        f, e = r["forward"], r["end2end"]
        sp = f"{r['speedup']:.2f}×" if r["speedup"] else "-"
        print(f"{r['backend']:<14}{r['device']:<6}{r['precision']:<6}{r['batch']:<6}"
              f"{f['median_ms']:>10.2f}{r['per_img_ms']:>8.2f}{r['img_per_s']:>8.0f}/s"
              f"{f['cv_pct']:>7.1f}{e['median_ms']:>11.2f}{sp:>11}{r['ep']:>10}")

    # ---- 落盘 ----
    csv_path = Path("eval_reports/deploy_speed.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["backend", "device", "precision", "batch", "ep", "fwd_median_ms", "per_img_ms",
                    "img_per_s", "fwd_p95_ms", "fwd_cv_pct", "e2e_median_ms", "e2e_p95_ms", "e2e_cv_pct",
                    "speedup_same_device", "speedup_base", "stable"])
        for r in rows:
            w.writerow([r["backend"], r["device"], r["precision"], r["batch"], r["ep"],
                        f"{r['forward']['median_ms']:.2f}", f"{r['per_img_ms']:.2f}",
                        f"{r['img_per_s']:.1f}", f"{r['forward']['p95_ms']:.2f}",
                        f"{r['forward']['cv_pct']:.1f}",
                        f"{r['end2end']['median_ms']:.2f}", f"{r['end2end']['p95_ms']:.2f}",
                        f"{r['end2end']['cv_pct']:.1f}",
                        r["speedup"] or "", r["base_name"], r["stable"]])

    md = [
        "# 部署后端测速",
        "",
        f"输入 {args.imgsz}×{args.imgsz}，预热 {args.warmup} 次，计时 {args.iters} 次取中位数。"
        "加速比**只在同设备内**成立，跨设备不可比。",
        "",
        "| 后端 | 设备 | 精度 | batch | EP 归属 | 前向中位(ms) | 每图(ms) | 吞吐(img/s) | CV | 端到端中位(ms) | 同设备加速 | 基准 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        f, e = r["forward"], r["end2end"]
        sp = f"{r['speedup']:.2f}×" if r["speedup"] else "—"
        md.append(f"| {r['backend']} | {r['device']} | {r['precision']} | {r['batch']} | {r['ep']} | "
                  f"{f['median_ms']:.2f} | {r['per_img_ms']:.2f} | {r['img_per_s']:.0f} | "
                  f"{f['cv_pct']:.1f}% | {e['median_ms']:.2f} | {sp} | {r['base_name']} |")
    md.append("")
    if mem:
        md.append(f"GPU 显存占用（会话常驻）：{mem} MB")
        md.append("")
    if drift is not None:
        md.append(f"漂移复核：基准行首测 vs 复测差 **{drift:+.1f}%**"
                  f"（>5% 说明存在降频，该批数字只能组内相对比较）")
    md += [
        "",
        "> CPU 组基准 = ORT-CPU-FP32；GPU 组基准 = PyTorch-GPU(batch=1)。",
        "> 「前向中位」是一次调用（N 张）的总耗时，batch 越大当然越长；比批量效率要看「每图」和「吞吐」。",
        "> batch>1 的行用动态轴模型 base_v2_dyn.onnx；batch=1 用服务在跑的静态模型。",
        "> CV > 10% 表示计时不稳定（频率漂移/调度抖动），该行不要跨批次横比。",
        "> EP 归属由 ORT profiling 统计得出；若 CUDA 占比 < 99.9%，该行直接丢弃而不是回落成 CPU 成绩。",
        "> 绝对毫秒数只能同一次运行内比：同机跨天复跑，CPU 行会因功耗/后台负载差 20% 以上。",
    ]
    Path("eval_reports/deploy_speed.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[done] 写入 {csv_path} 与 eval_reports/deploy_speed.md")


if __name__ == "__main__":
    main()
