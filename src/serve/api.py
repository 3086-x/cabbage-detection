"""甘蓝检测推理服务（FastAPI + ONNX Runtime）。

为什么是 ONNX Runtime 而不是 PyTorch
--------------------------------
推理服务不需要反向传播、优化器、数据加载器。带 torch 的镜像光依赖就 2~3 GB，
而 ONNX Runtime 只需要 ~50 MB。本仓库实测同硬件 ONNX 前向比 PyTorch CPU 快 1.66×，
所以服务端固定用 ONNX。

三处刻意的设计
------------
1. **推理必须离开事件循环**。ORT 的 run 是阻塞调用，直接在 async 函数里跑会把
   整个事件循环卡住，/health 都会超时。所以统一丢进线程池。

2. **并发靠信号量而不是靠调线程数**。ORT session 内部的 intra_op 线程池是被所有
   请求共享的：如果 24 个请求同时进来、每个都想用 12 个线程，实际是 288 个线程
   抢 12 个核。这里把 intra_op 固定成较小值，并用 Semaphore 限制同时推理的请求数，
   总线程数 = workers × intra_op ≈ 物理核数，不超订。

3. **返回结构化结论而不是一张标注图**。上层要的是「多少株、密度如何、置信度分布
   长什么样」，用来决定要不要换阈值重跑；画框图在浏览器里做比在服务端做便宜得多。

跑法
----
    uvicorn src.serve.api:app --host 0.0.0.0 --port 8000
    # 或
    python -m src.serve.api

接口
----
    GET  /health          模型状态、后端、预热耗时
    GET  /metrics         累计请求数、错误数、最近 60s QPS、延迟分位
    POST /predict         单图
    POST /predict/batch   多图
    GET  /                自述页
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from tools.onnx_common import build_session, postprocess, preprocess_array  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", "deploy/models/base_v2.onnx"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
# 同时推理的请求数上限。物理核 12 → 4 路并发 × 3 线程，正好吃满且不超订。
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
INTRA_THREADS = int(os.getenv("INTRA_THREADS", "3"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "16"))

# 密度分档阈值：test 75 张真实株数的 33/66 分位（46 / 52）。
# 用固定阈值而不是按请求动态算分位，是因为单张请求没有「总体」可言。
DENSITY_Q33, DENSITY_Q66 = 46.0, 52.0


# --------------------------------------------------------------------------
# 响应模型
# --------------------------------------------------------------------------
class Detection(BaseModel):
    box: list[float]
    score: float


class StageMs(BaseModel):
    preprocess: float
    inference: float
    postprocess: float
    total: float


class PredictResult(BaseModel):
    image: str
    width: int
    height: int
    count: int
    density: str
    conf_threshold: float
    iou_threshold: float
    detections: list[Detection]
    score_stats: dict
    stage_ms: StageMs
    backend: str
    truncated: bool = False


class BatchResult(BaseModel):
    n: int
    ok: int
    failed: int
    wall_ms: float
    results: list[PredictResult]
    errors: list[dict]


# --------------------------------------------------------------------------
# 推理核心
# --------------------------------------------------------------------------
def _density(count: int) -> str:
    if count < DENSITY_Q33:
        return "sparse"
    if count <= DENSITY_Q66:
        return "medium"
    return "dense"


def _score_stats(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "min": round(float(scores.min()), 4),
        "p25": round(float(np.percentile(scores, 25)), 4),
        "median": round(float(np.median(scores)), 4),
        "p75": round(float(np.percentile(scores, 75)), 4),
        "max": round(float(scores.max()), 4),
    }


class Detector:
    def __init__(self, model_path: Path, imgsz: int, intra_threads: int) -> None:
        self.model_path = model_path
        self.imgsz = imgsz
        self.intra_threads = intra_threads
        self.sess = None
        self.input_name = ""
        self.provider = ""
        self.load_ms = 0.0
        self.warmup_ms = 0.0

    def load(self) -> None:
        t0 = time.perf_counter()
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型不存在: {self.model_path}")
        self.sess = build_session(self.model_path, threads=self.intra_threads)
        self.input_name = self.sess.get_inputs()[0].name
        self.provider = self.sess.get_providers()[0]
        self.load_ms = (time.perf_counter() - t0) * 1000

        # 预热：ORT 第一次 run 会做内存分配和算子选择，不预热的话首个真实请求
        # 要背 200~400ms 的冷启动，P95 会被这一条拉爆。
        dummy = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        t1 = time.perf_counter()
        for _ in range(2):
            self.sess.run(None, {self.input_name: dummy})
        self.warmup_ms = (time.perf_counter() - t1) * 1000

    def infer(self, im: np.ndarray, conf: float, iou: float) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        x, r, pad = preprocess_array(im, self.imgsz)
        t1 = time.perf_counter()

        out = self.sess.run(None, {self.input_name: x})[0]
        t2 = time.perf_counter()

        det = postprocess(out, conf=conf, iou=iou, r=r, pad=pad, orig_shape=im.shape[:2])
        t3 = time.perf_counter()

        stage = {
            "preprocess": round((t1 - t0) * 1000, 2),
            "inference": round((t2 - t1) * 1000, 2),
            "postprocess": round((t3 - t2) * 1000, 2),
            "total": round((t3 - t0) * 1000, 2),
        }
        return det, stage


class Metrics:
    """只保留最近 N 条延迟做分位，避免长跑服务内存无限涨。"""

    def __init__(self, window: int = 2000) -> None:
        self.total = 0
        self.errors = 0
        self.latencies: deque[tuple[float, float]] = deque(maxlen=window)

    def record(self, ms: float, ok: bool = True) -> None:
        self.total += 1
        if not ok:
            self.errors += 1
        self.latencies.append((time.time(), ms))

    def snapshot(self) -> dict:
        arr = np.array([ms for _, ms in self.latencies], dtype=float) if self.latencies else np.array([])
        now = time.time()
        recent = [1 for ts, _ in self.latencies if now - ts <= 60]
        qps = len(recent) / 60.0
        def p(q: float) -> float:
            return round(float(np.percentile(arr, q)), 2) if arr.size else 0.0
        return {
            "requests_total": self.total,
            "errors_total": self.errors,
            "qps_last_60s": round(qps, 2),
            "latency_ms": {"p50": p(50), "p95": p(95), "p99": p(99), "max": p(100)},
            "samples_in_window": int(arr.size),
        }


detector: Detector | None = None
metrics = Metrics()
_sem: asyncio.Semaphore | None = None
_pool: ThreadPoolExecutor | None = None


# --------------------------------------------------------------------------
# 应用
# --------------------------------------------------------------------------
async def _lifespan(app: FastAPI):
    global detector, _sem, _pool
    detector = Detector(MODEL_PATH, IMGSZ, INTRA_THREADS)
    await asyncio.to_thread(detector.load)
    _sem = asyncio.Semaphore(MAX_WORKERS)
    _pool = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="infer")
    print(
        f"[serve] 模型已加载 {MODEL_PATH.name}  provider={detector.provider}  "
        f"load={detector.load_ms:.0f}ms warmup={detector.warmup_ms:.0f}ms  "
        f"workers={MAX_WORKERS} intra_op={INTRA_THREADS}"
    )
    yield
    _pool.shutdown(wait=False)


app = FastAPI(
    title="甘蓝检测推理服务",
    description="无人机航拍密集小目标检测 · ONNX Runtime 后端",
    version="0.1.0",
    lifespan=_lifespan,
)


def _decode(data: bytes, name: str) -> np.ndarray:
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{name} 超过 {MAX_UPLOAD_MB}MB 限制")
    im = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise HTTPException(400, f"{name} 不是可解码的图片")
    return im


async def _run_one(data: bytes, name: str, conf: float, iou: float, max_det: int) -> PredictResult:
    if detector is None:
        raise HTTPException(503, "模型尚未就绪")
    im = await asyncio.to_thread(_decode, data, name)

    async with _sem:  # 限制同时推理的请求数
        loop = asyncio.get_running_loop()
        det, stage = await loop.run_in_executor(_pool, detector.infer, im, conf, iou)

    scores = det["scores"]
    truncated = len(scores) > max_det
    boxes = det["boxes"][:max_det]
    scores_out = scores[:max_det]

    metrics.record(stage["total"])
    return PredictResult(
        image=name,
        width=int(im.shape[1]),
        height=int(im.shape[0]),
        count=int(len(scores)),          # 株数用未截断的总数，明细才截断
        density=_density(len(scores)),
        conf_threshold=conf,
        iou_threshold=iou,
        detections=[Detection(box=[round(float(v), 1) for v in b], score=round(float(s), 4))
                    for b, s in zip(boxes, scores_out)],
        score_stats=_score_stats(scores),
        stage_ms=StageMs(**stage),
        backend=f"onnxruntime/{detector.provider}",
        truncated=truncated,
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """<!doctype html><meta charset=utf-8><title>甘蓝检测服务</title>
<body style="font:15px/1.7 system-ui;max-width:640px;margin:48px auto;padding:0 16px">
<h2>甘蓝检测推理服务</h2>
<p>ONNX Runtime 后端 · 单类密集小目标检测</p>
<ul>
<li><code>GET /health</code> 健康检查</li>
<li><code>GET /metrics</code> 运行时指标</li>
<li><code>POST /predict</code> 单图（multipart: file）</li>
<li><code>POST /predict/batch</code> 多图（multipart: files）</li>
</ul>
<p>交互式文档：<a href="/docs">/docs</a></p></body>"""


@app.get("/health")
async def health() -> dict:
    if detector is None or detector.sess is None:
        raise HTTPException(503, "模型未加载")
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "imgsz": IMGSZ,
        "provider": detector.provider,
        "load_ms": round(detector.load_ms, 1),
        "warmup_ms": round(detector.warmup_ms, 1),
        "concurrency": MAX_WORKERS,
        "intra_op_threads": INTRA_THREADS,
    }


@app.get("/metrics")
async def get_metrics() -> dict:
    return metrics.snapshot()


@app.post("/predict", response_model=PredictResult)
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=0.99),
    iou: float = Query(0.7, ge=0.1, le=0.95),
    max_det: int = Query(100, ge=1, le=1000),
) -> PredictResult:
    data = await file.read()
    return await _run_one(data, file.filename or "upload", conf, iou, max_det)


@app.post("/predict/batch", response_model=BatchResult)
async def predict_batch(
    files: list[UploadFile] = File(...),
    conf: float = Query(0.25, ge=0.01, le=0.99),
    iou: float = Query(0.7, ge=0.1, le=0.95),
    max_det: int = Query(100, ge=1, le=1000),
) -> BatchResult:
    if len(files) > 64:
        raise HTTPException(413, "单次最多 64 张")

    t0 = time.perf_counter()
    payload = [(await f.read(), f.filename or f"upload_{i}") for i, f in enumerate(files)]
    results, errors = [], []

    # 并发提交，真正的并发度由 Semaphore 控制，不会因为提交 64 个就把 CPU 打爆
    async def one(data: bytes, name: str):
        try:
            return await _run_one(data, name, conf, iou, max_det), None
        except HTTPException as e:
            return None, {"image": name, "error": e.detail, "status": e.status_code}
        except Exception as e:  # noqa: BLE001
            return None, {"image": name, "error": str(e), "status": 500}

    for res, err in await asyncio.gather(*(one(d, n) for d, n in payload)):
        if res is not None:
            results.append(res)
        if err is not None:
            errors.append(err)

    return BatchResult(
        n=len(files),
        ok=len(results),
        failed=len(errors),
        wall_ms=round((time.perf_counter() - t0) * 1000, 1),
        results=results,
        errors=errors,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serve.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
