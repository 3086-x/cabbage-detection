"""ONNX 推理的公共层：预处理、后处理、会话构建。

为什么单独抽一层
--------------
测速、精度评估、HTTP 服务三处都要喂同一张图。如果各自写一份预处理，
最容易出的错是「服务里 BGR/RGB 搞反」或「忘记 /255」——这类错误不会报错，
只会让精度悄悄掉几个点，而且测速时预处理成本也对不上。

所以预处理只在这里写一次，三处都 import 它。

与 ultralytics 对齐的细节
----------------------
1. letterbox 用 cv2.copyMakeBorder，填充值 114（YOLO 系列约定），
   居中填充，round(d-0.1)/round(d+0.1) 保证左右/上下合计精确。
2. 通道顺序 BGR→RGB，因为 ultralytics 训练时是 RGB。
3. 归一化 /255 在预处理做，导出的 ONNX 图内部不再做归一化。
4. 后处理：输出 (1, 4+nc, N)，前 4 行是 cx/cy/w/h（输入尺度像素），
   后面是各类分数，取 max 作为置信度，再 NMS。
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

PAD_VALUE = 114


# --------------------------------------------------------------------------
# 预处理
# --------------------------------------------------------------------------
def letterbox(im: np.ndarray, new_shape: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    """等比缩放并居中填充到 new_shape×new_shape。

    返回 (处理后图像, 缩放比 r, (左侧填充, 顶部填充))，后两者用于把框映射回原图。
    """
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return im, r, (left, top)


def preprocess(path: str | Path, imgsz: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    """读图 → letterbox → BGR2RGB → CHW → /255 → NCHW float32。"""
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(f"读不到图片: {path}")
    return preprocess_array(im, imgsz)


def preprocess_array(im: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    im, r, pad = letterbox(im, imgsz)
    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    return x[None, ...], r, pad


# --------------------------------------------------------------------------
# 后处理
# --------------------------------------------------------------------------
def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.7) -> np.ndarray:
    """纯 numpy NMS，返回保留下标。逐类调用（本任务单类，等价于全局）。"""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.array(keep, dtype=np.int64)


def postprocess(
    output: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.7,
    r: float = 1.0,
    pad: tuple[float, float] = (0.0, 0.0),
    orig_shape: tuple[int, int] | None = None,
) -> dict:
    """解码 YOLO 输出并映射回原图坐标。

    参数
    ----
    output : (1, 4+nc, N)，前 4 行是 cx/cy/w/h。
    r, pad : letterbox 的返回，用于把坐标还原到原图。
    orig_shape : (h, w)，为 None 时不还原。

    返回 {boxes(N,4) xyxy 原图坐标, scores(N,), classes(N,)}。
    """
    pred = output[0] if output.ndim == 3 else output  # (4+nc, N)
    pred = pred.T  # (N, 4+nc)

    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    if cls_scores.shape[1] == 0:
        return {"boxes": np.empty((0, 4)), "scores": np.empty((0,)), "classes": np.empty((0,), dtype=int)}

    classes = cls_scores.argmax(axis=1)
    scores = cls_scores.max(axis=1)

    mask = scores >= conf
    if not mask.any():
        return {"boxes": np.empty((0, 4)), "scores": np.empty((0,)), "classes": np.empty((0,), dtype=int)}

    b, s, c = boxes_xywh[mask], scores[mask], classes[mask]

    # cxcywh -> xyxy（仍在 letterbox 尺度）
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # 还原到原图：先减填充，再除以缩放比
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / r
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / r

    if orig_shape is not None:
        oh, ow = orig_shape
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, ow)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, oh)

    keep = nms_numpy(xyxy, s, iou)
    return {"boxes": xyxy[keep], "scores": s[keep], "classes": c[keep]}


# --------------------------------------------------------------------------
# 会话
# --------------------------------------------------------------------------
def build_session(onnx_path: str | Path, providers: list[str] | None = None, threads: int = 0):
    """构建 ORT InferenceSession。

    threads=0 交给 ORT 自己按物理核决定；显式给值时关掉并行执行，
    避免多线程服务里线程数乘出几十个线程互相抢核。
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1 if threads else 0
    so.log_severity_level = 3

    if providers is None:
        avail = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else [
            "CPUExecutionProvider"
        ]

    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    return sess


def predict_array(sess, x: np.ndarray, conf: float = 0.25, iou: float = 0.7, r: float = 1.0,
                  pad: tuple[float, float] = (0.0, 0.0), orig_shape=None) -> dict:
    """对已预处理的 NCHW 数组跑一次前向 + 后处理。"""
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: x})[0]
    return postprocess(out, conf=conf, iou=iou, r=r, pad=pad, orig_shape=orig_shape)


def predict_file(sess, path: str | Path, imgsz: int = 640, conf: float = 0.25, iou: float = 0.7) -> dict:
    x, r, pad = preprocess(path, imgsz)
    im = cv2.imread(str(path))
    return predict_array(sess, x, conf=conf, iou=iou, r=r, pad=pad, orig_shape=im.shape[:2])


# --------------------------------------------------------------------------
# 计时
# --------------------------------------------------------------------------
class Timer:
    """用 perf_counter 记时，不依赖 CUDA event。

    ONNX Runtime 的 CPU 推理是同步的，perf_counter 就是准确耗时；
    GPU 推理（PyTorch）要额外 synchronize，调用方自己负责。
    """

    def __init__(self) -> None:
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._t0) * 1000


def summarize(samples_ms: list[float]) -> dict:
    """中位数 + P95 + 变异系数。CV 高说明计时不可信，必须报出来而不是藏起来。"""
    a = np.asarray(samples_ms, dtype=float)
    med = float(np.median(a))
    return {
        "n": int(a.size),
        "median_ms": med,
        "mean_ms": float(a.mean()),
        "p95_ms": float(np.percentile(a, 95)),
        "p05_ms": float(np.percentile(a, 5)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        # CV = 标准差/均值。>10% 基本说明有频率漂移或调度抖动，跨批次不可比
        "cv_pct": float(a.std(ddof=0) / a.mean() * 100) if a.mean() > 0 else float("nan"),
    }
