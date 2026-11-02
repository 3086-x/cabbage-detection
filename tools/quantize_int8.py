"""FP32 → INT8 静态量化（QDQ），校准集只取 train，绝不碰 test。

为什么用静态量化而不是动态
------------------------
动态量化只量化权重、激活在运行时算 scale，对卷积为主的检测网络加速有限；
静态量化在离线阶段用校准集统计每一层激活的分布，把激活也定成定点，
CPU 上才能走到 VNNI/oneDNN 的整数卷积路径。代价是**必须给校准集**，
而且校准集的分布决定掉点多少——这就是为什么校准只能看 train。

为什么校准不能碰 test
------------------
test 只有 75 张，它在这个仓库里的唯一用途是「最终报数」。
一旦拿它校准，report 出来的 mAP 就是自我实现的，和调参没区别。

跑法
----
    python tools/quantize_int8.py --onnx deploy/models/base_v2.onnx --calib 200
    python tools/quantize_int8.py --onnx deploy/models/base_v2.onnx --format QOperator

产物
----
    deploy/models/<tag>_int8.onnx
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class CalibReader:
    """把 train 里的图片按 ultralytics 的预处理喂给量化器。"""

    def __init__(self, files: list[Path], imgsz: int, input_name: str):
        from tools.onnx_common import preprocess

        self._files = files
        self._imgsz = imgsz
        self._name = input_name
        self._i = 0
        # 预热时就把预处理算好会占太多内存（200 张 × 640×640×3×4B ≈ 1GB），
        # 所以按需算。
        self._preprocess = staticmethod(preprocess)

    def get_next(self):
        if self._i >= len(self._files):
            return None
        f = self._files[self._i]
        self._i += 1
        x, _, _ = self._preprocess(f, self._imgsz)
        return {self._name: np.ascontiguousarray(x, dtype=np.float32)}

    def rewind(self):
        self._i = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", type=int, default=200, help="校准图片数，从 train 抽")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--format", default="QDQ", choices=["QDQ", "QOperator"])
    ap.add_argument("--per-channel", action="store_true", default=True)
    ap.add_argument(
        "--op-types",
        default="Conv",
        help=(
            "只量化这些算子，逗号分隔。默认只量化 Conv：YOLO 的算力 95%+ 在卷积上，"
            "而检测头的 Concat/Sigmoid 一旦被量化，分类分数会整体塌成 0（实测过）。"
        ),
    )
    ap.add_argument(
        "--calib-method",
        default="MinMax",
        choices=["MinMax", "Entropy", "Percentile"],
        help="Percentile 在部分检测头上会把范围截得过窄，导致激活折成常数，默认 MinMax 更稳。",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    src = Path(args.onnx)
    if not src.exists():
        raise SystemExit(f"找不到 {src}")

    tag = src.stem
    dst = Path(args.out) if args.out else src.with_name(f"{tag}_int8.onnx")

    # ---- 校准集：train 里等间隔抽样，保证覆盖不同光照/密度 ----
    train_imgs = sorted((ROOT / "dataset/train/images").glob("*.jpg"))
    if not train_imgs:
        raise SystemExit("找不到 train 图片，无法校准")
    step = max(1, len(train_imgs) // args.calib)
    calib = train_imgs[::step][: args.calib]
    print(f"[info] 校准集 {len(calib)} 张（取自 train，共 {len(train_imgs)} 张），test 不参与")

    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    reader = CalibReader(calib, args.imgsz, input_name)

    qformat = QuantFormat.QDQ if args.format == "QDQ" else QuantFormat.QOperator
    calib_method = getattr(CalibrationMethod, args.calib_method)
    # all = 不限制算子类型，让 ORT 尽可能量化整条链路（整数算子之间不用来回搬数据）
    op_types = None if args.op_types.strip().lower() == "all" else [
        t.strip() for t in args.op_types.split(",") if t.strip()
    ]
    print(f"[info] 格式={args.format}  量化算子={op_types or '全部'}  校准法={args.calib_method}")

    t0 = time.perf_counter()
    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=qformat,
        activation_type=QuantType.QUInt8,   # CPU 上 u8s8 组合走 oneDNN 最快
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        calibrate_method=calib_method,
        op_types_to_quantize=op_types,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    dt = time.perf_counter() - t0

    src_mb, dst_mb = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
    print(f"[ok] 量化完成 {dst}  用时 {dt:.1f}s")
    print(f"[ok] 体积 {src_mb:.2f} MB -> {dst_mb:.2f} MB（{dst_mb / src_mb:.2f}×）")

    # 一致性自检：形状对不代表内容对。检测头一旦被量化坏掉，
    # 坐标那 4 行看起来还正常，分类分数却会整体塌成 0——只看形状完全发现不了。
    q = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    f = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    from tools.onnx_common import preprocess

    x, _, _ = preprocess(sorted((ROOT / "dataset/test/images").glob("*.jpg"))[0], args.imgsz)
    out_q = q.run(None, {q.get_inputs()[0].name: x})[0]
    out_f = f.run(None, {f.get_inputs()[0].name: x})[0]

    coord_ratio = float(np.abs(out_q[:, :4, :] - out_f[:, :4, :]).mean() / (np.abs(out_f[:, :4, :]).mean() + 1e-9))
    conf_q_max = float(out_q[:, 4:, :].max())
    conf_f_max = float(out_f[:, 4:, :].max())
    print(f"[check] 输出形状 {out_q.shape}")
    print(f"[check] 坐标平均相对偏差 {coord_ratio * 100:.2f}%")
    print(f"[check] 分类分数峰值 FP32={conf_f_max:.4f} -> INT8={conf_q_max:.4f}")

    if conf_f_max > 0.1 and conf_q_max < 0.1 * conf_f_max:
        print("[FAIL] 分类分支被量化打塌（峰值几乎归零），这个模型不可用")
        raise SystemExit(1)
    if coord_ratio > 0.02:
        print("[FAIL] 坐标回归偏差过大，这个模型不可用")
        raise SystemExit(1)
    print("[ok] 量化后前向结果未退化")


if __name__ == "__main__":
    main()
