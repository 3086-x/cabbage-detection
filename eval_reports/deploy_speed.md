# 部署后端测速

输入 640×640，预热 25 次，计时 150 次取中位数。加速比**只在同设备内**成立，跨设备不可比。

| 后端 | 设备 | 精度 | batch | EP 归属 | 前向中位(ms) | 每图(ms) | 吞吐(img/s) | CV | 端到端中位(ms) | 同设备加速 | 基准 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch | CPU | FP32 | 1 | cpu | 76.44 | 76.44 | 13 | 6.6% | 98.47 | 0.55× | ORT-CPU-FP32 |
| ONNX Runtime | CPU | FP32 | 1 | cpu | 42.27 | 42.27 | 24 | 4.9% | 70.86 | 1.00× | ORT-CPU-FP32 |
| ONNX Runtime | CPU | INT8 | 1 | cpu | 51.93 | 51.93 | 19 | 6.7% | 76.60 | 0.81× | ORT-CPU-FP32 |
| PyTorch | GPU | FP32 | 1 | gpu | 12.90 | 12.90 | 78 | 8.3% | 28.93 | 1.00× | PyTorch-GPU-b1 |
| PyTorch | GPU | FP32 | 4 | gpu | 12.59 | 3.15 | 318 | 3.7% | 35.06 | 1.02× | PyTorch-GPU-b1 |
| PyTorch | GPU | FP32 | 8 | gpu | 15.85 | 1.98 | 505 | 16.1% | 45.45 | 0.81× | PyTorch-GPU-b1 |
| ONNX Runtime | GPU | FP32 | 1 | cuda 100% | 7.64 | 7.64 | 131 | 7.6% | 21.41 | 1.69× | PyTorch-GPU-b1 |
| ONNX Runtime | GPU | FP32 | 4 | cuda 100% | 14.07 | 3.52 | 284 | 8.9% | 32.47 | 0.92× | PyTorch-GPU-b1 |
| ONNX Runtime | GPU | FP32 | 8 | cuda 100% | 23.26 | 2.91 | 344 | 2.5% | 45.93 | 0.55× | PyTorch-GPU-b1 |
| ORT + CUDAGraph | GPU | FP32 | 1 | cuda 100% | 3.03 | 3.03 | 330 | 4.8% | 19.86 | 4.26× | PyTorch-GPU-b1 |
| ORT + CUDAGraph | GPU | FP32 | 8 | cuda 100% | 14.61 | 1.83 | 548 | 4.7% | 48.53 | 0.88× | PyTorch-GPU-b1 |

GPU 显存占用（会话常驻）：53, 180 MB

漂移复核：基准行首测 vs 复测差 **+3.8%**（>5% 说明存在降频，该批数字只能组内相对比较）

> CPU 组基准 = ORT-CPU-FP32；GPU 组基准 = PyTorch-GPU(batch=1)。
> 「前向中位」是一次调用（N 张）的总耗时，batch 越大当然越长；比批量效率要看「每图」和「吞吐」。
> batch>1 的行用动态轴模型 base_v2_dyn.onnx；batch=1 用服务在跑的静态模型。
> CV > 10% 表示计时不稳定（频率漂移/调度抖动），该行不要跨批次横比。
> EP 归属由 ORT profiling 统计得出；若 CUDA 占比 < 99.9%，该行直接丢弃而不是回落成 CPU 成绩。
> 绝对毫秒数只能同一次运行内比：同机跨天复跑，CPU 行会因功耗/后台负载差 20% 以上。