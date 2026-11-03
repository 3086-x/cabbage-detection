# 部署：ONNX 导出、量化取舍与 HTTP 服务

这份文档记录把训练好的权重变成可用服务的全过程，包括**两个会让精度悄悄掉点的坑**和**一个与主流说法相反的负结果**。

所有数字都可以用仓库里的脚本重跑，命令写在每节末尾。

## 结论先行

| 项 | 结果 |
|---|---|
| ONNX 导出是否无损 | **是**。同输入下 PyTorch 与 ONNX 的 mAP50-95 差 **0.0001**，框级坐标偏差 0.024px |
| ONNX CPU 比 PyTorch CPU 快多少 | **1.81×**（前向 76.44ms → 42.27ms，同一次运行内） |
| 同一个 GPU 上 ONNX 比 PyTorch 快多少 | **1.69×**（12.90ms → 7.64ms）；再开 CUDA Graph → **3.03ms，4.26×** |
| INT8 量化是否值得 | **不值得**。精度掉 0.56pp，速度**慢 23%**（51.93ms vs 42.27ms） |
| GPU 批量谁赢 | batch=8 裸 ORT **输**（344 vs 505 img/s），加 CUDA Graph **赢**（548 img/s） |
| 服务吞吐 | 4 路并发约 **30 QPS**，P95 **156ms**；再往上加并发只增延迟不增吞吐 |
| Docker 镜像 | 配置已就位，**本机无 Docker 未构建验证** |

> 版本说明：CPU 那三行是 ORT **1.26.0** 测的。上一版用的是 1.31.1（PyTorch 55.10 / ORT 33.25，1.66×）。
> 换版本是因为 1.31 的 CUDA EP 要 CUDA 13，而本机 torch 是 cu128（见第五节）。两个版本的 CPU 前向差别
> 与机器状态（功耗/后台负载）叠在一起，**没法彻底拆开**：PyTorch 那行也一起从 55 变到 76，说明至少有一半
> 是环境漂移。所以绝对值只能同一次运行内比，两次运行一致指向的是「ORT 比 PyTorch 快 1.7 倍上下」这件事。

## 一、导出与自检

`model.export(format="onnx")` 成功返回只说明图能被序列化，不说明输出对。所以导出后做两级检查：

**张量级**：同一输入，逐元素比 PyTorch 与 ORT 的原始输出。分「坐标 / 置信度」两段报，因为前者是 0~640 的像素、后者是 0~1 的概率，混在一起算会被坐标段主导，看不出分类头坏了没有。

**框级**：同一张真实图片，两条链路各自解码，比检出数和框坐标。这一级覆盖了 letterbox、NMS、坐标还原的完整链路。

| 模型 | 张量级 坐标 maxΔ | 张量级 置信度 maxΔ | 框级 检出数 | 框级 最大坐标偏差 |
|---|---|---|---|---|
| base_v2 | 1.5e-3 px（0.0002%） | 9.6e-6 | 51 vs 51 | **0.000 px** |
| cbam_v2 | 2.1e-4 px（0.0000%） | 1.0e-6 | 45 vs 45 | **0.000 px** |

CBAM 是自定义模块，ultralytics 里没有这个类。它能正确导出、且自检比基线还干净，说明「自定义注意力层可以直接 trace 成基础算子」——这一点对想加自定义结构的人有用。

```bash
python tools/export_onnx.py --weights runs/detect/train_base_v2/weights/best.pt --tag base_v2
python tools/export_onnx.py --weights runs/detect/train_cbam/weights/best.pt --tag cbam_v2
```

产物：`deploy/models/*.onnx` + `deploy/onnx_manifest.json`（输入输出签名 + 自检结论）。

## 二、坑 1：mAP 掉了 1.36pp，但不是精度问题

导出后直接跑 `model.val()`，ONNX 的 mAP50-95 是 0.7774，PyTorch 是 0.7910，掉了 1.36pp。

而框级自检显示两者**逐框一致**（51 vs 51，坐标偏差 0.000px）。矛盾，所以去查评估管线，发现 val 的输入张量是 **672×672**，不是 640：

```
val 的 img shape (3, 672, 672)   ← rect=True 时 ultralytics 把 640 补齐到 stride
我喂的 img shape (3, 640, 640)
```

ultralytics 的默认 `rect=True` 会按 stride 把输入尺寸往上补。固定 640 输入的 ONNX 模型拿到 672 的图后，会在后端内部**再缩一次到 640**——两次插值，细节被磨掉，mAP 就掉了。

把 `rect` 固定成 `False`，两边都用 640：

| 评估口径 | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| PyTorch @672（历史口径） | 0.9715 | 0.7910 | 0.9738 | 0.9472 |
| PyTorch @640 | 0.9636 | 0.7774 | 0.9536 | 0.9438 |
| ONNX FP32 @640 | 0.9636 | **0.7774** | 0.9536 | 0.9440 |

差 0.0001 —— 在舍入噪声量级。**导出无损，之前那 1.36pp 全是评估管线伪影。**

这个坑的通用教训：**评估固定输入尺寸的 ONNX 模型时，必须显式关掉 rect**，否则测的是「导出对不对 + 二次缩放」两件事的叠加。它不报错，只是数字变小。

顺带一个真实收益：同一模型 **672 输入比 640 高 1.35pp**（0.7910 vs 0.7774）。甘蓝直径在 640 下只有 20~40px，输入再大一点对小目标是有用的。本仓库的测速和部署仍按 640 走，是为了和已有数字可比。

```bash
python tools/eval_full.py --weights deploy/models/base_v2.onnx --split test \
    --conf 0.65 --iou 0.5 --rect false --name base_v2_onnx_fp32_c640
```

## 三、坑 2：INT8 把分类分支打塌了

第一次量化（全算子 + Percentile 校准），模型输出形状完全正常，但 mAP 是 **0.0000**、检出 0 个框。

拆开看原始输出：

| | 坐标区（前 4 行） | 分类分数峰值 |
|---|---|---|
| FP32 | max 656.4 / mean 155.9 | 0.8722 |
| INT8（坏） | max 648.9 / mean 155.2 | **0.0000** |

坐标回归完好，分类分支全塌成 0。原因是检测头的输出 logits 有正有负，全量化时被无符号量化截断。

试了四条路：

| 配置 | 分类峰值 | 可用 |
|---|---|---|
| 全算子 + Percentile + QDQ | 0.0000 | 否 |
| 全算子 + MinMax + QOperator | 0.0000 | 否 |
| 只量化 Conv + Percentile + QDQ | 0.0000 | 否 |
| **只量化 Conv + MinMax + QDQ** | **0.8842** | **是** |

结论：**检测头不能量化**，且校准法必须用 MinMax（Percentile 在部分层会把范围截得过窄，激活直接折成常数）。

所以量化脚本的自检不能只看输出形状——形状对、坐标对、分类全 0 的模型照样能"通过"。现在脚本会检查分类峰值，塌了就非零退出。

```bash
python tools/quantize_int8.py --onnx deploy/models/base_v2.onnx --calib 32 --out deploy/models/base_v2_int8.onnx
```

> 校准张数上限 32：超过会在 ORT 1.31 的 calibrate 阶段报 `bad allocation`（实测 64 张必现、32 张稳定）。32 张取自 train，test 全程不参与校准。

## 四、负结果：INT8 在这台机器上更慢

主流说法是「INT8 快 2~4 倍」。实测不是。

**前提：加速比只有在同一块硬件上才有意义。** 「PyTorch GPU 12ms，ONNX CPU 33ms，所以 ONNX 更慢」是错的——那是两块硬件。所以分两组测：

同一台机器（AMD Ryzen 7 8840HX，12 核 24 线程），输入 640×640，batch=1，预热 25 次、计时 150 次取中位数：

| 后端 | 设备 | 精度 | 前向中位 | CV | 端到端中位 | 相对 ORT-CPU-FP32 |
|---|---|---|---|---|---|---|
| PyTorch | CPU | FP32 | 76.44 ms | 6.6% | 98.47 ms | 0.55× |
| **ONNX Runtime** | CPU | **FP32** | **42.27 ms** | 4.9% | **70.86 ms** | **1.00×** |
| ONNX Runtime | CPU | INT8 | 51.93 ms | 6.7% | 76.60 ms | 0.81× |

两个读数：

1. **ONNX 相对 PyTorch CPU 快 1.81×**，这是真实的图优化 + 算子融合收益。
2. **INT8 比 FP32 慢 23%**。QDQ 和 QOperator 两种格式都试过，都是 50ms 上下。这个模型只有 2.6M 参数、卷积通道不宽，量化的「反量化 + 数据重排」固定开销超过了整数卷积省下的时间。加上只量化 Conv 会把图切碎，整数算子之间要反复搬数据。

所以部署选型是 **ORT CPU FP32**，不是 INT8。INT8 模型可以用 `tools/quantize_int8.py` 随时重生成（`deploy/models/` 下的 `.onnx` 不进仓库，见 `.gitignore`），但没有进服务端——「试过、量化了、测了精度、发现不划算」比「没试」有价值，但结论就是不划算。

```bash
python tools/bench_deploy.py --iters 150 --warmup 25 --gpu-batches 1,4,8
```

## 五、GPU 那一组：同一个 5060 上 ORT 比 PyTorch 快多少

这一节有三个坑、一个反超和一个负结果。坑的部分比数字本身更值得记：**它们都不会报错，只会给你一组看起来正常的错数字。**

### 5.1 装 ORT-GPU 踩了三个版本坑

**坑 A：provider 注册成功 ≠ 在 GPU 上跑。** 装完 `onnxruntime-gpu` 1.31，`get_available_providers()` 立刻列出 `CUDAExecutionProvider`。但一建会话，日志里是：

```
Error loading "onnxruntime_providers_cuda.dll" which depends on "cublasLt64_13.dll" which is missing
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 13.*
```

然后 `session.get_providers()` 返回 `['CPUExecutionProvider']`——**静默回落**。如果不看这行，接下来测到的是"GPU 和 CPU 一样慢"，而你会以为是 GPU 没用。

**坑 B：从 1.27 起 PyPI 默认包是 CUDA 13 构建。** 本机 torch 是 `2.11.0+cu128`（CUDA 12.8 + cuDNN 9.19），差一个大版本就加载不了。CUDA 12 的构建在另一个 index（`aiinfra.pkgs.visualstudio.com/.../onnxruntime-cuda-12`），国内拉 350MB 卡了 15 分钟没下完。最后用 **1.26.0**：1.27 之前的 PyPI 默认包就是 CUDA 12.8 构建，和 torch 天然对齐。

**坑 C：DLL 要手动指路。** ORT 不会自动去 torch 的 `lib` 目录找 cuBLAS/cuDNN，得 `onnxruntime.preload_dlls()` 或者 `os.add_dll_directory(torch/lib)`。缺 `nvrtc64_120_0.dll` 只影响 `EXHAUSTIVE` 卷积算法搜索——实测 `EXHAUSTIVE` 22.89ms vs `HEURISTIC` 23.40ms（batch=8），差 2%，可以不管。

### 5.2 怎么证明它真的在 GPU 上

判据只有一个：**profiling 里每个节点实际落在哪个 EP**。

```python
so.enable_profiling = True
# ... 跑几次 ...
events = json.loads(Path(sess.end_profiling()).read_text())
Counter(e["args"]["provider"] for e in events if e["cat"] == "Node")
# → {'CUDAExecutionProvider': 717}   717/717，100%
```

`tools/bench_deploy.py` 把这件事做成了硬门槛：CUDA 占比 < 99.9% 就**直接丢弃该行**，而不是把 CPU 成绩标成 GPU。宁可少一行，也不要一行假数据。

### 5.3 数字

RTX 5060 Laptop 8GB，输入 640×640，预热 25 次、计时 150 次取中位数。加速比全部相对 **PyTorch-GPU batch=1**（同设备、同一次进程）：

| 后端 | batch | 前向中位 | 每图 | 吞吐 | CV | 端到端中位 | 相对 PyTorch-b1 |
|---|---|---|---|---|---|---|---|
| PyTorch | 1 | 12.90 ms | 12.90 ms | 78 /s | 8.3% | 28.93 ms | 1.00× |
| PyTorch | 4 | 12.59 ms | 3.15 ms | 318 /s | 3.7% | 35.06 ms | 1.02× |
| PyTorch | 8 | 15.85 ms | 1.98 ms | 505 /s | 16.1% | 45.45 ms | — |
| **ONNX Runtime** | **1** | **7.64 ms** | 7.64 ms | 131 /s | 7.6% | **21.41 ms** | **1.69×** |
| ONNX Runtime | 4 | 14.07 ms | 3.52 ms | 284 /s | 8.9% | 32.47 ms | 0.92× |
| ONNX Runtime | 8 | 23.26 ms | 2.91 ms | 344 /s | 2.5% | 45.93 ms | 0.55× |
| **ORT + CUDA Graph** | **1** | **3.03 ms** | 3.03 ms | 330 /s | 4.8% | 19.86 ms | **4.26×** |
| **ORT + CUDA Graph** | 8 | 14.61 ms | 1.83 ms | **548 /s** | 4.7% | 48.53 ms | — |

漂移复核：基准行测完所有行之后再测一遍，7.64 → 7.93ms（**+3.8%**），说明这一批没有明显降频，比值的可信度高。CV 都在 10% 以内（批量行更高是因为单次调用只有 12~23ms，抖动占比大）。

### 5.4 为什么 CUDA Graph 能再快 2.5 倍

7.64ms 里有 **4.6ms 是内核启动开销**：这张图有 717 个节点，一次 launch 约 2µs，串起来就是 1.4ms 起步，再加上 ORT 逐层调度和 Python 侧的每次调用成本。CUDA Graph 把整串内核录成一张图，一次提交就完事，于是 7.64 → **3.03ms**。

代价是硬的：**只能静态 shape、输入必须是同一块固定显存**。服务里要用它，就得把 640×640 的输入缓冲固定住，每张图往同一块显存里覆盖（`OrtValue.update_inplace`）。所以它救不了动态 batch。

**为什么服务端没接这条路**：`src/serve/api.py` 是线程池并发，多个请求会同时往同一块固定显存里写——那是数据竞争，不是优化。要上 GPU + CUDA Graph，得把推理改成单线程队列（谁拿到缓冲谁跑），吞吐模型跟现在的 CPU 多线程版完全不同，得重做压测才能报数字。所以这里只把离线结论记下来，不假装服务已经能跑 3ms。

### 5.5 为什么 batch 放大后 PyTorch 反而赢

看每图耗时：PyTorch 12.90 → 1.98ms（6.5×），ORT 7.64 → 2.91ms（2.6×）。

原因是两边的固定开销不在一个位置：

- **PyTorch 的固定开销在 Python 侧**（每次调用的 dispatch、参数处理）。batch 一大，这笔钱摊到 8 张图上，每图成本立刻塌下来。
- **ORT 的固定开销在 GPU 侧**（717 个内核的启动）。batch 变大只是每次内核处理的数据变多，启动次数一次没少——所以每图成本降得慢。

这解释了为什么加了 CUDA Graph（把启动开销消掉）之后，batch=8 的 ORT 能到 **548 img/s**，重新超过 PyTorch 的 505 img/s。

所以结论不是"谁更快"，而是：**ORT 的优势全部来自「每次调用的开销更小」，batch 一大这个优势就被稀释；要在大 batch 上继续赢，必须上 CUDA Graph。**

### 5.6 数值一致性：74/75 张株数完全一致

速度只有在"两边算的是同一个模型"时才有意义。全 test 75 张、conf=0.65 下逐图比株数：

- **74/75 完全一致（98.7%）**，Δ 均值 **-0.01**
- 唯一差异 `img_155`：PyTorch 48 / ORT 49 / GT 49——那个框的分数正卡在 0.65 阈值上，ORT 侧反而与真值一致
- 单图逐框（按中心点一对一匹配）：坐标 max|Δ| **0.012px**，置信度 max|Δ| 2.45e-4

```bash
python tools/verify_ort_gpu.py          # 全 test 集株数一致性 + EP 归属
```

产物 `eval_reports/ort_gpu_parity.json`。

### 5.7 顺带修掉的一个自检 bug

导出动态 batch 版本时，框级自检报 `FAIL：框偏差 153.97px`，但张量级 max|Δ| 只有 1.5e-3px——两者矛盾。

根因是**判定方式**：原来按置信度排序后逐行比坐标。密集小目标里两个框的分数只差 1e-4，浮点累积差就足以让它们在排序中互换位置，于是第 5 行拿 A 框比 B 框，报出上百像素的"偏差"。

改成**按中心点一对一匹配**（贪心最近邻）后，同一个模型报 0.024px。这不是模型的问题，是尺子的问题——和之前评测里那几处「冤枉模型」是同一类错误。

### 5.8 GPU 组的局限

- 只有 1 次运行，没有像主实验那样做多种子/bootstrap，所以「1.69×」是单点测量值，不是置信区间
- PyTorch 侧没上 `torch.compile` / FP16，ORT 侧没上 FP16 / TensorRT，两边都是各自的默认配置
- TensorRT 未测：sm_120 需要单独装 TRT 10.x，本机没装

## 六、服务设计

`src/serve/api.py`，FastAPI + ONNX Runtime，四个接口：

| 接口 | 用途 |
|---|---|
| `GET /health` | 模型、后端、加载与预热耗时、并发配置 |
| `GET /metrics` | 累计请求/错误数、最近 60s QPS、P50/P95/P99 |
| `POST /predict` | 单图，`conf`/`iou`/`max_det` 走 query |
| `POST /predict/batch` | 多图，单次上限 64 张 |

三个刻意的设计：

**推理必须离开事件循环。** ORT 的 `run` 是阻塞调用，直接在 async 函数里跑会把整个事件循环卡住，连 `/health` 都会超时。所以统一丢进 `ThreadPoolExecutor`。

**并发靠信号量，不靠调线程数。** ORT session 内部的 intra_op 线程池被所有请求共享：24 个请求各要 12 线程，实际是 288 个线程抢 12 个核。这里把 intra_op 固定成 3，用 `Semaphore(4)` 限制同时推理的请求数，总线程数 ≈ 物理核数，不超订。

**返回结构化结论而不是标注图。** 上层要的是「多少株、密度档、置信度分布长什么样」，用来决定要不要换阈值重跑；画框图在浏览器里做比在服务端做便宜得多。

```json
{
  "image": "img_105.jpg", "width": 1024, "height": 1024,
  "count": 44, "density": "sparse",
  "score_stats": {"min": 0.6512, "p25": 0.7913, "median": 0.8271, "p75": 0.841, "max": 0.8722},
  "stage_ms": {"preprocess": 5.01, "inference": 53.02, "postprocess": 1.32, "total": 66.48},
  "backend": "onnxruntime/CPUExecutionProvider",
  "detections": [{"box": [428.4, 198.2, 541.7, 310.0], "score": 0.8722}, "…"]
}
```

密度档阈值取 test 75 张真实株数的 33/66 分位（46 / 52）。用固定阈值而非按请求动态算分位，因为单张请求没有"总体"可言。

**与离线评估的一致性**：抽 6 张图对比服务与 `tools/eval_full.py` 的株数，**6/6 完全一致**。这条必须验，否则服务端和报告里的数字会互相打脸。

## 七、压测

固定请求数、固定客户端并发，图片在 20 张里轮转（`tools/loadtest.py`）：

| 客户端并发 | QPS | P50 | P95 | P99 |
|---|---|---|---|---|
| 1 | 13.1 | 74 ms | 117 ms | 121 ms |
| **4** | **29.6** | 135 ms | **156 ms** | 166 ms |
| 8 | 32.5 | 238 ms | 278 ms | 281 ms |
| 16 | 30.5 | 503 ms | 575 ms | 604 ms |

吞吐在 4 路并发后基本饱和（约 30 QPS），继续加并发只是把延迟线性推高——典型的利特尔法则。

**是不是 GIL 卡的？** 起 2 个 uvicorn worker（多进程，绕开 GIL）重测，QPS **31.4**，几乎没变。所以瓶颈不是 Python 解释器，是 **CPU 算力本身已经打满**：单请求推理占 3 线程 × 53ms ≈ 0.16 核·秒，12 核就是 ~75 QPS 的上限，实际跑出 30 QPS（约 40% 效率，剩下是线程同步与内存带宽）。

这个结论决定了扩容方式：**加核比加 worker 有用**，或者换 GPU EP（第五节：单张 3.03ms，理论上单卡就能顶满这个服务）。

> 另一个试过的配置（intra_op=1 + 12 路并发）吞吐 31.6 QPS、P50 371ms——吞吐略高但延迟差 2.7 倍，不适合在线服务。

## 八、Docker

```bash
docker build -f deploy/Dockerfile -t cabbage-detect:0.1.0 .
docker run --rm -p 8000:8000 cabbage-detect:0.1.0
# 或
docker compose -f deploy/docker-compose.yml up --build
```

镜像**刻意不带 torch**：服务端只做前向，torch 那 2~3 GB 依赖没有任何收益。依赖清单 `deploy/requirements-serve.txt` 只有 onnxruntime / opencv-headless / numpy / fastapi / uvicorn。

`HEALTHCHECK` 会打 `/health`，模型没加载成功时容器直接标记 unhealthy，而不是等业务请求报错。

**未验证项**：本机没有 Docker，上述镜像**没有实际构建过**。Dockerfile 按 CPU-only 最小依赖编写，CUDA 相关一行没写。这条在 README 的局限里同样标注。

## 九、还没做的

- **TensorRT**：RTX 5060 是 sm_120，TensorRT 对 Blackwell 的支持要单独装 TRT 10.x，没做。
- **INT8 的另一种走法**：只量化骨干、把 head 整个留在 FP32，可能兼顾速度与精度。没试。注意这一思路在 GPU 上也意义不大——ORT 的 CUDA EP 没有 int8 卷积加速路径，QDQ 节点会退回 CPU。
- **FP16**：GPU 上 FP16 通常还能再快 1.5~2×，本仓库只测了 FP32。要和 CUDA Graph 一起做才有可比性（FP16 会改内存布局，graph 得重录）。
- **GPU 批量调优**：batch=8 那行 ORT 裸跑输给 PyTorch，只查到「启动开销 vs 图优化」这一层，没做逐层 profiling 定位到具体算子。
