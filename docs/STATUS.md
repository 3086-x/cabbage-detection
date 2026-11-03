# 项目总盘点

> 盘点时间：2026-11-03
> 覆盖：`D:\cabbage_project`（检测/复现/部署）+ `D:\cabbage-vision-agent`（Agent）
> 依据：实际扫盘 + `git log` + `eval_reports/*.json`，不是回忆

---

## 0. 一页结论

| 线 | 状态 | 一句话 |
|---|---|---|
| 研究（复现 + 噪声预算） | **完成，可离线复核** | 33 次训练（11 组 × 3 种子），10 个对比里只有 2 个扛得住 Bonferroni |
| 评测（业务指标） | **完成** | mAP50 饱和 → 改用 mAP50-95 + 计数 MAE/RMSE + 分密度 |
| 部署（ONNX + 服务） | **完成，Docker 除外** | CPU 1.81×、GPU 1.69×、CUDA Graph 4.26×；4 路并发 ~30 QPS |
| Agent（独立仓） | **可用且已评测** | 26 条任务 96%，76 个契约测试 |

两个仓库都已推送，工作区干净（`git status` 为空）。

| 仓库 | 地址 | HEAD | 入库文件 | 仓库体积 |
|---|---|---|---|---|
| 检测 | github.com/3086-x/cabbage-detection | `e950ac3` | 195 | 1.8 M |
| Agent | github.com/3086-x/cabbage-vision-agent | `f6f5a82` | 40 | 488 K |

---

## 1. 已做的

### 1.1 研究线（主仓）

| 做了什么 | 产出在哪 | 关键数字 |
|---|---|---|
| 点标注 → 自适应扩框（株距 10% 分位数 × k=0.53） | `Code/convert_points_to_boxes.py`、`DATA.md` | 18,811 个框；262/38/75 |
| 基线重训 + 三模块对比 | `runs/detect/train_base_v2` 等 | test mAP50 **0.9715**、mAP50-95 **0.7910** |
| 11 组配置 × 3 种子 = 33 run | `eval_reports/train_*_s1/s2.json`、`artifacts/results_csv/`（33 份） | 组内 σ 0.10~1.16pp |
| seed=0 复现检验 | `eval_reports/base_v2_repro.json` | 与 base_v2 逐位一致 |
| 配对 bootstrap（75 图 × 1000 次） | `eval_reports/bootstrap_base_v2.*` | 配对 CI 半宽 ±0.57pp（非配对 ±2.66pp） |
| 噪声预算 v1→v4 判定体系 | `eval_reports/noise_budget.*`、`REPRO_STATUS.md` 11.7 | 门槛 0.51~1.67pp，逐组实测 |
| 修正原论文的空实验 | `hyp_no_mixup.yaml`、`eval_reports/base_mixup.*` | ultralytics `mixup` 默认 0.0，原对照等于没改 |
| CBAM 真实训练（原论文权重里没有） | `tools/train_cbam_fixed.py`、`cfg/yolo11n-cbam.yaml` | best.pt 里 `cbam` 出现 >0，参数量确实变了 |
| 三种注意力 + 两种注入位置 | `eval_reports/{se,cbam_v2,ca,cbam_res_v2,cbam_p3}.json` | 全部**不可区分** |

判定结果（可直接引用的五条）：

1. YOLOv8n **−2.30pp**（σ 0.31pp，最硬的一条）
2. 加 MixUp 0.15 **−1.06pp**（唯一确定方向为负的策略改动）
3. SE / CBAM / CA：−0.36 / −0.77 / −0.19pp，全部在门槛内
4. mAP50 在 33 个 run 上极差 **0.06pp**，已饱和，无法区分任何配置
5. 单次训练不能用来排序（CBAM 单次 −2.15pp vs 三种子 −0.77pp，σ 1.20pp）

### 1.2 评测线（主仓）

| 做了什么 | 产出 | 数字 |
|---|---|---|
| 业务指标：MAE / RMSE / MAPE / 偏差 / 完全数对率 | `tools/eval_full.py` → `eval_reports/*.json` | base_v2 MAE **3.07**、偏差 **−2.27** |
| 计数阈值网格扫描（val 选参、test 报告） | `tools/count_sweep.py`、`count_sweep_test.json` | conf 0.65 / iou 0.5；统一阈值后 MAE 2.52~3.07 |
| 分密度分析（三档） | 同 json 的 `by_density` | 密集档漏检更重 |
| 一键汇总图表 | `tools/make_report.py` → `compare.md`、`compare_real.png` | 从 json 读，无硬编码 |

### 1.3 部署线（主仓）

| 做了什么 | 产出 | 数字 |
|---|---|---|
| ONNX 导出（静态 / 动态 / 固定 batch） | `tools/export_onnx.py`、`deploy/models/*.onnx` | 静态 b1、动态轴、静态 b8 各一份 |
| 两级数值自检（张量段 + 框级一对一匹配） | 同脚本 | 坐标 Δ 1.5e-3px、框 Δ 0.024px、mAP 差 **0.0001** |
| INT8 量化（含负结果） | `tools/quantize_int8.py`、`base_v2_int8.onnx` | 精度 −0.56pp、速度 **慢 23%**，弃用 |
| 三后端基准（CPU 组 / GPU 组，同设备比） | `tools/bench_deploy.py` → `deploy_speed.md` | CPU **1.81×**、GPU **1.69×**、CUDA Graph **4.26×** |
| EP 归属校验 | 同上（profiling 统计） | 717/717 节点在 CUDA 才算 GPU 成绩 |
| ORT-GPU 数值一致性 | `tools/verify_ort_gpu.py` → `ort_gpu_parity.json` | 75 张里 74 张株数一致，唯一差异在阈值边界 |
| FastAPI 服务 | `src/serve/api.py` | `/health` `/predict` `/predict/batch` `/metrics` |
| 压测 | `tools/loadtest.py` → `loadtest_w*.json` | 4 路并发 ~**30 QPS**，P95 **156ms** |
| Docker 配置 | `deploy/Dockerfile`、`compose`、`requirements-serve.txt` | **写完但没构建过** |

GPU 三行（batch=1）：PyTorch 12.90ms → ORT 7.64ms → ORT+CUDA Graph **3.03ms**。

### 1.4 Agent 线（独立仓）

| 做了什么 | 产出 | 数字 |
|---|---|---|
| 5 个按决策点划分的工具 | `src/cvagent/tools/spec.py` | inspect / detect / count / refine / batch |
| Agent 循环、历史压缩、步数收束 | `src/cvagent/agent/loop.py` | 观测预算 1400 字符，打满发无 tools 请求 |
| 错误分类学（不抛异常） | `src/cvagent/errors.py` | 统一 `{ok:false, error:{code, retryable, hint}}` |
| 供应商抽象 + 错误码归一化 | `src/cvagent/llm/*` | 任意 OpenAI 兼容网关 |
| CLI 7 个子命令 | `src/cvagent/cli.py` | ask / detect / tools / doctor / models / setup-weights / replay |
| 契约测试 | `tests/`（5 文件） | **76 个**，不依赖权重与网络 |
| 评测体系 | `eval/tasks.yaml`（26 条）+ `run_eval.py` | 五层判定，不用 LLM-as-judge |
| 真实模型实测 | `eval_reports/eval_dashscope.md` | 通过 **96%**、工具 100%、幻觉 **0%**、1.85 轮 |

---

## 2. 没做的

### 2.1 规划里点名、但确实没做

| 项 | 为什么没做 | 值不值得补 |
|---|---|---|
| **SAHI 切片推理** | 时间让给了噪声预算和部署 | 值。零训练，通常 +2~4pp mAP50-95 |
| **P2 小目标检测层** | **注意区分**：做的是把 CBAM 挪到 P3（`cbam_p3`），不是新增 P2 输出层 | 值。1024 图缩到 640 后甘蓝仅 20~40px |
| **TensorRT** | RTX 5060 是 sm_120，要单独装 TRT 10.x | 边际收益低，ORT+CUDA Graph 已接近上限 |
| **FP16** | 只测 FP32；FP16 要重录 CUDA Graph 才有可比性 | 值。GPU 上通常再快 1.5~2× |
| **Docker 镜像实际构建** | 本机没装 Docker | 值。否则「可部署」是纸面结论 |
| **服务上 GPU** | CUDA Graph 要求固定显存缓冲，线程池并发会数据竞争，得改成单队列 | 值，但要重做压测 |
| **Agent 多模型 × 多次运行** | 只跑了 `qwen3.8-27b` 一次 | 值。现在的 96% 是快照不是区间 |
| **MCP 封装**（简历版本 C） | 优先级排在部署之后 | 可选，腾讯 JD 提到 MCP |

### 2.2 做了，但有缺口

| 项 | 现状 | 缺口 |
|---|---|---|
| **目录重排（规划 Day 6）** | `src/ deploy/ docs/ cfg/ artifacts/ eval_reports/` 已建 | 根目录还堆着 **15 个 .py**（`train_cbam.py`、`flask_detect.py`、`gradio_app.py` 等）、**15 张 .png**、**3 个 .pt**，面试官点进来仍会看到混乱 |
| **图表可复现（硬伤 2）** | 新链路 `tools/make_report.py` 从 json 读 | 旧 `Code/generate_charts.py` **仍是硬编码**（第 9~35 行写死 96.4/96.2/94.4），且没删 |
| **计数改善** | MAE 3.07 → 2.52~2.80 | 只有 seed 0 + val 选参，**未做三种子验证**，按本仓库标准不算结论 |
| **RT-DETR** | 只有 `rtdetr_legacy.json`（20ep 跨环境） | 缺统一协议重训，不参与排名 |
| **测速** | 每配置 1 次（预热 25 + 计时 150） | 无重复实验，跨天绝对值差 25%+，只有同次运行内比值可信 |
| **Agent 任务集** | 26 条自拟，7 类 | 不代表真实用户分布 |
| **论文口径（硬伤 3）** | `DATA.md` 已改 7:1:2 | 论文 PDF 里仍写 7:2:1，文档与论文不一致（论文不改，但要知道） |

### 2.3 明确不做（已决策）

- 换更大模型（YOLOv11m/l）—— 8G 显存跑不动，也不加分
- 重新设计网络结构 —— 收益不确定
- 补拍 / 扩数据集 —— 属于另一个项目
- Flask → React 重写 —— 前端不是卖点
- INT8 继续优化 —— 已证伪，ORT CUDA EP 也没有 int8 卷积路径

---

## 3. 东西存在哪

### 3.1 主仓已入库（195 文件 / 1.8M）

| 路径 | 内容 | 该不该引用 |
|---|---|---|
| `README.md` | 30 秒首页：结论表 + 判定表 + 三层误差棒 | 主入口 |
| `docs/METHOD.md` | 方法细节（从 README 拆出） | 引用 |
| `docs/DEPLOY.md` | ONNX / INT8 / 服务 / 压测 / 五个坑 | 引用 |
| `docs/ENVIRONMENT.md` | 环境与 8G 显存约束 | 引用 |
| `DATA.md` | 数据结构、点转框、7:1:2、不公开原因 | 引用 |
| `REPRO_STATUS.md` | 复现沿革，11.7 节是判据作废史 | 引用 |
| `EXPERIMENT_CBAM.md` | CBAM 注入链路的三次尝试 | 引用 |
| `tools/`（24 个脚本） | 训练 / 评估 / 计数 / 统计 / 导出 / 量化 / 基准 / 压测 | 引用 |
| `cfg/`（5 个 yaml） | 四种注意力结构 + P3 变体 | 引用 |
| `eval_reports/`（约 100 文件） | 每次评估的 json+md、图表、测速、压测原始数据 | 引用，见其 README |
| `artifacts/results_csv/`（33 份） | 训练曲线快照，离线复算用 | 引用 |
| `src/serve/api.py` | FastAPI 服务 | 引用 |
| `deploy/` | Dockerfile / compose / 依赖 / `onnx_manifest.json` | 引用 |
| `reproduce.sh` | `env` / `train` / `eval` / `stats` / `all` 五个入口 | 引用 |
| `LICENSE`、`requirements.txt`、`data.yaml` | MIT、依赖、数据定义 | 引用 |

### 3.2 主仓未入库（本地保留）

| 路径 | 体积 | 为什么没入库 |
|---|---|---|
| `dataset/` | 113 M | 数据不公开 |
| `runs/` | **1.8 G** | 174 个目录（33 个正式训练 + 早期废弃 + 大量 val 输出），`.pt` 与曲线图太大 |
| `project/` | 154 M | 论文 / 开题 / 中期 / 答辩 PPT / 演示视频，**含真实姓名**，绝不公开 |
| `images/` | 111 M | 原始影像 |
| `ground_truth/`、`labels/` | — | 原始点标注（.mat）与中间标签 |
| `weights/` | 5.3 M | `best.pt` 备份 |
| `deploy/models/` | 44 M | ONNX 产物，可由脚本重新生成；清单入库存 `onnx_manifest.json` |

### 3.3 Agent 仓（40 文件 / 488K）

| 路径 | 内容 |
|---|---|
| `README.md` | 首页：一次运行、四个设计决策、报错对照表 |
| `docs/DESIGN.md` | 10 节设计说明，第 9 节是评测四个坑，第 10 节已知缺陷 |
| `src/cvagent/` | 18 个模块 2,173 行（vision / tools / agent / llm / report / cli / config / errors） |
| `tests/` | 76 个测试，5 个文件 |
| `eval/tasks.yaml` + `run_eval.py` + `README.md` | 26 条任务、五层判定、指标定义 |
| `eval_reports/` | `eval_dashscope.*`（真模型 96%）、`eval_mock.*`（基线 42%） |
| `examples/transcript_single_image.md` | 一条完整决策链的原始记录 |
| `weights/best.pt` | 5.3 M，另有 GitHub Release v0.1.0 分发 |

---

## 4. 哪些东西不能再引用

| 文件 | 为什么 |
|---|---|
| `eval_reports/base_v1.*` | 早期 run，与 base_v2 非同批次 |
| `eval_reports/cbam.*` | 早期 CBAM，配置与当前注入链路不一致 |
| `eval_reports/rtdetr_legacy.json` | 跨环境（20ep / AdamW / batch 4），只作存在性证明 |
| `*.json` 里的 `counting` 字段 | 固定 conf=0.25，跨模型比较请用 `count_sweep_test.json` |
| `Code/generate_charts.py` | 数字硬编码，已被 `tools/make_report.py` 取代 |
| 根目录 `train_cbam.py` / `yolo11_cbam.py` / `finetune_with_cbam.py` | 原论文的空实现（ImportError / 无训练代码），当前有效的是 `tools/train_cbam_fixed.py` |
| 根目录 15 张 `.png` | 早期手填图表，与 `eval_reports/*.png` 冲突 |
| `eval_report.md/json`（根目录） | 早期单份报告，已被 `eval_reports/` 取代 |

---

## 5. 下一步（三选一）

**A. 收尾门面**（半天）
清根目录 → 删旧图表脚本 → README 索引补 STATUS → 论文口径写说明。
收益：面试官点进来看到的是仓库不是垃圾堆。风险最低。

**B. 补一个真改进**（2~3 天）
P2 新增检测层 或 SAHI 切片推理，用同一套 `eval_full.py` 跑基线 vs 改进，三种子。
收益：唯一能写进简历的「精度提升」。风险：可能仍然不可区分，但那是诚实结论。

**C. 打通 Agent 与服务**（2 天）
把 Agent 的 vision 后端从 ultralytics `.pt` 换成 ORT 服务（GPU 单队列 + CUDA Graph），
端到端延迟从 ~200ms 降到 ~5ms，并重做压测。
收益：两个仓库从「并列」变成「一条链路」，这是 AI 应用岗最想看的东西。
