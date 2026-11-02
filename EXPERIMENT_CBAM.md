# 注意力机制与数据增强消融实验记录（可复核）

> 目的：验证「在 YOLOv11n 上加注意力模块能提升甘蓝检测」这个假设，
> 并复现原论文的 **SE / CBAM / CA 三模块消融（C 组）** 与 **数据增强消融（B 组）、交叉实验（D 组）**。
>
> 结论先说：
>
> 1. **三种注意力模块对检测精度都没有可测收益**：mAP50 极差仅 0.19pp（已饱和），
>    mAP50-95 的点估计全部略降。补齐三种子后，与基线的配对差是 **SE −0.36 / CBAM −0.77 / CA −0.19pp**，
>    而它们各自的 MDD 是 1.29 / 1.67 / 0.51pp —— **全部不可区分**。
>    它们唯一站得住的收益在计数侧：不再系统性漏检，MAE 下降 8.8%~17.8%，
>    fuse 后推理耗时无可测变化（<1ms）。
> 2. **数据增强里只有 MixUp 是确定有害的**：三种子实测 **−1.06pp**（逐 seed −1.33 / −0.84 / −1.01），
>    它的配对 σ 只有 0.25pp，是全部 10 个对比里**唯一能确定方向为负的训练策略改动**。
>    而「去 Mosaic +1.24pp」在补两个种子后缩到 **+0.59pp**，落进不可区分区 ——
>    那个 0.8034 是一次好签。~~「增强比注意力大一个量级」~~ 这个旧结论已作废。
> 3. **原论文的「无 MixUp」对照是空实验**：所有历史 run 的 `mixup` 都是 0.0（ultralytics 默认值就是 0），
>    所以「移除 MixUp」与「完整增强」在配置上完全等价。本工作改成「主动加入 mixup=0.15 vs 不加」才有真正的对照。
>
> **三次自我更正**（都写在第 14 节，保留痕迹）：
> ① 曾报「CBAM 使 mAP50-95 掉 9.7pp」→ 实为训练器丢弃预训练权重；
> ② 曾报「CBAM 显著有害 −2.15pp」→ 那是 seed 0 单次值 + 借用的 σ，补齐种子后只剩 −0.77pp；
> ③ 曾把「YOLOv8n 余量仅 1.1 倍」写进结论 → 那是借用的 σ 高估 2.4 倍，实测余量是 4.3 倍。
>
> 过程中还发现一个更值钱的问题：**ultralytics 的 `model.train()` 会静默丢弃手工加载的
> 预训练权重**，导致前两轮实验实际是「从零训练」——这直接造成了第一版错误结论。
>
> 覆盖范围与剩余缺口见 [REPRO_STATUS.md](REPRO_STATUS.md)：本文件覆盖原论文的 B / C / D 三组
> 与新增的注入形式实验；A 组（模型对比）已补 YOLOv8n，RT-DETR 未做。

---

## 1. 实验环境

| 项 | 值 |
|---|---|
| GPU | RTX 5060 Laptop 8GB（sm_120） |
| Python / torch | 3.13.14 / 2.11.0+cu128 |
| ultralytics | 8.4.194（原实验环境 8.4.37） |
| 训练配置 | 50 epoch，640×640，batch 8，SGD lr0=0.01，momentum 0.937，wd 5e-4 |
| 预训练 | yolo11n.pt（COCO） |
| 数据加载 | workers=0（16G 内存机器，workers≥4 会 OOM） |

> 注意：`optimizer=SGD / lr0=0.01 / batch=8` 与原论文一致，目的是可比；
> 不是最优配置，若追求指标应做超参搜索。

## 2. 数据集

| 项 | 值 |
|---|---|
| 来源 | 大疆 M3M 无人机航拍，湖北嘉鱼，1024×1024 |
| 规模 | 375 张，train 262 / val 38 / test 75（**实际 7:1:2**，论文写的 7:2:1 有误） |
| 标注 | 原始为点标注（.mat gt_point），按「点间距 10% 分位数估株距 + 扩框系数 k=0.53」转边界框 |
| 类别 | 单类 `cabbage`，全库 18,811 个框；test 集 3,765 个框 |
| 密度 | 单图 31~89 个目标（中位 49）→ **密集小目标** |

## 3. 十三轮训练（为什么跑这么多）

| # | 实验 | 预训练 | 注入点 | 形式 | val 最优 mAP50-95 |
|---|---|---|---|---|---|
| 1 | 基线（历史，8.4.37） | ✅ | — | — | 0.7742*（test） |
| 2 | 基线（同环境重训） | ✅ | — | — | **0.7426** |
| 3 | CBAM@SPPF | ❌ 被丢弃 | SPPF 后（20×20, 256ch） | `x * mask` | 0.6567 |
| 4 | 残差 CBAM@SPPF | ❌ 被丢弃 | SPPF 后 | `x + γ·CBAM(x)` | 0.6389 |
| 5 | CBAM@SPPF | ✅ | SPPF 后 | `x * mask` | 0.7359 |
| 6 | 残差 CBAM@SPPF | ✅ | SPPF 后 | `x + γ·CBAM(x)`，γ 初值 0 | 0.7401 |
| 7 | CBAM@P3 | ✅ | neck P3 输出（80×80, 64ch） | `x * mask` | 0.7369 |
| 8 | **SE@SPPF** | ✅ | SPPF 后 | `x * sigmoid(FC(GAP(x)))`，r=16 | 0.7410 |
| 9 | **CA@SPPF** | ✅ | SPPF 后 | `x * a_h * a_w`（CoordAtt），r=32 | **0.7420** |
| 10 | **基线·无 Mosaic**（B/D） | ✅ | — | mosaic=0.0 | 0.7520 |
| 11 | **基线·加 MixUp 0.15**（B） | ✅ | — | mixup=0.15 | 0.7344 |
| 12 | **CBAM·无 Mosaic**（D） | ✅ | SPPF 后 | mosaic=0.0 + `x * mask` | 0.7459 |
| 13 | **CA·无 Mosaic**（D） | ✅ | SPPF 后 | mosaic=0.0 + CoordAtt | 0.7395 |
| 14 | **YOLOv8n**（A） | ✅ | — | 换骨干，其余不变 | 0.7268 |

\* 第 1 行是 test 指标，历史 run 的 val 曲线不可比（不同 ultralytics 版本），故只用于交叉验证基线稳定性。

第 8、9 行是为了复现原论文的「SE / CBAM / CA 三模块消融」补做的：
原脚本 `from ultralytics.nn.modules import SE / CoordAtt` 在当前 ultralytics 8.4.194 里
会 ImportError（这两个类已被移除），因此在 `tools/attention_modules.py` 按原实现重写，
reduction 与原脚本对齐（SE=16，CA=32）。

第 10~13 行是复现原论文 B 组（数据增强消融）与 D 组（增强 × 注意力交叉实验）。
两个训练脚本新增 `--mosaic` / `--mixup` 覆盖参数，只在显式传参时写进 `train()`，
其余超参完全不变，保证与第 2/5/9 行可比。

**为什么 3/4 要重跑成 5/6**：见第 5 节。同一个"CBAM@SPPF"跑了两遍，唯一变量是「预训练权重有没有被训练器保留」，结果 mAP50-95 从 0.6567 变成 0.7359 —— 差 8pp。

## 4. 最终结果（test 集，75 张 / 3765 个真实框）

**四组注意力模块消融（对应原论文 C 组）**

| 模型 | mAP50 | mAP50-95 | P | R | 计数 MAE(个/图) | 计数 MAPE | 偏差 | 推理 ms/图 | 参数量 |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv11n 基线 | 0.9715 | **0.7910** | 0.9738 | 0.9472 | 3.07 | 5.90% | −2.27 | 10.7 | 2.6049M |
| + SE | 0.9714 | 0.7835 | 0.9668 | 0.9516 | 2.65 | 5.29% | +0.07 | 10.1 | 2.6133M |
| + CBAM | 0.9699 | 0.7695 | 0.9684 | 0.9511 | **2.52** | **5.10%** | +0.47 | 9.9 | 2.6132M |
| + CA | **0.9718** | 0.7852 | 0.9721 | 0.9515 | 2.80 | 5.70% | +1.07 | 11.1 | 2.6110M |

**注入形式扩展（原论文没有，本工作新增）**

| 模型 | mAP50 | mAP50-95 | P | R | 计数 MAE | 偏差 | 推理 ms/图 | 参数量 |
|---|---|---|---|---|---|---|---|---|
| 残差CBAM@SPPF | 0.9708 | 0.7817 | 0.9662 | 0.9538 | 2.75 | +1.23 | 10.4 | 2.6134M |
| CBAM@P3 | 0.9714 | 0.7742 | 0.9700 | 0.9493 | 2.76 | +0.04 | 10.9 | 2.6055M |

- 计数指标：conf=0.65 / NMS iou=0.5，**阈值在 val 上选定**后再在 test 报告（不在 test 上调参）
- 推理耗时：batch=1、640×640、模型 fuse 后，CUDA event 计时 100 次平均
- 图：`eval_reports/compare_real.png`（四联柱状图）、`eval_reports/count_sensitivity.png`（阈值敏感性）

**读法**：

| 指标 | 结论 |
|---|---|
| mAP50 | 六组极差 0.19pp（0.9699~0.9718），**在 75 张测试集上完全不可区分** |
| mAP50-95 | 基线最高（0.7910），三种注意力模块都略低（−0.6 ~ −2.2pp），说明注意力让**边界回归变粗** |
| 计数 MAE | 三种模块都比基线好（−8.8% ~ −17.8%），CBAM 最好；**机制见第 7 节** |
| 推理耗时 | fuse 后 9.7~10.8ms，极差 1.1ms，**落在测量噪声内，无可测差异**（见第 8 节） |

### 4.1 B 组：数据增强消融（基线架构，对应原论文 B 组）

| 配置 | mAP50 | mAP50-95 | P | R | 计数 MAE | 计数 MAPE | 偏差 | 推理 ms/图 |
|---|---|---|---|---|---|---|---|---|
| 完整增强（mosaic=1, mixup=0） | 0.9715 | 0.7910 | 0.9738 | 0.9472 | 3.07 | 5.90% | −2.27 | 10.7 |
| **无 Mosaic**（mosaic=0） | 0.9716 | **0.8034** | 0.9657 | 0.9538 | 2.64 | 5.26% | −1.12 | 9.6 |
| **加 MixUp 0.15**（mixup=0.15） | 0.9712 | 0.7776 | 0.9597 | 0.9543 | **2.49** | **4.92%** | −0.81 | 10.0 |

### 4.2 D 组：增强 × 注意力 交叉实验（对应原论文 D 组）

| 增强档位 | 无注意力 | +CBAM | +CA |
|---|---|---|---|
| 完整增强（mosaic=1） | **0.7910** | 0.7695 | 0.7852 |
| 无 Mosaic（mosaic=0） | **0.8034** | 0.7956 | 0.7886 |

（表中为 test mAP50-95；同一行的「无注意力」是对照基线）

### 4.3 这两组怎么读

| 观察 | 证据 | 判断 |
|---|---|---|
| mAP50 再次完全饱和 | 三档增强的 mAP50 是 0.9712~0.9716，**极差 0.04pp** | 这个数据集上 mAP50 已经没有任何区分度，只能看 mAP50-95 |
| Mosaic 在这套数据上**可能是负收益，但证据不足** | 基线 0.7910→0.8034（+1.24pp），CBAM 0.7695→0.7956（+2.61pp），CA 0.7852→0.7886（+0.34pp），**三个架构同向** | 方向一致，机制也说得通（Mosaic 把本已很小的植株再缩小 2×、拼接处切断边界）。**但三个架构都是同一个 seed**，不构成独立证据；补种子后基线档只剩 +0.59pp（14.2） |
| MixUp 有明确负收益 | 0.7910→0.7776（−1.34pp），是唯一让 mAP50-95 下降的增强 | 同类目标做 alpha 混合会产生"半株甘蓝"的伪目标，对计数任务尤其有害 |
| 注意力在两个档位下都低于同档基线 | 完整档 −2.2pp / −0.6pp；无 Mosaic 档 −0.8pp / −1.5pp | **注意力的无增益不是 Mosaic 造成的**，换档位也没有翻转 |
| 但排序会翻转 | 完整档：CBAM 最差、CA 次之；无 Mosaic 档：CA 最差、CBAM 次之 | 1~2pp 级别的排序在单次训练下不稳定，只能下"不可区分"的结论 |

> **必须写明的证据强度**：以上表格本身都是**单种子、单次训练**。
> 第一版在这里用「6 个完整增强模型（架构不同）的极差 2.15pp」当噪声参照 —— **这个参照是错的**：
> 不同架构之间的差异本身就是信号，把它当噪声等于系统性高估门槛，反而会掩盖真实效应。
> 真正的噪声参照是种子方差，而且**它按配置不同**：YOLOv8n 0.10pp、基线 0.23pp、
> 无 Mosaic 0.68pp、SE 0.89pp、**CBAM 1.16pp**、CA 0.28pp。
> 补完 3 个种子后（14.2、14.6）：**MixUp 的 −1.06pp 是确定的负效应**（σ 仅 0.25pp），
> 而 Mosaic 的 +0.59pp 低于 MDD 1.01pp —— 前者能当结论，后者不能。
> 完整判定见 14.5~14.7。

- 图：`eval_reports/aug_cross.png`（增强 × 注意力双因素三面板图）

### 4.4 A 组：模型对比（只补了 YOLOv8n，RT-DETR 未做）

| 模型 | 参数量 | mAP50 | mAP50-95 | 计数 MAE | 偏差 | 推理 ms/图 |
|---|---|---|---|---|---|---|
| YOLOv11n（基线） | 2.605M | **0.9715** | **0.7910** | **3.07** | −2.27 | 10.7 |
| YOLOv8n | 3.138M | 0.9710 | 0.7662 | 3.32 | −2.71 | 7.5 |

- 两者 mAP50 差 0.05pp（又一次证明饱和），但 YOLOv11n 的 mAP50-95 高 2.48pp，
  更容易理解为"更晚一代的架构在边界回归上更准"，而参数还少 17%。
- RT-DETR-L 未跑：8G 显存需 batch≤4、DETR 系列 epoch 需求与 YOLO 不同（原论文只跑 20 epoch），
  直接对比不成立，需单独设计协议。

## 5. 发现一：`model.train()` 会静默丢弃手工加载的预训练权重 ⚠️

这是本次实验最重要的收获，也是最容易踩的坑。

**现象**：用「读官方 yaml → 插入一层 → 加载预训练 → train」的流程训练，
第 1 轮 box_loss 是 **4.0**、mAP50 是 **0.0036**；而基线脚本第 1 轮 box_loss 是 **2.36**、mAP50 是 **0.89**。

**定位过程**：
1. 先怀疑注意力层把特征缩放了（`x * mask` 中 mask≈0.25，等于把特征缩小 4 倍）→ 于是写了残差版 γ=0 的恒等初始化；
2. 残差版第 1 轮依然是 0.001 → 说明不是注意力的问题；
3. 找到 `tools/probe_pretrained.py`：**只加载不训练**，直接 val。
   - 基线 yolo11n.pt：mAP50 = 0.0085（本来就没学过甘蓝，正常）
   - 残差版（加载后）：mAP50 = 0.0085 —— **和基线完全一致，说明权重迁移是对的**
   - 朴素 CBAM（加载后）：0.0000
   → 既然「加载后」是对的，那就只能是「训练开始时」被替换了。

**根因**（`ultralytics/engine/model.py:892`）：

```python
if not args.get("resume") and self.ckpt:      # ← 用 YOLO("xxx.yaml") 构建时 self.ckpt 是 None
    self.trainer.model = self.trainer.get_model(weights=self.model, cfg=self.model.yaml)
```

用 `YOLO("yaml")` 构建的模型 `self.ckpt` 为空，这个分支不执行；
`DetectionTrainer.setup_model()` 会**按 cfg 重新 new 一个随机初始化的模型**，
我们手工 `load_state_dict` 进去的权重被静默丢弃，没有任何警告。

**修法**（`tools/train_cbam_fixed.py`）：

```python
model.ckpt = {"model": model.model, "epoch": -1, "train_args": {}, "best_fitness": None}
```

并加了一个 `on_train_start` 回调做断言：比对训练器里的权重与加载前的快照，
本次输出为 `[check] 训练器权重与加载的预训练一致: 416/416 个张量`。

**这个坑的教训**：对比实验里「配置项」远不止超参，**权重从哪来、有没有真的生效**也要断言。
如果没做这一步，会得出「CBAM 让 mAP50-95 掉 9.7pp」这种完全错误的结论（我第一版就是这么写的）。

## 6. 发现二：三种注意力模块在预训练微调下都没有检测增益

修好之后 SE、CBAM、CA 三组全部收敛到与基线几乎相同的水平（mAP50 极差 0.19pp，
其中 +CA 的 0.9718 只比基线高 0.03pp），且 mAP50-95 一致略低。可能的原因：

1. **数据集太"简单"**：无人机正射影像、单类、目标大且纹理清晰，mAP50 已经 97%，
   COCO 预训练特征足够，注意力没有可补的空间；
2. **注入位置分辨率太低**：SPPF 之后是 20×20，7×7 的空间注意力卷积在这个尺度上等于抹平局部响应，
   对边界回归有害；改插到 P3（80×80）后 mAP50-95 从 0.7695 回到 0.7742，符合这个解释，
   但仍未超过基线（0.7910）；
3. **三种模块的初始缩放都在压预训练特征**：朴素形式 `x * mask` 的 mask 由 sigmoid 给出，
   SE 初值 ≈0.50、CBAM ≈0.25、CA ≈0.25（实测见 `tools/attention_modules.py` 自检输出），
   等于一插进去就把预训练特征整体缩了一半到四分之一。残差版本（γ 初值 0）本来是为了绕开这点，
   但 γ 最终只学到 ±0.003 —— **优化器自己判断"不需要它"**；
4. **训练预算不足**：50 epoch、无超参搜索，注意力层是随机初始化的，可能还没学到位。
   但结合第 3 点，更可能的解释是"没有可学的"。

**这次三模块对比最有价值的一点**：原论文用 mAP50 把 CBAM 选成"最优模型"，
但在本复现里 CBAM 的 mAP50（0.9699）与 mAP50-95（0.7695）**都是四组最低**。
0.1~0.2pp 的差距在 38~75 张的评估集上不构成任何结论，
"用 mAP50 排序选模型"这个方法本身就是失效的 —— 这也是要补 mAP50-95 与计数指标的原因。

## 7. 发现三：计数指标对阈值高度敏感，固定阈值的排名不可信

第一版用 conf=0.25 / NMS iou=0.7 数框，算出基线 MAE 5.71、CBAM 4.51，
看起来"CBAM 让计数误差降 21%"。换一批同样配置重训后，结论又反过来了（8.39 / 9.57）。

原因：甘蓝密集，相邻植株框的 IoU 大量落在 0.5~0.7，**iou=0.7 的 NMS 抑制不掉这些近重复框**，
同一株被数两次，MAE 完全由「阈值碰巧抑制掉多少重复框」决定。

于是写了 `tools/count_sweep.py`：每张图只做一次前向（conf=0.01, iou=0.9），
把框和分数存下来，再事后网格扫描 (conf, iou)，18 组配置只花一次推理；
**配置在 val 上选**（最小化各模型平均 MAE），再在 test 上报告。

结果（`eval_reports/count_sensitivity.png`）：val 上 conf 从 0.25 走到 0.65，
基线的 MAE 从 5.6 降到 4.3，CBAM 从 7.0 降到 3.5 —— **排名在 conf=0.55 附近翻转**。
这解释了之前两次结论为什么互相矛盾：不是模型变了，是阈值没定。

按 val 选出的 conf=0.65 / iou=0.5 报告后，六组 MAE 稳定在 2.5~3.1，三种注意力模块都优于基线。

**但这里要小心解读**：MAE 的改善几乎全部来自「偏差 bias」而不是「随机误差」。

| 模型 | 偏差 bias | MAE |
|---|---|---|
| 基线 | **−2.27**（系统性漏检） | 3.07 |
| + SE | +0.07 | 2.65 |
| + CBAM | +0.47 | 2.52 |
| + CA | +1.07 | 2.80 |

基线在 conf=0.65 处是持续**少数**2.27 个/图的，注意力模块把这个系统性偏移拉回 0 附近，
MAE 才降下来。也就是说：**注意力模块并不是"数得更准"，而是"不再系统性地漏"**，
代价是轻微过检（bias 转正）。这个解释和召回率的变化是一致的（三模块 R 都高于基线）。
在农业计数场景里，"少 2 个"和"偏差 ±0.5 个"对下游决策的影响完全不同，
所以这个结论仍然有实用价值 —— 但它不是"精度提升"，别把它讲成精度提升。

## 8. 发现四：「注意力让推理慢 15~27%」这个结论被推翻了 ⚠️

第一版报告里写的是「CBAM 慢 14.8%~26.7%，慢在自适应池化的 kernel 启动开销」，
数字来自 11.7ms（基线）vs 14.9 / 14.3 / 13.4ms（三个 CBAM 变体）。

补做 SE / CA 之后，为了公平，把**全部 6 组放在同一进程、同一会话、紧挨着依次测**，
结果变成了：

| 模型 | 未 fuse ms/图 | **fuse 后 ms/图** | 相对基线 |
|---|---|---|---|
| 基线 | 12.23 | **10.45** | — |
| + SE | 12.59 | 9.70 | −7.2% |
| + CBAM | 13.73 | 10.44 | −0.1% |
| + CA | 13.05 | 9.98 | −4.5% |
| 残差CBAM | 14.13 | 10.75 | +2.9% |
| CBAM@P3 | 12.79 | 10.39 | −0.6% |

**fuse 后六组全在 9.7~10.8ms，极差 1.1ms，而且 SE 反而"比基线快 7%"。**
后者显然不可能是真的 —— 结构上 SE 只多了 8.5K 参数，不可能变快。
结论只能是：**这 1ms 的差异是测量噪声（GPU 频率、后台进程、CUDA event 抖动），不是模型差异。**

旧测速为什么偏高：那批数字是在训练/评估任务还在后台跑的时候测的，
显存和 SM 都被占着，绝对值和相对排序都不可信。这是对比实验里非常典型的一类坑 ——
**测速必须独占 GPU，且各组要在同一批次内测**。

仍然成立的部分：**未 fuse 时差异是真实的**（12.23 vs 13.73 / 14.13 = +12%~16%）。
BN 折叠（`model.fuse()`）会把注意力模块里紧邻卷积的 BN 融掉，这部分开销在部署形态下被吸收。
所以准确的说法是：

> 注意力模块带来的额外算力开销，在 fuse 后的部署形态下无法被测出（<1ms，噪声级）；
> 但在未 fuse 的 PyTorch 训练/推理形态下有 +12%~16% 的开销。

这条更正比原来的"慢 27%"更有信息量：它说明**这个模块在部署上其实没有代价**，
真正的问题只是它不提升精度。

### 8.1 再往下挖一层：绝对时延在笔记本上根本不可复现

补 A 组时又测了两次，同一个 `base_v2` 权重得到三个数：

| 第几次 | 场景 | `base_v2` fuse 后 ms |
|---|---|---|
| 1 | 训练结束后立刻测 | 10.45 |
| 2 | 冷机、独立批次（11 组） | **7.84** |
| 3 | 紧接着重测同一批 | **12.04** |
| 4 | 再跑一遍 11 组 | 10.71 |

**同一个文件、同一台机器、同一份代码，7.84 ~ 12.04ms，差 53%。**
原因是笔记本 GPU 的频率会随温度/功耗墙在 1.2~2.4GHz 之间浮动，`bench_speed.py` 里
没有锁频（`nvidia-smi -lgc`）也没有预热到热稳态。

结论要再收一层：

> 在这台机器上，**"X 比 Y 快几毫秒"这类结论一律不可信**。
> 能报的只有「同一次测量内、各组在 9.6~11.2ms 之间，差异不超过 ±1.5ms」，
> 以及"未 fuse 时注意力有 +12%~16% 开销"这个**同批次内**的相对结论。

如果要拿时延去做结论（比如"部署端可接受"），必须：
锁频（`nvidia-smi -lgc <固定频率>`）、预热到热稳态、每组重复 3 次取中位数、关闭所有其他 GPU 任务。
这件事本仓库没做，所以 `speed.csv` 只能作为**相对参考**，不能作为绝对性能指标。
这也是"为什么测速比训练更容易出错"的典型例子：训练有 loss 曲线自证，测速没有。

## 9. 发现五：原论文的「无 MixUp」对照是一个空实验 ⚠️

复现 B 组时，先看原项目的 `hyp_no_mixup.yaml`：

```yaml
# 移除 MixUp 增强
mixup: 0.0
```

看起来没问题。但把本项目**全部历史 run 的 `args.yaml` 都 dump 出来**后发现：

```
train-3   mosaic=1.0  mixup=0.0
train-7   mosaic=1.0  mixup=0.0
...
train_se  mosaic=1.0  mixup=0.0
```

**每一个 run 的 `mixup` 都是 0.0** —— 因为 ultralytics 的默认值就是 0.0（`default.yaml: mixup: 0.0`）。
也就是说「完整增强」这一档本身就没开 MixUp，那么「移除 MixUp」和「完整增强」在配置上**完全相同**，
原论文 B 组里那两组实际上是同一个实验跑了两次，差异纯来自随机性。

这是消融实验里最隐蔽的一类错误：**参数被"改成"的值等于它的默认值时，消融是空操作**。
如果只看 yaml 文件，两边都写着 0.0，很容易以为对照成立。

本工作的处理：既然"移除 MixUp"没有意义，就反过来做 **"主动加入 MixUp"**——
设 `mixup=0.15`（mosaic 保持 1.0），这才是真正改变了一个变量的对照。结果见 4.1 节：
mAP50-95 从 0.7910 降到 0.7776（−1.34pp），说明这个增强**在这套数据上是有害的**，
反过来解释了为什么原作者最终没开它。

> 可复用的检查方法：任何消融开跑前，先 `cat runs/detect/<name>/args.yaml`，
> 确认要改的那个参数**确实不等于默认值**。ultralytics 会把最终生效的全部超参写进 args.yaml，
> 这是比读自己写的 yaml 更可靠的证据。

## 10. 复现

```bash
cd D:/cabbage_project

# --- 训练：基线 + 三模块消融（对应原论文 C 组） ---
python tools/train_baseline.py    --epochs 50 --batch 8 --device 0 --workers 0 --name train_base_v2
python tools/train_cbam_fixed.py  --variant se --epochs 50 --batch 8 --device 0 --workers 0 --name train_se
python tools/train_cbam_fixed.py  --epochs 50 --batch 8 --device 0 --workers 0 --name train_cbam_v2
python tools/train_cbam_fixed.py  --variant ca --epochs 50 --batch 8 --device 0 --workers 0 --name train_ca

# --- 训练：注入形式扩展（原论文没有） ---
python tools/train_cbam_fixed.py --variant residual --epochs 50 --batch 8 --device 0 --workers 0 --name train_cbam_res_v2
python tools/train_cbam_fixed.py --insert-index 16 --epochs 50 --batch 8 --device 0 --workers 0 --name train_cbam_p3

# --- 训练：数据增强消融与交叉实验（对应原论文 B 组 / D 组） ---
python tools/train_baseline.py   --mosaic 0.0 --epochs 50 --batch 8 --device 0 --workers 0 --name train_base_nomosaic
python tools/train_baseline.py   --mixup 0.15 --epochs 50 --batch 8 --device 0 --workers 0 --name train_base_mixup
python tools/train_cbam_fixed.py --variant ca    --mosaic 0.0 --epochs 50 --batch 8 --device 0 --workers 0 --name train_ca_nomosaic
python tools/train_cbam_fixed.py --variant plain --mosaic 0.0 --epochs 50 --batch 8 --device 0 --workers 0 --name train_cbam_nomosaic

# --- 评估（每个模型一次，写 eval_reports/<name>.json） ---
python tools/eval_full.py --weights runs/detect/train_base_v2/weights/best.pt     --split test --name base_v2
python tools/eval_full.py --weights runs/detect/train_se/weights/best.pt          --split test --name se
python tools/eval_full.py --weights runs/detect/train_cbam_v2/weights/best.pt     --split test --name cbam_v2
python tools/eval_full.py --weights runs/detect/train_ca/weights/best.pt          --split test --name ca
python tools/eval_full.py --weights runs/detect/train_cbam_res_v2/weights/best.pt --split test --name cbam_res_v2
python tools/eval_full.py --weights runs/detect/train_cbam_p3/weights/best.pt     --split test --name cbam_p3
python tools/eval_full.py --weights runs/detect/train_base_nomosaic/weights/best.pt  --split test --name base_nomosaic
python tools/eval_full.py --weights runs/detect/train_base_mixup/weights/best.pt     --split test --name base_mixup
python tools/eval_full.py --weights runs/detect/train_cbam_nomosaic/weights/best.pt  --split test --name cbam_nomosaic
python tools/eval_full.py --weights runs/detect/train_ca_nomosaic/weights/best.pt    --split test --name ca_nomosaic

# --- 阈值扫描（必须在同一次调用里给全所有模型，否则阈值就不是共同选出的） ---
python tools/count_sweep.py --models \
    base_v2:runs/detect/train_base_v2/weights/best.pt \
    se:runs/detect/train_se/weights/best.pt \
    cbam_v2:runs/detect/train_cbam_v2/weights/best.pt \
    ca:runs/detect/train_ca/weights/best.pt \
    cbam_res_v2:runs/detect/train_cbam_res_v2/weights/best.pt \
    cbam_p3:runs/detect/train_cbam_p3/weights/best.pt \
    base_nomosaic:runs/detect/train_base_nomosaic/weights/best.pt \
    base_mixup:runs/detect/train_base_mixup/weights/best.pt \
    cbam_nomosaic:runs/detect/train_cbam_nomosaic/weights/best.pt \
    ca_nomosaic:runs/detect/train_ca_nomosaic/weights/best.pt

# --- 测速（同样要一次测全，且确保没有别的任务在占用 GPU） ---
python tools/bench_speed.py --models <同上 11 组 name:path>

python tools/make_report.py
```

单次 50 epoch 训练约 8~13 分钟（RTX 5060 实测 10 分钟；GTX 1650 约 1.5~2 小时）。

## 11. 局限

1. ~~**单次运行，无重复种子**~~ → **已补齐**：11 组配置 × 3 个种子 = **33 个 run，全部实测**
   （σ = 0.10 ~ 1.16pp，差 11 倍）。早期版本用「6 个完整增强模型（架构不同）mAP50-95 极差 2.15pp」
   当噪声参照，这个参照是**错的** —— 架构间差异本身是信号，不是噪声，拿它当噪声会把门槛高估，
   从而掩盖真实效应。正确参照是**逐配置实测的种子方差**。
   另外新增 `tools/check_pairing.py`：24 对种子 run 逐字段比对 114 个超参 + 权重结构签名，
   用机器证明它们只差 seed，而不是靠"我记得一样"。
2. **A 组只补了 YOLOv8n**：RT-DETR-L 未跑（8G 显存要 batch≤4，且 DETR 的 epoch 需求与 YOLO 不同，
   原论文只跑 20 epoch，直接对比不成立）。覆盖矩阵见 [REPRO_STATUS.md](REPRO_STATUS.md)。
3. **50 epoch、单一学习率**：注意力层随机初始化，训练预算对它们偏紧。
4. **计数指标的偏差方向不一致**：基线倾向漏检（−2.27），注意力组倾向过检（+0.07~+1.07），
   MAE 相近但误差结构不同，下游若对"漏检"和"过检"代价不对称，需要按代价重新选阈值。
5. **测速只测了 PyTorch**：ONNX / TensorRT 的部署时延未测，fuse 后的结论不能直接外推到量化引擎。
6. **MixUp 只测了一个强度**（0.15）且只跑了基线架构，不能断言"任何强度的 MixUp 都有害"。
   它是 10 个对比里第二硬的结论，但余量只有 1.5 倍，保守口径还差一点。
7. **n=3 的 σ 本身不可靠**：σ 的 90% CI 约为 [0.58s, 4.4s]，所以判定只到量级，不报 p 值。
   要真正收紧，唯一有效的手段是**扩 test 集**，而不是继续加种子。

## 12. 这份实验怎么讲（简历 / 面试）

**简历版（三行，已按 14.5~14.7 的最终判定校正）**：

> - 复现并纠偏 YOLOv11n 的注意力与数据增强消融（A/B/C/D/E 共 11 组配置、**37 个训练 run，
>   其中 11 组各 3 种子全部实测**）：定位到 ultralytics 训练器会静默丢弃手工加载的预训练权重
>   （修复后预训练命中 499/499、训练器权重一致性 416/416），修正后证明 SE / CBAM / CA
>   三模块与原论文「CBAM 最优」的排序**均无法复现**。
> - 发现原论文「移除 MixUp」对照是**空实验**（dump 全部历史 run 的 args.yaml，mixup 全为默认值 0），
>   改为「主动加入 0.15 vs 不加」重做对照，三种子实测 **−1.06pp**，是该数据集上唯一能确定
>   方向为负的训练策略改动。
> - 自建三层评估（mAP50-95 + 计数 MAE/RMSE/偏差 + 分密度）与**噪声预算**体系：先用配对 bootstrap
>   纠正「非配对 CI 把门槛高估 5 倍」（2.66→0.57pp），再实测 11 组种子方差发现训练端 σ 差 11 倍
>   （0.10~1.16pp），据此把 10 个对比的结论从「3 个显著」收缩到 **「2 个显著」**
>   —— 全程记录并撤销了 7 个自己下过的错误结论。

**面试可以主动说的七句话**：

1. "第一版我得出「CBAM 让 mAP50-95 掉 9.7pp」，后来发现是训练器重建模型把预训练权重丢了，
   这个 bug 我加了断言防复发 —— 现在每条实验都会打印『权重一致性 499/499』。"
2. "mAP50 在 97% 饱和了，三种子的标准差只有 0.01pp —— 它不是'看起来差不多'，数学上就是个常数。
   后来我把 33 个 run 全拉出来看，全部落在 0.9699~0.9718，最大的组间差 0.06pp 比它自己的
   噪声还小。所以我把评估换成 mAP50-95 + 计数误差。"
3. "计数误差我一开始也报错了，因为我用固定阈值；做了阈值扫描才发现排名会翻转，
   现在阈值在 val 上选，test 上只报告。"
4. "我一开始还报过『CBAM 慢 27%』，后来把 6 组放在同一批次重测，发现是测速时后台还在训练 ——
   fuse 后差异不到 1ms。测速必须独占 GPU，这条比数字本身重要。"
5. "做增强消融时我发现原论文的『无 MixUp』组是空实验：ultralytics 的 mixup 默认就是 0，
   『移除』和『保留』是同一份配置。所以我改成主动加入 mixup 才有对照，结果它是负收益。"
6. **"同一个结论我改过三次，每次都是往更保守的方向。第一次：我原来用非配对 bootstrap 估误差棒，
   得出『这套数据分辨不了 2.7pp 以内的差异』。后来意识到两个模型跑的是同一批图，某张图难
   会让两者同时掉分 —— 这部分该抵消，不该叠进误差棒。改成配对重采样后门槛从 2.66pp 缩到 0.57pp，
   于是得出『CBAM 显著有害 −2.15pp』。第二次：我给 SE/CBAM/CA 各补了两个种子，
   发现 CBAM 的种子 σ 是 1.2pp —— 基线的 5 倍 —— 三种子平均只剩 −0.77pp，那句话又撤了。
   第三次：还剩 6 个配置在借用 σ，而唯一过线的 YOLOv8n 恰好就在里面。把剩下 12 个 run 补完后，
   它的 σ 实测只有 0.31pp，是借用值的三分之一 —— 结论没变，但余量从 1.1 倍变成 4.3 倍。
   所以借用参数不只影响显著性，还会让本来很硬的结论看起来摇摇欲坠。"**
7. "还有个更有意思的：『哪个权重更好』和『这个方法更好』可以给出不同的答案。
   去 Mosaic 在 seed 0 上配对显著赢 1.25pp，但换两个种子重训只剩 0.59pp；
   无Mosaic+CA 更极端，单次 −0.24pp，三种子平均 +0.75pp，**连方向都反了**。
   所以我现在说任何结论前，会先问自己问的是哪一个问题。"

## 13. 文件清单

| 文件 | 作用 |
|---|---|
| `tools/cbam_module.py` | CBAM / CBAMResidual 模块（必须在可 import 的模块里，否则 pickle 成 `__main__` 后无法加载） |
| `tools/attention_modules.py` | SE / CoordAtt 模块（8.4.194 已从 ultralytics 移除，按原实现重写，reduction 对齐原脚本） |
| `tools/train_cbam_fixed.py` | 生成注入 yaml、按索引重映射预训练权重、修 `model.ckpt`、训练、断言权重一致性（`--variant plain/residual/se/ca`，`--mosaic/--mixup` 覆盖增强） |
| `tools/train_baseline.py` | 同环境基线训练（同样支持 `--mosaic` / `--mixup`） |
| `tools/eval_full.py` | 检测指标 + 计数指标 + 分密度分析，按 `--name` 写 `eval_reports/<name>.json` |
| `tools/count_sweep.py` | 计数指标的 (conf, iou) 敏感性扫描，val 选参 / test 报告 |
| `tools/bench_speed.py` | 参数量与 fused 推理耗时基准 |
| `tools/make_report.py` | 汇总对比表与图表 |
| `tools/probe_pretrained.py` | 诊断脚本：确认训练前 mAP 是否与基线一致 |
| `tools/bootstrap_ci.py` | **单模型** bootstrap 置信区间（非配对，不能用作模型间比较门槛） |
| `tools/seed_variance.py` | 种子方差、合并标准差与 MDD |
| `tools/paired_bootstrap.py` | **配对** bootstrap：同一重采样索引作用于两个模型，Δ 的 CI + Bonferroni 校正 |
| `tools/noise_budget.py` | 把评估端 SE 与训练端 σ 合成，输出每个配置的**最终判定** |
| `tools/experiment_registry.py` | 11 组配置 ↔ 33 个 run 的**唯一**映射来源，其余脚本 import |
| `tools/check_pairing.py` | 校验种子 run 与原 run 只差 seed（114 个超参 + 权重结构签名） |
| `tools/train_seeds.sh` | 可重入的种子训练驱动：跳过已完成、崩溃重试一次、半成品自动归档 |
| `tools/eval_seeds.sh` | 可重入的种子评估驱动：已有 json 的跳过 |
| `eval_reports/` | 11 组模型评估 json/md、五组对比表、`compare_real.png`（注意力 6 组）、`aug_cross.png`（增强×注意力 2×3 矩阵）、`count_sensitivity.png`、`seed_variance.png`、`noise_budget.png`、`speed.csv` |
| `REPRO_STATUS.md` | 原论文实验矩阵 vs 本次复现覆盖情况 |

## 14. 补充：噪声预算（F 组）

前面第 1~13 节的所有结论都建立在「单次训练 + 75 张 test」上。这一节回答一个前置问题：
**这些差异里，有多少是真的？**

### 14.1 复现检验：同 seed 重跑是否一致

`base_v2_repro` 与 `train_base_v2` 除 `--name` 外完全同配置。结果：

| 指标 | 逐位相同的轮数 | 差值 |
|---|---|---|
| `metrics/mAP50(B)` | 50 / 50 | `+0.00e+00` |
| `metrics/mAP50-95(B)` | 50 / 50 | `+0.00e+00` |
| `train/box_loss` | 50 / 50 | `+0.00e+00` |
| 第 50 轮 box_loss | — | 两边都是 `0.69875` |

→ `deterministic=True`（本版本默认）确实生效，这条管线**完全可复现**。
所以 14.2 测到的是**纯种子效应**，不含 cudnn 算子噪声。
另外 `base_v2_repro` 的 test 指标（0.9715 / 0.7910）也与 `base_v2` 完全一致，进一步确认。

### 14.2 种子方差（11 组 × 3 个 seed）

只改 `--seed`，其余（yolo11n.pt、50ep、batch 8、640、SGD lr0=0.01、同一 data.yaml）完全不动。
新增的 24 个种子 run 都用 `tools/check_pairing.py` 逐字段校验过：**114 个超参 + 权重结构签名
完全一致，只有 seed 不同**（22/22 对通过）。

| 配置 | seed 0 | seed 1 | seed 2 | test mAP50-95 | test mAP50 | 配对 Δ | 自己的 MDD |
|---|---|---|---|---|---|---|---|
| 完整增强 | 0.7910 | 0.7877 | 0.7921 | **0.7903 ± 0.0023** | 0.9715 ± 0.0001 | — | — |
| 无 Mosaic | 0.8034 | 0.7900 | 0.7951 | 0.7962 ± 0.0068 | 0.9716 ± 0.0001 | +0.59pp | 1.01pp |
| +SE | 0.7835 | 0.7797 | 0.7966 | 0.7866 ± 0.0089 | 0.9715 ± 0.0001 | −0.36pp | 1.29pp |
| +CBAM | 0.7695 | 0.7868 | 0.7915 | 0.7826 ± 0.0116 | 0.9709 ± 0.0009 | −0.77pp | 1.67pp |
| +CA | 0.7852 | 0.7890 | 0.7907 | 0.7883 ± 0.0028 | 0.9715 ± 0.0003 | −0.19pp | 0.51pp |
| CBAM@P3 | 0.7742 | 0.7881 | 0.7840 | 0.7854 ± 0.0100 | 0.9713 ± 0.0002 | −0.49pp | 1.45pp |
| 残差CBAM@SPPF | 0.7817 | 0.7790 | 0.7838 | 0.7848 ± 0.0058 | 0.9713 ± 0.0005 | −0.54pp | 0.88pp |
| 加 MixUp 0.15 | 0.7776 | 0.7830 | 0.7785 | 0.7797 ± 0.0022 | 0.9713 ± 0.0001 | **−1.06pp** | **0.45pp** |
| 无Mosaic+CBAM | 0.7956 | 0.8017 | 0.7959 | 0.7977 ± 0.0035 | 0.9709 ± 0.0005 | +0.74pp | 0.59pp |
| 无Mosaic+CA | 0.7886 | 0.8005 | 0.7941 | 0.7977 ± 0.0080 | 0.9708 ± 0.0007 | +0.75pp | 1.17pp |
| YOLOv8n | 0.7662 | 0.7707 | 0.7646 | 0.7672 ± 0.0010 | 0.9710 ± 0.0002 | **−2.30pp** | **0.35pp** |

**σ 差一个量级**：YOLOv8n 0.10pp、基线 0.23pp、MixUp 0.22pp、CA 0.28pp、无 Mosaic 0.68pp、
SE 0.89pp、CBAM@P3 1.00pp、**CBAM 1.16pp（基线的 5 倍）**。
这条直接否掉"所有配置共享一个 σ"的做法。

按各自的 MDD 判，**只有 3 个配置过线**（YOLOv8n −2.30 / MixUp −1.06 / 无Mosaic+CBAM +0.74），
其余 7 个全部不可区分。加上评估端噪声与多重比较校正后（14.6），只剩 2 个。

**推翻了第 4 节 B 组的结论。** 那里写「去 Mosaic +1.24pp」，是拿 `base_nomosaic` 单次结果
（0.8034）与基线单次结果（0.7910）比出来的。补两个种子后该组均值降到 0.7962，
而 0.8034 恰好是全部 37 个 run 的最高值 —— 它是一次好签，不是一种更好的配置。

**也推翻了 14.6 上一版对 CBAM 的判定。** CBAM 的 0.7695 是它三个种子里最差的一次
（另两次 0.7868 / 0.7915），逐 seed 的 Δ 是 −2.15 / −0.09 / −0.06pp。

**还有一个更值得注意的现象**：**6 个配置的结论在补种子后翻转**（单次看"可区分"，
三种子看"不可区分"）—— 无Mosaic、+CBAM、+CA、CBAM@P3、残差CBAM、无Mosaic+CBAM。
10 个里有 6 个。这是「单种子消融表不可信」最直接的证据。

### 14.3 评估端噪声：bootstrap

对 base_v2 的 75 张 test 预测做 1000 次按图重采样（`tools/bootstrap_ci.py`）：

| 指标 | 点估计 | 95% CI | 半宽 |
|---|---|---|---|
| mAP50 | 0.9715 | [0.9695, 0.9809] | ±0.57pp |
| mAP50-95 | 0.7910 | [0.7629, 0.8161] | **±2.66pp** |

**这个数字要读对**：它描述的是**单个模型**在这批图上的 mAP 有多不确定 —— 换一批同样大小的图，
同一个权重的 mAP50-95 会在这个区间里晃。它**不能**直接当作「两个模型差异」的门槛，
14.5 会说明为什么。

### 14.4 mAP50 的饱和有了定量证据

11 组 × 3 种子下来，mAP50 的表现可以一句话说清：**没有任何一种改动能让它移动超过它自己的种子波动**。

- 全部 33 个 run 的 mAP50 落在 **0.9699 ~ 0.9718**（0.19pp 宽）；
- 基线 σ = **0.01pp**，而 +CBAM 0.09pp、无Mosaic+CA 0.07pp —— σ 本身也差 7 倍；
- 三十三个 run 里最大的配对差也只有 **−0.06pp**（无Mosaic+CA），而它的 σ 是 0.07pp
  —— **差异比噪声还小**。

也就是说：在这批数据上 mAP50 已经饱和，它区分不了任何两个配置。
加 CBAM 之后连 mAP50 都开始随种子晃（9 倍于基线），这从侧面印证了 14.2 里
它的 mAP50-95 σ 最大（1.16pp）—— 它不只是"没用"，还让训练变得更不稳。

### 14.5 配对 bootstrap：非配对 CI 不能拿来比较模型

上一版在这里下过一个结论，**是错的**：

> ~~评估端误差 2.66pp 是训练端种子波动的 4~10 倍，想提高分辨率唯一手段是扩大 test 集。~~

错在把「单模型的不确定性」当成了「两个模型差异的不确定性」。

两个模型跑的是**同一批 75 张图**。某张图特别难，会让两个模型**同时掉分** —— 这是
「图难度的公共成分」，在比较差异时应当抵消，不该叠进误差棒。分别独立重采样两个模型，
等于假设两次评估用了不同的图片，门槛被人为放大了。

正确做法是**配对 bootstrap**（`tools/paired_bootstrap.py`）：同一次重采样抽出的索引
**同时**作用于两个模型，算 Δ = mAP(当前) − mAP(基线)，重复 1000 次。

| 口径 | mAP50-95 CI 半宽 | 回答的问题 |
|---|---|---|
| 非配对（单模型 CI） | ±2.66pp | 这个权重本身有多不确定 |
| **配对（Δ 的 CI）** | **±0.57pp** | 换模型、不换图，差异还在吗 |

**收窄 4.7 倍。** 配对前，实现方式是给每个模型单独收集 per-image 统计量（batch=1，
每张图对应一份 `(tp, conf, pred_cls, target_cls)`），并以「每张图的 GT 框数序列」作指纹
校验两个模型的图顺序一致（不一致直接退出）。

结果（10 个配置 vs 基线，Bonferroni α=0.005）。
**这一列只含评估端噪声、且是 seed 0 单次** —— 把训练端噪声合进来后大部分会跨 0，见 14.6：

| 配置 | Δ mAP50-95 | 配对 95% CI | 配对 Bonferroni 后（未含训练端） |
|---|---|---|---|
| 无 Mosaic | **+1.25pp** | [+0.62, +1.76] | **显著为正** |
| 无Mosaic+CBAM | +0.46pp | [−0.22, +0.94] | 不可区分 |
| 无Mosaic+CA | −0.24pp | [−0.87, +0.40] | 不可区分 |
| +CA | −0.57pp | [−1.07, −0.13] | 仅未校正显著 |
| +SE | −0.74pp | [−1.29, −0.23] | 仅未校正显著 |
| 残差CBAM@SPPF | −0.93pp | [−1.48, −0.42] | **显著为负** |
| 加 MixUp 0.15 | −1.35pp | [−1.95, −0.74] | **显著为负** |
| CBAM@P3 | −1.68pp | [−2.24, −1.25] | **显著为负** |
| +CBAM | −2.15pp | [−2.76, −1.51] | **显著为负** |
| YOLOv8n | −2.52pp | [−3.07, −1.86] | **显著为负** |

**但这张表只代表 seed 0 那一次训练的两个权重** —— 配对 CI 只证明了「这两个已训好的权重之间有差别」，
没证明「这个配置换一次训练还会有差别」。后半句要靠训练端噪声，见 14.6。

### 14.6 两个噪声源合成

评估端（数据抽样）与训练端（随机种子）是**独立**的两个来源，合成方式：

```
SE_total = sqrt( SE_eval² + (σ_paired / √n)² )      CI = Δ ± z · SE_total
```

三处口径细节：

- **效应量取三种子配对均值**，不用 seed 0 单次值；
- **SE_train 要除以 √n** —— Δ 是 3 次训练的平均，其标准误是 σ/√3。此前直接拿 σ 当 SE，
  等于把 CI 放大了 √3 ≈ 1.7 倍（偏保守，不会造假阳性，但会掩盖真实效应）。
  同时保留一列"不除 √n"的**保守口径**，两种都过线结论才算硬；
- **SE_eval 不除以 √n** —— 75 张 test 是固定的一份样本，"这份测试集偏难/偏易"是三个模型
  共有的偏差，多训种子消不掉。

**本版关键改动**：训练端 σ 从"2 个配置实测 + 8 个配置借用"变成 **11 组全部实测**
（33 个 run）。逐 seed 校验见 14.2。

最终判定（Bonferroni α=0.05/10，z=2.807）：

| 配置 | n | Δ（三seed均值） | 配对 σ | 合成 95% CI | 合成 Bonferroni CI | 最终判定 |
|---|---|---|---|---|---|---|
| **YOLOv8n** | 3 | **−2.30pp** | 0.31pp | [−3.00, −1.61] | [−3.31, −1.30] | **显著为负** |
| **加 MixUp 0.15** | 3 | **−1.06pp** | 0.25pp | [−1.73, −0.39] | [−2.01, −0.11] | **显著为负** |
| +CBAM | 3 | −0.77pp | 1.20pp | [−2.26, +0.73] | [−2.90, +1.37] | 不可区分 |
| 残差CBAM@SPPF | 3 | −0.54pp | 0.45pp | [−1.27, +0.19] | [−1.59, +0.51] | 不可区分 |
| CBAM@P3 | 3 | −0.49pp | 1.12pp | [−1.85, +0.88] | [−2.44, +1.46] | 不可区分 |
| +SE | 3 | −0.36pp | 0.71pp | [−1.32, +0.60] | [−1.74, +1.01] | 不可区分 |
| +CA | 3 | −0.19pp | 0.35pp | [−0.81, +0.42] | [−1.07, +0.69] | 不可区分 |
| 无 Mosaic | 3 | +0.59pp | 0.57pp | [−0.27, +1.45] | [−0.64, +1.83] | 不可区分 |
| 无Mosaic+CBAM | 3 | +0.74pp | 0.25pp | [+0.10, +1.39] | [−0.18, +1.67] | 不可区分（仅 95%） |
| 无Mosaic+CA | 3 | +0.75pp | 0.91pp | [−0.47, +1.96] | [−0.99, +2.48] | 不可区分 |

**没有一行带「借用」脚注了。**

**只剩 2 个过线，余量差别很大**：

| 配置 | 当前配对 σ | 临界 σ | 余量 | 保守口径 |
|---|---|---|---|---|
| YOLOv8n | 0.31pp | 1.32pp | **4.3×** | 仍显著 |
| 加 MixUp 0.15 | 0.25pp | 0.38pp | **1.5×** | 刚跨 0（[−2.17, +0.06]） |

YOLOv8n 这条已经相当硬：σ 要涨到 4.3 倍才翻转，而它的实测 σ 是 11 组里**最小**的
（三种子 0.7662 / 0.7707 / 0.7646）。MixUp 只有 1.5 倍余量，保守口径还差一点，
所以只能说"**方向稳定为负、量级约 1.1pp，但没到铁证**"。

**交叉验证**：11 个配置同时有配对 bootstrap 与三种子配对两组数字：

| 配置 | seed 0 配对 bootstrap Δ | 三种子配对 Δ | 偏差 |
|---|---|---|---|
| **+CBAM** | **−2.15pp** | **−0.77pp** | **−1.38pp** |
| **CBAM@P3** | **−1.68pp** | **−0.49pp** | **−1.19pp** |
| **无Mosaic+CA** | **−0.24pp** | **+0.75pp** | **−0.98pp（方向翻转）** |
| 无 Mosaic | +1.25pp | +0.59pp | +0.66pp |
| 其余 7 个 | — | — | −0.22 ~ −0.38pp |

**10 个里 6 个量级缩水 40~70%，1 个连方向都翻转**（无Mosaic+CA：−0.24 → +0.75pp）。
这就是整个实验最值得写下来的一点：**「哪个权重更好」和「这个方法更好」是两个不同问题，
答案可以不一样。**

**已知局限**：n=3 时 σ 自身不可靠（90% CI 约 [0.58s, 4.4s]），所以合成只到量级，
不当精确 p 值用。另外三种子均值的 SE_eval 仍取 seed 0 那一对的配对半宽作为代表值
（严格做法是对 3 个权重各做一次配对 bootstrap 再合并，但评估端 SE 主要由 75 张图决定）。

### 14.7 判据

判据不是"一个固定门槛"，而是**按配置查它自己的 MDD**（= 2×合并标准差）：

| 配置 | MDD | 配置 | MDD |
|---|---|---|---|
| YOLOv8n | 0.35pp | 残差CBAM | 0.88pp |
| 加 MixUp | 0.45pp | 无 Mosaic | 1.01pp |
| +CA | 0.51pp | 无Mosaic+CA | 1.17pp |
| 无Mosaic+CBAM | 0.59pp | +SE | 1.29pp |
| — | — | CBAM@P3 | 1.45pp |
| — | — | +CBAM | 1.67pp |

用固定门槛会错：拿 MDD=1.01pp 去判 CBAM 的 −0.77pp，结论"不可区分"是**碰巧对**的；
但拿它去判一个 −1.3pp 的效应就会误判成可区分，而 CBAM 的真实阈值是 1.67pp。
反过来，用 1.01pp 判 MixUp 的 −1.06pp 也会得出"可区分"，同样碰巧对 ——
但余量只有 0.05pp，而按它自己的 0.45pp 门槛，余量是 0.61pp。**门槛错了，余量就错了。**

按这把尺子 + 14.6 的双噪声合成重判：**只有 YOLOv8n（−2.30pp）与 MixUp（−1.06pp）过线**；
CBAM、CBAM@P3、残差 CBAM、SE、CA、无 Mosaic、两个交叉格 **全部不可区分**。

**三次作废说明**：

| 版本 | 门槛 | 错在哪 |
|---|---|---|
| v1 | <1.0 / 1.0~2.7 / >2.7pp | 用非配对 CI 当配对门槛，高估约 5 倍 |
| v2 | ±1.15pp / Bonferroni ±1.64pp | 门槛量级对了，但当成所有配置的通用值 —— 等于假设"加注意力不增加训练随机性" |
| v3 | 查各配置自己的 MDD，但仅 5 组实测 σ | 仍有 6 个配置借用，而唯一过线的 YOLOv8n 恰好在借用名单里 |
| v4（当前） | 11 组全实测 σ + 双噪声合成 + Bonferroni + 保守口径 | — |

v2 的错法让 CBAM 从"差异在噪声内"变成"确定有害"，而实测 σ 证明它只是抖了一下。
v3 的错法更隐蔽：**借用 σ 高估了 2.4 倍**（0.74 vs 实测 0.31pp），把 YOLOv8n 的余量
从 4.3 倍压到 1.1 倍，差一点就被当成"不可靠"丢掉。**借用 σ 不只影响显著性判定，
还会让本来很硬的结论看起来摇摇欲坠。**

### 14.8 复现命令

```bash
python tools/train_baseline.py --seed 0 --name base_v2_repro      # 复现检验
python tools/train_baseline.py --seed 1 --name base_v2_s1
python tools/train_baseline.py --seed 2 --name base_v2_s2
python tools/train_baseline.py --seed 1 --mosaic 0.0 --name base_nomosaic_s1
python tools/train_baseline.py --seed 2 --mosaic 0.0 --name base_nomosaic_s2

# 注意力组种子（把「借用 σ」变成「实测 σ」）
python tools/train_cbam_fixed.py --variant se    --seed 1 --name train_se_s1
python tools/train_cbam_fixed.py --variant se    --seed 2 --name train_se_s2
python tools/train_cbam_fixed.py --variant plain --seed 1 --name train_cbam_s1
python tools/train_cbam_fixed.py --variant plain --seed 2 --name train_cbam_s2
python tools/train_cbam_fixed.py --variant ca    --seed 1 --name train_ca_s1
python tools/train_cbam_fixed.py --variant ca    --seed 2 --name train_ca_s2

# 其余 6 个配置的种子（本版新增，共 12 个 run）—— 命令由脚本管，避免手写漏配
bash tools/train_seeds.sh --dry-run     # 先看清单
bash tools/train_seeds.sh               # 可重入：跳过已完成、崩溃重试一次、半成品自动归档

# 校验配对性：逐字段比对 114 个超参 + 权重结构签名
python -m tools.check_pairing           # 输出 `22/22 对通过` 才说明只差 seed

# 评估所有种子 run（可重入：已有 json 的跳过）
bash tools/eval_seeds.sh
# 注意：评估与训练**不能并行**。实测过一边训练一边评估，16G 内存被打爆，
#      训练在第 46 轮因 OpenCV `Insufficient memory` 崩溃。

python tools/bootstrap_ci.py --weights runs/detect/train_base_v2/weights/best.pt --name base_v2 --n 1000

# 统计：不给 --group 时自动读 experiment_registry 的 11 组
python tools/seed_variance.py

# 配对 bootstrap（全部 10 组 vs 基线一次跑完，约 3 分钟）
python -m tools.paired_bootstrap \
    --ref base_v2=runs/detect/train_base_v2/weights/best.pt \
    --model se=runs/detect/train_se/weights/best.pt --label "+SE" \
    --model cbam_v2=runs/detect/train_cbam_v2/weights/best.pt --label "+CBAM" \
    --model ca=runs/detect/train_ca/weights/best.pt --label "+CA" \
    --model base_nomosaic=runs/detect/train_base_nomosaic/weights/best.pt --label "无 Mosaic" \
    --n 1000

# 两个噪声源合成 + Bonferroni（读上面两个 json，秒级）
python -m tools.noise_budget
```

> **顺序很重要**：`seed_variance.py` 必须先跑完再跑 `noise_budget.py` ——
> 后者从 `seed_variance.json` 读各组实测 σ，靠 `experiment_registry.PB_KEY_TO_GROUP`
> 把配置名映射到种子组名。

> **组名只维护一份**：11 组 ↔ 33 个 run 的映射写在 `tools/experiment_registry.py`，
> `seed_variance` / `noise_budget` / `check_pairing` 全部 import 它。
> 此前这份映射在三个地方各写一遍，已经因为命名漂移（`train_se_s1` vs `se_s1`）
> 静默读不到数据一次 —— 所以现在不再手写命令行 `--group`。

> 坑：`bootstrap_ci.py` 构造 validator 参数时必须显式 `rect=True`，否则 mAP 会低 1.36pp
> （0.7774 vs 0.7910）——`model.val()` 内部是 `custom={"rect": True}`，而 `get_cfg` 默认是 False。
> 另外 `validator.stats` 被 `get_stats()` 清空，per-image 统计量必须在 `get_stats` 调用前截获。
