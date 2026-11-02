# eval_reports 文件说明

## 当前有效（可以引用）

| 文件 | 内容 |
|---|---|
| `base_v2.json/md` | 基线，同环境重训，test 75 张（= 完整增强） |
| `se.json/md` | +SE @SPPF（复现原论文 C 组） |
| `cbam_v2.json/md` | +CBAM @SPPF（复现原论文 C 组） |
| `ca.json/md` | +CoordAtt @SPPF（复现原论文 C 组） |
| `cbam_res_v2.json/md` | 残差 CBAM @SPPF（本工作新增） |
| `cbam_p3.json/md` | CBAM 改插 P3（本工作新增） |
| `base_nomosaic.json/md` | 基线 + mosaic=0（复现原论文 B/D 组） |
| `base_mixup.json/md` | 基线 + mixup=0.15（本工作替换原论文的空实验） |
| `cbam_nomosaic.json/md` | CBAM + mosaic=0（复现原论文 D 组） |
| `ca_nomosaic.json/md` | CA + mosaic=0（复现原论文 D 组） |
| `yolov8n.json/md` | YOLOv8n 50ep（复现原论文 A 组，RT-DETR 未做） |
| `count_sweep_test.json/md` | 11 组计数指标，阈值在 val 上选定（conf=0.65 / iou=0.5） |
| `count_sweep_val.csv` | val 上的 (conf, iou) 网格，用于选阈值 |
| `speed.csv` | 11 组参数量与推理耗时（`ms` 未 fuse，`ms_fused` fuse 后）。**绝对时延跨批次不可复现（7.8~12.0ms），只作同批次相对参考** |
| `compare.md` | 汇总对比表（五张，按实验分组：C / 注入形式 / B / D / A） |
| `compare_real.png` | 注意力 6 组四联柱状图：mAP50 / mAP50-95 / 计数 MAE / 耗时 |
| `aug_cross.png` | 增强 × 注意力双因素三面板图（B 组 + D 组） |
| `count_sensitivity.png` | 计数误差随 conf 阈值变化（11 条曲线） |

> 注意：`*.json` 里的 `counting` 字段是 **conf=0.25 固定阈值**下的结果，
> 仅供单模型自查；**跨模型比较请用 `count_sweep_test.json`**（阈值在 val 上选定）。
> 例如 `base_nomosaic` 在固定阈值下 MAE 是 7.40，统一阈值后是 2.64 —— 差值全部来自阈值选择。

## 历史留档（不要引用数字）

| 文件 | 为什么不能引用 |
|---|---|
| `base_v1.json/md` | 早期基线 run，与 `base_v2` 不是同一批次，仅用于确认基线稳定性 |
| `cbam.json/md` | 早期 CBAM run，配置与当前注入链路不一致（详见 `EXPERIMENT_CBAM.md` 第 3 节） |

这两组的核心用途是记录"同一个配置跑两遍能差多少"，用来判断后续差异是否显著，
不作为结论依据。需要清理时可直接删除，不影响 `tools/make_report.py`。

---

## 2026-11-01 新增：F 组噪声预算

| 文件 | 内容 |
|---|---|
| `base_v2_repro.json/md` | seed=0 重跑，用于复现检验（结果与 `base_v2` 完全一致，0.9715 / 0.7910） |
| `base_v2_s1.json/md`、`base_v2_s2.json/md` | 完整增强，seed = 1 / 2 |
| `base_nomosaic_s1.json/md`、`base_nomosaic_s2.json/md` | 无 Mosaic，seed = 1 / 2 |
| `train_se_s1/s2.json/md` | +SE，seed = 1 / 2 |
| `train_cbam_s1/s2.json/md` | +CBAM，seed = 1 / 2 |
| `train_ca_s1/s2.json/md` | +CA，seed = 1 / 2 |
| `train_cbam_p3_s1/s2.json/md` | CBAM@P3，seed = 1 / 2（**本轮新增**） |
| `train_cbam_res_s1/s2.json/md` | 残差CBAM@SPPF，seed = 1 / 2（**本轮新增**） |
| `train_base_mixup_s1/s2.json/md` | 加 MixUp 0.15，seed = 1 / 2（**本轮新增**） |
| `train_cbam_nomosaic_s1/s2.json/md` | 无Mosaic+CBAM，seed = 1 / 2（**本轮新增**） |
| `train_ca_nomosaic_s1/s2.json/md` | 无Mosaic+CA，seed = 1 / 2（**本轮新增**） |
| `train_yolov8n_s1/s2.json/md` | YOLOv8n，seed = 1 / 2（**本轮新增**） |
| `rtdetr_legacy.json` | RT-DETR-L 20ep 的 test 指标（**跨环境**，不参与排名） |
| `bootstrap_base_v2.json/md` | **单模型**置信区间（非配对），75 图 × 1000 次；mAP50-95 半宽 ±2.66pp。**不能当作模型间比较的门槛**，见下 |
| `seed_variance.json/md/png` | 种子方差（**11 组 × 3 seed**）、配对 Δ 与各自的 MDD |
| `paired_bootstrap.json/md/png` | **配对** bootstrap：10 个配置 vs 基线，同一重采样索引同时作用于两个模型；Δ 的 CI 半宽仅 **±0.57pp**（比非配对窄 4.7 倍）+ Bonferroni 多重比较校正 |
| `noise_budget.json/md/png` | **最终判定表**：评估端 SE 与训练端 σ 合成，给出每个配置能否作为结论的结论。读这一份就够 |

**总数**：37 个训练 run（11 组 × 3 seed = 33，加 1 个复现检验 run、2 个作废 run、1 个崩溃归档）。
其中 11 组配置的 seed 0 进入 `compare.md` 的分组表（RT-DETR 单列标注），
24 个种子 run 只用于噪声分析、不进对比表。

**引用时注意**：
- 判据以 `noise_budget.md` 为准，而且**门槛不是常数**：每个配置查它自己的 MDD。
  YOLOv8n 0.35pp / MixUp 0.45pp / CA 0.51pp / 无Mosaic+CBAM 0.59pp / 残差CBAM 0.88pp /
  无 Mosaic 1.01pp / 无Mosaic+CA 1.17pp / SE 1.29pp / CBAM@P3 1.45pp / **CBAM 1.67pp**。
- 十个对比里只有 2 个差异站得住：**YOLOv8n（−2.30pp，余量 4.3 倍）** 与
  **加 MixUp 0.15（−1.06pp，余量 1.5 倍）**。其余 8 个全部不可区分。
- **CBAM 的「显著为负 −2.15pp」已作废**：那是 seed 0 单次值 + 借用的 σ。
  补齐三个种子后效应是 **−0.77pp**，配对 σ 就有 1.20pp，不可区分。
- **YOLOv8n 的「余量仅 1.1 倍」也作废**：借用 σ=0.74pp 高估了 2.4 倍，实测 0.31pp。
- 三次判据作废记录见 `REPRO_STATUS.md` 第 11.7 节，不要再用「±1.15pp / ±1.64pp」
  或「<1.0 / 1.0~2.7 / >2.7pp」这类固定门槛。
