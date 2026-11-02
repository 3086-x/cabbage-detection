# 运行环境说明
#
# 复现所需的完整环境。若只做离线统计复算（见 README「可复核性」），
# 只需 Python + numpy + matplotlib，无需 GPU。

## 实测环境

| 项 | 值 |
|---|---|
| OS | Windows 11 |
| Python | 3.13.12 |
| ultralytics | **8.4.194**（训练指标对此版本敏感，必须固定） |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA RTX 5060 Laptop，8 GB 显存 |
| 内存 | 16 GB |

## 安装

```bash
python -m venv .venv
source .venv/Scripts/activate           # Windows Git Bash
# 或 .venv\Scripts\activate            # CMD / PowerShell

pip install -r requirements.txt

# PyTorch 请按本机 CUDA 版本装对应 wheel，例如 cu128：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 训练配置（全部 33 个 run 共用）

| 参数 | 值 |
|---|---|
| epochs | 50 |
| batch | 8 |
| imgsz | 640 |
| optimizer | SGD，lr0 = 0.01 |
| 预训练 | COCO `yolo11n.pt` |
| 冻结 | 不冻结，全量微调 |

除 `seed` 与所研究的变量外，其余 114 个超参与权重结构完全一致，
由 `tools/check_pairing.py` 逐字段机器校验（22/22 对通过）。

## 8 GB 显存下的两个实操约束

1. **训练与评估必须串行**。训练中途启动评估会因 `cv2` 分配不到内存使训练崩溃
   （本项目实际踩过：`train_cbam_p3_s1` 在第 46 轮 OOM）。`tools/train_seeds.sh`
   与 `tools/eval_seeds.sh` 已按串行设计，且支持断点续跑。
2. **判断训练是否完成要看 `results.csv` 行数 = 50**，不能看 `best.pt` 是否存在——
   ultralytics 从第 1 个 epoch 就开始写 `best.pt`，崩溃的 run 也会留下权重文件。

## 复现性实测

`base_v2_repro` 与 `train_base_v2` 为同配置同种子重跑：50/50 轮，所有指标逐位相同
（box_loss 差值 `+0.00e+00`）。说明在固定 seed + 固定版本下，本环境完全确定性。
