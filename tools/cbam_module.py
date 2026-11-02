"""CBAM 注意力模块（ultralytics 兼容版）。

为什么单独放一个文件
------------------
ultralytics 保存 best.pt 时会把模型对象 pickle 起来，pickle 记录的是类的
"定义位置"（__module__）。如果 CBAM 直接写在训练脚本里，位置会变成
`__main__.CBAM`——之后任何别的脚本（推理、Flask、评估）加载这个权重都会
报 `Can't get attribute 'CBAM' on <module '__main__'>`。

把它定义在可 import 的模块里，保存的位置就是 `tools.cbam_module.CBAM`，
任何脚本只要 `from tools.cbam_module import CBAM` 就能加载。

关于 args 的坑（踩过）
------------------
ultralytics 的 parse_model 只会给「白名单模块」（Conv/C3k2/SPPF 等）自动
补上 `[c1, c2, *args]`，自定义模块走的是 `else` 分支，yaml 里写什么就原样
传什么。所以 yaml 里的通道必须写成**缩放后的实际通道**（yolo11n 的 SPPF
输出是 1024×0.25 = 256），而不是 1024。写 1024 会得到 c1=1024 c2=16 的
诡异错误。

用法
----
    from tools.cbam_module import register
    register()                       # 让 parse_model 能按名字找到 CBAM
    YOLO("cfg/yolo11n-cbam.yaml")    # 之后才能构建带 CBAM 的模型
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力：平均池化 + 最大池化过共享 MLP，相加后 sigmoid。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    """空间注意力：沿通道维做 avg/max，拼成 2 通道后 7x7 卷积。"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1))) * x


class CBAM(nn.Module):
    """ultralytics 模块约定：yaml 里写 [channels, reduction, kernel]。

    channels 必须是**该层实际通道数**（已按 width 缩放）。见文件头说明。
    """

    def __init__(self, c1: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.c1 = int(c1)
        self.channel_attention = ChannelAttention(self.c1, int(reduction))
        self.spatial_attention = SpatialAttention(int(kernel_size))

    def forward(self, x):
        if x.shape[1] != self.c1:
            raise RuntimeError(
                f"CBAM 通道不匹配：配置 {self.c1}，实际输入 {x.shape[1]}。"
                f"检查 cfg 里 CBAM 的通道是否写成了缩放后的值（yolo11n 应为 256）。"
            )
        return self.spatial_attention(self.channel_attention(x))


class CBAMResidual(nn.Module):
    """残差式 CBAM：out = x + gamma * CBAM(x)。

    为什么需要它
    -----------
    朴素 CBAM 的 forward 是 `x * mask`，mask 由两层 sigmoid 相乘得到，
    初始值约 0.5 * 0.5 = 0.25 —— 等于把预训练特征整体缩小 4 倍。加载
    yolo11n.pt 后第一轮 mAP 只有 0.0036（基线 0.89），就是因为预训练
    特征被这个缩放打乱了，只能从头重学。

    这里用 ReZero 思路：gamma 是可学习参数、**初始化为 0**，所以训练
    开始时该模块严格等于恒等映射 `out = x`，预训练特征一点不受影响；
    之后再靠梯度把注意力"慢慢接进来"。

    gamma 是逐通道的（shape = [c1, 1, 1]），这样它能学到"只在部分通道
    上使用注意力"，比整层一个标量更灵活。

    yaml 写法与 CBAM 完全一致：[-1, 1, "CBAMResidual", [channels, reduction, kernel]]
    """

    def __init__(self, c1: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.c1 = int(c1)
        self.block = CBAM(self.c1, int(reduction), int(kernel_size))
        self.gamma = nn.Parameter(torch.zeros(self.c1, 1, 1))

    def forward(self, x):
        if x.shape[1] != self.c1:
            raise RuntimeError(
                f"CBAMResidual 通道不匹配：配置 {self.c1}，实际输入 {x.shape[1]}。"
            )
        return x + self.gamma * self.block(x)


def register(verbose: bool = True) -> None:
    """把 CBAM / CBAMResidual 注入 ultralytics.nn.tasks 命名空间，供 parse_model 按名字查找。"""
    import ultralytics.nn.tasks as tasks

    tasks.CBAM = CBAM
    tasks.CBAMResidual = CBAMResidual
    if verbose:
        print("[ok] CBAM / CBAMResidual 已注册到 ultralytics.nn.tasks")


if __name__ == "__main__":
    # 自检：形状不变 + 反向传播能走通 + 参数量可算
    x = torch.randn(2, 256, 20, 20, requires_grad=True)
    m = CBAM(256, 16, 7)
    y = m(x)
    y.sum().backward()
    print("输入", tuple(x.shape), "输出", tuple(y.shape))
    print("参数量", sum(p.numel() for p in m.parameters()))
