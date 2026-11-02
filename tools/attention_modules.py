"""SE 与 CoordAtt 注意力模块（ultralytics 兼容版）。

为什么还要单独一个文件
--------------------
`tools/cbam_module.py` 里只有 CBAM / CBAMResidual。原论文的注意力消融是
**三个模块对比**：SE、CBAM、CoordAtt（CA）。2024 版的 ultralytics（原实验用的
8.4.37）里带了 SE 和 CoordAtt，但当前环境 8.4.194 已经把这两个删掉了
（`hasattr(ultralytics.nn.modules, "SE") == False`），所以必须自己补，
否则「三模块消融」根本复现不了。

签名为什么要写成 (c1, reduction, kernel_size)
--------------------------------------------
`tools/train_cbam_fixed.py::build_cbam_yaml` 会统一生成
`[-1, 1, <模块名>, [channels, reduction, kernel]]` 这一层配置。
自定义模块走 parse_model 的 `else` 分支，yaml 里给几个参数就原样传几个，
所以 SE / CoordAtt 必须能吃下这 3 个参数——用不到的 kernel_size 显式吸收，
这样三个模块共用同一条注入/重映射/断言链路，**唯一变量才是模块本身**。

与原论文实现的差异
----------------
1. `reduction`：SE 用 16、CA 用 32，与原脚本 `train_se.py` / `train_ca.py` 一致。
2. 与原脚本一样保持**朴素形式** `x * mask`，不做残差化。代价是初始 mask ≈ 0.5
   （SE）或 ≈ 0.25（CA，两级 sigmoid 相乘），会整体缩放预训练特征；CBAM 那边
   已有对照实验（plain vs residual）量化过这件事，这里保持原样以便和论文数字对照。
3. BN 层保留在 CoordAtt 内（原实现就有），SE 不加 BN（原实现也没有）。

用法
----
    from tools.attention_modules import SE, CoordAtt, register_extra
    register_extra()                       # 注入 ultralytics.nn.tasks
    YOLO("cfg/yolo11n-SE.yaml")
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SE(nn.Module):
    """Squeeze-and-Excitation：全局平均池化 → 两层 FC 瓶颈 → sigmoid 通道权重。

    yaml 写法：[-1, 1, SE, [channels, reduction, kernel]]
    kernel 参数仅为与其他注意力模块共享配置格式而存在，SE 不使用。
    """

    def __init__(self, c1: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.c1 = int(c1)
        hidden = max(self.c1 // int(reduction), 4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 用 1x1 conv 而不是 Linear：省掉 flatten/reshape，且对非方形输入更稳
        self.fc = nn.Sequential(
            nn.Conv2d(self.c1, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.c1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.shape[1] != self.c1:
            raise RuntimeError(
                f"SE 通道不匹配：配置 {self.c1}，实际输入 {x.shape[1]}。"
                f"检查 cfg 里通道是否写成了缩放后的值（yolo11n 的 SPPF 输出应为 256）。"
            )
        return x * self.fc(self.avgpool(x))


class h_sigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu6(x + 3.0) / 6.0


class h_swish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.h_sigmoid = h_sigmoid(inplace)

    def forward(self, x):
        return x * self.h_sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention（CVPR 2021）。

    与 SE 的区别：SE 把 H×W 压成 1×1，位置信息全丢；CoordAtt 分别沿 H 和 W
    做一维池化，得到两组带方向信息的描述子，再共享 1×1 卷积后拆开、
    分别生成 H、W 方向的注意力图，最后 `x * a_h * a_w`。

    对密集小目标检测来说，这个"保留坐标"的动机是成立的（相邻甘蓝的区分
    依赖空间位置而非通道响应），所以这一组是整个消融里最值得看的一项。

    yaml 写法：[-1, 1, CoordAtt, [channels, reduction, kernel]]
    """

    def __init__(self, c1: int, reduction: int = 32, kernel_size: int = 7):
        super().__init__()
        self.c1 = int(c1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        hidden = max(8, self.c1 // int(reduction))
        self.conv1 = nn.Conv2d(self.c1, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(hidden, self.c1, 1, bias=False)
        self.conv_w = nn.Conv2d(hidden, self.c1, 1, bias=False)

    def forward(self, x):
        if x.shape[1] != self.c1:
            raise RuntimeError(
                f"CoordAtt 通道不匹配：配置 {self.c1}，实际输入 {x.shape[1]}。"
            )
        n, c, h, w = x.size()

        x_h = self.pool_h(x)                      # (n, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (n, c, w, 1)

        y = torch.cat([x_h, x_w], dim=2)          # (n, c, h+w, 1)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)             # (n, c, 1, w)

        a_h = self.conv_h(x_h).sigmoid()          # (n, c, h, 1)
        a_w = self.conv_w(x_w).sigmoid()          # (n, c, 1, w)
        return x * a_h * a_w


def register_extra(verbose: bool = True) -> None:
    """把 SE / CoordAtt 注入 ultralytics.nn.tasks，供 parse_model 按名字查找。"""
    import ultralytics.nn.tasks as tasks

    tasks.SE = SE
    tasks.CoordAtt = CoordAtt
    if verbose:
        print("[ok] SE / CoordAtt 已注册到 ultralytics.nn.tasks")


if __name__ == "__main__":
    # 自检：形状不变、反向可走、初始缩放系数有多大（解释"朴素注意力会压缩预训练特征"）
    torch.manual_seed(0)
    x = torch.randn(2, 256, 20, 20)
    for cls, red in ((SE, 16), (CoordAtt, 32)):
        m = cls(256, red, 7)
        y = m(x)
        y.sum().backward()
        ratio = (y.detach() / x).mean().item()
        n_p = sum(p.numel() for p in m.parameters())
        print(
            f"{cls.__name__:9s} 输入{tuple(x.shape)} → 输出{tuple(y.shape)}  "
            f"参数量 {n_p:>6d}  初始输出/输入均值 {ratio:.3f}"
        )
