import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(卷积 => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样: MaxPool2d => DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样 => 裁剪/填充 => 拼接(Skip Connection) => DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # 如果使用双线性插值，卷积层需要处理通道减半
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 这里的 kernel_size=2, stride=2 会让尺寸精确翻倍
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 是当前层上采样后的结果
        # x2 是从 Encoder 传过来的 Skip Connection 特征图
        x1 = self.up(x1)
        
        # --- 处理尺寸不匹配问题 ---
        # 如果输入图像尺寸是奇数，MaxPool 会向下取整，导致上采样后 x1 比 x2 小 1 个像素
        # 这里动态计算 padding 来补齐 x1
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # --- 拼接 (Skip Connection) ---
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # --- 1. Encoder (收缩路径) ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # --- 2. Bottleneck (中间层) ---
        # factor 用于处理双线性插值的通道对齐，默认为 1
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # === 加入 Dropout 防止过拟合 ===
        self.dropout = nn.Dropout(p=0.5) 

        # --- 3. Decoder (扩张路径) ---
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # --- 4. Output Head ---
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)        # 64
        x2 = self.down1(x1)     # 128
        x3 = self.down2(x2)     # 256
        x4 = self.down3(x3)     # 512
        x5 = self.down4(x4)     # 1024 (Bottleneck)
        
        # Dropout 在最深层特征处使用
        x5 = self.dropout(x5)
        
        # Decoder (带跳跃连接)
        x = self.up1(x5, x4)    # 1024 -> 512
        x = self.up2(x, x3)     # 512 -> 256
        x = self.up3(x, x2)     # 256 -> 128
        x = self.up4(x, x1)     # 128 -> 64
        
        logits = self.outc(x)
        return logits