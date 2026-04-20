import torch
import torch.nn as nn

class SimpleVSR(nn.Module):
    def __init__(self, num_feat=64, num_frames=5):
        super(SimpleVSR, self).__init__()
        
        # 1. 初始卷积层
        # 输入是 5 帧 RGB 图像，所以通道数是 5 * 3 = 15
        self.conv_first = nn.Conv2d(num_frames * 3, num_feat, 3, 1, 1)
        
        # 2. 特征提取层（简单的残差结构）
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # 3. 上采样层 (Upsampling)
        # 目标是放大 4 倍。PixelShuffle 会把通道（Channel）转换成像素。
        # 要得到 3 通道 (RGB) 的 4x 放大图，我们需要 3 * (4*4) = 48 个通道
        self.conv_upsample = nn.Conv2d(num_feat, 48, 3, 1, 1)
        self.upsample = nn.PixelShuffle(4)
        
        # 4. 最后的细化层
        self.conv_last = nn.Conv2d(3, 3, 3, 1, 1)
        
    def forward(self, x):
        # x 的形状: (B, 5, 3, H, W)
        b, n, c, h, w = x.size()
        
        # --- 核心步骤：融合维度 ---
        # 把 (B, 5, 3, H, W) 变成 (B, 15, H, W)
        # 这样模型就能同时看到 5 帧的信息
        x = x.view(b, n * c, h, w)
        
        # 提取特征
        feat = self.conv_first(x)
        feat = self.feature_extraction(feat)
        
        # 放大分辨率
        out = self.upsample(self.conv_upsample(feat))
        
        # 细化输出
        out = self.conv_last(out)
        
        return out