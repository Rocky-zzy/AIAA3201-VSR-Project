import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d 
import torch.nn.functional as F

class AlignmentModule(nn.Module):
    def __init__(self, n_feats, offset_clip=10.0):
        super().__init__()
        self.offset_clip = offset_clip
        # 偏移量预测卷积
        self.offset_conv = nn.Conv2d(n_feats * 2, 2 * 3 * 3, 3, 1, 1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        
        self.weight = nn.Parameter(torch.Tensor(n_feats, n_feats, 3, 3))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, ref):
        combined = torch.cat([x, ref], dim=1)
        offset = self.offset_conv(combined)
        
        #  对偏移量进行截断，防止过大位移
        offset = torch.clamp(offset, -self.offset_clip, self.offset_clip)
        
        with torch.cuda.amp.autocast(enabled=False):
            aligned_feat = deform_conv2d(
                x.float(), offset.float(),
                weight=self.weight.float(),
                padding=1
            )
        return aligned_feat.to(x.dtype)


class ResidualBlock(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        )
        # 残差分支末层零初始化 → 初始时近似恒等映射
        nn.init.zeros_(self.main[2].weight)
        nn.init.zeros_(self.main[2].bias)

    def forward(self, x):
        return x + self.main(x)

class AdvancedVSR(nn.Module):
    def __init__(self, num_feat=64, num_frames=5, num_blocks=10):
        super().__init__()
        self.num_frames = num_frames
        self.center_idx = num_frames // 2

        # 1. 特征提取
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(num_feat)
        )

        # 2. 对齐模块
        self.align_module = AlignmentModule(num_feat)

        # 3. 融合 + 深层残差
        self.res_blocks = nn.Sequential(
            nn.Conv2d(num_feat * num_frames, num_feat, 3, 1, 1),
            *[ResidualBlock(num_feat) for _ in range(num_blocks)]
        )

        # 4. 重建上采样
        self.conv_upsample = nn.Conv2d(num_feat, 48, 3, 1, 1)
        self.upsample = nn.PixelShuffle(4)
        self.conv_last = nn.Conv2d(3, 3, 3, 1, 1)


    def forward(self, x):
        b, n, c, h, w = x.size()

        # A. 逐帧特征提取
        frames = x.view(-1, c, h, w)
        feats = self.feat_extract(frames)
        feats = feats.view(b, n, -1, h, w)

        # B. 以中间帧为参考，对齐周围帧
        ref_feat = feats[:, self.center_idx, :, :, :]
        aligned_feats = []
        for i in range(n):
            if i == self.center_idx:
                aligned_feats.append(ref_feat)
            else:
                aligned_feats.append(self.align_module(feats[:, i, :, :, :], ref_feat))

        # C. 特征拼接与融合
        feat_fusion = torch.cat(aligned_feats, dim=1)
        feat = self.res_blocks(feat_fusion)

        # --- 之前的特征提取和融合不变 ---
        
        # D. 改良版上采样重建
        out_feat = self.upsample(self.conv_upsample(feat)) # 此时是 3 通道高频特征
        
        center_lr = x[:, self.center_idx, :, :, :]
        # 直接把 Bicubic 放大图作为“基础支架”
        base_img = F.interpolate(center_lr, scale_factor=4, mode='bilinear', align_corners=False)
        
        # 让卷积层去处理 (高频特征 + 底图) 的融合，而不是直接暴力相加
        res = self.conv_last(out_feat)
        out = base_img + res * 0.1
        
        return torch.clamp(out, 0.0, 1.0)
