import torch
import torch.nn as nn

class ResidualBlockNoBN(nn.Module):
    """没有 Batch Normalization 的残差块（超分任务的标准配置）"""
    def __init__(self, num_feat=64):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out # 残差连接