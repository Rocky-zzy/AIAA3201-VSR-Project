import torch.nn as nn

class VGGStyleDiscriminator(nn.Module):
    """一个经典的 VGG 风格的判别器，用于区分真假高分辨率图像"""
    def __init__(self, in_channels=3, num_feat=64):
        super(VGGStyleDiscriminator, self).__init__()
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((128, 128))
        self.conv0_0 = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat)
        
        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2)
        
        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4)

        # 修正：128x128 输入经过3次下采样(每次2倍)变为 16x16
        # 通道数: num_feat * 4 = 64 * 4 = 256
        # 总特征数: 256 * 16 * 16 = 65536
        self.linear1 = nn.Linear(num_feat * 4 * 16 * 16, 100)
        self.linear2 = nn.Linear(100, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        # 关键修改：在开头使用 adaptive_pool 确保输入尺寸统一
        x = self.adaptive_pool(x)
        
        x = self.lrelu(self.conv0_0(x))
        x = self.lrelu(self.bn0_1(self.conv0_1(x)))
        x = self.lrelu(self.bn1_0(self.conv1_0(x)))
        x = self.lrelu(self.bn1_1(self.conv1_1(x)))
        x = self.lrelu(self.bn2_0(self.conv2_0(x)))
        x = self.lrelu(self.bn2_1(self.conv2_1(x)))
        
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear1(x))
        x = self.linear2(x)
        return x