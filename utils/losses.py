import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # 加载预训练的 VGG19
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # 通常我们截取到第 35 层 (ReLU5_4)，这包含了非常丰富的高级纹理特征
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:36]).eval()
        
        # 冻结 VGG 参数，我们只用它做特征提取，不训练它
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        # VGG 期望的输入是归一化到 ImageNet 标准的
        # 你的输出是 0-1 范围，可以直接送入，或者做简单的均值方差归一化（这里为了简便直接提取）
        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        return self.criterion(x_features, y_features)