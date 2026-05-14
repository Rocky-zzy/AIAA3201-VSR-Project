import os
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import lpips

# 导入FlowMatching模型
from part3_flow_matching_vsr import Part3_FlowMatchingVSR
from utils.metrics import calculate_psnr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def pad_to_multiple(x, multiple=16):
    """确保图像尺寸是16的倍数"""
    h, w = x.size(2), x.size(3)
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return x, h, w

def test_flowmatching():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载FlowMatching模型权重
    model_path = 'experiments_flowmatching/flowmatching_best.pth'
    if not os.path.exists(model_path):
        model_path = 'experiments_flowmatching/flowmatching_latest.pth'
    
    model = Part3_FlowMatchingVSR(device=device).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 初始化LPIPS评估器
    lpips_vgg = lpips.LPIPS(net='alex').to(device)
    
    # 结果保存目录
    os.makedirs("results_flowmatching", exist_ok=True)
    to_tensor = ToTensor()
    to_pil = ToPILImage()

    # 测试图像路径（与原test.py对齐）
    lr_image_path = "data/train_sharp_bicubic/010/00000010.png"
    hr_image_path = "data/train_sharp/010/00000010.png"

    # 加载图像并转换为Tensor
    lr_tensor = to_tensor(Image.open(lr_image_path).convert('RGB')).unsqueeze(0).to(device)
    hr_tensor = to_tensor(Image.open(hr_image_path).convert('RGB')).unsqueeze(0).to(device)

    # 3. Padding（保证尺寸为16的倍数）
    lr_tensor_padded, h, w = pad_to_multiple(lr_tensor, multiple=16)
    
    # 4. 构造5帧序列 [1,5,3,H_padded,W_padded]
    lr_video = lr_tensor_padded.unsqueeze(1).repeat(1, 5, 1, 1, 1)

    # 5. 推理
    with torch.no_grad():
        output = model(lr_video)  # [1,3,4H,4W]
        # 裁剪回原始尺寸（去除padding）
        output = output[:, :, :h*4, :w*4]
        
        # 转换为numpy用于PSNR计算
        out_np = output.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        hr_np = hr_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        
        # 归一化到0-255
        out_np = (out_np.clip(0, 1) * 255.0).round().astype(np.uint8)
        hr_np = (hr_np.clip(0, 1) * 255.0).round().astype(np.uint8)

        # 边缘裁剪（去除边界效应）
        crop_size = 4
        out_c_np = out_np[crop_size:-crop_size, crop_size:-crop_size, :]
        hr_c_np = hr_np[crop_size:-crop_size, crop_size:-crop_size, :]
        
        # 计算PSNR
        psnr_val = calculate_psnr(out_c_np, hr_c_np)
        
        # 计算LPIPS（需要归一化到[-1,1]）
        out_lpips = (output[:, :, crop_size:-crop_size, crop_size:-crop_size] * 2.0 - 1.0)
        hr_lpips = (hr_tensor[:, :, crop_size:-crop_size, crop_size:-crop_size] * 2.0 - 1.0)
        lpips_val = lpips_vgg(out_lpips, hr_lpips).item()

    # 保存结果
    to_pil(output.squeeze(0).cpu().clamp(0, 1)).save("results_flowmatching/flowmatching_output_010.png")

    # 打印评估结果
    print("\n" + "="*30)
    print(f"📊 评估指标 (FlowMatching Model)")
    print(f"PSNR : {psnr_val:.2f} dB")
    print(f"LPIPS: {lpips_val:.4f}")
    print("="*30)

if __name__ == '__main__':
    test_flowmatching()