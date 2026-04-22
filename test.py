import os
import warnings
import torch
import torch.nn.functional as F
import numpy as np 
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import lpips

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from models.advanced_vsr import AdvancedVSR
from utils.metrics import calculate_psnr
def pad_to_multiple(x, multiple=16):
    """确保图像尺寸是 16 的倍数，防止 DCN 等模块在边缘报错"""
    h, w = x.size(2), x.size(3)
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return x, h, w

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载 GAN 微调后的最新/最佳权重
    model_path = 'experiments_v3_gan/vsr_gan_G_best.pth'
    if not os.path.exists(model_path):
        model_path = 'experiments_v3_gan/vsr_gan_G_latest.pth'
    
    model = AdvancedVSR(num_blocks=16).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 初始化 LPIPS 评估器 (使用 alexnet 模式是学术界标配)
    lpips_vgg = lpips.LPIPS(net='alex').to(device)
    
    os.makedirs("results", exist_ok=True)
    to_tensor = ToTensor()
    to_pil = ToPILImage()

    # 读取测试图像 (请确保路径正确)
    lr_image_path = "data/train_sharp_bicubic/010/00000010.png"
    hr_image_path = "data/train_sharp/010/00000010.png"

    lr_tensor = to_tensor(Image.open(lr_image_path).convert('RGB')).unsqueeze(0).to(device)
    hr_tensor = to_tensor(Image.open(hr_image_path).convert('RGB')).unsqueeze(0).to(device)

    # 2.先在 4D 状态下进行 Padding
    _, _, h, w = lr_tensor.size()
    multiple = 16
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h > 0 or pad_w > 0:
        # 在 4D (B, C, H, W) 模式下，(0, pad_w, 0, pad_h) 是标准用法
        lr_tensor_padded = torch.nn.functional.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='replicate')
    else:
        lr_tensor_padded = lr_tensor

    # 3.将 Padding 后的单帧重复 5 次，构造 5D 序列 (1, 5, 3, H_padded, W_padded)
    lr_video = lr_tensor_padded.unsqueeze(1).repeat(1, 5, 1, 1, 1) 

    with torch.no_grad():
        output_video = model(lr_video)
        output = output_video[:, 0, :, :h*4, :w*4] if output_video.dim() == 5 else output_video[:, :, :h*4, :w*4]
        
        out_np = output.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        hr_np = hr_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        
        out_np = (out_np.clip(0, 1) * 255.0).round().astype(np.uint8)
        hr_np = (hr_np.clip(0, 1) * 255.0).round().astype(np.uint8)

        crop_size = 4
        out_c_np = out_np[crop_size:-crop_size, crop_size:-crop_size, :]
        hr_c_np = hr_np[crop_size:-crop_size, crop_size:-crop_size, :]
        
        # 调用 calculate_psnr 
        psnr_val = calculate_psnr(out_c_np, hr_c_np)
        
        # LPIPS 需要 Tensor 模式
        out_lpips = (output[:, :, crop_size:-crop_size, crop_size:-crop_size] * 2.0 - 1.0)
        hr_lpips = (hr_tensor[:, :, crop_size:-crop_size, crop_size:-crop_size] * 2.0 - 1.0)
        lpips_val = lpips_vgg(out_lpips, hr_lpips).item()

    # 保存结果
    to_pil(output.squeeze(0).cpu().clamp(0, 1)).save("results/gan_output_010.png")

    print("\n" + "="*30)
    print(f"📊 评估指标 (GAN Model)")
    print(f"PSNR : {psnr_val:.2f} dB")
    print(f"LPIPS: {lpips_val:.4f}")
    print("="*30)

if __name__ == '__main__':
    test()