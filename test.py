import os
import cv2
import torch
import numpy as np
from models.advanced_vsr import AdvancedVSR
from utils.metrics import calculate_psnr, calculate_ssim, bgr2ycbcr

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 你可以根据需要切换 best.pth 或 latest.pth
    weight_path = "experiments_v2/vsr_advanced_best.pth"
    
    print(f"🚀 加载模型权重: {weight_path}")
    if not os.path.exists(weight_path):
        print(f"❌ 权重文件不存在！")
        return

    # 初始化模型
    model = AdvancedVSR(num_blocks=16).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    # 测试目标：第 010 序列的第 10 帧
    seq_folder = "010"
    frame_idx = 10
    lr_dir = os.path.join("data/train_sharp_bicubic", seq_folder)
    hr_dir = os.path.join("data/train_sharp", seq_folder)

    # 1. 读取 5 帧 LR 作为输入 (中间帧是 frame_idx)
    lr_frames = []
    for i in range(frame_idx - 2, frame_idx + 3):
        img_path = os.path.join(lr_dir, f"{i:08d}.png")
        img = cv2.imread(img_path).astype(np.float32) / 255.
        lr_frames.append(img)
    
    # 转换为 Tensor (B, N, C, H, W)
    input_tensor = torch.from_numpy(np.stack(lr_frames, axis=0)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    # 2. 模型推理
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # 打印通道均值，用于诊断亮度偏差
    ch_mean = output_tensor.mean(dim=[0, 2, 3])
    print(f"输出 clamp 前通道均值 (B,G,R): {ch_mean[0]:.3f}, {ch_mean[1]:.3f}, {ch_mean[2]:.3f}")

    # 3. 后处理保存图像
    output_img = output_tensor.squeeze(0).cpu().numpy()
    output_img = np.transpose(output_img, (1, 2, 0))
    output_img = (output_img * 255.0).round().clip(0, 255).astype(np.uint8)

    # 4. 读取 HR 图像进行对比
    hr_img = cv2.imread(os.path.join(hr_dir, f"{frame_idx:08d}.png"))
    if hr_img is None:
        raise FileNotFoundError(f"HR 图像未找到: {hr_dir}")

    # 5. 核心指标计算
    output_y = bgr2ycbcr(output_img, only_y=True)
    hr_y = bgr2ycbcr(hr_img, only_y=True)
    
    hr_float = hr_y.astype(np.float32)
    out_float = output_y.astype(np.float32)

    # 诊断 1：确认真值均值
    print(f"🔍 真实 HR (Y通道) 均值: {hr_float.mean():.3f}")
    print(f"🔍 模型输出 (Y通道) 均值: {out_float.mean():.3f}")

    # 诊断 2：正常裁剪 vs 极度裁剪 (绕开边缘爆炸区)
    psnr_crop4 = calculate_psnr(output_y, hr_y, crop_border=4)
    psnr_center = calculate_psnr(output_y, hr_y, crop_border=128) # 直接切掉四周 128 圈像素，只看最核心！

    # 诊断 3：生成老旧的 Bicubic 作为底线基准
    lr_center_img = cv2.imread(os.path.join(lr_dir, f"{frame_idx:08d}.png"))
    bicubic_img = cv2.resize(lr_center_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    bicubic_y = bgr2ycbcr(bicubic_img, only_y=True)
    psnr_bicubic = calculate_psnr(bicubic_y, hr_y, crop_border=4)

    print("-" * 40)
    print(f"--- 核心诊断结果 ---")
    print(f"📍 你的底线 (Bicubic Baseline): {psnr_bicubic:.2f} dB")
    print(f"📍 模型全图 PSNR (Crop 4)   : {psnr_crop4:.2f} dB")
    print(f"📍 模型核心区 PSNR (Crop 128) : {psnr_center:.2f} dB")
    print("-" * 40)

    # 保存结果
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"test_result_{seq_folder}_{frame_idx}.png")
    cv2.imwrite(save_path, hr_img)
    cv2.imwrite(save_path, output_img)
    print(f"✅ 图像已保存至 {save_dir} 文件夹")

if __name__ == "__main__":
    test()