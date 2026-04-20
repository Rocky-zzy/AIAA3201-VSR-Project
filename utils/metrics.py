import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

def bgr2ycbcr(img, only_y=True):
    # 如果输入已经是 uint8，先转 float 计算更准
    img_float = img.astype(np.float32)
    ycbcr = cv2.cvtColor(img_float, cv2.COLOR_BGR2YCrCb)
    if only_y:
        return ycbcr[:, :, 0] # 返回 Y 通道 (0-255 范围)
    return ycbcr

def crop_boundary(img, crop_border=4):
    """
    裁剪边界像素。
    超分模型在边缘处往往会有填充效应（Padding artifacts），
    学术界通常会根据放大倍数（如 scale=4）裁剪掉边缘部分再计算指标。
    """
    if crop_border == 0:
        return img
    return img[crop_border:-crop_border, crop_border:-crop_border]

def calculate_psnr(img1, img2, crop_border=0, input_order='HWC'):
    """
    计算 PSNR。
    img1: 模型生成的图 (H, W, C)
    img2: 真值图 (H, W, C)
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = crop_boundary(img1, crop_border)
        img2 = crop_boundary(img2, crop_border)

    return psnr_func(img1, img2, data_range=255)

def calculate_ssim(img1, img2, crop_border=0):
    img1 = crop_boundary(img1, crop_border)
    img2 = crop_boundary(img2, crop_border)
    
    # 自动判断维度
    if img1.ndim == 2: # 如果是单通道 (Y通道)
        return ssim_func(img1, img2, data_range=255)
    else: # 如果是 BGR 三通道
        return ssim_func(img1, img2, data_range=255, channel_axis=2)

    
    return ssim_func(img1, img2, data_range=255, channel_axis=2)