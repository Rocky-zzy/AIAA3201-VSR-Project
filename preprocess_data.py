import os
import cv2
from tqdm import tqdm

def preprocess_reds_full():
    # 1. 设定路径 
    input_root = "data/train_sharp"  
    output_root = "data/train_sharp_bicubic"
    scale = 4

    if not os.path.exists(input_root):
        print(f"❌ 错误：找不到路径 {input_root}。")
        return

    # 获取所有视频序列文件夹 (000, 001, ..., 239)
    sub_folders = sorted([f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))])
    print(f"  {len(sub_folders)} 个视频序列，开始全量生成 LR 图像...")

    for sub in tqdm(sub_folders, desc="Overall Progress"):
        input_sub_dir = os.path.join(input_root, sub)
        output_sub_dir = os.path.join(output_root, sub)
        
        os.makedirs(output_sub_dir, exist_ok=True)

        img_list = sorted([f for f in os.listdir(input_sub_dir) if f.endswith('.png')])
        
        for img_name in img_list:
            img_path = os.path.join(input_sub_dir, img_name)
            save_path = os.path.join(output_sub_dir, img_name)

            # 如果文件已存在，跳过（防止中断后重跑浪费时间）
            if os.path.exists(save_path):
                continue

            img = cv2.imread(img_path)
            if img is None: continue

            h, w = img.shape[:2]
            # 缩放 4 倍
            lr_size = (w // scale, h // scale) 

            # 标准双三次插值
            img_lr = cv2.resize(img, lr_size, interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(save_path, img_lr)

    print(f"数据预处理完成！保存在: {output_root}")

if __name__ == "__main__":
    preprocess_reds_full()