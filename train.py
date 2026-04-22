import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.reds_dataset import REDSDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F

plt.switch_backend('Agg')

def plot_loss_curve(loss_list, save_path='experiments_v2/loss_curve.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Training Loss (L1)', color='#2c3e50', linewidth=2)
    plt.title('VSR Training Loss Curve', fontsize=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100         
    batch_size = 32       
    learning_rate = 5e-5
    # 【关键1】新建 GAN 实验文件夹，绝对不能覆盖之前 v2 的纯净权重
    save_dir = 'experiments_v3_gan' 

    epoch_losses_G = []
    best_loss_G = float('inf')

    train_dataset = REDSDataset(
        lr_path="data/train_sharp_bicubic",
        hr_path="data/train_sharp",
        patch_size=128,      
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True
    )

    from models.advanced_vsr import AdvancedVSR
    from models.discriminator import VGGStyleDiscriminator
    from utils.losses import PerceptualLoss

    # 1. 初始化生成器 (Generator) 并进行【热启动 Warm-start】
    model_G = AdvancedVSR(num_blocks=16).to(device)
    # 加载你之前跑到 25dB 的权重作为起点，这对于 GAN 的稳定收敛极其重要！
    if os.path.exists('experiments_v2/vsr_advanced_best.pth'):
        model_G.load_state_dict(torch.load('experiments_v2/vsr_advanced_best.pth'))
        print("🚀 成功加载 v2 基础权重，启动 GAN 微调！")
    
    # 2. 初始化判别器 (Discriminator)
    model_D = VGGStyleDiscriminator().to(device)
    
    # 3. 定义损失函数体系
    def charbonnier_loss(pred, target, eps=1e-6):
        return torch.sqrt((pred - target) ** 2 + eps).mean()
    
    criterion_pixel = lambda x, y: charbonnier_loss(x, y) # 像素基础损失
    criterion_vgg = PerceptualLoss().to(device)           # 提取高频纹理特征
    criterion_gan = nn.BCEWithLogitsLoss().to(device)     # 判别真假
    
    # 损失权重分配
    lambda_pixel = 1.0
    lambda_vgg = 1.0
    lambda_gan = 0.1 # GAN 损失占比 10% 即可，避免画面彻底崩坏

    # 4. 定义双优化器与双调度器
    optimizer_G = optim.Adam(model_G.parameters(), lr=1e-5) # GAN 微调阶段，基础 LR 调低
    optimizer_D = optim.Adam(model_D.parameters(), lr=1e-5)
    
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs, eta_min=1e-7)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs, eta_min=1e-7)

    print(f"开始在 {device} 上进行对抗感知(GAN)训练。")
    print(f"当前配置: Blocks=16, BatchSize={batch_size}, patch_size=128")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model_G.train()
        model_D.train()
        
        running_loss_G = 0.0
        running_loss_D = 0.0

        for i, batch in enumerate(train_loader):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)

            #  阶段一：训练判别器 (Discriminator)
            optimizer_D.zero_grad()
            
            # 生成假图（此时不更新 Generator 的梯度）
            with torch.no_grad():
                fake_hr = model_G(lr)
                
            # 判别器看真图，期望输出 1
            real_preds = model_D(hr)
            loss_D_real = criterion_gan(real_preds, torch.ones_like(real_preds))
            
            # 判别器看假图，期望输出 0
            fake_preds = model_D(fake_hr.detach())
            loss_D_fake = criterion_gan(fake_preds, torch.zeros_like(fake_preds))
            
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()
            
            running_loss_D += loss_D.item()

            #  阶段二：训练生成器 (Generator)
            optimizer_G.zero_grad()
            
            # 重新生成假图放入计算图
            fake_hr = model_G(lr)
            
            # 1. 像素损失 (维持大局结构)
            loss_pixel = criterion_pixel(fake_hr, hr)
            
            # 2. 感知损失 (VGG 高频纹理逼近)
            loss_perceptual = criterion_vgg(fake_hr, hr)
            
            # 3. 对抗损失 (欺骗判别器，期望输出 1)
            fake_preds_for_G = model_D(fake_hr)
            loss_adv = criterion_gan(fake_preds_for_G, torch.ones_like(fake_preds_for_G))
            
            # 组合损失
            loss_G = (lambda_pixel * loss_pixel) + (lambda_vgg * loss_perceptual) + (lambda_gan * loss_adv)
            
            loss_G.backward()
            optimizer_G.step()

            running_loss_G += loss_G.item()

            if i % 20 == 0:
                ch_mean = fake_hr.mean(dim=[0, 2, 3])
                hr_ch_mean = hr.mean(dim=[0, 2, 3])
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}]")
                print(f"  Loss_G: {loss_G.item():.4f} (Pix:{loss_pixel.item():.4f}, VGG:{loss_perceptual.item():.4f}, GAN:{loss_adv.item():.4f}) | Loss_D: {loss_D.item():.4f}")
                print(f"  Output channel means (B,G,R): {ch_mean[0]:.3f}, {ch_mean[1]:.3f}, {ch_mean[2]:.3f}")
                print(f"  HR channel means (B,G,R):     {hr_ch_mean[0]:.3f}, {hr_ch_mean[1]:.3f}, {hr_ch_mean[2]:.3f}")

        scheduler_G.step()
        scheduler_D.step()
        
        avg_loss_G = running_loss_G / len(train_loader)
        epoch_losses_G.append(avg_loss_G)

        print(f"✅ Epoch {epoch+1} 完成, G平均Loss: {avg_loss_G:.6f}, 当前 LR: {scheduler_G.get_last_lr()[0]:.2e}")

        # 保存最新的 G 和 D 权重
        torch.save(model_G.state_dict(), f'{save_dir}/vsr_gan_G_latest.pth')
        torch.save(model_D.state_dict(), f'{save_dir}/vsr_gan_D_latest.pth')
        
        # 只根据 G 的损失来保存 best 模型
        if avg_loss_G < best_loss_G:
            best_loss_G = avg_loss_G
            torch.save(model_G.state_dict(), f'{save_dir}/vsr_gan_G_best.pth')
            print(f" 🌟 发现更低 G_Loss，已保存 best 生成器模型！")

        if (epoch + 1) % 5 == 0:
            plot_loss_curve(epoch_losses_G, save_path=f'{save_dir}/loss_curve_G.png')

    print("🎉 对抗感知训练结束！") 

if __name__ == "__main__":
    train()