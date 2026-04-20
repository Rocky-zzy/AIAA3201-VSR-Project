import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.reds_dataset import REDSDataset
import matplotlib.pyplot as plt

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
    num_epochs = 100         # 增加到 100 epoch，给足收敛时间
    batch_size = 64          
    learning_rate = 5e-5
    save_dir = 'experiments_v2' # 【关键】新实验文件夹，隔离旧权重

    epoch_losses = []
    best_loss = float('inf')

    train_dataset = REDSDataset(
        lr_path="data/train_sharp_bicubic",
        hr_path="data/train_sharp",
        patch_size=128,      
        is_train=True
    )
    # 使用 12 个 worker，适配最佳系统吞吐量
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True
    )

    from models.advanced_vsr import AdvancedVSR
    # 【关键】网络加深到 16 层，增强特征提取能力
    model = AdvancedVSR(num_blocks=16).to(device)
    
    def charbonnier_loss(pred, target, eps=1e-6):
        return torch.sqrt((pred - target) ** 2 + eps).mean()
    criterion = lambda x, y: charbonnier_loss(x, y)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    print(f"开始在 {device} 上进行全局残差训练。")
    print(f"当前配置: Blocks=16, BatchSize={batch_size}, patch_size=128")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)

            optimizer.zero_grad()

            output = model(lr)
            loss = criterion(output, hr)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 20 == 0:
                ch_mean = output.mean(dim=[0, 2, 3])
                hr_ch_mean = hr.mean(dim=[0, 2, 3])
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
                print(f"  Output channel means (B,G,R): {ch_mean[0]:.3f}, {ch_mean[1]:.3f}, {ch_mean[2]:.3f}")
                print(f"  HR channel means (B,G,R):     {hr_ch_mean[0]:.3f}, {hr_ch_mean[1]:.3f}, {hr_ch_mean[2]:.3f}")

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        print(f"✅ Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.6f}, 当前 LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  LR input range: [{lr.min():.3f}, {lr.max():.3f}], mean={lr.mean():.3f}")
        print(f"  HR target range: [{hr.min():.3f}, {hr.max():.3f}], mean={hr.mean():.3f}")
        print(f"  Model output range: [{output.min():.3f}, {output.max():.3f}], mean={output.mean():.3f}")

        # 保存到新目录
        torch.save(model.state_dict(), f'{save_dir}/vsr_advanced_latest.pth')
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{save_dir}/vsr_advanced_best.pth')
            print(f" 🌟 发现更低 Loss，已保存 best 模型！")

        if (epoch + 1) % 5 == 0:
            plot_loss_curve(epoch_losses, save_path=f'{save_dir}/loss_curve.png')

    print("🎉 训练结束！")

if __name__ == "__main__":
    train()