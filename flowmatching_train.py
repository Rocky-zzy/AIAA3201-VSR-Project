import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.reds_dataset import REDSDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 导入FlowMatching模型
from part3_flow_matching_vsr import Part3_FlowMatchingVSR

plt.switch_backend('Agg')

def plot_loss_curve(loss_list, save_path='experiments_flowmatching/loss_curve.png'):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Training Loss (L1)', color='#2c3e50', linewidth=2)
    plt.title('FlowMatching VSR Training Loss Curve', fontsize=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def charbonnier_loss(pred, target, eps=1e-6):
    """Charbonnier损失（鲁棒的L1损失）"""
    return torch.sqrt((pred - target) ** 2 + eps).mean()

def train_flowmatching():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100
    batch_size = 32
    learning_rate = 5e-5
    save_dir = 'experiments_flowmatching'  # 独立的实验文件夹
    os.makedirs(save_dir, exist_ok=True)

    # 数据集加载（与原训练脚本路径对齐）
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

    # 初始化FlowMatching模型
    model = Part3_FlowMatchingVSR(device=device).to(device)
    
    # 损失函数与优化器
    criterion = charbonnier_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7
    )

    # 训练记录
    epoch_losses = []
    best_loss = float('inf')

    print(f"开始在 {device} 上训练 FlowMatching VSR 模型")
    print(f"配置: BatchSize={batch_size}, patch_size=128, Epochs={num_epochs}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            lr = batch['lr'].to(device, non_blocking=True)  # [B,5,3,H,W]
            hr = batch['hr'].to(device, non_blocking=True)  # [B,3,4H,4W]

            # 前向传播
            optimizer.zero_grad()
            sr = model(lr)  # 模型输出 [B,3,4H,4W]
            
            # 计算损失
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 打印中间结果
            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 每个epoch后更新
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        scheduler.step()

        print(f"✅ Epoch {epoch+1} 完成 | 平均损失: {avg_loss:.6f} | 当前LR: {scheduler.get_last_lr()[0]:.2e}")

        # 保存最新模型
        torch.save(model.state_dict(), f'{save_dir}/flowmatching_latest.pth')
        
        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{save_dir}/flowmatching_best.pth')
            print(f"🌟 保存最优模型 (Loss: {best_loss:.6f})")

        # 每5个epoch绘制损失曲线
        if (epoch + 1) % 5 == 0:
            plot_loss_curve(epoch_losses, save_path=f'{save_dir}/loss_curve.png')

    print("🎉 FlowMatching VSR 训练完成！")
    print(f"最优损失: {best_loss:.6f} | 模型保存路径: {save_dir}")

if __name__ == "__main__":
    train_flowmatching()