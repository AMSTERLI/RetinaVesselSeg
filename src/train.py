import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # 显示进度条
import os
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import DriveDataset

# Dice Loss 
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # 将结果展平为一维
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

# --- 2. 训练主配置 ---
def train_model():
    # 超参数设置
    DATA_PATH = './data/DRIVE'
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_DIR = './checkpoints'

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # --- 3. 初始化数据、模型、优化器 ---
    train_ds = DriveDataset(root_path=DATA_PATH, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet(n_channels=1, n_classes=1).to(DEVICE) # 输入是绿色单通道
    
    # 组合损失函数: BCE 稳定 + Dice 针对细小血管
    criterion_bce = nn.BCELoss()
    criterion_dice = DiceLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 初始化 Writer ---
    # log_dir 是日志保存路径，建议按时间或实验名区分，例如 'runs/exp1_unet'
    writer = SummaryWriter(log_dir='runs/experiment_1')
    step = 0

    print(f"开始在 {DEVICE} 上训练...")

    # --- 4. 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # 使用 tqdm 包装 loader，产生酷炫进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, masks in progress_bar:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            # 前向传播
            preds = model(imgs)
            
            # 计算混合损失
            loss_bce = criterion_bce(preds, masks)
            loss_dice = criterion_dice(preds, masks)
            loss = loss_bce + loss_dice 

            # 反向传播
            optimizer.zero_grad() # 梯度清零
            loss.backward()       # 计算梯度
            optimizer.step()      # 更新参数

            # ---【修改 3】记录 Loss 到 TensorBoard ---
            # 每 10 个 batch 记录一次，避免日志文件过大
            if step % 10 == 0:
                # add_scalar('图表标题', Y轴数值, X轴数值)
                writer.add_scalar('Training/Loss', loss.item(), step)
            
            step += 1
            progress_bar.set_postfix(loss=loss.item())

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # 每个 Epoch 结束，打印平均 Loss 并保存模型
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/unet_epoch_{epoch+1}.pth")
    # 训练结束关闭 writer
    writer.close()

if __name__ == "__main__":
    train_model()
