import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import csv
import time
import datetime

# 从本地文件导入
from model import UNet 
from dataset import DriveDataset  # 👈 关键：从 dataset.py 导入类

# ==========================================
# Loss 函数定义
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

# ==========================================
# 训练主程序
# ==========================================
def train_model():
    # --- 配置参数 ---
    DATA_PATH = './data/DRIVE'
    BATCH_SIZE = 32       
    LEARNING_RATE = 2e-4  
    EPOCHS = 50
    PATIENCE = 5         # 早停耐心值
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 结果路径配置
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULT_DIR = os.path.join('./results', f'exp_{current_time}')
    CHECKPOINT_DIR = os.path.join(RESULT_DIR, 'checkpoints')
    LOG_CSV_PATH = os.path.join(RESULT_DIR, 'training_log.csv')
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, 'tensorboard_logs'), exist_ok=True)

    # --- 数据加载 ---
    train_ds = DriveDataset(root_path=DATA_PATH, mode="train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_ds = DriveDataset(root_path=DATA_PATH, mode="val")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # --- 模型、损失函数、优化器 ---
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    criterion_bce = nn.BCEWithLogitsLoss() 
    criterion_dice = DiceLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    writer = SummaryWriter(log_dir=os.path.join(RESULT_DIR, 'tensorboard_logs'))
    
    # 初始化 CSV 日志
    with open(LOG_CSV_PATH, mode='w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val Dice', 'LR', 'Time(s)'])

    print(f"🚀 开始训练... 设备: {DEVICE} | 训练集规模: {len(train_ds)}")
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for imgs, masks in train_bar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds_logits = model(imgs)
            loss_bce = criterion_bce(preds_logits, masks)
            loss_dice = criterion_dice(torch.sigmoid(preds_logits), masks)
            loss = 0.5 * loss_bce + 1.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss, val_dice_score = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds_logits = model(imgs)
                preds_probs = torch.sigmoid(preds_logits)
                
                v_loss = 0.5 * criterion_bce(preds_logits, masks) + 1.5 * criterion_dice(preds_probs, masks)
                val_loss += v_loss.item()
                val_dice_score += (1 - criterion_dice(preds_probs, masks).item())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice_score / len(val_loader)
        
        # 更新 LR
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - epoch_start
        
        # --- 日志记录与保存 ---
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} | LR: {current_lr:.1e}")
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Metric/Dice', avg_val_dice, epoch)

        with open(LOG_CSV_PATH, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, avg_val_loss, avg_val_dice, current_lr, epoch_duration])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"✅ 最佳模型已保存")
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print("🛑 触发早停")
                break

    writer.close()

if __name__ == "__main__":
    train_model()