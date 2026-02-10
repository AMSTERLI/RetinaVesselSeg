import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import csv
import time
import datetime
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import DriveDataset

# ... DiceLoss 保持不变 ...
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

def train_model():
    # --- 1. 基础配置 ---
    DATA_PATH = './data/DRIVE'
    BATCH_SIZE = 32     # 【建议】显存够的话改为 32 或 64
    LEARNING_RATE = 1e-3
    EPOCHS = 100        # 【建议】设置大一点，反正有早停机制会帮我们停
    PATIENCE = 5       # 【新增】早停耐心值：如果验证集 Loss 连续 10 轮不下降，就停止
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULT_DIR = os.path.join('./results', f'exp_{current_time}')
    CHECKPOINT_DIR = os.path.join(RESULT_DIR, 'checkpoints')
    LOG_CSV_PATH = os.path.join(RESULT_DIR, 'training_log.csv')
    REPORT_PATH = os.path.join(RESULT_DIR, 'final_report.txt')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    with open(LOG_CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 修改表头，增加验证集数据
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val Dice', 'LR', 'Time(s)'])

    # --- 2. 数据加载 ---
    # 训练集：使用 training 文件夹全部数据
    train_ds = DriveDataset(root_path=DATA_PATH, mode="train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 验证集：使用 test 文件夹中的前 5 张
    val_ds = DriveDataset(root_path=DATA_PATH, mode="val")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    
    criterion_bce = nn.BCELoss()
    criterion_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir=os.path.join(RESULT_DIR, 'tensorboard_logs'))
    
    print(f"🚀 开始训练... 设备: {DEVICE} | Batch: {BATCH_SIZE} | Patience: {PATIENCE}")
    start_time = time.time()

    # --- 早停相关变量 ---
    best_val_loss = float('inf')
    early_stop_counter = 0  # 计数器

    # --- 3. 训练循环 ---
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # =========== 训练阶段 ===========
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for imgs, masks in train_bar:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            preds = torch.sigmoid(preds) # 确保输出是 0-1

            loss = 0.5*criterion_bce(preds, masks) + 1.5*criterion_dice(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # =========== 验证阶段 (新增) ===========
        model.eval() # 切换到评估模式 (关闭 Dropout 等)
        val_loss = 0.0
        val_dice_score = 0.0 # 记录纯 Dice 分数用于观察
        
        # 验证时不计算梯度，节省显存
        with torch.no_grad():
            # 这里不用 tqdm 也可以，避免进度条刷屏
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                
                preds = model(imgs)
                preds = torch.sigmoid(preds)

                # 计算验证 Loss
                v_loss = criterion_bce(preds, masks) + criterion_dice(preds, masks)
                val_loss += v_loss.item()
                
                # 计算纯 Dice 系数 (1 - DiceLoss) 用于人类观察
                # 注意：DiceLoss 返回的是 1-Dice，所以我们反推一下
                d_loss = criterion_dice(preds, masks)
                val_dice_score += (1 - d_loss.item())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice_score / len(val_loader)
        
        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # --- 记录与打印 ---
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        # TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Metric/Val_Dice', avg_val_dice, epoch)

        # CSV
        with open(LOG_CSV_PATH, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch+1, f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}", f"{avg_val_dice:.4f}", current_lr, f"{epoch_duration:.2f}"])

        # =========== 早停机制核心逻辑 ===========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0 # 重置计数器
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"✅ 验证集 Loss 降低，模型已保存！(Patience: 0/{PATIENCE})")
        else:
            early_stop_counter += 1
            print(f"⚠️ 验证集 Loss 未降低，计数器: {early_stop_counter}/{PATIENCE}")
            
            if early_stop_counter >= PATIENCE:
                print(f"🛑 触发早停机制！训练在 Epoch {epoch+1} 停止。")
                break # 跳出 Epoch 循环

    # --- 4. 结束报告 ---
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    
    final_report = f"""
    Training Finished.
    Total Time: {total_time}
    Best Val Loss: {best_val_loss:.4f}
    Stopped at Epoch: {epoch+1}
    """
    with open(REPORT_PATH, "w") as f:
        f.write(final_report)
    print(final_report)
    writer.close()

if __name__ == "__main__":
    train_model()