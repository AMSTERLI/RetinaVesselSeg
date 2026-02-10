import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
import csv
import time
import datetime
from model import UNet 

# ==========================================
# 1. 修复后的 Dataset 类
# ==========================================
class DriveDataset(Dataset):
    def __init__(self, root_path, mode="train"):
        self.path = root_path
        self.mode = mode
        
        data_folder = "training" if mode == "train" else "test"
        
        self.img_list = sorted(glob.glob(os.path.join(self.path, data_folder, 'images/*.tif')))
        self.mask_list = sorted(glob.glob(os.path.join(self.path, data_folder, '1st_manual/*.gif')))

        # 数据切分逻辑
        if mode == "val":
            self.img_list = self.img_list[:5]
            self.mask_list = self.mask_list[:5]
        elif mode == "test":
            self.img_list = self.img_list[5:]
            self.mask_list = self.mask_list[5:]
        # train 模式使用全部 training 文件夹数据

        # 设置重复倍数 (仅训练集)
        self.repeat = 50 if mode == "train" else 1

        # --- Transform 配置 ---
        if mode == "train":
            self.transform = A.Compose([
                A.RandomCrop(height=128, width=128),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])
        else:
            # 【重要修复】验证集不裁剪，但必须 Pad 到 32 的倍数
            # 否则 UNet 下采样 4 次后再上采样，尺寸会对应不上
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.img_list) * self.repeat

    def __getitem__(self, index):
        index = index % len(self.img_list)
        
        # 读取图片
        img = cv2.imread(self.img_list[index])
        
        # 【优化】提取绿色通道 (Green Channel)，血管对比度最高
        # OpenCV 是 BGR 格式，所以 G 通道是 index 1
        img = img[:, :, 1] 
        
        # 读取 Mask
        mask = np.array(Image.open(self.mask_list[index]))
        mask = (mask > 0).astype(np.float32) 

        # 应用变换
        augmented = self.transform(image=img, mask=mask)
        img = augmented['image'] # Tensor: [H, W] (因为是灰度输入)
        mask = augmented['mask'] # Tensor: [H, W]
        
        # 【重要修复】手动增加 Channel 维度 [H, W] -> [1, H, W]
        # 卷积层需要 (Batch, Channel, H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask

# ==========================================
# 2. Loss 函数
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs 必须是已经经过 Sigmoid 的概率值 (0-1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

# ==========================================
# 3. 训练主程序
# ==========================================
def train_model():
    # --- 配置参数 ---
    DATA_PATH = './data/DRIVE'
    BATCH_SIZE = 32       # 显存如果不够，改为 8 或 4
    LEARNING_RATE = 1e-3  # 【建议】降低学习率，1e-3 容易震荡
    EPOCHS = 50
    PATIENCE = 5         # 早停耐心值
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 结果目录
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULT_DIR = os.path.join('./results', f'exp_{current_time}')
    CHECKPOINT_DIR = os.path.join(RESULT_DIR, 'checkpoints')
    LOG_CSV_PATH = os.path.join(RESULT_DIR, 'training_log.csv')
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, 'tensorboard_logs'), exist_ok=True)

    # 初始化 CSV
    with open(LOG_CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val Dice', 'LR', 'Time(s)'])

    # --- 数据加载 ---
    train_ds = DriveDataset(root_path=DATA_PATH, mode="train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    val_ds = DriveDataset(root_path=DATA_PATH, mode="val")
    # 验证集 Batch Size 必须为 1 (因为图片没做 Crop，原始尺寸大)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # --- 模型与优化器 ---
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    
    # 【重要修复】使用 BCEWithLogitsLoss (自带 Sigmoid，更稳定)
    criterion_bce = nn.BCEWithLogitsLoss() 
    criterion_dice = DiceLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    writer = SummaryWriter(log_dir=os.path.join(RESULT_DIR, 'tensorboard_logs'))
    
    print(f"🚀 开始训练... 设备: {DEVICE} | 训练集数量: {len(train_ds)}")
    start_time = time.time()
    
    best_val_loss = float('inf')
    early_stop_counter = 0

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # >>>>>> 训练阶段 <<<<<<
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for imgs, masks in train_bar:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            # 1. 前向传播 (得到 Logits，未经过 Sigmoid)
            preds_logits = model(imgs)

            # 2. 计算 Loss
            # BCE 直接吃 Logits
            loss_bce = criterion_bce(preds_logits, masks)
            
            # Dice 需要吃概率 (0-1)，所以这里手动 Sigmoid
            preds_probs = torch.sigmoid(preds_logits)
            loss_dice = criterion_dice(preds_probs, masks)

            # 组合 Loss (你可以调整权重)
            loss = 0.5 * loss_bce + 1.5 * loss_dice

            # 3. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item(), bce=loss_bce.item(), dice=loss_dice.item())

        avg_train_loss = train_loss / len(train_loader)

        # >>>>>> 验证阶段 <<<<<<
        model.eval()
        val_loss = 0.0
        val_dice_score = 0.0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                
                # Forward
                preds_logits = model(imgs)
                preds_probs = torch.sigmoid(preds_logits)
                
                # Loss
                v_loss_bce = criterion_bce(preds_logits, masks)
                v_loss_dice = criterion_dice(preds_probs, masks)
                
                total_v_loss = 0.5 * v_loss_bce + 1.5 * v_loss_dice
                val_loss += total_v_loss.item()
                
                # 记录 Dice Score (1 - DiceLoss) 用于直观展示
                val_dice_score += (1 - v_loss_dice.item())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice_score / len(val_loader)
        
        # 更新学习率调度器
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_duration = time.time() - epoch_start
        
        # >>>>>> 日志与保存 <<<<<<
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {current_lr:.1e}")

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Metric/Dice', avg_val_dice, epoch)

        with open(LOG_CSV_PATH, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, avg_val_loss, avg_val_dice, current_lr, epoch_duration])

        # 早停与最佳模型保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"✅ 模型已保存 (Best Loss: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"⏳ Loss 未下降 ({early_stop_counter}/{PATIENCE})")
            
            if early_stop_counter >= PATIENCE:
                print("🛑 触发早停机制，训练结束。")
                break

    writer.close()
    print("Training Completed.")

if __name__ == "__main__":
    train_model()