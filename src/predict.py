import os
import torch
import cv2
import numpy as np
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, jaccard_score

# 导入你之前的模块
from dataset import DriveDataset
from model import UNet

# ==========================================
# 1. 核心指标计算函数
# ==========================================
def calculate_metrics(y_true, y_prob):
    # 展平并转为整型标签
    y_true = (y_true > 0.5).astype(np.uint8).ravel()
    y_prob = y_prob.ravel()
    y_pred = (y_prob > 0.5).astype(np.uint8)

    # 计算各项指标
    roc_auc = roc_auc_score(y_true, y_prob)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    f1 = f1_score(y_true, y_pred, zero_division=1)
    iou = jaccard_score(y_true, y_pred, zero_division=1)
    
    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "F1-Score": f1,
        "IoU": iou
    }

# ==========================================
# 2. 结果保存函数 (原图|标签|预测)
# ==========================================
def save_result_combined(image, mask, pred, save_path):
    # image: [1, H, W], mask/pred: [1, H, W]
    img_np = (image[0].cpu().numpy() * 0.5 + 0.5) * 255
    mask_np = (mask[0].cpu().numpy() * 255)
    pred_np = (pred[0].cpu().numpy() > 0.5) * 255
    
    combined = np.hstack([img_np, mask_np, pred_np]).astype(np.uint8)
    cv2.imwrite(save_path, combined)

# ==========================================
# 3. 主预测程序
# ==========================================
def run_predict():
    # --- 配置区 ---
    DATA_PATH = './data/DRIVE'
    MODEL_WEIGHTS = './results/best_model.pth' # 修改为你的实际路径
    SAVE_DIR = './predict_results'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 加载测试集 (mode="test")
    test_loader = DataLoader(DriveDataset(DATA_PATH, mode="test"), batch_size=1, shuffle=False)
    
    # 加载模型
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()

    all_metrics = []
    metric_names = ["ROC-AUC", "PR-AUC", "F1-Score", "IoU"]

    print(f"🚀 开始测试... 设备: {DEVICE}")

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(test_loader)):
            imgs = imgs.to(DEVICE)
            
            # 推理
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            # 计算指标
            m = calculate_metrics(masks.cpu().numpy(), probs.cpu().numpy())
            all_metrics.append(list(m.values()))

            # 保存拼接图
            save_path = os.path.join(SAVE_DIR, f"test_result_{i+1:02d}.png")
            save_result_combined(imgs[0], masks[0], probs[0], save_path)

    # --- 保存 CSV 报告 ---
    avg_metrics = np.mean(all_metrics, axis=0)
    report_path = os.path.join(SAVE_DIR, "metrics_report.csv")
    
    with open(report_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image_ID"] + metric_names)
        for idx, row in enumerate(all_metrics):
            writer.writerow([f"Test_{idx+1}"] + [f"{v:.4f}" for v in row])
        writer.writerow([])
        writer.writerow(["AVERAGE"] + [f"{v:.4f}" for v in avg_metrics])

    print(f"\n✅ 评估完成！")
    print(f"平均 F1-Score: {avg_metrics[2]:.4f}")
    print(f"结果文件已保存至: {SAVE_DIR}")

if __name__ == "__main__":
    run_predict()