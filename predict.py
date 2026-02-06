import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import UNet  # 导入你写的模型类
import os

def preprocess_image(img_path):
    """
    预处理函数：必须与 Dataset 里的逻辑严格一致
    """
    # 1. 读取并提取绿色通道
    img = cv2.imread(img_path)
    green_ch = img[:, :, 1]
    
    # 2. CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(green_ch)
    
    # 3. 归一化
    img_normalized = img_enhanced.astype(np.float32) / 255.0
    
    # 4. 转为 Tensor 并增加维度 [B, C, H, W]
    # H, W -> 1, H, W -> 1, 1, H, W
    input_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
    return input_tensor, img

def predict():
    # --- 配置参数 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './checkpoints/unet_epoch_50.pth'  # 你训练好的模型路径
    test_img_path = './data/DRIVE/test/images/01_test.tif' # 挑一张测试图
    threshold = 0.5 # 阈值：概率大于 0.5 判定为血管
    
    # 1. 初始化模型并加载权重
    model = UNet(n_channels=1, n_classes=1).to(device)
    # map_location 确保在没有 GPU 的电脑上也能加载 GPU 训练的模型
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 切换到评估模式（关闭 Dropout 和 Batch Norm 的训练行为）

    # 2. 预处理图片
    input_tensor, original_img = preprocess_image(test_img_path)
    input_tensor = input_tensor.to(device)

    # 3. 推理 (Inference)
    with torch.no_grad(): # 推理阶段不需要计算梯度
        output = model(input_tensor)
        # 将输出转回 CPU 并从 Tensor 转为 Numpy
        pred_mask = output.squeeze().cpu().numpy()
        
    # 4. 二值化处理
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255

    # --- 5. 结果可视化 ---
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Probability Map")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Binary Segmentation")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    # 也可以保存结果
    # cv2.imwrite('result_01.png', binary_mask)

if __name__ == "__main__":
    predict()