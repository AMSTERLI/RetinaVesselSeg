import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A  # 数据增强库，支持同时增强图像和掩码
from albumentations.pytorch import ToTensorV2

class DriveDataset(Dataset):
    def __init__(self, root_path, is_train=True):
        self.mode = "training" if is_train else "test"
        # 获取所有图像和对应掩码的路径
        self.img_list = sorted(glob.glob(os.path.join(self.path, 'images/*.tif')))
        self.mask_list = sorted(glob.glob(os.path.join(self.path, '1st_manual/*.gif')))

        # --- 定义数据增强流水线 ---
        if is_train:
            self.transform = A.Compose([
                # 1. 随机裁剪: 这一步直接解决了数据量少和显存不足的问题
                # 每次随机切出一个 128x128 的区域用于训练
                A.RandomCrop(height=128, width=128),
                
                # 2. 几何变换: 翻转和旋转
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # 3. 弹性形变: 模拟生物组织扭曲 (关键!)
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                
                # 4. 像素增强: 模拟不同相机的光照和噪点
                A.RandomGamma(p=0.2),
                A.GaussNoise(p=0.2),
                
                # 5. 归一化并转为 Tensor
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])
        else:
            # 测试集不做增强，只做必要的归一化
            self.transform = A.Compose([
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])

    def __getitem__(self, index):
        # 读取图片 (OpenCV 读取的是 HWC 格式)
        img = cv2.imread(self.img_list[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转灰度或提取绿色通道
        
        # 读取掩码
        from PIL import Image
        mask = np.array(Image.open(self.mask_list[index]))
        mask = (mask > 0).astype(np.float32) # 转为 0/1

        # --- 应用增强 ---
        # Albumentations 会自动同时处理 image 和 mask
        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        
        # 此时 mask 的维度可能是 [H, W]，需要增加一个维度变成 [1, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask