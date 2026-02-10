import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class DriveDataset(Dataset):
    def __init__(self, root_path, mode="train"):
        self.path = root_path
        self.mode = mode
        
        data_folder = "training" if mode == "train" else "test"
        
        self.img_list = sorted(glob.glob(os.path.join(self.path, data_folder, 'images/*.tif')))
        self.mask_list = sorted(glob.glob(os.path.join(self.path, data_folder, '1st_manual/*.gif')))

        # 数据切分
        if mode == "val":
            self.img_list = self.img_list[:5]
            self.mask_list = self.mask_list[:5]
        elif mode == "test":
            self.img_list = self.img_list[5:]
            self.mask_list = self.mask_list[5:]
        
        # 训练集重复数据以增加一个 Epoch 的迭代次数
        self.repeat = 50 if mode == "train" else 1

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
            # 【修复】验证集 Pad 到 32 的倍数，防止 UNet 拼接报错
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.img_list) * self.repeat

    def __getitem__(self, index):
        index = index % len(self.img_list)
        
        img = cv2.imread(self.img_list[index])
        # 【优化】使用绿色通道 (G channel) 替代简单的灰度转换
        # OpenCV 读入是 BGR，所以 G 通道是 index 1
        img = img[:, :, 1] 
        
        mask = np.array(Image.open(self.mask_list[index]))
        mask = (mask > 0).astype(np.float32) # 0, 1

        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        
        # 【修复】关键：手动增加 Channel 维度
        # Albumentations 对灰度图返回 [H, W]，我们需要 [1, H, W]
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask