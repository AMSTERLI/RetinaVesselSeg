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
        """
        Args:
            root_path: 数据根目录 (./data/DRIVE)
            mode: "train", "val", 或 "test"
        """
        self.path = root_path
        self.mode = mode
        
        # 1. 确定文件夹路径
        # 注意：val 和 test 都读取 test 文件夹
        data_folder = "training" if mode == "train" else "test"
        
        # 获取该文件夹下所有的图片和掩码
        all_imgs = sorted(glob.glob(os.path.join(self.path, data_folder, 'images/*.tif')))
        all_masks = sorted(glob.glob(os.path.join(self.path, data_folder, '1st_manual/*.gif')))

        # 2. 实现数据切分逻辑 (如果是 test 文件夹，取前 5 个作为验证集)
        if mode == "val":
            self.img_list = all_imgs[:5]   # 前 5 个
            self.mask_list = all_masks[:5]
        elif mode == "test":
            self.img_list = all_imgs[5:]   # 5 个以后的全部作为测试
            self.mask_list = all_masks[5:]
        else: # train
            self.img_list = all_imgs
            self.mask_list = all_masks

        # 3. 设置数据重复倍数 (仅训练集需要)
        self.repeat = 50 if mode == "train" else 1

        # 4. 设置变换 (仅训练集需要数据增强)
        if mode == "train":
            self.transform = A.Compose([
                A.RandomCrop(height=128, width=128),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.5), 
                A.RandomGamma(p=0.2),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])
        else:
            # val 和 test 模式只做归一化，不裁剪，不做增强
            self.transform = A.Compose([
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.img_list) * self.repeat

    def __getitem__(self, index):
        index = index % len(self.img_list)
        
        img = cv2.imread(self.img_list[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
        mask = np.array(Image.open(self.mask_list[index]))
        mask = (mask > 0).astype(np.float32) 

        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask