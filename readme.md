# RetinaVesselSeg: 基于 U-Net 的眼底血管分割

本项目实现了一个基于 **U-Net** 架构的眼底图像血管分割系统。通过对 DRIVE 数据集进行绿色通道提取、CLAHE 增强以及混合损失函数（BCE + Dice）训练，实现了对视网膜血管的高精度自动化分割。

---

## 🚀 项目特性

* **模型架构**: 标准 U-Net，支持灵活的通道配置。
* **预处理优化**: 提取血管对比度最高的 **绿色通道 (Green Channel)**，并应用 **CLAHE**（限制对比度自适应直方图均衡化）。
* **混合损失函数**: 结合 `BCEWithLogitsLoss` 和 `DiceLoss`，有效应对极端的类别不平衡（血管仅占图像极小比例）。
* **专业评估**: 自动计算并输出 **ROC-AUC, PR-AUC, F1-Score (Dice)** 以及 **IoU** 等医学影像分割关键指标。

---

## 🛠️ 环境配置

1.  **创建虚拟环境**:
    ```bash
    conda create -n retina python=3.9
    conda activate retina
    ```

2.  **安装依赖**:
    ```bash
    pip install torch torchvision
    pip install opencv-python albumentations tqdm scikit-learn numpy
    ```

---

## 📂 项目结构

```text
├── data/               # 存放 DRIVE 数据集 (包含 training 和 test 文件夹)
├── src/                # 源代码
│   ├── dataset.py      # 数据读取、增强与预处理逻辑
│   ├── model.py        # U-Net 网络结构定义
│   ├── train.py        # 模型训练脚本
│   └── predict.py      # 测试集推理与专业指标评估
├── results/            # 训练日志、TensorBoard 记录及模型权重
└── predict_results/    # 预测生成的对比图及 CSV 指标报告
```


## 📈 实验结果


### 专业指标评估 (DRIVE 测试集)
经过 50 Epoch 训练，模型在 15 张测试集图片上的表现非常稳定：

| 指标 | 平均得分 (AVERAGE) | 备注 |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9828** | 整体分类排序能力极高 |
| **PR-AUC** | **0.9069** | 对细小血管捕捉能力的核心指标 |
| **F1-Score (Dice)** | **0.8211** | 预测结果与专家标注的重叠度 |
| **IoU** | **0.6967** | 交并比表现，反映分割边界准确性 |

### 结果可视化

![左侧为原始图像（绿色通道增强后），中间为专家手动标注结果（Ground Truth），右侧为本模型 U-Net 的预测结果。可见模型能够精准还原复杂的血管网络拓扑结构，包括末梢细小分支。](./predict_results/result_01.png)

---

## 📖 使用指南

1. **训练**:
   运行 `python src/train.py`。训练完成后，最佳权重将保存至 `results/exp_xxx/checkpoints/best_model.pth`。

2. **预测**:
   修改 `predict.py` 中的 `MODEL_WEIGHTS` 路径，运行 `python src/predict.py`。

3. **查看报告**:
   在 `predict_results/metrics_report.csv` 中可查看每张测试图的详细评估数据。

---

## 📝 参考

* **数据集**: [DRIVE: Digital Retinal Images for Vessel Extraction](https://drive.grand-challenge.org/)
* **算法架构**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation."