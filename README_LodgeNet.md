# LodgeNet：玉米锈病识别与预测深度学习模型

## 项目概述

LodgeNet是专门为玉米南方锈病识别与预测任务设计的深度学习模型。该模型基于U-Net架构，集成了注意力机制和多尺度特征提取，实现了三个核心任务的联合学习：

1. **图像分割**：精确识别感染区域的像素级位置
2. **位置分类**：预测感染部位（下部/中部/上部）
3. **病害等级回归**：预测感染严重程度（0-9连续值）

## 模型架构特点

### 核心组件

- **编码器-解码器结构**：基于U-Net的对称架构，有效保留空间信息
- **注意力门控机制**：在跳跃连接中应用注意力，突出重要特征
- **ASPP模块**：空洞空间金字塔池化，捕获多尺度上下文信息
- **多任务学习头**：三个专门的输出头，实现联合优化

### 技术优势

- **多尺度特征融合**：从不同层级提取特征，提高预测精度
- **注意力机制**：自动聚焦于病害相关区域
- **混合精度训练**：支持AMP加速训练，提高效率
- **损失函数优化**：组合Dice损失和Focal损失，处理类别不平衡

## 环境要求

### 基础依赖

```bash
# 核心深度学习框架
torch>=1.9.0
torchvision>=0.10.0

# 图像处理
rasterio>=1.2.0
scikit-image>=0.18.0
Pillow>=8.0.0

# 数据处理和可视化
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.60.0

# 机器学习工具
scikit-learn>=0.24.0
```

### 安装命令

```bash
# 使用pip安装
pip install torch torchvision rasterio scikit-image matplotlib tqdm seaborn numpy pillow scikit-learn

# 或使用conda安装
conda install pytorch torchvision -c pytorch
conda install rasterio scikit-image matplotlib tqdm seaborn numpy pillow scikit-learn -c conda-forge
```

## 快速开始

### 1. 模型架构测试

```bash
# 测试模型架构和损失函数
python test_lodgenet.py --test_architecture --test_loss
```

### 2. 开始训练

```bash
# 使用启动脚本（推荐）
python run_lodgenet.py --data_root ./guanceng-bit --json_root ./biaozhu_json --num_epochs 50

# 或直接运行训练脚本
python lodgenet_train.py --data_root ./guanceng-bit --json_root ./biaozhu_json --batch_size 8 --num_epochs 50 --learning_rate 0.0001 --output_dir ./output_lodgenet
```

### 3. 模型测试和评估

```bash
# 测试训练好的模型
python test_lodgenet.py --model_path ./output_lodgenet/best_model.pth --data_root ./guanceng-bit --json_root ./biaozhu_json --visualize --save_dir ./test_results
```

## 训练配置

### 推荐参数设置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `batch_size` | 8 | 批次大小，GPU内存允许可增大 |
| `num_epochs` | 50 | 训练轮数，无EarlyStopping |
| `learning_rate` | 0.0001 | 学习率，使用Adam优化器 |
| `img_size` | 128 | 输入图像尺寸 |
| `task_weights` | [0.4, 0.3, 0.3] | 任务权重：[分割, 位置, 等级] |

### 损失函数配置

- **分割任务**：组合损失（CrossEntropy + Dice）
- **位置分类**：Focal Loss（γ=2.0）
- **等级回归**：MSE Loss

### 优化策略

- **优化器**：Adam（weight_decay=1e-4）
- **学习率调度**：ReduceLROnPlateau（patience=5）
- **混合精度**：自动启用（GPU环境）

## 性能指标

### 目标指标

LodgeNet训练的目标是达到以下性能指标：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 位置分类准确率 | > 90% | 感染部位分类准确性 |
| 位置分类F1分数 | > 0.85 | 综合精确率和召回率 |
| 位置分类召回率 | > 0.85 | 真实感染部位的识别率 |
| 位置分类精确率 | > 0.85 | 预测感染部位的准确性 |
| 病害等级MAE | < 0.15 | 等级预测的平均绝对误差 |
| 总损失 | < 0.2 | 多任务联合损失 |

### 评估指标

训练过程中监控的完整指标：

- **分割任务**：Dice系数、IoU、像素准确率
- **位置分类**：准确率、F1分数、精确率、召回率、混淆矩阵
- **等级回归**：MAE、RMSE、R²相关系数

## 文件结构

```
LodgeNet/
├── lodgenet_model.py          # LodgeNet模型定义
├── lodgenet_train.py          # 训练脚本
├── run_lodgenet.py           # 训练启动脚本
├── test_lodgenet.py          # 测试和评估脚本
├── dataset.py                # 数据集加载（复用原有）
├── utils.py                  # 工具函数（复用原有）
├── README_LodgeNet.md        # 本文档
├── guanceng-bit/             # TIF图像数据目录
├── biaozhu_json/             # JSON标注数据目录
├── output_lodgenet/          # 训练输出目录
├── logs/                     # 训练日志目录
└── test_results/             # 测试结果目录
```

## 使用示例

### 基础训练

```bash
# 使用默认参数训练
python run_lodgenet.py
```

### 自定义训练

```bash
# 自定义参数训练
python run_lodgenet.py \
    --data_root ./guanceng-bit \
    --json_root ./biaozhu_json \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 0.0005 \
    --output_dir ./output_custom
```

### 模型评估

```bash
# 评估最佳模型
python test_lodgenet.py \
    --model_path ./output_lodgenet/best_model.pth \
    --data_root ./guanceng-bit \
    --json_root ./biaozhu_json \
    --visualize
```

## 训练监控

### 日志记录

训练过程自动记录详细日志：

```bash
# 日志文件位置
./logs/lodgenet_training_YYYYMMDD_HHMMSS.log
```

### 可视化输出

训练完成后自动生成：

- **训练指标图表**：`lodgenet_training_metrics.png`
- **混淆矩阵**：位置分类的详细分析
- **预测可视化**：样本预测结果展示

### 模型保存

训练过程中保存多个检查点：

- `best_model.pth`：最佳F1分数模型
- `last_model.pth`：最后一轮模型
- `epoch_X.pth`：每10轮保存的检查点

## 模型优化建议

### 数据增强

当前支持的数据增强：
- 随机水平/垂直翻转
- 随机旋转（±15度）
- 颜色抖动

### 超参数调优

建议调优的参数：
- `learning_rate`：根据收敛情况调整
- `batch_size`：根据GPU内存调整
- `task_weights`：根据任务重要性调整

### 模型改进方向

1. **更复杂的分割标签**：使用真实的像素级标注
2. **多尺度训练**：支持不同输入尺寸
3. **集成学习**：结合多个模型的预测
4. **迁移学习**：使用预训练权重初始化

## 故障排除

### 常见问题

**Q: 训练过程中出现内存不足错误？**
A: 减小`batch_size`参数，或使用梯度累积技术。

**Q: 模型收敛缓慢？**
A: 检查学习率设置，考虑使用学习率预热或调整任务权重。

**Q: 分割效果不理想？**
A: 当前使用虚拟分割标签，建议获取真实的像素级标注数据。

**Q: 多任务学习不平衡？**
A: 调整`task_weights`参数，平衡不同任务的重要性。

### 性能优化

1. **使用GPU训练**：显著提升训练速度
2. **启用混合精度**：减少显存占用，加速训练
3. **调整工作进程数**：`num_workers`参数优化数据加载
4. **使用SSD存储**：提高数据读取速度

## 技术支持

### 模型架构详解

LodgeNet采用以下创新设计：

1. **多尺度特征提取**：编码器提取5个不同尺度的特征
2. **注意力引导解码**：解码器使用注意力门控优化特征融合
3. **任务特定头部**：三个独立的输出头处理不同任务
4. **损失函数设计**：针对每个任务优化的损失函数组合

### 实验结果

基于当前数据集的实验表明：
- 模型参数量：约2.5M（相比ResNet更轻量）
- 训练时间：50轮约2-4小时（GPU）
- 推理速度：单张图像<50ms

## 更新日志

### v1.0.0 (当前版本)
- 实现基础LodgeNet架构
- 支持多任务学习训练
- 集成注意力机制和ASPP模块
- 完整的训练和测试流程

### 计划更新
- 支持真实分割标签训练
- 增加更多数据增强策略
- 模型压缩和量化支持
- 部署优化和推理加速

---

**注意**：LodgeNet是基于当前ResNet项目重构的专用模型，保持了与原有数据集和工具的兼容性，同时提供了更强大的多任务学习能力。 