# YOTO - 机械损伤检测项目

YOTO是一个基于深度学习的机械损伤检测项目，主要用于轴承剩余使用寿命(RUL)预测。该项目使用了混合专家模型(Mixture of Experts, MoE)架构，通过稀疏门控机制提高模型性能和计算效率。

## 项目结构

```
YOTO/
├── data/                  # 数据目录
├── data_provider/         # 数据加载和预处理模块
│   ├── __init__.py
│   └── xjtu_dataset.py    # XJTU-SY轴承数据集类
├── layers/                # 神经网络层模块
│   ├── __init__.py
│   └── MoE_layer.py       # 混合专家模型层实现
├── model/                 # 模型定义
│   └── DualStreamMoE.py   # 双流混合专家模型
├── scripts/               # 脚本目录
│   └── train_SparseMoE.sh # 训练脚本
├── utils/                 # 工具函数
│   ├── __init__.py
│   ├── losses.py          # 损失函数
│   ├── masking.py         # 掩码相关函数
│   ├── md2img.py          # 机械数据转图像函数
│   ├── m4_summary.py      # 摘要生成函数
│   ├── metrics.py         # 评估指标
│   ├── timefeatures.py    # 时间特征提取
│   └── tools.py           # 通用工具函数
└── main_SparseMoE.py      # 主程序
```

## 核心功能

1. **剩余使用寿命(RUL)预测**：预测轴承的剩余使用寿命
2. **混合专家模型**：使用稀疏混合专家模型提高模型性能和计算效率

## 模型架构

项目使用了双流混合专家模型(DualStreamMoE)架构，主要包括以下组件：

1. **特征投影层**：将原始5维特征投影到高维空间
2. **骨干网络(Backbone)**：使用Transformer结构，每个Transformer层中的前馈网络被替换为混合专家模型(MoE)层
3. **RUL预测头**：使用全局注意力机制和多层感知机预测剩余使用寿命

### 优化的模型参数

为了提高训练速度和减少资源消耗，模型参数已优化为：

- **隐藏层维度(d_model)**: 128（原为768）
- **注意力头数(num_heads)**: 4（原为8）
- **Transformer层数(num_layers)**: 2（原为6）
- **专家数量(num_experts)**: 4（原为16）
- **专家网络隐藏层维度**: 2倍输入维度（原为4倍）

## 训练策略

当前训练策略专注于RUL预测，训练过程简化为单一阶段：

- **RUL训练**：训练模型预测剩余使用寿命

## 数据集

项目使用XJTU-SY轴承数据集，该数据集包含多个轴承在不同工况下的振动信号数据，以及对应的剩余使用寿命标签。

## 使用方法

### 环境配置

```bash
# 在Linux系统中初始化conda
source ~/miniconda3/bin/activate
# 或者
source /root/miniconda3/bin/activate

# 创建conda环境
conda create -n yoto python=3.8
conda activate yoto

# 配置更快的镜像源
# 清华大学镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 或者使用中国科技大学镜像源
# pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn h5py tqdm tensorboard matplotlib
```

### 训练模型

```bash
cd YOTO
bash scripts/train_SparseMoE.sh
```

### 自定义训练参数

可以通过修改`scripts/train_SparseMoE.sh`中的参数来自定义训练过程：

```bash
# 模型参数
D_MODEL=128       # 隐藏层维度
NUM_HEADS=4       # 注意力头数
NUM_LAYERS=2      # Transformer层数
NUM_EXPERTS=4     # 专家数量

# 训练参数
BATCH_SIZE=16     # 批处理大小
EPOCHS=40         # 训练轮数
LR=5e-4           # 学习率
```

或者直接修改`main_SparseMoE.py`中的默认参数：

```python
# 模型参数
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_experts', type=int, default=4)
```

## 性能优化建议

如果训练速度仍然不理想，可以考虑以下优化措施：

1. **使用混合精度训练**：
   - 在训练过程中使用FP16（半精度浮点数）可以显著减少内存使用并提高计算速度
   - 可以使用PyTorch的`torch.cuda.amp`模块实现

2. **数据预处理优化**：
   - 确保数据预处理在训练前完成，而不是在每个epoch中重复
   - 使用更高效的数据加载方式，如内存映射文件

3. **梯度累积**：
   - 如果内存是瓶颈，可以使用梯度累积技术，使用更小的批处理大小，但累积多个批次的梯度后再更新

4. **进一步减小模型规模**：
   - 如果计算资源有限，可以考虑进一步减小模型规模，如减少`d_model`到64或32

## 待完成功能

- [x] 优化模型结构，提高训练速度
- [x] 简化训练流程，专注于RUL预测
- [ ] 添加模型评估和可视化功能
- [ ] 实现模型部署和推理接口
- [ ] 添加更多数据集的支持
- [ ] 实现混合精度训练

## 参考资料

- [Mixture of Experts论文](https://arxiv.org/abs/1701.06538)
- [XJTU-SY轴承数据集](https://biaowang.tech/xjtu-sy-bearing-datasets/)
- [Transformer架构](https://arxiv.org/abs/1706.03762)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)