# YOTO - 轴承剩余寿命预测模型架构

## 整体架构

YOTO是一个基于Transformer和MoE(Mixture of Experts)的轴承剩余寿命预测模型。该模型采用单流结构，通过MoE机制实现了对轴承振动特征的自适应建模。

## 核心组件

### 1. RULPredictionModel

主模型，包含以下关键组件：
- 输入: [batch_size, seq_len, 1] 的轴承振动特征
- 特征投影层: 将1维特征投影到d_model维
- 单流骨干网络: SingleStreamBackbone
- RUL预测头: RULHead
- 输出: 剩余寿命预测值和MoE损失

### 2. SingleStreamBackbone

单流骨干网络，由多个TransformerLayerWithMoE组成：
- 层数: num_layers (默认3)
- 每层包含:
  - 多头自注意力机制
  - MoE层替代传统FFN
- 输出: 特征表示和累积的MoE损失

### 3. TransformerLayerWithMoE

结合MoE的Transformer层：
- 多头自注意力层 (MultiheadAttention)
  - 输入维度: d_model
  - 注意力头数: num_heads
- MoE层
  - 专家数量: num_experts
  - 隐藏维度: 2*d_model (可配置)
  - 激活函数: GELU

### 4. RULHead

RUL预测头部网络：
- 全局注意力层
- 特征聚合
- MLP预测网络
  - 输出维度: 1

## 模型参数

- d_model: 384 (模型维度)
- num_heads: 8 (注意力头数)
- num_layers: 3 (Transformer层数)
- num_experts: 8 (MoE专家数量)
- dropout: 0.1 (Dropout比率)

## 损失函数

总损失 = RUL_Loss + moe_weight * MoE_Loss
- RUL_Loss: 回归损失
- MoE_Loss: 专家平衡损失

## 数据处理

- 输入: 1维振动特征
- 特征投影: 1维 → 384维

## 可视化与分析

模型提供了get_active_experts方法用于分析：
- 可获取每一层的专家激活模式
- 用于分析不同输入对专家的选择偏好

## 模型结构图

振动特征 [B, L, 1]
        ↓
   特征投影层 (1→384)
        ↓
   Transformer+MoE
        ↓
     RUL预测头
        ↓
    寿命预测 