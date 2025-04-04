import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.transformer import TransformerLayerWithMoE
from layers.heads import RegressionHead

class SingleStreamBackbone(nn.Module):
    """
    单流MoE骨干网络，专注于处理振动特征
    
    参数:
        d_model (int): 模型维度，默认384
        num_heads (int): 注意力头数，默认8
        num_layers (int): 层数，默认3
        num_experts (int): 专家数量，默认8
        dropout (float): Dropout比率，默认0.1
        hidden_dim (int, optional): MoE层的隐藏维度，默认为None时使用2*d_model
    """
    def __init__(self, d_model=384, num_heads=8, num_layers=3, num_experts=8, dropout=0.1, hidden_dim=None):
        super().__init__()
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerLayerWithMoE(d_model, num_heads, num_experts, dropout, hidden_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        total_moe_loss = 0
        
        # 序列特征处理
        features = x
        for layer in self.layers:
            features, moe_loss = layer(features, mask)
            total_moe_loss += moe_loss
            
        return features, total_moe_loss

class RULPredictionModel(nn.Module):
    """
    单流RUL预测模型
    
    参数:
        d_model (int): 模型维度，默认384
        num_heads (int): 注意力头数，默认8
        num_layers (int): 层数，默认3
        num_experts (int): 专家数量，默认8
        hidden_dim (int, optional): MoE层的隐藏维度，默认为None时使用2*d_model
    """
    def __init__(self, d_model=384, num_heads=8, num_layers=3, num_experts=8, hidden_dim=None):
        super().__init__()
        # 振动特征投影层
        self.feature_projection = nn.Linear(1, d_model)  # 1维振动特征
        
        # 骨干网络
        self.backbone = SingleStreamBackbone(d_model, num_heads, num_layers, num_experts, hidden_dim=hidden_dim)
        
        # RUL预测头 (使用RegressionHead替代)
        self.rul_head = RegressionHead(d_model, output_dim=1, pool_type='attention')
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为[batch_size, seq_len, feature_dim]
            其中feature_dim=1，表示1维振动特征
            
        返回:
            dict: 包含预测结果的字典
        """
        # 特征投影
        features = self.feature_projection(x)
        
        # 特征提取
        features, moe_loss = self.backbone(features)
        
        # RUL预测
        rul_pred = self.rul_head(features)
        
        # 返回结果
        return {'rul': rul_pred, 'moe_loss': moe_loss}
    
    def get_active_experts(self, x):
        """
        获取当前激活的专家信息，用于分析
        
        参数:
            x: 输入特征，形状为[batch_size, seq_len, feature_dim]
            
        返回:
            list: 包含每一层的专家激活模式
        """
        features = self.feature_projection(x)
        
        expert_patterns = []
        x = features
        
        # 获取每一层的专家激活信息
        for layer in self.backbone.layers:
            norm_x = layer.norm2(x)
            gates, indices = layer.moe.get_active_experts(norm_x)
            expert_patterns.append((gates, indices))
            x, _ = layer(x)
            
        return expert_patterns 