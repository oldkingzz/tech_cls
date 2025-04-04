import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.transformer import TransformerLayerWithMoE

class ClassificationHead(nn.Module):
    """
    分类头，用于多分类任务
    
    参数:
        d_model (int): 输入特征维度
        num_classes (int): 分类类别数
        dropout (float): Dropout比率
    """
    def __init__(self, d_model, num_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # 全局平均池化
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class SingleStreamBackbone(nn.Module):
    """
    单流MoE骨干网络，专注于处理振动特征
    
    参数:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        num_layers (int): 层数
        num_experts (int): 专家数量
        dropout (float): Dropout比率，默认0.1
        hidden_dim (int, optional): MoE层的隐藏维度，默认为None时使用2*d_model
    """
    def __init__(self, d_model, num_heads, num_layers, num_experts, dropout=0.1, hidden_dim=None):
        super().__init__()
        
        # 如果没有指定隐藏维度，则使用2倍的模型维度
        if hidden_dim is None:
            hidden_dim = 2 * d_model
            
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerLayerWithMoE(
                d_model=d_model, 
                num_heads=num_heads, 
                num_experts=num_experts, 
                dropout=dropout, 
                hidden_dim=hidden_dim
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为[batch_size, seq_len, d_model]
            mask: 可选的注意力掩码
            
        返回:
            tuple: (输出特征, MoE损失)
        """
        total_moe_loss = 0
        
        # 序列特征处理
        features = x
        for layer in self.layers:
            features, moe_loss = layer(features, mask)
            total_moe_loss += moe_loss
            
        return features, total_moe_loss

class ClassificationMoE(nn.Module):
    """
    基于MoE的分类模型
    
    参数:
        d_model (int): 模型维度，默认768
        num_heads (int): 注意力头数，默认12
        num_layers (int): 层数，默认6
        num_experts (int): 专家数量，默认16
        num_classes (int): 分类类别数
        hidden_dim (int, optional): MoE层的隐藏维度，默认为None时使用2*d_model
    """
    def __init__(self, d_model=768, num_heads=12, num_layers=6, num_experts=16, 
                 num_classes=2, hidden_dim=None, dropout=0.1, **kwargs):
        super().__init__()
        # 振动特征投影层
        self.feature_projection = nn.Linear(1, d_model)  # 1维振动特征
        
        # 骨干网络
        self.backbone = SingleStreamBackbone(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            num_experts=num_experts,
            dropout=dropout,
            hidden_dim=hidden_dim
        )
        
        # 分类头
        self.classification_head = ClassificationHead(d_model, num_classes, dropout)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为[batch_size, 1, seq_len]
            其中seq_len是序列长度
            
        返回:
            dict: 包含预测结果的字典
        """
        # 调整输入维度
        batch_size, channels, seq_len = x.shape
        x = x.transpose(1, 2)  # [batch_size, seq_len, 1]
        
        # 特征投影
        features = self.feature_projection(x)  # [batch_size, seq_len, d_model]
        
        # 特征提取
        features, moe_loss = self.backbone(features)
        
        # 分类预测
        logits = self.classification_head(features)
        
        # 返回结果
        return {'logits': logits, 'moe_loss': moe_loss}
    
    def get_active_experts(self, x):
        """
        获取当前激活的专家信息，用于分析
        
        参数:
            x: 输入特征，形状为[batch_size, 1, seq_len]
            
        返回:
            list: 包含每一层的专家激活模式
        """
        # 调整输入维度
        batch_size, _, seq_len = x.shape
        x = x.transpose(1, 2)  # [batch_size, seq_len, 1]
        
        # 特征投影
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