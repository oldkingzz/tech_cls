import torch
import torch.nn as nn
from .transformer import TransformerLayerWithMoE

class TransformerMoEBackbone(nn.Module):
    """
    基于Transformer和MoE的骨干网络，用于处理特征序列
    
    参数:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        num_layers (int): Transformer层数
        num_experts (int): 专家数量
        dropout (float): Dropout比率，默认0.1
        hidden_dim (int, optional): MoE隐藏层维度，默认为None时使用2*d_model
    """
    def __init__(self, d_model, num_heads, num_layers, num_experts, dropout=0.1, hidden_dim=None):
        super().__init__()
        
        # 如果没有指定隐藏维度，则使用2倍的模型维度
        if hidden_dim is None:
            hidden_dim = 2 * d_model
            
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
            
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
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为[batch_size, seq_len, d_model]
            mask: 可选的注意力掩码
            
        返回:
            tuple: (输出特征, MoE损失)
        """
        # 添加位置编码
        x = self.pos_encoder(x)
        
        total_moe_loss = 0
        
        # 序列特征处理
        for layer in self.layers:
            x, moe_loss = layer(x, mask)
            total_moe_loss += moe_loss
            
        # 最终归一化
        x = self.norm(x)
            
        return x, total_moe_loss
    
    def get_active_experts(self, x, mask=None):
        """
        获取激活的专家信息，用于分析
        
        参数:
            x: 输入特征，形状为[batch_size, seq_len, d_model]
            mask: 可选的注意力掩码
            
        返回:
            list: 每层激活的专家信息
        """
        # 添加位置编码
        x = self.pos_encoder(x)
        
        expert_patterns = []
        
        # 获取每一层的专家激活信息
        for layer in self.layers:
            # 自注意力部分
            norm_x = layer.norm1(x)
            attn_output = layer.self_attn(norm_x, norm_x, norm_x, mask=mask)[0]
            x = x + layer.dropout1(attn_output)
            
            # MoE部分
            norm_x = layer.norm2(x)
            gates, indices = layer.moe.get_active_experts(norm_x)
            expert_patterns.append((gates, indices))
            
            x, _ = layer(x, mask)
            
        return expert_patterns

class PositionalEncoding(nn.Module):
    """
    位置编码模块，为Transformer添加位置信息
    
    参数:
        d_model (int): 模型维度
        dropout (float): Dropout比率
        max_len (int): 最大序列长度
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码
        
        参数:
            x: 输入序列 [batch_size, seq_len, d_model]
            
        返回:
            添加位置编码后的序列
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x) 