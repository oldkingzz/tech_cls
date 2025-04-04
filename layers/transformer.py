import torch
import torch.nn as nn
from .MoE_layer import MoELayer

class TransformerLayerWithMoE(nn.Module):
    """
    结合MoE的Transformer层
    
    参数:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        num_experts (int): 专家数量
        dropout (float): Dropout比率
        hidden_dim (int, optional): MoE层的隐藏维度，默认为None时使用2*d_model
    """
    def __init__(self, d_model, num_heads, num_experts=8, dropout=0.1, hidden_dim=None):
        super().__init__()
        
        # 使用PyTorch原生的MultiheadAttention
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 使用MoE替代传统的FFN
        self.moe = MoELayer(
            dim=d_model,
            num_experts=num_experts,
            hidden_dim=hidden_dim,  # 允许外部指定hidden_dim
            activation=nn.GELU
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 [B, L, D]
            mask: 注意力掩码，形状为 [B, L]
            
        返回:
            tuple: (处理后的张量, moe_loss)
        """
        # 自注意力
        residual = x
        x = self.norm1(x)
        
        # 注意力计算
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = residual + self.dropout(attn_output)
        
        # MoE层
        residual = x
        x = self.norm2(x)
        x, moe_loss = self.moe(x)
        x = residual + self.dropout(x)
        
        return x, moe_loss 