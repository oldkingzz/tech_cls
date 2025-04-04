import torch
import torch.nn as nn
from layers.transformer import TransformerLayerWithMoE

class SingleStreamBackbone(nn.Module):
    """
    单流骨干网络，使用Transformer层和MoE
    
    参数:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        num_layers (int): 层数
        num_experts (int): 专家数量
        hidden_dim (int, optional): MoE层的隐藏维度，默认为None时使用2*d_model
        window_size (int): 滑动窗口大小，默认为128
    """
    def __init__(self, d_model, num_heads, num_layers, num_experts, hidden_dim=None, window_size=128):
        super().__init__()
        
        # 如果没有指定隐藏维度，则使用2倍的模型维度
        if hidden_dim is None:
            hidden_dim = 2 * d_model
            
        # 创建多层Transformer
        self.layers = nn.ModuleList([
            TransformerLayerWithMoE(
                d_model=d_model,
                num_heads=num_heads,
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                window_size=window_size
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为[batch_size, seq_len, d_model]
            
        返回:
            tuple: (输出特征, MoE损失)
        """
        total_moe_loss = 0
        
        # 依次通过每一层
        for layer in self.layers:
            x, moe_loss = layer(x)
            total_moe_loss += moe_loss
            
        return x, total_moe_loss 