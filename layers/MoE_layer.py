"""
混合专家模型(Mixture of Experts)层的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """
    混合专家模型层
    
    参数:
        dim (int): 输入特征维度
        num_experts (int): 专家数量
        hidden_dim (int): 专家隐藏层维度
        activation (nn.Module): 激活函数
        noisy_gating (bool): 是否使用噪声门控
        k (int): 每个token选择的专家数量
    """
    
    def __init__(self, dim, num_experts=8, hidden_dim=None, activation=nn.GELU, 
                 noisy_gating=True, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.k = k
        self.noisy_gating = noisy_gating
        
        if hidden_dim is None:
            hidden_dim = 2 * dim  # 减少隐藏层维度
            
        # 专家网络 - 使用ModuleList更高效
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
        # 噪声门控参数
        if noisy_gating:
            self.noise_epsilon = 1e-2
            
        # 负载均衡损失的权重
        self.balance_loss_weight = 0.01
            
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征，形状为[batch_size, seq_len, dim]
            
        返回:
            tuple: (输出特征, 负载均衡损失)
        """
        batch_size, seq_len, _ = x.shape
        
        # 将输入展平为二维张量
        flat_x = x.reshape(-1, self.dim)  # [batch_size * seq_len, dim]
        
        # 计算门控值
        gate_logits = self.gate(flat_x)  # [batch_size * seq_len, num_experts]
        
        # 添加噪声（如果启用）
        if self.noisy_gating and self.training:
            noise = torch.randn_like(gate_logits) * self.noise_epsilon
            gate_logits = gate_logits + noise
            
        # 计算门控概率
        gates = F.softmax(gate_logits, dim=-1)  # [batch_size * seq_len, num_experts]
        
        # 选择top-k专家
        top_k_gates, top_k_indices = torch.topk(gates, k=self.k, dim=-1)
        top_k_gates = top_k_gates / torch.sum(top_k_gates, dim=-1, keepdim=True)  # 重新归一化
        
        # 计算负载均衡损失 - 使用向量化操作
        # 创建one-hot编码的专家选择矩阵
        expert_mask = torch.zeros(flat_x.size(0), self.num_experts, device=x.device)
        expert_mask.scatter_(1, top_k_indices, 1)  # 将选中的专家位置设为1
        
        # 计算每个专家的使用频率
        expert_usage = expert_mask.sum(0) / (batch_size * seq_len)
        
        # 计算负载均衡损失
        balance_loss = torch.mean((expert_usage - (1.0 / self.num_experts)) ** 2) * self.num_experts
        
        # 使用批处理方式计算专家输出
        expert_outputs = torch.zeros_like(flat_x)
        
        # 对每个专家进行批处理计算
        for expert_idx in range(self.num_experts):
            # 找到选择了当前专家的样本
            expert_mask = torch.any(top_k_indices == expert_idx, dim=1)
            if not expert_mask.any():
                continue
                
            # 提取选择了当前专家的样本
            expert_inputs = flat_x[expert_mask]
            
            # 计算专家输出
            expert_output = self.experts[expert_idx](expert_inputs)
            
            # 找到当前专家在top-k中的位置
            expert_positions = (top_k_indices == expert_idx).float()
            
            # 提取对应的门控值
            expert_gates = torch.sum(top_k_gates * expert_positions, dim=1)
            expert_gates = expert_gates[expert_mask].unsqueeze(1)
            
            # 加权求和
            expert_outputs[expert_mask] += expert_output * expert_gates
        
        # 重塑回原始形状
        outputs = expert_outputs.reshape(batch_size, seq_len, self.dim)
        
        return outputs, balance_loss * self.balance_loss_weight
    
    def get_active_experts(self, x):
        """
        获取激活的专家信息
        
        参数:
            x (torch.Tensor): 输入特征
            
        返回:
            tuple: (门控值, 专家索引)
        """
        flat_x = x.reshape(-1, self.dim)
        gate_logits = self.gate(flat_x)
        gates = F.softmax(gate_logits, dim=-1)
        top_k_gates, top_k_indices = torch.topk(gates, k=self.k, dim=-1)
        
        return gates, top_k_indices 