import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    """
    分类头，用于多分类任务
    
    参数:
        d_model (int): 输入特征维度
        num_classes (int): 分类类别数
        pool_type (str): 池化类型，可选'mean'、'max'或'attention'
        dropout (float): Dropout比率
        hidden_size (int): 隐藏层大小，如果为None则不使用隐藏层
    """
    def __init__(self, d_model, num_classes, pool_type='mean', dropout=0.1, hidden_size=None):
        super().__init__()
        self.pool_type = pool_type
        
        # 如果使用注意力池化
        if pool_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )
        
        # 隐藏层（可选）
        if hidden_size is not None:
            self.fc1 = nn.Linear(d_model, hidden_size)
            self.dropout1 = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
        else:
            self.fc = nn.Linear(d_model, num_classes)
            
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入序列特征 [batch_size, seq_len, d_model]
            
        返回:
            分类logits [batch_size, num_classes]
        """
        # 根据池化类型进行池化操作
        if self.pool_type == 'mean':
            # 全局平均池化
            x = torch.mean(x, dim=1)
        elif self.pool_type == 'max':
            # 全局最大池化
            x = torch.max(x, dim=1)[0]
        elif self.pool_type == 'attention':
            # 注意力池化
            attn_weights = self.attention(x).squeeze(-1)
            attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
            x = torch.bmm(attn_weights, x).squeeze(1)
        else:
            raise ValueError(f"Unsupported pool type: {self.pool_type}")
        
        # 应用分类器
        if self.hidden_size is not None:
            x = self.dropout1(F.gelu(self.fc1(x)))
            x = self.norm(x)
            x = self.fc2(x)
        else:
            x = self.dropout(x)
            x = self.fc(x)
            
        return x

class RegressionHead(nn.Module):
    """
    回归头，用于回归任务
    
    参数:
        d_model (int): 输入特征维度
        output_dim (int): 输出维度，默认为1
        pool_type (str): 池化类型，可选'mean'、'max'或'attention'
        dropout (float): Dropout比率
        hidden_size (int): 隐藏层大小，如果为None则不使用隐藏层
    """
    def __init__(self, d_model, output_dim=1, pool_type='mean', dropout=0.1, hidden_size=None):
        super().__init__()
        self.pool_type = pool_type
        
        # 如果使用注意力池化
        if pool_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )
        
        # 隐藏层（可选）
        if hidden_size is not None:
            self.fc1 = nn.Linear(d_model, hidden_size)
            self.dropout1 = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_dim)
        else:
            self.fc = nn.Linear(d_model, output_dim)
            
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入序列特征 [batch_size, seq_len, d_model]
            
        返回:
            回归输出 [batch_size, output_dim]
        """
        # 根据池化类型进行池化操作
        if self.pool_type == 'mean':
            # 全局平均池化
            x = torch.mean(x, dim=1)
        elif self.pool_type == 'max':
            # 全局最大池化
            x = torch.max(x, dim=1)[0]
        elif self.pool_type == 'attention':
            # 注意力池化
            attn_weights = self.attention(x).squeeze(-1)
            attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
            x = torch.bmm(attn_weights, x).squeeze(1)
        else:
            raise ValueError(f"Unsupported pool type: {self.pool_type}")
        
        # 应用回归器
        if self.hidden_size is not None:
            x = self.dropout1(F.gelu(self.fc1(x)))
            x = self.norm(x)
            x = self.fc2(x)
        else:
            x = self.dropout(x)
            x = self.fc(x)
            
        return x 