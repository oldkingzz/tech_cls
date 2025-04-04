import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.feature_extractor import VibrationFeatureExtractor
from layers.backbone import TransformerMoEBackbone
from layers.heads import ClassificationHead

class EnhancedClassificationMoE(nn.Module):
    """
    增强版基于MoE的分类模型，使用先进的特征提取和更灵活的结构
    
    参数:
        in_channels (int): 输入通道数，默认1
        base_filters (int): 基础滤波器数量，默认32
        d_model (int): 模型维度，默认768
        seq_len (int): 序列长度，默认16
        num_heads (int): 注意力头数，默认12
        num_layers (int): Transformer层数，默认6
        num_experts (int): 专家数量，默认8（比原来减少）
        hidden_dim (int, optional): MoE隐藏层维度，默认None
        num_classes (int): 分类类别数
        pool_type (str): 池化类型，默认'attention'
        dropout (float): Dropout比率，默认0.2（增加正则化）
        use_fft (bool): 是否使用FFT特征，默认True
    """
    def __init__(
        self, 
        in_channels=1,
        base_filters=32,
        d_model=768, 
        seq_len=16,
        num_heads=12, 
        num_layers=4,  # 减少层数以降低复杂度
        num_experts=8,  # 减少专家数量
        hidden_dim=None, 
        num_classes=5, 
        pool_type='attention',
        dropout=0.2,  # 增加dropout
        use_fft=True,
        **kwargs
    ):
        super().__init__()
        
        # 保存配置
        self.d_model = d_model
        self.num_experts = num_experts
        
        # 特征提取器
        self.feature_extractor = VibrationFeatureExtractor(
            in_channels=in_channels,
            base_filters=base_filters,
            d_model=d_model,
            seq_len=seq_len,
            use_fft=use_fft
        )
        
        # 骨干网络
        self.backbone = TransformerMoEBackbone(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            num_experts=num_experts,
            dropout=dropout,
            hidden_dim=hidden_dim
        )
        
        # 分类头
        self.classification_head = ClassificationHead(
            d_model=d_model, 
            num_classes=num_classes, 
            pool_type=pool_type,
            dropout=dropout,
            hidden_size=d_model // 2  # 使用隐藏层
        )
        
        # 辅助分类器（在特征提取器之后直接预测，提供梯度）
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为[batch_size, channels, seq_len]
            
        返回:
            dict: 包含预测结果的字典
        """
        # 特征提取
        features = self.feature_extractor(x)  # [batch_size, seq_len, d_model]
        
        # 辅助分类器（可选使用）
        aux_logits = self.aux_classifier(features.transpose(1, 2))
        
        # 骨干网络
        features, moe_loss = self.backbone(features)
        
        # 分类预测
        logits = self.classification_head(features)
        
        # 返回结果
        return {
            'logits': logits, 
            'aux_logits': aux_logits,
            'moe_loss': moe_loss
        }
    
    def get_active_experts(self, x):
        """
        获取当前激活的专家信息，用于分析
        
        参数:
            x: 输入特征，形状为[batch_size, channels, seq_len]
            
        返回:
            list: 包含每一层的专家激活模式
        """
        # 特征提取
        features = self.feature_extractor(x)
        
        # 获取专家激活信息
        expert_patterns = self.backbone.get_active_experts(features)
        
        return expert_patterns 