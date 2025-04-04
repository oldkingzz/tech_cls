import torch
import numpy as np
from model import EnhancedClassificationMoE

def test_model():
    """
    测试增强版分类MoE模型
    """
    print("测试增强版分类MoE模型...")
    
    # 创建模型实例
    model = EnhancedClassificationMoE(
        in_channels=1,
        base_filters=32,
        d_model=256,  # 减小模型维度
        seq_len=16,
        num_heads=4,  # 减少注意力头数
        num_layers=2,  # 减少层数
        num_experts=4,  # 减少专家数量
        num_classes=5,
        pool_type='attention',
        dropout=0.2,
        use_fft=True
    )
    
    # 打印模型结构
    print("模型参数数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # 创建一个样本输入
    batch_size = 4
    seq_len = 1024
    x = torch.randn(batch_size, 1, seq_len)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    # 打印输出结果
    print("输出形状:")
    print(f"logits: {output['logits'].shape}")
    print(f"aux_logits: {output['aux_logits'].shape}")
    print(f"moe_loss: {output['moe_loss']}")
    
    # 测试专家激活分析
    expert_patterns = model.get_active_experts(x)
    print(f"专家激活模式数量: {len(expert_patterns)}")
    
    # 测试不同输入长度
    seq_lens = [512, 1024, 2048, 4096]
    for sl in seq_lens:
        print(f"\n测试序列长度: {sl}")
        x = torch.randn(batch_size, 1, sl)
        with torch.no_grad():
            output = model(x)
        print(f"logits: {output['logits'].shape}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_model() 