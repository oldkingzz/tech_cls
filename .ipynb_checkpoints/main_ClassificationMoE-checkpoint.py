import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model.ClassificationMoE import ClassificationMoE
from model.EnhancedClassificationMoE import EnhancedClassificationMoE  # 添加增强版模型
import argparse
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data_provider.data_factory import DataFactory
from utils.tools import get_cosine_schedule_with_warmup, cleanup_temp_files
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import shutil
import random

# 解析命令行参数
parser = argparse.ArgumentParser()
# 模型参数
parser.add_argument('--model_type', type=str, default='enhanced', choices=['standard', 'enhanced'], help='选择模型类型: standard或enhanced')
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_experts', type=int, default=8)
parser.add_argument('--num_classes', type=int, default=5)  # 修改为5类：normal、inner、outer、cage、ball
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--window_size', type=int, default=128)  # 添加窗口大小参数

# 训练参数
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--warmup_steps', type=int, default=500)

# 损失权重
parser.add_argument('--cls_weight', type=float, default=1.0)
parser.add_argument('--moe_weight', type=float, default=0.01)
parser.add_argument('--aux_weight', type=float, default=0.3)  # 辅助分类器权重（适用于增强版模型）

# 数据集参数
parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp')
parser.add_argument('--max_seq_len', type=int, default=4096)
parser.add_argument('--use_all_samples', action='store_true', help='是否使用所有样本')
parser.add_argument('--train_ratio', type=float, default=1.0, help='训练样本比例')
parser.add_argument('--val_ratio', type=float, default=1.0, help='验证样本比例')
parser.add_argument('--balance_classes', action='store_true', default=True, help='是否平衡各个类别样本数量')
parser.add_argument('--max_samples_per_class', type=int, default=5000, help='每个类别最大样本数量，None表示不限制')
parser.add_argument('--train_datasets', type=str, default='CASE,MFPT,XJTU,OTTAWA', help='训练集数据集列表，用逗号分隔')
parser.add_argument('--val_datasets', type=str, default='HUST', help='验证集数据集列表，用逗号分隔')

# 其他
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--save_dir', type=str, default='/root/autodl-tmp/checkpoints')
parser.add_argument('--exp_name', type=str, default='EnhancedModel')
parser.add_argument('--log_dir', type=str, default='/root/tf-logs')

args = parser.parse_args()
device = torch.device(args.device)

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子: {seed}")

set_seed(args.seed)

# 创建保存目录
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

# 清理之前的日志
if os.path.exists(args.log_dir):
    shutil.rmtree(args.log_dir)
os.makedirs(args.log_dir)

# 初始化tensorboard
writer = SummaryWriter(
    log_dir=os.path.join(args.log_dir, args.exp_name),
    flush_secs=120  # 增加刷新间隔，减少IO操作
)

print(f"TensorBoard日志保存在: {os.path.join(args.log_dir, args.exp_name)}")

# 记录超参数（改用scalar方式记录，避免创建过多子目录）
writer.add_scalar('hyperparameters/model_type', 1 if args.model_type == 'enhanced' else 0, 0)
writer.add_scalar('hyperparameters/d_model', args.d_model, 0)
writer.add_scalar('hyperparameters/num_heads', args.num_heads, 0)
writer.add_scalar('hyperparameters/num_layers', args.num_layers, 0)
writer.add_scalar('hyperparameters/num_experts', args.num_experts, 0)
writer.add_scalar('hyperparameters/num_classes', args.num_classes, 0)
writer.add_scalar('hyperparameters/batch_size', args.batch_size, 0)
writer.add_scalar('hyperparameters/lr', args.lr, 0)
writer.add_scalar('hyperparameters/weight_decay', args.weight_decay, 0)
writer.add_scalar('hyperparameters/warmup_steps', args.warmup_steps, 0)
writer.add_scalar('hyperparameters/cls_weight', args.cls_weight, 0)
writer.add_scalar('hyperparameters/moe_weight', args.moe_weight, 0)
writer.add_scalar('hyperparameters/max_seq_len', args.max_seq_len, 0)
writer.add_scalar('hyperparameters/window_size', args.window_size, 0)
writer.add_scalar('hyperparameters/balance_classes', 1 if args.balance_classes else 0, 0)

# 解析数据集列表
train_datasets = args.train_datasets.split(',')
val_datasets = args.val_datasets.split(',')

# 创建数据工厂，先不启用类别平衡，以便查看原始分布
print("\n创建原始数据集（未平衡）")
data_factory = DataFactory(
    data_dir=args.data_dir,
    batch_size=args.batch_size,
    normalize=True,
    max_seq_len=args.max_seq_len,
    use_all_samples=args.use_all_samples,
    train_ratio=args.train_ratio,
    val_ratio=args.val_ratio,
    balance_classes=False,  # 先不启用类别平衡
    max_samples_per_class=None
)

# 配置数据集
print("\n设置数据集")
data_factory.setup(
    train_datasets=train_datasets,
    val_datasets=val_datasets,
    data_type=0  # 故障分类任务
)

# 打印数据集信息和类别分布
print("\n原始训练集信息:")
print(f"训练集样本数: {len(data_factory.train_dataset)}")
train_class_dist = data_factory.get_class_distribution('train')
print(f"训练集类别分布: {train_class_dist}")

print("\n原始验证集信息:")
print(f"验证集样本数: {len(data_factory.val_dataset)}")
val_class_dist = data_factory.get_class_distribution('val')
print(f"验证集类别分布: {val_class_dist}")

# 保存原始数据集分布到TensorBoard
for class_idx, count in train_class_dist.items():
    writer.add_scalar(f'Data/Original_Train_{class_idx}', count, 0)
for class_idx, count in val_class_dist.items():
    writer.add_scalar(f'Data/Original_Val_{class_idx}', count, 0)

# 现在创建新的数据工厂，启用类别平衡
if args.balance_classes:
    print("\n创建平衡后的数据集")
    balanced_data_factory = DataFactory(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        normalize=True,
        max_seq_len=args.max_seq_len,
        use_all_samples=args.use_all_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        balance_classes=True,
        max_samples_per_class=args.max_samples_per_class
    )
    
    # 配置数据集
    balanced_data_factory.setup(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        data_type=0
    )
    
    # 打印平衡后的数据集信息
    print("\n平衡后训练集信息:")
    print(f"平衡后训练集样本数: {len(balanced_data_factory.train_dataset)}")
    balanced_train_class_dist = balanced_data_factory.get_class_distribution('train')
    print(f"平衡后训练集类别分布: {balanced_train_class_dist}")
    
    # 保存平衡后的数据集分布到TensorBoard
    for class_idx, count in balanced_train_class_dist.items():
        writer.add_scalar(f'Data/Balanced_Train_{class_idx}', count, 0)
    
    # 使用平衡后的数据集
    data_factory = balanced_data_factory

# 获取数据加载器
train_loader = data_factory.get_train_dataloader()
val_loader = data_factory.get_val_dataloader()

# 打印最终使用的数据集信息
print("\n训练集信息:")
print(f"训练集样本数: {len(train_loader.dataset)}")
print(f"训练集类别分布: {data_factory.get_class_distribution('train')}")
print("\n验证集信息:")
print(f"验证集样本数: {len(val_loader.dataset)}")
print(f"验证集类别分布: {data_factory.get_class_distribution('val')}")

# 创建模型
if args.model_type == 'enhanced':
    print("\n使用增强版分类MoE模型")
    model = EnhancedClassificationMoE(
        in_channels=1,
        base_filters=32,
        d_model=args.d_model,
        seq_len=16,  # 特征提取器输出序列长度
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        num_classes=args.num_classes,
        pool_type='attention',
        dropout=0.2,
        use_fft=True
    ).to(device)
else:
    print("\n使用标准分类MoE模型")
    model = ClassificationMoE(
        d_model=768,  # 增加基础维度
        num_heads=12,  # 增加注意力头数
        num_layers=6,  # 增加层数
        num_experts=16,  # 增加专家数量
        num_classes=args.num_classes,
        hidden_dim=2048,  # 增加隐藏层维度
        window_size=args.window_size
    ).to(device)

# 打印模型信息
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数总量: {total_params:,}")
print(f"模型结构:\n{model}")

# 打印GPU信息
try:
    import subprocess
    import time
    
    def get_gpu_memory_usage():
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_used, memory_total = map(int, result.strip().split(','))
        return memory_used, memory_total
    
    # 测试数据批次通过模型
    for test_batch in train_loader:
        x = test_batch['x'].float().to(device)
        labels = test_batch['label'].float().to(device)
        
        print(f"\n输入数据维度: {x.shape}")
        print(f"期望的序列长度: {args.max_seq_len}")
        
        # 前向传播前的GPU内存
        before_forward_used, before_forward_total = get_gpu_memory_usage()
        print(f"前向传播前GPU内存: {before_forward_used}MB / {before_forward_total}MB")
        
        # 前向传播
        outputs = model(x)
        torch.cuda.synchronize()  # 确保GPU操作完成
        
        # 前向传播后的GPU内存
        after_forward_used, after_forward_total = get_gpu_memory_usage()
        print(f"前向传播后GPU内存: {after_forward_used}MB / {after_forward_total}MB")
        print(f"模型使用内存: {after_forward_used - before_forward_used}MB")
        
        break
except Exception as e:
    print(f"获取GPU信息时出错: {e}")

# 优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.BCEWithLogitsLoss()  # 修改为BCEWithLogitsLoss以支持多标签分类
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    args.warmup_steps,
    len(train_loader) * args.epochs
)

# 训练循环
best_f1 = 0.0
print("开始分类训练")

for epoch in range(args.epochs):
    # 训练
    model.train()
    train_loss = 0
    train_preds = []
    train_labels_all = []
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    # 添加打印输入维度的标志，只打印第一个批次
    print_first_batch = True
    
    for step, batch in enumerate(pbar):
        optimizer.zero_grad()
        x = batch['x'].float().to(device)
        labels = batch['label'].float().to(device)  # 确保标签是float类型
        
        # 打印第一个批次的输入维度
        if print_first_batch:
            print(f"\n输入数据维度: {x.shape}")
            print(f"期望的序列长度: {args.max_seq_len}")
            print_first_batch = False
        
        outputs = model(x)
        logits = outputs['logits']
        
        # 计算主分类损失
        cls_loss = criterion(logits, labels)
        loss = args.cls_weight * cls_loss
        
        # 添加MoE损失（如果存在）
        if 'moe_loss' in outputs:
            loss += args.moe_weight * outputs['moe_loss']
        
        # 添加辅助分类器损失（适用于增强版模型）
        if 'aux_logits' in outputs and args.model_type == 'enhanced':
            aux_loss = criterion(outputs['aux_logits'], labels)
            loss += args.aux_weight * aux_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        # 使用sigmoid和阈值0.5来获取预测结果
        predicted = (torch.sigmoid(logits) > 0.5).float()
        train_preds.extend(predicted.cpu().numpy())
        train_labels_all.extend(labels.cpu().numpy())
        
        # 计算当前批次的F1分数
        current_f1 = f1_score(train_labels_all, train_preds, average='macro', zero_division=0)
        
        pbar.set_postfix({
            'loss': loss.item(),
            'f1': current_f1
        })
        
        # 记录到tensorboard
        global_step = epoch * len(train_loader) + step
        writer.add_scalar('Loss/train', loss.item(), global_step)
        writer.add_scalar('F1/train', current_f1, global_step)
    
    # 计算整个训练集的F1分数
    train_f1 = f1_score(train_labels_all, train_preds, average='macro', zero_division=0)
    print(f'Epoch {epoch}: Train F1 = {train_f1:.4f}')
    
    # 验证
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels_all = []
    
    # 检查验证集是否为空
    if len(val_loader.dataset) == 0:
        print("警告：验证集为空，跳过验证")
        val_f1 = train_f1  # 使用训练集F1代替
        # 在验证集为空的情况下，保存当前模型而不是只保存"最佳"模型
        checkpoint_path = os.path.join(args.save_dir, f'{args.exp_name}_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'f1_score': train_f1,  # 使用训练集F1代替
        }, checkpoint_path)
        print(f"已保存模型到 {checkpoint_path}")
    else:
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].float().to(device)
                labels = batch['label'].float().to(device)  # 确保标签是float类型
                
                outputs = model(x)
                logits = outputs['logits']
                val_loss += criterion(logits, labels).item()
                
                # 使用sigmoid和阈值0.5来获取预测结果
                predicted = (torch.sigmoid(logits) > 0.5).float()
                val_preds.extend(predicted.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())
        
        # 计算验证集F1分数
        val_f1 = f1_score(val_labels_all, val_preds, average='macro', zero_division=0)
        writer.add_scalar('Validation/F1', val_f1, epoch)
        writer.add_scalar('Validation/Loss', val_loss / len(val_loader), epoch)
        
        print(f'Epoch {epoch}: Val F1 = {val_f1:.4f}')
        
        # 记录每个类别的F1分数
        class_f1 = f1_score(val_labels_all, val_preds, average=None, zero_division=0)
        fault_types = ['normal', 'inner', 'outer', 'cage', 'ball']
        print("验证集各故障类型F1分数:")
        
        # 遍历每个故障类型
        for i, fault_type in enumerate(fault_types):
            writer.add_scalar(f'Validation/F1_{fault_type}', class_f1[i], epoch)
            print(f'  - {fault_type}: F1 = {class_f1[i]:.4f}')
    
    # 保存最佳模型
    if val_f1 > best_f1:
        best_f1 = val_f1
        checkpoint_path = os.path.join(args.save_dir, f'{args.exp_name}_best.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'f1_score': val_f1,
        }, checkpoint_path)
        print(f"保存最佳模型到 {checkpoint_path}")

# 训练结束后显示最终结果
print(f"\n最佳验证F1分数: {best_f1:.4f}")

# 如果验证集非空，加载最佳模型并计算详细的分类指标
if len(val_loader.dataset) > 0:
    print("\n加载最佳模型计算详细性能指标")
    best_model_path = os.path.join(args.save_dir, f'{args.exp_name}_best.pth')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].float().to(device)
            labels = batch['label'].float().to(device)
            
            outputs = model(x)
            logits = outputs['logits']
            
            predicted = (torch.sigmoid(logits) > 0.5).float()
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # 打印分类报告
    fault_types = ['normal', 'inner', 'outer', 'cage', 'ball']
    print("\n分类报告:")
    labels_idx = [np.argmax(label) for label in test_labels]
    preds_idx = [np.argmax(pred) for pred in test_preds]
    report = classification_report(labels_idx, preds_idx, target_names=fault_types)
    print(report)
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels_idx, preds_idx)
    print("\n混淆矩阵:")
    print(cm)

cleanup_temp_files()
writer.close() 