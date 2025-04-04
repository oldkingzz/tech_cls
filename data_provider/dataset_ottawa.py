"""
Ottawa数据集类，继承自基类dataset_yoto
"""

import os
import torch
import numpy as np
import pandas as pd
from glob import glob

# 使用相对导入，但在直接运行时会处理这个问题
try:
    from .dataset_yoto import dataset_yoto
except ImportError:
    # 当直接运行此文件时，使用绝对导入
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_provider.dataset_yoto import dataset_yoto


class dataset_ottawa(dataset_yoto):
    """
    渥太华轴承数据集类
    
    数据格式：
    - 文件路径: "/root/autodl-tmp/ottawa_resampled"
    - H开头: 正常（normal）
    - I开头: 内圈故障（inner）
    - O开头: 外圈故障（outer）
    """
    
    def __init__(self, data_dir="/root/autodl-tmp/ottawa_resampled", data_type=0, 
                 seq_len=4096, normalize=True, dataset_train_or_val="train", 
                 train_ratio=0.8):
        """
        初始化函数
        
        参数:
            data_dir (str): 数据目录路径
            data_type (int): 数据类型，0表示故障分类，1表示寿命预测
            seq_len (int): 序列长度，默认为4096
            normalize (bool): 是否归一化数据
            dataset_train_or_val (str): 训练集或验证集
            train_ratio (float): 训练集比例，默认0.8
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.train_ratio = train_ratio
        
        # Ottawa数据集文件名前缀到故障类型的映射
        self.ottawa_fault_map = {
            'H': 'normal',
            'I': 'inner',
            'O': 'outer'
        }
        
        # 调用父类初始化
        super().__init__(
            data_type=data_type,
            seq_len=seq_len,
            dataset_name="Ottawa",
            dataset_description="Ottawa University Bearing Dataset",
            sampling_rate=51200,  # 重采样后的采样率
            dataset_train_or_val=dataset_train_or_val,
            data_dir=data_dir
        )
    
    def _load_samples(self, data_type):
        """
        加载渥太华数据集样本
        
        参数:
            data_type (int): 数据类型，0表示故障分类，1表示寿命预测
            
        返回:
            list: 样本列表
        """
        if data_type != 0:
            raise ValueError("渥太华数据集只支持故障分类任务(data_type=0)")
        
        samples = []
        
        # 检查数据目录是否存在
        if not os.path.exists(self.data_dir):
            print(f"警告: 渥太华数据集目录 {self.data_dir} 不存在")
            return samples
            
        # 获取所有CSV文件
        csv_files = glob(os.path.join(self.data_dir, "*.csv"))
        if len(csv_files) == 0:
            print(f"警告: 在 {self.data_dir} 中未找到CSV文件")
            return samples
        
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        # 划分训练集和验证集
        np.random.seed(42)  # 确保可重复性
        np.random.shuffle(csv_files)
        split_idx = int(len(csv_files) * self.train_ratio)
        
        if self.dataset_train_or_val == "train":
            selected_files = csv_files[:split_idx]
        else:
            selected_files = csv_files[split_idx:]
        
        print(f"选择 {len(selected_files)} 个文件用于{self.dataset_train_or_val}集")
        
        for file_path in selected_files:
            file_name = os.path.basename(file_path)
            
            # 根据文件名前缀确定故障类型
            fault_type = None
            for prefix, fault in self.ottawa_fault_map.items():
                if file_name.startswith(prefix):
                    fault_type = fault
                    break
                    
            if fault_type is None:
                print(f"警告: 无法确定文件 {file_name} 的故障类型，跳过该文件")
                continue
            
            # 读取CSV文件
            try:
                df = pd.read_csv(file_path)
                
                # 检查必要的列是否存在
                if 'Channel_1' not in df.columns and 'Channel_2' not in df.columns:
                    print(f"警告: 文件 {file_name} 缺少通道数据，跳过该文件")
                    continue
                
                # 获取振动数据（这里使用Channel_1优先，如果没有则使用Channel_2）
                vibration_data = df['Channel_1'].values if 'Channel_1' in df.columns else df['Channel_2'].values
                
                # 处理太短的数据
                if len(vibration_data) < self.seq_len:
                    print(f"警告: 文件 {file_name} 数据长度不足 {self.seq_len}，跳过该文件")
                    continue
                
                # 归一化处理
                if self.normalize:
                    vibration_data = (vibration_data - np.mean(vibration_data)) / (np.std(vibration_data) + 1e-8)
                
                # 创建滑动窗口样本
                for i in range(0, len(vibration_data) - self.seq_len + 1, self.seq_len // 2):  # 50%重叠
                    segment = vibration_data[i:i+self.seq_len]
                    
                    # 创建样本，确保只包含基类中定义的字段
                    sample = {
                        'x': torch.tensor(segment, dtype=torch.float32).unsqueeze(0),  # [1, seq_len]
                        'y': torch.tensor(self.fault_type_map[fault_type], dtype=torch.float32),
                        'metadata': {
                            'subdir': '',  # 空子目录
                            'rpm': 0.0,  # 没有RPM信息
                            'load': 0.0,  # 没有负载信息
                            'fault_size': 0.0  # 没有故障尺寸信息
                        }
                    }
                    samples.append(sample)
                
            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")
                continue
        
        print(f"渥太华数据集共加载了 {len(samples)} 个样本")
        return samples


# 测试代码
if __name__ == "__main__":
    # 添加项目根目录到Python路径
    import sys
    import os
    
    # 获取当前文件所在目录的父目录的父目录（项目根目录）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    # 确保能正确导入dataset_yoto
    try:
        from YOTO.data_provider.dataset_yoto import dataset_yoto
        print("成功导入dataset_yoto")
    except ImportError:
        try:
            from data_provider.dataset_yoto import dataset_yoto
            print("成功导入dataset_yoto")
        except ImportError as e:
            print(f"导入dataset_yoto失败: {e}")
            sys.exit(1)
    
    # 确定数据目录
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/root/autodl-tmp/ottawa_resampled"  # 默认目录
    
    # 创建故障分类数据集
    print("\n测试故障分类数据集")
    try:
        dataset = dataset_ottawa(
            data_dir=data_dir,
            data_type=0,  # 故障分类
            seq_len=4096,
            normalize=True,
            dataset_train_or_val="train"
        )
        
        # 查看样本数量
        print(f"样本数量: {len(dataset)}")
        
        # 获取一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本x形状: {sample['x'].shape}")
            print(f"样本标签: {sample['label']}")
            print(f"样本元数据: {sample['metadata']}")
            
            # 尝试绘制波形（如果有matplotlib）
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 4))
                plt.plot(sample['x'][0].numpy())
                plt.title(f"故障类型: {list(dataset.fault_type_map.keys())[sample['label'].argmax().item()]}")
                plt.xlabel("采样点")
                plt.ylabel("振幅")
                plt.savefig("ottawa_sample.png")
                print("已保存波形图: ottawa_sample.png")
            except ImportError:
                print("未安装matplotlib，跳过波形绘制")
                
        # 创建验证集
        print("\n测试验证集")
        val_dataset = dataset_ottawa(
            data_dir=data_dir,
            data_type=0,
            seq_len=4096,
            normalize=True,
            dataset_train_or_val="val"
        )
        print(f"验证集样本数量: {len(val_dataset)}")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc() 