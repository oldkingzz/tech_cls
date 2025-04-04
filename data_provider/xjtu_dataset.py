"""
XJTU数据集类，继承自基类dataset_yoto
"""

import os
import numpy as np
import torch
import pandas as pd
import glob
from .dataset_yoto import dataset_yoto


class XJTU_Dataset(dataset_yoto):
    """XJTU数据集类，继承自基类dataset_yoto"""
    
    def __init__(self, data_dir, data_type=0, seq_len=32768, normalize=True, 
                 dataset_train_or_val="train", use_all_samples=False):
        """
        初始化函数
        
        参数:
            data_dir (str): 数据目录
            data_type (int): 数据类型，0表示故障分类，1表示寿命预测
            seq_len (int): 序列长度，默认为32768（对应0.64秒@51.2kHz）
            normalize (bool): 是否标准化
            dataset_train_or_val (str): 训练集或验证集，"train"或"val"
            use_all_samples (bool): 是否使用所有样本
        """
        # XJTU数据集的采样率设置为统一标准
        sampling_rate = 51200
        
        self.data_dir = data_dir
        self.normalize = normalize
        self.use_all_samples = use_all_samples
        
        # 轴承故障类型映射
        self.bearing_fault_map = {
            'Bearing1_1': 'outer',
            'Bearing1_2': 'outer',
            'Bearing1_3': 'outer',
            'Bearing1_4': 'cage',
            'Bearing2_1': 'inner',
            'Bearing2_2': 'outer',
            'Bearing2_3': 'cage',
            'Bearing2_4': 'outer',
            'Bearing2_5': 'outer',
            'Bearing3_1': 'outer',
            'Bearing3_3': 'inner',
            'Bearing3_4': 'inner',
            'Bearing3_5': 'outer',
        }
        
        # 调用父类初始化方法
        super().__init__(
            data_type=data_type,
            seq_len=seq_len,
            dataset_name="XJTU",
            dataset_description="西安交通大学轴承数据集",
            sampling_rate=sampling_rate,
            dataset_train_or_val=dataset_train_or_val,
            data_dir=self.data_dir
        )
    
    def _load_samples(self, data_type):
        """
        加载XJTU数据集的样本
        
        参数:
            data_type (int): 数据类型，0表示故障分类，1表示寿命预测
            
        返回:
            list: 样本列表
        """
        samples = []
        
        # 检查数据目录是否存在
        if not os.path.exists(self.data_dir):
            print(f"警告: 目录 {self.data_dir} 不存在")
            return samples
        
        # 遍历所有测试条件目录（35Hz12kN, 37.5Hz11kN, 40Hz10kN）
        condition_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for condition_dir in condition_dirs:
            condition_path = os.path.join(self.data_dir, condition_dir)
            bearing_dirs = [d for d in os.listdir(condition_path) if os.path.isdir(os.path.join(condition_path, d))]
            
            for bearing_dir in bearing_dirs:
                if bearing_dir not in self.bearing_fault_map:
                    print(f"警告: 未知轴承 {bearing_dir}，跳过")
                    continue
                
                fault_type = self.bearing_fault_map[bearing_dir]
                bearing_path = os.path.join(condition_path, bearing_dir)
                
                # 获取所有CSV文件
                csv_files = [f for f in os.listdir(bearing_path) if f.endswith('.csv')]
                
                # 如果设置了使用所有样本，或者文件不多，则全部使用
                if self.use_all_samples or len(csv_files) <= 20:
                    selected_files = csv_files
                else:
                    # 每隔10个文件选择一个，但至少选择10个文件
                    selected_files = csv_files[::10]
                    if len(selected_files) < 10 and len(csv_files) >= 10:
                        selected_files = csv_files[:10]
                    elif len(selected_files) < 5:
                        selected_files = csv_files  # 如果文件总数少于5个，则全部使用
                
                print(f"轴承 {bearing_dir} 选择了 {len(selected_files)}/{len(csv_files)} 个文件")
                
                for csv_file in selected_files:
                    try:
                        file_path = os.path.join(bearing_path, csv_file)
                        df = pd.read_csv(file_path)
                        
                        # 提取振动信号
                        if '振动信号' in df.columns:
                            vibration = df['振动信号'].values[:self.seq_len]
                            
                            # 如果序列长度不足，用0填充
                            if len(vibration) < self.seq_len:
                                vibration = np.pad(vibration, (0, self.seq_len - len(vibration)))
                            
                            # 从CSV文件提取条件信息
                            try:
                                # 先从CSV文件内容获取频率和负载信息
                                freq = df['频率(Hz)'].iloc[0] if '频率(Hz)' in df.columns else float(condition_dir.split('Hz')[0])
                                load = df['负载(kN)'].iloc[0] if '负载(kN)' in df.columns else float(condition_dir.split('Hz')[1].split('kN')[0])
                                
                                # 如果CSV文件中有故障类型列，则使用它；否则使用目录名
                                fault_type_csv = df['故障类型'].iloc[0] if '故障类型' in df.columns else None
                                if fault_type_csv and fault_type_csv in self.fault_type_map.keys():
                                    fault_type = fault_type_csv
                            except (ValueError, IndexError):
                                print(f"警告: 无法解析条件 {condition_dir}，使用默认值")
                                freq = 35.0
                                load = 10.0
                            
                            # 标准化数据
                            if self.normalize:
                                vibration = (vibration - np.mean(vibration)) / (np.std(vibration) + 1e-8)
                            
                            # 根据data_type创建样本
                            x = torch.FloatTensor(vibration).unsqueeze(0)  # [1, seq_len]
                            
                            if data_type == 0:  # 故障分类
                                y = self.fault_type_map[fault_type]
                                
                                # 创建样本
                                samples.append({
                                    'x': x,
                                    'y': torch.FloatTensor(y),
                                    'metadata': {
                                        'rpm': freq * 60,  # 将Hz转换为rpm
                                        'load': load,
                                        'bearing': bearing_dir,
                                        'file': csv_file,
                                        'fault_size': 0.0,  # 添加必要的元数据
                                        'subdir': bearing_dir
                                    }
                                })
                            elif data_type == 1:  # 寿命预测
                                # 提取文件序号作为RUL的指标
                                try:
                                    file_num = int(csv_file.split('.')[0])
                                    total_files = len(csv_files)
                                    
                                    # 计算RUL值（剩余寿命，简单使用线性模型）
                                    # 假设文件按时间顺序命名，越靠后的文件RUL越小
                                    rul_value = 1.0 - (file_num / total_files)
                                    
                                    samples.append({
                                        'x': x,
                                        'y': torch.FloatTensor([rul_value]),
                                        'metadata': {
                                            'rpm': freq * 60,
                                            'load': load,
                                            'bearing': bearing_dir,
                                            'file': csv_file,
                                            'file_num': file_num,
                                            'total_files': total_files,
                                            'fault_size': 0.0,
                                            'subdir': bearing_dir
                                        }
                                    })
                                except ValueError:
                                    print(f"警告: 无法从文件名 {csv_file} 提取序号，跳过")
                        else:
                            print(f"警告: CSV文件 {csv_file} 中缺少'振动信号'列")
                                
                    except Exception as e:
                        print(f"加载文件 {csv_file} 出错: {e}")
        
        print(f"XJTU数据集共加载了 {len(samples)} 个样本")
        return samples


if __name__ == "__main__":
    # 测试代码
    import sys
    import matplotlib.pyplot as plt
    
    # 添加项目根目录到Python路径
    import sys
    import os
    # 获取当前文件所在目录的父目录的父目录（项目根目录）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    # 现在可以导入YOTO模块
    from YOTO.data_provider.dataset_yoto import dataset_yoto
    
    # 确定数据目录
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/root/autodl-tmp"  # 默认目录
    
    # 创建故障分类数据集
    print("\n测试故障分类数据集")
    classification_dataset = XJTU_Dataset(
        data_dir=data_dir,
        data_type=0,  # 故障分类
        seq_len=1024,
        normalize=True,
        dataset_train_or_val="train",
        use_all_samples=False
    )
    
    # 查看样本数量
    print(f"样本数量: {len(classification_dataset)}")
    
    # 获取一个样本
    if len(classification_dataset) > 0:
        sample = classification_dataset[0]
        print(f"样本x形状: {sample['x'].shape}")
        print(f"样本y: {sample['y']}")
        print(f"样本元数据: {sample['metadata']}")
        
        # 绘制波形
        plt.figure(figsize=(12, 4))
        plt.plot(sample['x'][0].numpy())
        plt.title(f"故障类型: {list(classification_dataset.fault_type_map.keys())[sample['y'].argmax().item()]}")
        plt.xlabel("采样点")
        plt.ylabel("振幅")
        plt.savefig("xjtu_classification_sample.png")
        print("已保存波形图: xjtu_classification_sample.png")
    
    # 创建寿命预测数据集
    print("\n测试寿命预测数据集")
    rul_dataset = XJTU_Dataset(
        data_dir=data_dir,
        data_type=1,  # 寿命预测
        seq_len=1024,
        normalize=True,
        dataset_train_or_val="train",
        use_all_samples=False
    )
    
    # 查看样本数量
    print(f"样本数量: {len(rul_dataset)}")
    
    # 获取一个样本
    if len(rul_dataset) > 0:
        sample = rul_dataset[0]
        print(f"样本x形状: {sample['x'].shape}")
        print(f"剩余寿命值: {sample['y'].item():.4f}")
        print(f"样本元数据: {sample['metadata']}")
        
        # 绘制波形
        plt.figure(figsize=(12, 4))
        plt.plot(sample['x'][0].numpy())
        plt.title(f"剩余寿命: {sample['y'].item():.4f}")
        plt.xlabel("采样点")
        plt.ylabel("振幅")
        plt.savefig("xjtu_rul_sample.png")
        print("已保存波形图: xjtu_rul_sample.png") 