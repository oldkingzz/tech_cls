"""
MFPT数据集类，继承自基类dataset_yoto
"""

import os
import numpy as np
import torch
import pandas as pd
from .dataset_yoto import dataset_yoto


class MFPT_Dataset(dataset_yoto):
    """MFPT数据集类，继承自基类dataset_yoto"""
    
    def __init__(self, data_dir, data_type=0, seq_len=32768, normalize=True, 
                 dataset_train_or_val="train"):
        """
        初始化函数
        
        参数:
            data_dir (str): 数据目录
            data_type (int): 数据类型，0表示故障分类，1表示寿命预测
            seq_len (int): 序列长度，默认为32768（对应0.64秒@51.2kHz）
            normalize (bool): 是否标准化
            dataset_train_or_val (str): 训练集或验证集，"train"或"val"
        """
        # MFPT数据集的采样率设置为统一标准
        sampling_rate = 51200  # 设置为统一标准51.2kHz
        
        self.data_dir = data_dir
        self.normalize = normalize
        
        # 调用父类初始化方法
        super().__init__(
            data_type=data_type,
            seq_len=seq_len,
            dataset_name="MFPT",
            dataset_description="机械故障预防技术学会轴承数据集",
            sampling_rate=sampling_rate,
            dataset_train_or_val=dataset_train_or_val,
            data_dir=self.data_dir
        )
    
    def _load_samples(self, data_type):
        """
        加载MFPT数据集的样本
        
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
        
        # 获取所有子目录
        subdirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.')]
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.data_dir, subdir)
            
            # 获取目录中的所有CSV文件
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv') and not f.startswith('.')]
            
            for csv_file in csv_files:
                try:
                    file_path = os.path.join(subdir_path, csv_file)
                    df = pd.read_csv(file_path)
                    
                    # 确保振动信号列存在
                    if '振动信号' not in df.columns:
                        print(f"警告: CSV文件 {csv_file} 没有'振动信号'列")
                        continue
                    
                    # 提取振动信号和负载
                    vibration = df['振动信号'].values
                    load = df['load'].iloc[0] if 'load' in df.columns else 0.0
                    
                    # 确定故障类型
                    if 'baseline' in csv_file.lower():
                        fault_type = 'normal'
                    elif 'inner' in csv_file.lower():
                        fault_type = 'inner'
                    elif 'outer' in csv_file.lower():
                        fault_type = 'outer'
                    else:
                        print(f"警告: 无法确定文件 {csv_file} 的故障类型，跳过")
                        continue
                    
                    # 将长振动信号分割为多个样本
                    total_samples = len(vibration) // self.seq_len
                    
                    for i in range(total_samples):
                        # 提取一个完整的样本
                        sample_data = vibration[i * self.seq_len:(i + 1) * self.seq_len]
                        
                        # 标准化数据
                        if self.normalize:
                            sample_data = (sample_data - np.mean(sample_data)) / (np.std(sample_data) + 1e-8)
                        
                        # 根据data_type创建样本
                        x = torch.FloatTensor(sample_data).unsqueeze(0)  # [1, seq_len]
                        
                        if data_type == 0:  # 故障分类
                            y = self.fault_type_map[fault_type]
                            
                            # 创建样本
                            samples.append({
                                'x': x,
                                'y': torch.FloatTensor(y),
                                'metadata': {
                                    'load': load,
                                    'file': csv_file,
                                    'subdir': subdir,
                                    'segment': i,
                                    'fault_size': 0.0
                                }
                            })
                        
                        elif data_type == 1:  # 寿命预测（MFPT数据集可能不支持）
                            print("警告: MFPT数据集不支持寿命预测任务")
                    
                except Exception as e:
                    print(f"加载文件 {csv_file} 出错: {e}")
        
        print(f"MFPT数据集共加载了 {len(samples)} 个样本")
        return samples
    
    def get_dataset_info(self):
        """获取数据集信息"""
        return {
            'sample_count': len(self.samples),
            'fault_types': list(self.fault_type_map.keys()),
            'sequence_length': self.seq_len
        }


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
    
    # 确定数据目录
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/root/autodl-tmp"  # 默认目录
    
    # 创建故障分类数据集
    print("\n测试故障分类数据集")
    dataset = MFPT_Dataset(
        data_dir=data_dir,
        data_type=0,  # 故障分类
        seq_len=1024,
        normalize=True,
        dataset_train_or_val="train"
    )
    
    # 查看样本数量
    print(f"样本数量: {len(dataset)}")
    print(f"数据集信息: {dataset.get_dataset_info()}")
    
    # 获取一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"样本x形状: {sample['x'].shape}")
        print(f"样本y: {sample['y']}")
        print(f"样本元数据: {sample['metadata']}")
        
        # 绘制波形
        plt.figure(figsize=(12, 4))
        plt.plot(sample['x'][0].numpy())
        plt.title(f"故障类型: {list(dataset.fault_type_map.keys())[sample['y'].argmax().item()]}")
        plt.xlabel("采样点")
        plt.ylabel("振幅")
        plt.savefig("mfpt_sample.png")
        print("已保存波形图: mfpt_sample.png") 