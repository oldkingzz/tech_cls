"""
HUST数据集类，继承自基类dataset_yoto
"""

import os
import numpy as np
import torch
import pandas as pd
from .dataset_yoto import dataset_yoto


class HUST_Dataset(dataset_yoto):
    """HUST数据集类，继承自基类dataset_yoto"""
    
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
        # HUST数据集的采样率设置为统一标准
        sampling_rate = 51200  # 设置为统一标准51.2kHz
        
        self.data_dir = data_dir
        self.normalize = normalize
        
        # HUST数据集的故障类型映射（中文到英文）
        self.hust_fault_map = {
            '内圈故障': 'inner',
            '外圈故障': 'outer',
            '滚动体故障': 'ball',
            '正常': 'normal'
        }
        
        # 调用父类初始化方法
        super().__init__(
            data_type=data_type,
            seq_len=seq_len,
            dataset_name="HUST",
            dataset_description="华中科技大学轴承数据集",
            sampling_rate=sampling_rate,
            dataset_train_or_val=dataset_train_or_val,
            data_dir=self.data_dir
        )
    
    def _load_samples(self, data_type):
        """
        加载HUST数据集的样本
        
        参数:
            data_type (int): 数据类型，0表示故障分类，1表示寿命预测
            
        返回:
            list: 样本列表
        """
        samples = []
        
        # 检查数据目录是否存在
        # 首先检查self.data_dir/processed_data是否存在
        processed_dir = os.path.join(self.data_dir, 'processed_data')
        if not os.path.exists(processed_dir):
            # 如果不存在，尝试self.data_dir/HUST/processed_data
            processed_dir = os.path.join(self.data_dir, 'HUST', 'processed_data')
            if not os.path.exists(processed_dir):
                print(f"警告: 找不到HUST数据集目录，尝试的路径: {os.path.join(self.data_dir, 'processed_data')} 或 {processed_dir}")
                return samples
            else:
                print(f"使用HUST数据路径: {processed_dir}")
        
        # 获取所有CSV文件
        csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv') and not f.startswith('._')]
        
        # 只保留单一故障类型的文件（只有一个字母开头的文件）
        valid_files = []
        for csv_file in csv_files:
            # 检查文件名是否只由一个大写字母开头
            if csv_file[0].isalpha() and csv_file[0].isupper() and len(csv_file) > 1:
                # 检查第二个字符是否是数字（例如B500.csv）
                if csv_file[1].isdigit():
                    valid_files.append(csv_file)
                else:
                    print(f"跳过复合故障文件: {csv_file}")
        
        print(f"找到 {len(valid_files)} 个有效的单一故障类型文件")
        
        # 解析文件名确定故障类型
        for csv_file in valid_files:
            try:
                file_path = os.path.join(processed_dir, csv_file)
                df = pd.read_csv(file_path)
                
                # 从文件名确定故障类型
                fault_type = None
                if csv_file.startswith('B'):  # B开头表示Ball
                    fault_type = 'ball'
                elif csv_file.startswith('I'):  # I开头表示Inner
                    fault_type = 'inner'
                elif csv_file.startswith('O'):  # O开头表示Outer
                    fault_type = 'outer'
                elif csv_file.startswith('N'):  # N开头表示Normal
                    fault_type = 'normal'
                else:
                    print(f"警告: 无法从文件名 {csv_file} 确定故障类型，跳过")
                    continue
                
                # 确保振动信号列存在
                vibration_col = None
                for col in df.columns:
                    if '振动' in col or 'vib' in col.lower() or 'accelerate' in col.lower():
                        vibration_col = col
                        break
                
                if vibration_col is None and len(df.columns) > 0:
                    # 如果找不到明确的振动信号列，使用第一列
                    vibration_col = df.columns[0]
                    print(f"未找到明确的振动信号列，使用第一列 {vibration_col}")
                
                if vibration_col is None:
                    print(f"警告: CSV文件 {csv_file} 没有可用的振动信号列")
                    continue
                
                # 提取振动信号
                vibration = df[vibration_col].values
                
                # 提取转速信息 (如果存在)
                rpm_col = None
                for col in df.columns:
                    if 'rpm' in col.lower() or '转速' in col:
                        rpm_col = col
                        break
                
                rpm = df[rpm_col].iloc[0] if rpm_col else 0.0
                
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
                                'rpm': rpm,
                                'file': csv_file,
                                'segment': i,
                                'subdir': os.path.basename(processed_dir)  # 兼容其他数据集的元数据字段
                            }
                        })
                    
                    elif data_type == 1:  # 寿命预测（HUST数据集可能不支持）
                        print("警告: HUST数据集不支持寿命预测任务")
                
            except Exception as e:
                print(f"加载文件 {csv_file} 出错: {e}")
        
        print(f"HUST数据集共加载了 {len(samples)} 个样本")
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
    
    # 确定数据目录
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/root/autodl-tmp"  # 默认目录
    
    # 创建故障分类数据集
    print("\n测试故障分类数据集")
    dataset = HUST_Dataset(
        data_dir=data_dir,
        data_type=0,  # 故障分类
        seq_len=1024,
        normalize=True,
        dataset_train_or_val="train"
    )
    
    # 查看样本数量
    print(f"样本数量: {len(dataset)}")
    
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
        plt.savefig("hust_sample.png")
        print("已保存波形图: hust_sample.png") 