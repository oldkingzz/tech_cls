"""
数据集基类，所有数据集类都需要继承自这个基类
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np


class dataset_yoto(Dataset):
    """轴承数据集基类，所有的数据集类都应该继承这个类"""
    
    def __init__(self, data_type, seq_len=32768, dataset_name="", dataset_description="", 
                 sampling_rate=51200, dataset_train_or_val="train", data_dir=""):
        """
        初始化函数
        
        参数:
            data_type (int): 数据类型，0表示故障分类，1表示寿命预测
            seq_len (int): 序列长度，默认为32768（对应0.64秒@51.2kHz）
            dataset_name (str): 数据集名称
            dataset_description (str): 数据集描述
            sampling_rate (float): 采样率，默认为51.2kHz (51200Hz)
            dataset_train_or_val (str): 训练集或验证集，"train"或"val"
            data_dir (str): 数据目录路径
        """
        self.data_type = data_type
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_description = dataset_description
        self.sampling_rate = sampling_rate
        self.dataset_train_or_val = dataset_train_or_val
        self.data_dir = data_dir
        
        # 检查数据目录是否存在
        if data_dir and not os.path.exists(data_dir):
            error_msg = f"错误: 数据目录 {data_dir} 不存在"
            print(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 故障类型映射
        self.fault_type_map = {
            'normal': [1, 0, 0, 0, 0],
            'inner': [0, 1, 0, 0, 0],
            'outer': [0, 0, 1, 0, 0],
            'cage': [0, 0, 0, 1, 0],
            'ball': [0, 0, 0, 0, 1]
        }
        
        # 加载样本
        self.samples = self._load_samples(self.data_type)
    
    def __getitem__(self, index):
        """获取样本"""
        sample = self.samples[index]
        metadata = sample['metadata'].copy()  # 创建一个副本，避免修改原始数据
        
        # 确保metadata包含所有必要字段
        if 'fault_size' not in metadata:
            metadata['fault_size'] = 0.0
        if 'rpm' not in metadata:
            metadata['rpm'] = 0.0
        if 'subdir' not in metadata:
            metadata['subdir'] = ""
        if 'load' not in metadata:
            metadata['load'] = 0.0
        if 'file' not in metadata:
            metadata['file'] = ""
        if 'segment' not in metadata:
            metadata['segment'] = 0
        
        return {
            'x': sample['x'],
            'y': sample['y'],
            'metadata': metadata
        }
    
    def __len__(self):
        """获取样本数量"""
        return len(self.samples)
    
    def _load_samples(self, data_type):
        """
        加载样本的方法，子类需要实现这个方法
        
        参数:
            data_type (int): 数据类型，0表示故障分类，1表示寿命预测
            
        返回:
            list: 样本列表，每个样本包含
                x = [channel, length]
                if data_type = 0, y = [fault_type]
                elif data_type = 1, y = [rul_value]
        """
        raise NotImplementedError("子类必须实现_load_samples方法") 