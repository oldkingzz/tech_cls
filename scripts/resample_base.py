#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
轴承数据集重采样基类
提供通用的重采样功能，各数据集特定的重采样类可以继承这个基类
"""

import os
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import json
import tqdm
from abc import ABC, abstractmethod


class ResampleBase(ABC):
    """轴承数据集重采样基类"""
    
    def __init__(self, source_dir, target_dir=None, target_rate=51200):
        """
        初始化函数
        
        参数:
            source_dir (str): 源数据目录
            target_dir (str): 目标数据目录，如果为None，则使用子类的默认设置
            target_rate (int): 目标采样率，默认为51.2kHz (51200Hz)
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.target_rate = target_rate
        
        # 保存采样率信息
        self.sampling_info = {
            'target_rate': target_rate
        }
    
    def resample_data(self, data, original_fs):
        """
        对数据进行重采样
        
        参数:
            data (numpy.ndarray): 原始数据
            original_fs (int): 原始采样率
            
        返回:
            numpy.ndarray: 重采样后的数据
        """
        # 如果原始采样率已经是目标采样率，直接返回原始数据
        if original_fs == self.target_rate:
            return data
        
        # 计算重采样比例
        resample_ratio = self.target_rate / original_fs
        
        # 确定重采样后的数据长度
        target_length = int(len(data) * resample_ratio)
        
        # 使用scipy的resample函数进行重采样
        resampled_data = signal.resample(data, target_length)
        
        return resampled_data
    
    def trim_or_pad_data(self, data, target_length):
        """
        剪裁或填充数据到指定长度
        
        参数:
            data (numpy.ndarray): 输入数据
            target_length (int): 目标长度
            
        返回:
            numpy.ndarray: 处理后的数据
        """
        current_length = len(data)
        
        if current_length == target_length:
            return data
        elif current_length > target_length:
            # 剪裁数据
            return data[:target_length]
        else:
            # 填充数据
            return np.pad(data, (0, target_length - current_length))
    
    def normalize_data(self, data):
        """
        对数据进行标准化
        
        参数:
            data (numpy.ndarray): 输入数据
            
        返回:
            numpy.ndarray: 标准化后的数据
        """
        return (data - np.mean(data)) / (np.std(data) + 1e-8)
    
    def save_metadata(self, metadata, file_path=None):
        """
        保存元数据
        
        参数:
            metadata (dict): 元数据字典
            file_path (str): 保存路径，默认为target_dir/metadata.json
        """
        if file_path is None:
            file_path = os.path.join(self.target_dir, 'metadata.json')
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def plot_sample(self, csv_file, duration=0.1, output_file=None):
        """
        绘制一个样本的波形图
        
        参数:
            csv_file (str): CSV文件路径
            duration (float): 要绘制的时长（秒），默认0.1秒
            output_file (str): 输出文件路径，默认为CSV文件名_plot.png
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 提取振动信号和采样率
            if '振动信号' in df.columns:
                vibration = df['振动信号'].values
            else:
                print(f"警告: CSV文件 {csv_file} 中未找到'振动信号'列")
                return
            
            if '目标采样率' in df.columns:
                fs = df['目标采样率'].iloc[0]
            else:
                fs = self.target_rate
            
            # 提取故障类型和负载（如果存在）
            fault_type = df['故障类型'].iloc[0] if '故障类型' in df.columns else 'unknown'
            load = df['load'].iloc[0] if 'load' in df.columns else 0.0
            
            # 计算要绘制的样本点数
            n_samples = min(int(duration * fs), len(vibration))
            samples = vibration[:n_samples]
            time = np.arange(len(samples)) / fs
            
            # 绘制波形图
            plt.figure(figsize=(12, 4))
            plt.plot(time, samples)
            plt.title(f"故障类型: {fault_type}, 负载: {load}, 采样率: {fs} Hz")
            plt.xlabel("时间 (秒)")
            plt.ylabel("振幅")
            plt.grid(True)
            
            # 设置输出文件路径
            if output_file is None:
                output_file = csv_file.replace('.csv', '_plot.png')
            
            # 保存图像
            plt.savefig(output_file)
            plt.close()
            
            print(f"波形图已保存至 {output_file}")
            
        except Exception as e:
            print(f"绘制样本 {csv_file} 出错: {e}")
    
    @abstractmethod
    def process_file(self, file_path):
        """
        处理单个文件的抽象方法，子类必须实现
        
        参数:
            file_path (str): 文件路径
            
        返回:
            str: 保存的CSV文件路径，如果处理失败则返回None
        """
        pass
    
    @abstractmethod
    def process_all(self):
        """
        处理所有文件的抽象方法，子类必须实现
        
        返回:
            int: 成功处理的文件数量
        """
        pass 