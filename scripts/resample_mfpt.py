#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MFPT数据集重采样程序
将MFPT数据集从原始采样率（97,656 Hz或48,828 Hz）重采样到51,200 Hz
"""

import os
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import loadmat
import argparse
import tqdm
import shutil
import json
from resample_base import ResampleBase


class MFPTResample(ResampleBase):
    """MFPT数据集重采样类，继承自ResampleBase"""
    
    def __init__(self, source_dir, target_dir=None, target_rate=51200):
        """
        初始化函数
        
        参数:
            source_dir (str): 源数据目录，存放原始.mat文件的位置
            target_dir (str): 目标数据目录，重采样后的CSV文件存放位置，默认为source_dir下的mfpt_csv目录
            target_rate (int): 目标采样率，默认为51.2kHz (51200Hz)
        """
        # 如果未指定目标目录，则使用默认设置
        if target_dir is None:
            target_dir = os.path.join(source_dir, 'mfpt_csv')
        
        # 调用父类初始化方法
        super().__init__(source_dir, target_dir, target_rate)
        
        # 每种文件类型对应的采样率和标签信息
        self.file_info = {
            'baseline': {'fs': 97656, 'fault_type': 'normal', 'load': 270.0},
            'OuterRaceFault': {'fs': 97656, 'fault_type': 'outer', 'load': 270.0},
            '1': {'fs': 48828, 'fault_type': 'outer', 'load': 25.0},
            '2': {'fs': 48828, 'fault_type': 'outer', 'load': 50.0},
            '3': {'fs': 48828, 'fault_type': 'outer', 'load': 100.0},
            '4': {'fs': 48828, 'fault_type': 'outer', 'load': 150.0},
            '5': {'fs': 48828, 'fault_type': 'outer', 'load': 200.0},
            '6': {'fs': 48828, 'fault_type': 'outer', 'load': 250.0},
            '7': {'fs': 48828, 'fault_type': 'outer', 'load': 300.0},
            'InnerRaceFault_vload_1': {'fs': 48828, 'fault_type': 'inner', 'load': 0.0},
            'InnerRaceFault_vload_2': {'fs': 48828, 'fault_type': 'inner', 'load': 50.0},
            'InnerRaceFault_vload_3': {'fs': 48828, 'fault_type': 'inner', 'load': 100.0},
            'InnerRaceFault_vload_4': {'fs': 48828, 'fault_type': 'inner', 'load': 150.0},
            'InnerRaceFault_vload_5': {'fs': 48828, 'fault_type': 'inner', 'load': 200.0},
            'InnerRaceFault_vload_6': {'fs': 48828, 'fault_type': 'inner', 'load': 250.0},
            'InnerRaceFault_vload_7': {'fs': 48828, 'fault_type': 'inner', 'load': 300.0}
        }
        
        # 更新采样率信息
        self.sampling_info.update({
            'original_rates': [97656, 48828],
            'description': 'MFPT数据集包含两种采样率：97,656 Hz和48,828 Hz，已重采样为51,200 Hz'
        })
    
    def resample_data(self, data, original_fs):
        """
        对数据进行重采样
        
        参数:
            data (numpy.ndarray): 原始数据
            original_fs (int): 原始采样率
            
        返回:
            numpy.ndarray: 重采样后的数据
        """
        # 计算重采样比例
        resample_ratio = self.target_rate / original_fs
        
        # 确定重采样后的数据长度
        target_length = int(len(data) * resample_ratio)
        
        # 使用scipy的resample函数进行重采样
        resampled_data = signal.resample(data, target_length)
        
        return resampled_data
        
    def process_file(self, mat_file):
        """
        处理单个.mat文件，将其重采样并保存为CSV
        
        参数:
            mat_file (str): .mat文件的路径
            
        返回:
            str: 保存的CSV文件路径，如果处理失败则返回None
        """
        try:
            # 加载.mat文件
            mat_data = loadmat(mat_file)
            
            # 提取文件名（不包括扩展名）
            base_name = os.path.basename(mat_file).replace('.mat', '')
            
            # 检查是否为已知的文件类型
            file_type = None
            for key in self.file_info.keys():
                if key in base_name:
                    file_type = key
                    break
            
            if file_type is None:
                print(f"警告: 无法确定文件 {mat_file} 的类型，跳过")
                return None
            
            # 获取原始采样率和标签信息
            original_fs = self.file_info[file_type]['fs']
            fault_type = self.file_info[file_type]['fault_type']
            load = self.file_info[file_type]['load']
            
            # 提取振动数据
            # MFPT数据集中，数据通常存储在变量名为'bearing'或以'bearing'开头的变量中
            vibration_data = None
            for key in mat_data.keys():
                if key.startswith('bearing') or key == 'bearing':
                    vibration_data = mat_data[key].ravel()
                    break
            
            if vibration_data is None:
                print(f"警告: 在文件 {mat_file} 中未找到振动数据，跳过")
                return None
            
            # 进行重采样（使用父类的方法）
            resampled_data = self.resample_data(vibration_data, original_fs)
            
            # 创建子目录
            subdir = os.path.join(self.target_dir, fault_type)
            os.makedirs(subdir, exist_ok=True)
            
            # 保存为CSV文件，包含重采样后的振动数据和元数据
            csv_file = os.path.join(subdir, f"{base_name}.csv")
            
            # 创建DataFrame
            df = pd.DataFrame({
                '振动信号': resampled_data,
                '故障类型': fault_type,
                'load': load,
                '原始采样率': original_fs,
                '目标采样率': self.target_rate
            })
            
            # 保存CSV
            df.to_csv(csv_file, index=False)
            
            return csv_file
            
        except Exception as e:
            print(f"处理文件 {mat_file} 时出错: {e}")
            return None
    
    def process_all(self):
        """
        处理所有.mat文件，将其重采样并保存为CSV
        
        返回:
            int: 成功处理的文件数量
        """
        # 确保目标目录存在
        os.makedirs(self.target_dir, exist_ok=True)
        
        # 获取所有.mat文件
        mat_files = []
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
        
        # 保存数据集信息
        metadata = {
            'dataset_name': 'MFPT',
            'dataset_description': '机械故障预防技术学会轴承数据集',
            'sampling_rates': self.sampling_info,
            'file_count': len(mat_files),
            'fault_types': ['normal', 'inner', 'outer'],
            'process_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 使用父类的方法保存元数据
        self.save_metadata(metadata)
        
        # 使用tqdm显示进度
        successful_count = 0
        for mat_file in tqdm.tqdm(mat_files, desc="处理MFPT数据集"):
            result = self.process_file(mat_file)
            if result is not None:
                successful_count += 1
        
        print(f"成功处理 {successful_count}/{len(mat_files)} 个文件")
        return successful_count
    
    def plot_sample(self, csv_file, duration=0.1):
        """
        绘制一个样本的波形图
        
        参数:
            csv_file (str): CSV文件路径
            duration (float): 要绘制的时长（秒），默认0.1秒
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 提取振动信号和采样率
            vibration = df['振动信号'].values
            fs = df['目标采样率'].iloc[0]
            fault_type = df['故障类型'].iloc[0]
            load = df['load'].iloc[0]
            
            # 计算要绘制的样本点数
            n_samples = int(duration * fs)
            samples = vibration[:n_samples]
            time = np.arange(len(samples)) / fs
            
            # 绘制波形图
            plt.figure(figsize=(12, 4))
            plt.plot(time, samples)
            plt.title(f"故障类型: {fault_type}, 负载: {load} lbs, 采样率: {fs} Hz")
            plt.xlabel("时间 (秒)")
            plt.ylabel("振幅")
            plt.grid(True)
            
            # 保存图像
            output_file = csv_file.replace('.csv', '_plot.png')
            plt.savefig(output_file)
            plt.close()
            
            print(f"波形图已保存至 {output_file}")
            
        except Exception as e:
            print(f"绘制样本 {csv_file} 出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='MFPT数据集重采样工具')
    parser.add_argument('source_dir', type=str, help='源数据目录，包含原始.mat文件')
    parser.add_argument('--target_dir', type=str, default=None, help='目标数据目录，默认为source_dir下的mfpt_csv目录')
    parser.add_argument('--target_rate', type=int, default=51200, help='目标采样率（Hz），默认为51200 Hz')
    parser.add_argument('--plot_samples', action='store_true', help='是否为每个重采样后的文件生成波形图')
    
    args = parser.parse_args()
    
    # 创建重采样器
    resampler = MFPTResample(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        target_rate=args.target_rate
    )
    
    # 处理所有文件
    successful_count = resampler.process_all()
    
    # 如果需要，绘制样本波形
    if args.plot_samples and successful_count > 0:
        print("正在生成样本波形图...")
        
        # 为每个子目录的第一个CSV文件生成波形图
        for root, _, files in os.walk(resampler.target_dir):
            csv_files = [f for f in files if f.endswith('.csv') and f != 'metadata.csv']
            if csv_files:
                resampler.plot_sample(os.path.join(root, csv_files[0]))
    
    print(f"重采样完成。数据已保存至 {resampler.target_dir}")


if __name__ == "__main__":
    main() 