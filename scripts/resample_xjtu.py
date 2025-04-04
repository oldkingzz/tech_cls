#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XJTU数据集重采样程序
将XJTU数据集从原始采样率（25.6 kHz）重采样到51.2 kHz
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tqdm
import json
import glob
from resample_base import ResampleBase


class XJTUResample(ResampleBase):
    """XJTU数据集重采样类，继承自ResampleBase"""
    
    def __init__(self, source_dir, target_dir=None, target_rate=51200):
        """
        初始化函数
        
        参数:
            source_dir (str): 源数据目录，存放原始数据文件的位置
            target_dir (str): 目标数据目录，重采样后的CSV文件存放位置，默认为source_dir下的xjtu_csv目录
            target_rate (int): 目标采样率，默认为51.2kHz (51200Hz)
        """
        # 如果未指定目标目录，则使用默认设置
        if target_dir is None:
            target_dir = os.path.join(source_dir, 'xjtu_csv')
        
        # 调用父类初始化方法
        super().__init__(source_dir, target_dir, target_rate)
        
        # XJTU数据集的原始采样率是25.6 kHz
        self.original_rate = 25600
        
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
        
        # 更新采样率信息
        self.sampling_info.update({
            'original_rate': self.original_rate,
            'description': 'XJTU数据集原始采样率为25.6 kHz，已重采样为51.2 kHz'
        })
    
    def process_file(self, file_path):
        """
        处理单个数据文件，将其重采样并保存为CSV
        
        参数:
            file_path (str): 数据文件路径
            
        返回:
            str: 保存的CSV文件路径，如果处理失败则返回None
        """
        try:
            # 提取文件名和路径信息
            file_name = os.path.basename(file_path)
            dir_name = os.path.basename(os.path.dirname(file_path))
            
            # 提取上一级目录信息（工况目录，如35Hz12kN）
            condition_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            
            # 检查轴承故障类型是否已知
            if dir_name not in self.bearing_fault_map:
                print(f"警告: 未知轴承类型 {dir_name}，跳过")
                return None
            
            fault_type = self.bearing_fault_map[dir_name]
            
            # 读取数据文件（CSV格式）
            df = pd.read_csv(file_path)
            
            # 确保水平振动信号列存在
            if 'Horizontal_vibration_signals' not in df.columns:
                print(f"警告: 文件 {file_path} 中没有水平振动信号列，跳过")
                return None
            
            # 提取振动信号和时间信息
            vibration = df['Horizontal_vibration_signals'].values
            
            # 尝试从工况目录中提取信息
            try:
                freq = float(condition_dir.split('Hz')[0])
                load = float(condition_dir.split('Hz')[1].split('kN')[0])
            except (ValueError, IndexError):
                print(f"警告: 无法从目录 {condition_dir} 中提取工况信息，使用默认值")
                freq = 0.0
                load = 0.0
            
            # 重采样数据
            resampled_data = self.resample_data(vibration, self.original_rate)
            
            # 创建保存目录结构
            subdir = os.path.join(self.target_dir, condition_dir, dir_name)
            os.makedirs(subdir, exist_ok=True)
            
            # 保存为CSV文件
            csv_file = os.path.join(subdir, file_name)
            
            # 创建DataFrame
            df_out = pd.DataFrame({
                '振动信号': resampled_data,
                '故障类型': fault_type,
                '频率(Hz)': freq,
                '负载(kN)': load,
                '原始采样率': self.original_rate,
                '目标采样率': self.target_rate
            })
            
            # 保存CSV
            df_out.to_csv(csv_file, index=False)
            
            return csv_file
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return None
    
    def process_all(self):
        """
        处理所有数据文件，将其重采样并保存为CSV
        
        返回:
            int: 成功处理的文件数量
        """
        # 确保目标目录存在
        os.makedirs(self.target_dir, exist_ok=True)
        
        # 获取所有需要处理的文件
        # XJTU数据集的目录结构为: source_dir/工况(如35Hz12kN)/轴承ID/文件.csv
        data_files = []
        for condition_dir in os.listdir(self.source_dir):
            condition_path = os.path.join(self.source_dir, condition_dir)
            if not os.path.isdir(condition_path):
                continue
                
            for bearing_dir in os.listdir(condition_path):
                bearing_path = os.path.join(condition_path, bearing_dir)
                if not os.path.isdir(bearing_path) or bearing_dir not in self.bearing_fault_map:
                    continue
                    
                for file in os.listdir(bearing_path):
                    if file.endswith('.csv'):
                        data_files.append(os.path.join(bearing_path, file))
        
        # 保存数据集信息
        metadata = {
            'dataset_name': 'XJTU',
            'dataset_description': '西安交通大学轴承数据集',
            'sampling_rates': self.sampling_info,
            'file_count': len(data_files),
            'fault_types': list(set(self.bearing_fault_map.values())),
            'process_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 使用父类的方法保存元数据
        self.save_metadata(metadata)
        
        # 使用tqdm显示进度
        successful_count = 0
        for data_file in tqdm.tqdm(data_files, desc="处理XJTU数据集"):
            result = self.process_file(data_file)
            if result is not None:
                successful_count += 1
        
        print(f"成功处理 {successful_count}/{len(data_files)} 个文件")
        return successful_count


def main():
    parser = argparse.ArgumentParser(description='XJTU数据集重采样工具')
    parser.add_argument('source_dir', type=str, help='源数据目录，包含原始数据文件')
    parser.add_argument('--target_dir', type=str, default=None, help='目标数据目录，默认为source_dir下的xjtu_csv目录')
    parser.add_argument('--target_rate', type=int, default=51200, help='目标采样率（Hz），默认为51200 Hz')
    parser.add_argument('--plot_samples', action='store_true', help='是否为每个工况和轴承类型生成样本波形图')
    
    args = parser.parse_args()
    
    # 创建重采样器
    resampler = XJTUResample(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        target_rate=args.target_rate
    )
    
    # 处理所有文件
    successful_count = resampler.process_all()
    
    # 如果需要，绘制样本波形
    if args.plot_samples and successful_count > 0:
        print("正在生成样本波形图...")
        
        # 为每个工况和轴承类型的第一个CSV文件生成波形图
        plotted = set()
        for root, _, files in os.walk(resampler.target_dir):
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files and os.path.basename(root) in resampler.bearing_fault_map:
                bearing_type = os.path.basename(root)
                condition = os.path.basename(os.path.dirname(root))
                key = f"{condition}_{bearing_type}"
                
                if key not in plotted:
                    resampler.plot_sample(os.path.join(root, csv_files[0]))
                    plotted.add(key)
    
    print(f"重采样完成。数据已保存至 {resampler.target_dir}")


if __name__ == "__main__":
    main() 