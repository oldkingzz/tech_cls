#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
凯西储大学数据集重采样程序
将凯西储大学数据集从原始采样率（12 kHz或48 kHz）重采样到51.2 kHz
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


class CaseWesternResample(ResampleBase):
    """凯西储大学数据集重采样类，继承自ResampleBase"""
    
    def __init__(self, source_dir, target_dir=None, target_rate=51200):
        """
        初始化函数
        
        参数:
            source_dir (str): 源数据目录，存放原始CSV文件的位置
            target_dir (str): 目标数据目录，重采样后的CSV文件存放位置，默认为source_dir父目录下的case_western_resampled目录
            target_rate (int): 目标采样率，默认为51.2kHz (51200Hz)
        """
        # 如果未指定目标目录，则使用默认设置
        if target_dir is None:
            target_dir = os.path.join(os.path.dirname(source_dir), 'case_western_resampled')
        
        # 调用父类初始化方法
        super().__init__(source_dir, target_dir, target_rate)
        
        # 数据集中的采样率
        self.sample_rates = {
            '12k': 12000,
            '48k': 48000
        }
        
        # 中文故障类型到英文的映射
        self.fault_type_map = {
            '正常': 'normal',
            '内圈': 'inner',
            '外圈': 'outer',
            '滚动体': 'ball'
        }
        
        # 更新采样率信息
        self.sampling_info.update({
            'original_rates': list(self.sample_rates.values()),
            'description': '凯西储大学数据集包含两种采样率：12 kHz和48 kHz，已重采样为51.2 kHz'
        })
    
    def process_file(self, csv_file):
        """
        处理单个CSV文件，将其重采样并保存为新的CSV
        
        参数:
            csv_file (str): CSV文件的路径
            
        返回:
            str: 保存的CSV文件路径，如果处理失败则返回None
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 确保必要的列存在 - 寻找振动信号列
            vibration_col = None
            for col in df.columns:
                if '_DE_time' in col or '_FE_time' in col or '振动信号' in col or 'vibration' in col:
                    vibration_col = col
                    break
            
            if vibration_col is None:
                print(f"警告: CSV文件 {csv_file} 中未找到振动信号列，跳过。列名为: {df.columns.tolist()}")
                return None
            
            # 提取振动信号 - 保留所有数据点
            vibration = df[vibration_col].values
            
            # 提取采样率信息
            if '采样频率(Hz)' in df.columns:
                original_rate = df['采样频率(Hz)'].iloc[0]
            else:
                # 从路径中推断采样率
                if '12k' in csv_file:
                    original_rate = self.sample_rates['12k']
                elif '48k' in csv_file:
                    original_rate = self.sample_rates['48k']
                else:
                    print(f"警告: 无法确定文件 {csv_file} 的采样率，跳过")
                    return None
            
            # 提取故障类型 - 首先从文件路径判断
            fault_type = 'unknown'
            
            # 从目录路径和名称推断故障类型
            if 'Normal Baseline' in csv_file or 'normal' in csv_file.lower():
                fault_type = 'normal'
            elif 'Inner Race' in csv_file or '/IR' in csv_file:
                fault_type = 'inner'
            elif 'Outer Race' in csv_file or '/OR' in csv_file:
                fault_type = 'outer'
            elif 'Ball' in csv_file or '/B' in csv_file:
                fault_type = 'ball'
            
            # 如果从路径无法确定，则尝试从CSV文件内容确定
            if fault_type == 'unknown' and '故障类型' in df.columns:
                fault_type_zh = df['故障类型'].iloc[0]
                fault_type = self.fault_type_map.get(fault_type_zh, 'unknown')
            
            # 输出调试信息
            if fault_type == 'unknown':
                print(f"警告: 无法确定文件 {csv_file} 的故障类型")
            
            # 提取故障尺寸信息
            fault_size = 0.0
            if '故障尺寸(inch)' in df.columns:
                fault_size = df['故障尺寸(inch)'].iloc[0]
            
            # 提取转速信息
            rpm = 0.0
            for col in df.columns:
                if 'RPM' in col:
                    rpm = df[col].iloc[0]
                    break
            
            # 重采样数据 - 保留全部数据
            resampled_data = self.resample_data(vibration, original_rate)
            
            # 创建存储目录结构
            subdir = os.path.join(self.target_dir, fault_type)
            os.makedirs(subdir, exist_ok=True)
            
            # 保存文件名: 保留原始文件名但添加前缀表示位置(DE/FE)和采样率
            file_name = os.path.basename(csv_file)
            if '12k Drive End' in csv_file:
                prefix = 'DE12k_'
            elif '12k Fan End' in csv_file:
                prefix = 'FE12k_'
            elif '48k Drive End' in csv_file:
                prefix = 'DE48k_'
            else:
                prefix = ''
            
            csv_out_file = os.path.join(subdir, prefix + file_name)
            
            # 创建DataFrame
            df_out = pd.DataFrame({
                '振动信号': resampled_data,
                '故障类型': fault_type,
                '故障尺寸(inch)': fault_size,
                'RPM': rpm,
                '原始采样率': original_rate,
                '目标采样率': self.target_rate
            })
            
            # 保存CSV
            df_out.to_csv(csv_out_file, index=False)
            
            print(f"处理文件 {csv_file}，重采样后数据点数: {len(resampled_data)}")
            
            return csv_out_file
            
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
            return None
    
    def process_all(self):
        """
        处理所有CSV文件，将其重采样并保存
        
        返回:
            int: 成功处理的文件数量
        """
        # 确保目标目录存在
        os.makedirs(self.target_dir, exist_ok=True)
        
        # 获取所有CSV文件
        csv_files = []
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith('.csv') and not file.startswith('._'):  # 排除隐藏文件
                    csv_files.append(os.path.join(root, file))
        
        # 保存数据集信息
        metadata = {
            'dataset_name': 'Case Western',
            'dataset_description': '凯西储大学轴承数据集',
            'sampling_rates': self.sampling_info,
            'file_count': len(csv_files),
            'fault_types': ['normal', 'inner', 'outer', 'ball'],
            'process_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 使用父类的方法保存元数据
        self.save_metadata(metadata)
        
        # 使用tqdm显示进度
        successful_count = 0
        for csv_file in tqdm.tqdm(csv_files, desc="处理凯西储大学数据集"):
            result = self.process_file(csv_file)
            if result is not None:
                successful_count += 1
        
        print(f"成功处理 {successful_count}/{len(csv_files)} 个文件")
        return successful_count


def main():
    parser = argparse.ArgumentParser(description='凯西储大学数据集重采样工具')
    parser.add_argument('source_dir', type=str, help='源数据目录，包含原始CSV文件')
    parser.add_argument('--target_dir', type=str, default=None, help='目标数据目录，默认为source_dir父目录下的case_western_resampled目录')
    parser.add_argument('--target_rate', type=int, default=51200, help='目标采样率（Hz），默认为51200 Hz')
    parser.add_argument('--plot_samples', action='store_true', help='是否为每个故障类型生成样本波形图')
    
    args = parser.parse_args()
    
    # 创建重采样器
    resampler = CaseWesternResample(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        target_rate=args.target_rate
    )
    
    # 处理所有文件
    successful_count = resampler.process_all()
    
    # 如果需要，绘制样本波形
    if args.plot_samples and successful_count > 0:
        print("正在生成样本波形图...")
        
        # 为每个故障类型的第一个CSV文件生成波形图
        for fault_type in ['normal', 'inner', 'outer', 'ball']:
            fault_dir = os.path.join(resampler.target_dir, fault_type)
            if os.path.exists(fault_dir):
                csv_files = [f for f in os.listdir(fault_dir) if f.endswith('.csv')]
                if csv_files:
                    resampler.plot_sample(os.path.join(fault_dir, csv_files[0]))
    
    print(f"重采样完成。数据已保存至 {resampler.target_dir}")


if __name__ == "__main__":
    main() 