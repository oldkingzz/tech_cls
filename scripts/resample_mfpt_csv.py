#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MFPT CSV数据集重采样程序
将已存在的MFPT CSV数据从原始采样率（97,656 Hz或48,828 Hz）重采样到51,200 Hz
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


class MFPTCSVResample(ResampleBase):
    """MFPT CSV数据集重采样类，继承自ResampleBase"""
    
    def __init__(self, source_dir, target_dir=None, target_rate=51200):
        """
        初始化函数
        
        参数:
            source_dir (str): 源数据目录，存放原始CSV文件的位置
            target_dir (str): 目标数据目录，重采样后的CSV文件存放位置，默认为source_dir下的mfpt_resampled目录
            target_rate (int): 目标采样率，默认为51.2kHz (51200Hz)
        """
        # 如果未指定目标目录，则使用默认设置
        if target_dir is None:
            target_dir = os.path.join(os.path.dirname(source_dir), 'mfpt_resampled')
        
        # 调用父类初始化方法
        super().__init__(source_dir, target_dir, target_rate)
        
        # 可能的采样率
        self.possible_rates = [97656, 48828]
        
        # 更新采样率信息
        self.sampling_info.update({
            'original_rates': self.possible_rates,
            'description': 'MFPT数据集包含两种采样率：97,656 Hz和48,828 Hz，已重采样为51,200 Hz'
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
            
            # 确保必要的列存在
            required_columns = ['振动信号', 'sr']
            for col in required_columns:
                if col not in df.columns:
                    print(f"警告: CSV文件 {csv_file} 缺少必要的列 {col}，跳过")
                    return None
            
            # 提取振动信号和采样率
            vibration = df['振动信号'].values
            original_rate = df['sr'].iloc[0]
            
            # 如果原始采样率不在可能的列表中，打印警告
            if original_rate not in self.possible_rates:
                print(f"警告: CSV文件 {csv_file} 的采样率 {original_rate} 不在预期列表中")
            
            # 提取其他信息
            load = df['load'].iloc[0] if 'load' in df.columns else 0.0
            rate = df['rate'].iloc[0] if 'rate' in df.columns else 0.0
            
            # 确定故障类型
            fault_type = 'unknown'
            file_name = os.path.basename(csv_file)
            dir_name = os.path.basename(os.path.dirname(csv_file))
            
            if 'OuterRaceFault' in file_name or 'outer' in file_name.lower():
                fault_type = 'outer'
            elif 'InnerRaceFault' in file_name or 'inner' in file_name.lower():
                fault_type = 'inner'
            elif 'baseline' in file_name.lower() or 'normal' in file_name.lower():
                fault_type = 'normal'
            
            # 重采样数据
            resampled_data = self.resample_data(vibration, original_rate)
            
            # 创建存储目录结构
            # 保持与原目录结构相似，但将所有文件按故障类型分组
            subdir = os.path.join(self.target_dir, fault_type)
            os.makedirs(subdir, exist_ok=True)
            
            # 保存为CSV文件
            new_file_name = f"{dir_name.replace(' - ', '_')}_{file_name}"
            if len(new_file_name) > 100:  # 防止文件名过长
                new_file_name = file_name
            
            csv_out_file = os.path.join(subdir, new_file_name)
            
            # 创建DataFrame
            df_out = pd.DataFrame({
                '振动信号': resampled_data,
                '故障类型': fault_type,
                'load': load,
                'rate': rate,
                '原始采样率': original_rate,
                '目标采样率': self.target_rate
            })
            
            # 保存CSV
            df_out.to_csv(csv_out_file, index=False)
            
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
            'dataset_name': 'MFPT',
            'dataset_description': '机械故障预防技术学会轴承数据集',
            'sampling_rates': self.sampling_info,
            'file_count': len(csv_files),
            'fault_types': ['normal', 'inner', 'outer'],
            'process_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 使用父类的方法保存元数据
        self.save_metadata(metadata)
        
        # 使用tqdm显示进度
        successful_count = 0
        for csv_file in tqdm.tqdm(csv_files, desc="处理MFPT CSV数据集"):
            result = self.process_file(csv_file)
            if result is not None:
                successful_count += 1
        
        print(f"成功处理 {successful_count}/{len(csv_files)} 个文件")
        return successful_count


def main():
    parser = argparse.ArgumentParser(description='MFPT CSV数据集重采样工具')
    parser.add_argument('source_dir', type=str, help='源数据目录，包含原始CSV文件')
    parser.add_argument('--target_dir', type=str, default=None, help='目标数据目录，默认为source_dir父目录下的mfpt_resampled目录')
    parser.add_argument('--target_rate', type=int, default=51200, help='目标采样率（Hz），默认为51200 Hz')
    parser.add_argument('--plot_samples', action='store_true', help='是否为每个故障类型生成样本波形图')
    
    args = parser.parse_args()
    
    # 创建重采样器
    resampler = MFPTCSVResample(
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
        for fault_type in ['normal', 'inner', 'outer']:
            fault_dir = os.path.join(resampler.target_dir, fault_type)
            if os.path.exists(fault_dir):
                csv_files = [f for f in os.listdir(fault_dir) if f.endswith('.csv')]
                if csv_files:
                    resampler.plot_sample(os.path.join(fault_dir, csv_files[0]))
    
    print(f"重采样完成。数据已保存至 {resampler.target_dir}")


if __name__ == "__main__":
    main() 