import os
import pandas as pd
import numpy as np
from scipy import signal
import glob

def resample_mfpt_data(input_dir, output_dir, target_fs=25600):
    """
    将MFPT数据集重采样到目标采样率
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        target_fs: 目标采样率（Hz）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith('.')]
    
    # MFPT数据集的原始采样率是48kHz
    current_fs = 48000
    print(f"MFPT数据集的原始采样率: {current_fs} Hz")
    
    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        print(f"处理目录: {subdir_path}")
        
        # 创建对应的输出子目录
        output_subdir = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        
        # 获取所有CSV文件
        csv_files = glob.glob(os.path.join(subdir_path, '*.csv'))
        
        for csv_file in csv_files:
            print(f"处理文件: {csv_file}")
            
            try:
                # 读取CSV文件
                df = pd.read_csv(csv_file)
                
                # 获取振动信号
                vibration = df['振动信号'].values
                
                # 计算重采样因子
                resample_factor = target_fs / current_fs
                
                # 重采样
                resampled_vibration = signal.resample(vibration, int(len(vibration) * resample_factor))
                
                # 创建新的DataFrame
                new_df = pd.DataFrame({
                    'rate': target_fs,
                    'load': df['load'].iloc[0],
                    '振动信号': resampled_vibration,
                    'sr': target_fs
                })
                
                # 保存新的CSV文件
                output_file = os.path.join(output_subdir, os.path.basename(csv_file))
                new_df.to_csv(output_file, index=False)
                print(f"已保存到: {output_file}")
                
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {str(e)}")

if __name__ == '__main__':
    # 处理MFPT数据集
    input_dir = '/root/autodl-tmp/mfpt_csv'
    output_dir = '/root/autodl-tmp/mfpt_csv/resampled_data'
    resample_mfpt_data(input_dir, output_dir) 