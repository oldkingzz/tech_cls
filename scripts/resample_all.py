#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
轴承数据集重采样主脚本
用于方便地运行所有数据集的重采样程序
"""

import os
import argparse
import subprocess
import sys
import time


def run_resampler(script_name, source_dir, target_dir=None, target_rate=51200, plot_samples=False):
    """
    运行重采样脚本
    
    参数:
        script_name (str): 脚本名称
        source_dir (str): 源数据目录
        target_dir (str): 目标目录
        target_rate (int): 目标采样率
        plot_samples (bool): 是否绘制样本波形
    
    返回:
        int: 进程返回码
    """
    cmd = [sys.executable, script_name, source_dir]
    
    if target_dir:
        cmd.extend(['--target_dir', target_dir])
    
    cmd.extend(['--target_rate', str(target_rate)])
    
    if plot_samples:
        cmd.append('--plot_samples')
    
    print(f"运行命令: {' '.join(cmd)}")
    
    start_time = time.time()
    process = subprocess.run(cmd)
    end_time = time.time()
    
    print(f"执行时间: {end_time - start_time:.2f} 秒")
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description='轴承数据集重采样工具')
    parser.add_argument('data_dir', type=str, help='数据目录，包含所有原始数据集的根目录')
    parser.add_argument('--target_dir', type=str, default=None, help='目标目录，默认为各数据集内的子目录')
    parser.add_argument('--target_rate', type=int, default=51200, help='目标采样率（Hz），默认为51200 Hz')
    parser.add_argument('--plot_samples', action='store_true', help='是否为重采样后的数据生成样本波形图')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        choices=['mfpt', 'xjtu', 'case_western', 'all'], default=['all'], 
                        help='要处理的数据集，可以是mfpt、xjtu、case_western或all')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确定需要处理的数据集
    datasets_to_process = []
    if 'all' in args.datasets or 'mfpt' in args.datasets:
        datasets_to_process.append({
            'name': 'MFPT',
            'script': os.path.join(script_dir, 'resample_mfpt_csv.py'),
            'source_dir': os.path.join(args.data_dir, 'mfpt_csv'),
            'target_subdir': 'mfpt_resampled'
        })
    
    if 'all' in args.datasets or 'xjtu' in args.datasets:
        datasets_to_process.append({
            'name': 'XJTU',
            'script': os.path.join(script_dir, 'resample_xjtu.py'),
            'source_dir': os.path.join(args.data_dir, 'XJTU'),
            'target_subdir': 'xjtu_resampled'
        })
    
    if 'all' in args.datasets or 'case_western' in args.datasets:
        datasets_to_process.append({
            'name': 'Case Western',
            'script': os.path.join(script_dir, 'resample_case_western.py'),
            'source_dir': os.path.join(args.data_dir, '凯西储大学/processed_csv_data'),
            'target_subdir': 'case_western_resampled_fixed'
        })
    
    # 处理每个数据集
    for dataset in datasets_to_process:
        print(f"\n================ 处理 {dataset['name']} 数据集 ================")
        
        # 确定目标目录
        if args.target_dir:
            target_dir = os.path.join(args.target_dir, dataset['target_subdir'])
        else:
            target_dir = None  # 使用默认设置
        
        # 运行重采样脚本
        return_code = run_resampler(
            script_name=dataset['script'],
            source_dir=dataset['source_dir'],
            target_dir=target_dir,
            target_rate=args.target_rate,
            plot_samples=args.plot_samples
        )
        
        if return_code != 0:
            print(f"警告: {dataset['name']} 数据集处理失败，返回码: {return_code}")
    
    print("\n所有数据集处理完成!")


if __name__ == "__main__":
    main() 