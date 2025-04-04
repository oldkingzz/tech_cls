"""
数据工厂模块，用于统一创建各种数据集的数据加载器
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from collections import Counter
import os
import pandas as pd

from .dataset_yoto import dataset_yoto
from .xjtu_dataset import XJTU_Dataset
from .hust_dataset import HUST_Dataset
from .case_dataset import Case_Dataset
from .mfpt_dataset import MFPT_Dataset
from .dataset_ottawa import dataset_ottawa


class DataFactory:
    """
    数据工厂类，用于创建和管理数据加载器
    
    该类提供了统一的接口来创建不同数据集的数据加载器，并支持配置各种参数
    """
    
    def __init__(self, data_dir: str, batch_size: int = 32, normalize: bool = True, 
                max_seq_len: Optional[int] = None, use_all_samples: bool = False,
                num_workers: int = 4, pin_memory: bool = True,
                train_ratio: float = 1.0, val_ratio: float = 1.0,
                balance_classes: bool = False, max_samples_per_class: Optional[int] = None):
        """
        初始化数据工厂
        
        参数:
            data_dir (str): 数据目录路径
            batch_size (int): 批次大小
            normalize (bool): 是否标准化特征
            max_seq_len (int, optional): 最大序列长度，用于截断过长序列
            use_all_samples (bool): 是否使用所有样本（对于XJTU等大型数据集）
            num_workers (int): 数据加载的工作进程数
            pin_memory (bool): 是否将数据固定在内存中，用于加速GPU训练
            train_ratio (float): 训练集采样比例，范围[0,1]
            val_ratio (float): 验证集采样比例，范围[0,1]
            balance_classes (bool): 是否平衡类别分布
            max_samples_per_class (int, optional): 每个类别的最大样本数，为None时不限制
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_seq_len = max_seq_len
        self.use_all_samples = use_all_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.balance_classes = balance_classes
        self.max_samples_per_class = max_samples_per_class
        
        # 创建数据集实例
        self.train_dataset = None
        self.val_dataset = None
        
        # 数据集配置字典，用于存储不同数据集的配置参数
        self.dataset_configs = {
            "XJTU": {
                "class": XJTU_Dataset,
                "params": {
                    "data_dir": f"{self.data_dir}/xjtu_resampled",
                    "data_type": 0,  # 默认为故障分类
                    "seq_len": self.max_seq_len if self.max_seq_len else 32768,
                    "normalize": self.normalize,
                    "use_all_samples": self.use_all_samples
                }
            },
            "HUST": {
                "class": HUST_Dataset,
                "params": {
                    "data_dir": self.data_dir,
                    "data_type": 0,
                    "seq_len": self.max_seq_len if self.max_seq_len else 32768,
                    "normalize": self.normalize
                }
            },
            "CASE": {
                "class": Case_Dataset,
                "params": {
                    "data_dir": f"{self.data_dir}/case_western_resampled_fixed",
                    "data_type": 0,
                    "seq_len": self.max_seq_len if self.max_seq_len else 32768,
                    "normalize": self.normalize
                }
            },
            "MFPT": {
                "class": MFPT_Dataset,
                "params": {
                    "data_dir": f"{self.data_dir}/mfpt_resampled",
                    "data_type": 0,
                    "seq_len": self.max_seq_len if self.max_seq_len else 32768,
                    "normalize": self.normalize
                }
            },
            "OTTAWA": {
                "class": dataset_ottawa,
                "params": {
                    "data_dir": f"{self.data_dir}/ottawa_resampled",
                    "data_type": 0,
                    "seq_len": self.max_seq_len if self.max_seq_len else 32768,
                    "normalize": self.normalize
                }
            }
        }
    
    def setup(self, train_datasets: List[str], val_datasets: List[str], data_type: int = 0):
        """
        设置训练集和验证集
        
        参数:
            train_datasets (List[str]): 用于训练的数据集列表，例如 ["XJTU", "HUST"]
            val_datasets (List[str]): 用于验证的数据集列表，例如 ["CASE"]
            data_type (int): 数据类型，0表示故障分类，1表示寿命预测
        """
        # 创建训练集
        train_datasets_list = []
        for dataset_name in train_datasets:
            if dataset_name in self.dataset_configs:
                config = self.dataset_configs[dataset_name]
                params = config["params"].copy()
                params["data_type"] = data_type
                params["dataset_train_or_val"] = "train"
                
                dataset = config["class"](**params)
                train_datasets_list.append(dataset)
                print(f"已加载训练集: {dataset_name}, 样本数: {len(dataset)}")
            else:
                print(f"警告: 数据集 {dataset_name} 未定义在配置中")
        
        # 创建验证集
        val_datasets_list = []
        for dataset_name in val_datasets:
            if dataset_name in self.dataset_configs:
                config = self.dataset_configs[dataset_name]
                params = config["params"].copy()
                params["data_type"] = data_type
                params["dataset_train_or_val"] = "val"
                
                dataset = config["class"](**params)
                val_datasets_list.append(dataset)
                print(f"已加载验证集: {dataset_name}, 样本数: {len(dataset)}")
            else:
                print(f"警告: 数据集 {dataset_name} 未定义在配置中")
        
        # 合并数据集
        if train_datasets_list:
            if len(train_datasets_list) == 1:
                self.train_dataset = train_datasets_list[0]
            else:
                # 合并多个数据集
                self.train_dataset = self._combine_datasets(train_datasets_list)
        
        if val_datasets_list:
            if len(val_datasets_list) == 1:
                self.val_dataset = val_datasets_list[0]
            else:
                # 合并多个数据集
                self.val_dataset = self._combine_datasets(val_datasets_list)
        
        # 过滤数据集，只保留inner和outer两种类型
        if self.train_dataset and data_type == 0:
            # 获取标签映射
            label_map = self.train_dataset.fault_type_map
            inner_label = None
            outer_label = None
            
            # 找到inner和outer对应的标签索引
            for label, one_hot in label_map.items():
                if 'inner' in label.lower():
                    inner_label = one_hot
                elif 'outer' in label.lower():
                    outer_label = one_hot
            
            if inner_label is None or outer_label is None:
                print("警告: 未找到inner或outer标签")
                return
            
            # 过滤训练集
            filtered_samples = []
            for sample in self.train_dataset.samples:
                label = sample['y']
                if torch.equal(label, torch.tensor(inner_label)) or torch.equal(label, torch.tensor(outer_label)):
                    filtered_samples.append(sample)
            
            original_count = len(self.train_dataset.samples)
            self.train_dataset.samples = filtered_samples
            print(f"训练集样本数量：过滤前 {original_count}，过滤后 {len(filtered_samples)}，只保留inner和outer类型")
            
            # 更新标签映射，只保留inner和outer
            new_label_map = {}
            for label, one_hot in label_map.items():
                if 'inner' in label.lower() or 'outer' in label.lower():
                    new_label_map[label] = one_hot
            self.train_dataset.fault_type_map = new_label_map
            
            # 过滤验证集
            if self.val_dataset:
                filtered_samples = []
                for sample in self.val_dataset.samples:
                    label = sample['y']
                    if torch.equal(label, torch.tensor(inner_label)) or torch.equal(label, torch.tensor(outer_label)):
                        filtered_samples.append(sample)
                
                original_count = len(self.val_dataset.samples)
                self.val_dataset.samples = filtered_samples
                print(f"验证集样本数量：过滤前 {original_count}，过滤后 {len(filtered_samples)}，只保留inner和outer类型")
                
                # 更新验证集的标签映射
                self.val_dataset.fault_type_map = new_label_map
        
        # 如果需要，应用类别平衡
        if self.balance_classes and self.train_dataset:
            self.train_dataset = self._balance_dataset_classes(self.train_dataset)
        
        print(f"训练集样本数: {len(self.train_dataset) if self.train_dataset else 0}")
        print(f"验证集样本数: {len(self.val_dataset) if self.val_dataset else 0}")
    
    def _combine_datasets(self, datasets: List[dataset_yoto]) -> dataset_yoto:
        """
        合并多个数据集
        
        参数:
            datasets (List[dataset_yoto]): 要合并的数据集列表
            
        返回:
            dataset_yoto: 合并后的数据集
        """
        # 合并样本
        combined_samples = []
        for dataset in datasets:
            combined_samples.extend(dataset.samples)
        
        # 创建一个新的数据集实例，复制第一个数据集的参数
        combined_dataset = datasets[0].__class__(
            data_dir=datasets[0].data_dir,
            data_type=datasets[0].data_type,
            seq_len=datasets[0].seq_len,
            normalize=False,  # 已经标准化过了
            dataset_train_or_val=datasets[0].dataset_train_or_val
        )
        
        # 根据数据集类型应用不同的采样比例
        ratio = self.train_ratio if combined_dataset.dataset_train_or_val == "train" else self.val_ratio
        
        # 应用采样比例
        if ratio < 1.0:
            # 确保随机性，但每次运行结果一致
            np.random.seed(42)
            sample_indices = np.random.permutation(len(combined_samples))
            sample_count = int(len(combined_samples) * ratio)
            selected_indices = sample_indices[:sample_count]
            selected_samples = [combined_samples[i] for i in selected_indices]
            print(f"应用采样比例 {ratio:.2f}，从 {len(combined_samples)} 个样本中选择了 {len(selected_samples)} 个样本")
            combined_samples = selected_samples
        
        # 替换样本
        combined_dataset.samples = combined_samples
        
        return combined_dataset
    
    def _balance_dataset_classes(self, dataset: dataset_yoto) -> dataset_yoto:
        """
        平衡数据集类别分布
        
        参数:
            dataset (dataset_yoto): 需要平衡的数据集
            
        返回:
            dataset_yoto: 平衡后的数据集
        """
        # 获取样本的标签
        samples = dataset.samples
        
        # 计算每个类别的样本索引
        class_indices = {}
        for i, sample in enumerate(samples):
            label = sample['y'].argmax().item()  # 获取类别索引
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        
        # 统计每个类别的样本数量
        class_counts = {label: len(indices) for label, indices in class_indices.items()}
        print(f"原始类别分布: {class_counts}")
        
        # 确定采样数量
        if self.max_samples_per_class:
            samples_per_class = min(self.max_samples_per_class, min(class_counts.values()))
        else:
            samples_per_class = min(class_counts.values())
        
        print(f"平衡后每个类别的样本数: {samples_per_class}")
        
        # 对每个类别进行采样
        balanced_indices = []
        for label, indices in class_indices.items():
            if len(indices) > samples_per_class:
                # 随机采样
                np.random.seed(42 + label)  # 确保每个类别的随机性不同但可重现
                selected = np.random.choice(indices, samples_per_class, replace=False)
                balanced_indices.extend(selected)
            else:
                # 如果样本数少于目标数量，保留所有样本
                balanced_indices.extend(indices)
        
        # 创建新的数据集
        balanced_samples = [samples[i] for i in balanced_indices]
        print(f"平衡前样本数: {len(samples)}, 平衡后样本数: {len(balanced_samples)}")
        
        # 更新数据集的样本
        balanced_dataset = dataset
        balanced_dataset.samples = balanced_samples
        
        return balanced_dataset
    
    def collate_fn(self, batch):
        """
        自定义的collate_fn函数，用于处理数据批次
        """
        # 提取batch中的各个字段
        x = torch.stack([item['x'] for item in batch])
        y = torch.stack([item['y'] for item in batch])
        metadata = [item['metadata'] for item in batch]
        
        return {
            'x': x,
            'y': y,
            'metadata': metadata
        }
    
    def get_train_dataloader(self) -> DataLoader:
        """
        获取训练数据加载器
        
        返回:
            DataLoader: 训练数据加载器
        """
        if self.train_dataset is None:
            raise ValueError("训练数据集未初始化，请先调用setup方法")
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
    
    def get_val_dataloader(self) -> DataLoader:
        """
        获取验证数据加载器
        
        返回:
            DataLoader: 验证数据加载器
        """
        if self.val_dataset is None:
            raise ValueError("验证数据集未初始化，请先调用setup方法")
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
    
    def add_dataset_config(self, name: str, dataset_class, params: Dict[str, Any]):
        """
        添加新的数据集配置
        
        参数:
            name (str): 数据集名称
            dataset_class: 数据集类
            params (Dict[str, Any]): 数据集参数
        """
        self.dataset_configs[name] = {
            "class": dataset_class,
            "params": params
        }
        print(f"已添加数据集配置: {name}")
    
    def update_dataset_config(self, name: str, params: Dict[str, Any]):
        """
        更新数据集配置
        
        参数:
            name (str): 数据集名称
            params (Dict[str, Any]): 要更新的参数
        """
        if name in self.dataset_configs:
            self.dataset_configs[name]["params"].update(params)
            print(f"已更新数据集配置: {name}")
        else:
            print(f"警告: 数据集 {name} 未定义在配置中")
    
    def get_class_distribution(self, dataset_type='train'):
        """
        获取数据集的类别分布
        
        参数:
            dataset_type (str): 'train'或'val'，表示要查看的数据集类型
            
        返回:
            Counter: 类别计数器对象
        """
        dataset = self.train_dataset if dataset_type == 'train' else self.val_dataset
        
        if dataset is None:
            print(f"警告: {dataset_type} 数据集未初始化")
            return None
        
        # 标签名称到对应的索引位置的映射
        label_to_index = {}
        for fault_type, one_hot in dataset.fault_type_map.items():
            index = torch.tensor(one_hot).argmax().item()
            label_to_index[index] = fault_type
        
        # 统计每个类别的样本数
        class_counts = Counter()
        for sample in dataset.samples:
            label_index = sample['y'].argmax().item()  # 获取类别索引
            label_name = label_to_index.get(label_index, f"未知类别_{label_index}")
            class_counts[label_name] += 1
        
        return class_counts

    def _load_dataset(self, dataset_name, data_type):
        """加载单个数据集"""
        dataset_path = os.path.join(self.data_dir, dataset_name)
        if not os.path.exists(dataset_path):
            raise ValueError(f"数据集 {dataset_name} 不存在于 {dataset_path}")
        
        all_data = []
        all_labels = []
        
        # 遍历数据集目录
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if not file.endswith('.csv'):
                    continue
                
                file_path = os.path.join(root, file)
                try:
                    # 确定故障类型
                    fault_type = self._determine_fault_type(file)
                    if fault_type is None:
                        print(f"警告: 无法确定文件 {file} 的故障类型，跳过")
                        continue
                    
                    # 读取数据
                    data = pd.read_csv(file_path, header=None).values
                    if data.shape[1] != 1:  # 确保数据是单通道
                        data = data[:, 0].reshape(-1, 1)
                    
                    # 创建标签（5个类别）
                    label = np.zeros(5)  # [normal, inner, outer, cage, ball]
                    if fault_type == 'normal':
                        label[0] = 1
                    elif fault_type == 'inner':
                        label[1] = 1
                    elif fault_type == 'outer':
                        label[2] = 1
                    elif fault_type == 'cage':
                        label[3] = 1
                    elif fault_type == 'ball':
                        label[4] = 1
                    
                    # 添加到数据集
                    all_data.append(data)
                    all_labels.append(label)
                    
                except Exception as e:
                    print(f"加载文件 {file} 出错: {str(e)}")
                    continue
        
        if not all_data:
            raise ValueError(f"数据集 {dataset_name} 中没有有效数据")
        
        return all_data, all_labels


# 使用示例
if __name__ == "__main__":
    # 数据目录
    data_dir = "/root/autodl-tmp"
    
    # 创建数据工厂
    print("\n创建数据工厂")
    data_factory = DataFactory(
        data_dir=data_dir,
        batch_size=32,
        normalize=True,
        max_seq_len=32768,  # 可选：限制序列长度
        use_all_samples=False,
        balance_classes=True,  # 启用类别平衡
        max_samples_per_class=1000  # 每个类别最多1000个样本
    )
    
    # 设置数据集
    print("\n设置数据集")
    data_factory.setup(
        train_datasets=["CASE", "OTTAWA", "XJTU","MFPT"],  # 将Ottawa数据集添加到训练集
        val_datasets=["HUST"],  # 使用XJTU数据集作为验证集
        data_type=0  # 故障分类任务
    )
    
    # 获取类别分布
    print("\n训练集类别分布:")
    print(data_factory.get_class_distribution('train'))
    
    print("\n验证集类别分布:")
    print(data_factory.get_class_distribution('val'))
    
    # 获取数据加载器
    train_loader = data_factory.get_train_dataloader()
    val_loader = data_factory.get_val_dataloader()
    
    # 获取一个批次的数据
    for batch in train_loader:
        x = batch['x']
        label = batch['label']
        print(f"\n批次数据形状:")
        print(f"x: {x.shape}")
        print(f"label: {label.shape}")
        break 