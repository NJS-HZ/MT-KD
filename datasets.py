
# import torch
# import torchvision
# import torchvision.transforms as tt


# def cic2017(root='./'):
#     # 在数据加载时确保数据类型
#     transform = tt.Compose([
#         tt.ToTensor(),
#         tt.ConvertImageDtype(torch.float32)  # 确保转换为float32
#     ])

#     train_ds = torchvision.datasets.MNIST(root='./', train=True, download=False, transform=transform)
#     valid_ds = torchvision.datasets.MNIST(root='./', train=False, download=False, transform=transform)

#     return train_ds, valid_ds

# from torch.utils.data import Dataset
# import numpy as np
# import random
# from scipy.stats import skewnorm

# class CIC2017WithZeroDay(Dataset):
#     def __init__(self, base_dataset, unknown_ratio=0.1):
#         self.base_dataset = base_dataset
#         self.unknown_ratio = unknown_ratio
        
#         # 初始化类别信息
#         if hasattr(base_dataset, 'classes'):
#             self.classes = base_dataset.classes + ['unknown']
#         else:
#             # 如果基类没有classes属性，创建默认类别
#             self.classes = [str(i) for i in range(10)] + ['unknown']
            
#         self.unknown_class = len(self.classes) - 1  # 未知类别索引
        
#         # 生成未知攻击样本
#         self.unknown_samples = self._generate_zero_day_samples()
        
#     @property
#     def targets(self):
#         """获取所有样本的标签"""
#         base_targets = self.base_dataset.targets
#         unknown_targets = [self.unknown_class] * len(self.unknown_samples)
#         return base_targets + unknown_targets
        
#     def _generate_zero_day_samples(self):
#         # 分析已知攻击的统计特征
#         known_stats = self._analyze_known_attacks()
        
#         # 生成未知攻击样本
#         num_unknown = int(len(self.base_dataset) * self.unknown_ratio)
#         samples = []
        
#         for _ in range(num_unknown):
#             # 方法1：基于已知攻击的变异
#             sample = self._mutate_known_attack(known_stats)
            
#             # 方法2：随机生成异常特征
#             if random.random() < 0.5:
#                 sample = self._random_abnormal_sample(known_stats)
            
#             samples.append((sample, self.unknown_class))
            
#         return samples
    
#     def _analyze_known_attacks(self):
#         # 收集已知攻击的统计特征
#         stats = {
#             'means': [],
#             'stds': [],
#             'skews': []
#         }
        
#         for data, label in self.base_dataset:
#             if label >= 0:  # 排除正常流量
#                 # 将数据转换为numpy数组
#                 if isinstance(data, torch.Tensor):
#                     data = data.cpu().numpy()
                
#                 # 计算统计量
#                 stats['means'].append(np.mean(data))
#                 stats['stds'].append(np.std(data))
#                 stats['skews'].append(skewnorm.stats(data))
                
#         return {
#             'mean_range': (np.min(stats['means']), np.max(stats['means'])),
#             'std_range': (np.min(stats['stds']), np.max(stats['stds'])),
#             'skew_range': (np.min(stats['skews']), np.max(stats['skews']))
#         }
    
#     def _mutate_known_attack(self, stats):
#         # 随机选择一个已知攻击样本
#         idx = random.randint(0, len(self.base_dataset)-1)
#         sample, _ = self.base_dataset[idx]
        
#         # 确保sample是numpy数组
#         if isinstance(sample, torch.Tensor):
#             sample = sample.numpy()
        
#         # 添加随机扰动
#         mutation_strength = random.uniform(0.1, 0.5)
#         noise = np.random.normal(
#             loc=random.uniform(*stats['mean_range']),
#             scale=random.uniform(*stats['std_range']),
#             size=sample.shape
#         )
        
#         # 使用torch的clip代替numpy的clip
#         result = sample + mutation_strength * noise
#         result = np.clip(result, 0, 1)
        
#         # 转换为tensor并确保float32类型
#         return torch.from_numpy(result).float()
        
#     def _random_abnormal_sample(self, stats):
#         # 基于统计特征生成随机样本
#         size = self.base_dataset[0][0].shape
#         sample = skewnorm.rvs(
#             a=random.uniform(*stats['skew_range']),
#             loc=random.uniform(*stats['mean_range']),
#             scale=random.uniform(*stats['std_range']),
#             size=size
#         )
        
#         # 使用torch的clip代替numpy的clip
#         sample = np.clip(sample, 0, 1)
        
#         # 转换为tensor并确保float32类型
#         return torch.from_numpy(sample).float()
    
#     def __len__(self):
#         return len(self.base_dataset) + len(self.unknown_samples)
    
#     def __getitem__(self, idx):
#         if idx < len(self.base_dataset):
#             return self.base_dataset[idx]
#         else:
#             return self.unknown_samples[idx - len(self.base_dataset)]



import torch
import torchvision
import torchvision.transforms as tt
from torch.utils.data import Dataset
import numpy as np
import random
from scipy.stats import skewnorm
from torchattacks import FGSM  # 引入对抗攻击库

# 数据加载函数
def cic2017(root='./'):
    transform = tt.Compose([
        tt.ToTensor(),
        tt.ConvertImageDtype(torch.float32)  # 确保转换为float32
    ])

    train_ds = torchvision.datasets.MNIST(root='./', train=True, download=False, transform=transform)
    valid_ds = torchvision.datasets.MNIST(root='./', train=False, download=False, transform=transform)

    return train_ds, valid_ds

# 零日攻击数据集类
class CIC2017WithZeroDay(Dataset):
    def __init__(self, base_dataset, unknown_ratio=0.1, model=None):
        self.base_dataset = base_dataset
        self.unknown_ratio = unknown_ratio
        self.model = model  # 用于生成对抗样本的模型
        
        # 初始化类别信息
        if hasattr(base_dataset, 'classes'):
            self.classes = base_dataset.classes + ['unknown']
        else:
            self.classes = [str(i) for i in range(10)] + ['unknown']
            
        self.unknown_class = len(self.classes) - 1  # 未知类别索引
        
        # 生成未知攻击样本
        self.unknown_samples = self._generate_zero_day_samples()
        
    @property
    def targets(self):
        """获取所有样本的标签"""
        base_targets = self.base_dataset.targets
        unknown_targets = [self.unknown_class] * len(self.unknown_samples)
        return base_targets + unknown_targets
        
    def _generate_zero_day_samples(self):
        # 分析已知攻击的统计特征
        known_stats = self._analyze_known_attacks()
    
        # 生成未知攻击样本
        num_unknown = int(len(self.base_dataset) * self.unknown_ratio)
        samples = []
    
        for _ in range(num_unknown):
            # 随机选择生成方法
            method = random.choice(['mutate', 'random', 'adversarial'])
    
            sample = None  # 初始化 sample 为 None
    
            if method == 'mutate':
                # 方法1：基于已知攻击的变异
                sample = self._mutate_known_attack(known_stats)
            elif method == 'random':
                # 方法2：随机生成异常特征
                sample = self._random_abnormal_sample(known_stats)
            elif method == 'adversarial' and self.model is not None:
                try:
                    # 方法3：生成对抗样本
                    sample = self._generate_adversarial_sample()
                except Exception as e:
                    print(f"Error generating adversarial sample: {e}")
                    continue  # 出现异常则跳过本次循环
    
            if sample is not None:
                samples.append((sample, self.unknown_class))
    
        return samples

    
    def _analyze_known_attacks(self):
        # 收集已知攻击的统计特征
        stats = {
            'means': [],
            'stds': [],
            'skews': []
        }
        
        for data, label in self.base_dataset:
            if label >= 0:  # 排除正常流量
                # 将数据转换为numpy数组
                if isinstance(data, torch.Tensor):
                    data = data.cpu().numpy()
                
                # 计算统计量
                stats['means'].append(np.mean(data))
                stats['stds'].append(np.std(data))
                stats['skews'].append(skewnorm.stats(data))
                
        return {
            'mean_range': (np.min(stats['means']), np.max(stats['means'])),
            'std_range': (np.min(stats['stds']), np.max(stats['stds'])),
            'skew_range': (np.min(stats['skews']), np.max(stats['skews']))
        }
    
    def _mutate_known_attack(self, stats):
        # 随机选择一个已知攻击样本
        idx = random.randint(0, len(self.base_dataset)-1)
        sample, _ = self.base_dataset[idx]
        
        # 确保sample是numpy数组
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        
        # 添加随机扰动
        mutation_strength = random.uniform(0.1, 0.5)
        noise = np.random.normal(
            loc=random.uniform(*stats['mean_range']),
            scale=random.uniform(*stats['std_range']),
            size=sample.shape
        )
        
        
        result = sample + mutation_strength * noise
        result = np.clip(result, 0, 1)
        
        # 转换为tensor并确保float32类型
        return torch.from_numpy(result).float()
        
    def _random_abnormal_sample(self, stats):
        # 基于统计特征生成随机样本
        size = self.base_dataset[0][0].shape
        sample = skewnorm.rvs(
            a=random.uniform(*stats['skew_range']),
            loc=random.uniform(*stats['mean_range']),
            scale=random.uniform(*stats['std_range']),
            size=size
        )
        
        # 使用torch的clip代替numpy的clip
        sample = np.clip(sample, 0, 1)
        
        # 转换为tensor并确保float32类型
        return torch.from_numpy(sample).float()
    
    def _generate_adversarial_sample(self):
        # 使用FGSM生成对抗样本
        idx = random.randint(0, len(self.base_dataset)-1)
        sample, label = self.base_dataset[idx]
        sample = sample.unsqueeze(0).to('cuda')  # 添加batch维度并移动到GPU
        
        # 生成对抗样本
        attack = FGSM(self.model, eps=0.03)
        adversarial_sample = attack(sample, torch.tensor([label]).to('cuda'))
        
        return adversarial_sample.squeeze(0).cpu()  # 移除batch维度并返回CPU
    
    def __len__(self):
        return len(self.base_dataset) + len(self.unknown_samples)
    
    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            img, label = self.base_dataset[idx]
            return img, torch.tensor(label)  # 确保标签是张量
        else:
            img, label = self.unknown_samples[idx - len(self.base_dataset)]
            return img, torch.tensor(label)  # 确保标签是张量


import matplotlib.pyplot as plt

def visualize_samples(original_samples, attacked_samples, title="Sample Comparison"):
    """
    显示原始图片和零日攻击后的图片。
    
    参数:
        original_samples: 原始图片列表 (list of torch.Tensor)
        attacked_samples: 零日攻击后的图片列表 (list of torch.Tensor)
        title: 图的标题
    """
    plt.figure(figsize=(10, 6))
    for i in range(3):
        # 显示原始图片
        plt.subplot(2, 3, i + 1)
        plt.imshow(original_samples[i].squeeze().numpy(), cmap='gray')
        plt.title(f"Original Sample {i + 1}")
        plt.axis('off')
        
        # 显示零日攻击后的图片
        plt.subplot(2, 3, i + 4)
        plt.imshow(attacked_samples[i].squeeze().numpy(), cmap='gray')
        plt.title(f"Attacked Sample {i + 1}")
        plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

