import numpy as np
import torch
import pandas as pd
import os
import copy
from collections import OrderedDict


def get_normalized_score(total_reward, env_name):
    """
    
    参考:https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/gym_mujoco/__init__.py
    
    
    
    """
    
    
    reference_scores = {
        'hopper': {'min_score': -20.272305, 'max_score': 3234.3},
        'walker2d': {'min_score': 1.629008, 'max_score': 4592.3},
        'halfcheetah': {'min_score': -280.178953, 'max_score': 12135.0},
        'ant': {'min_score': -325.6, 'max_score': 3879.7},
    }

    # 直接检查 env_name 是否包含基础环境名称
    base_env_name = None
    for key in reference_scores.keys():
        if key in env_name.lower():  # 使用 lower() 处理大小写不敏感的匹配
            base_env_name = key
            break

    if base_env_name is None:
        raise ValueError(f"No reference scores found for environment: {env_name}")

    random_score = reference_scores[base_env_name]['min_score']
    expert_score = reference_scores[base_env_name]['max_score']

    normalized_score = (total_reward - random_score) / (expert_score - random_score) * 100

    return normalized_score



def initialize_env(env_name):
    if env_name == 'hopper':
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env_targets = [5000, 2500]
        scale = 1000.
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
    return env_targets, scale

def log_and_print(text_list, message):
    print(message)
    text_list.append(message)

class RunningMeanStd:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.ones(shape)  # 避免初始除零

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        batch_mean = np.mean(x, axis=0)
        batch_S = np.sum((x - batch_mean) ** 2, axis=0)
        batch_n = x.shape[0]

        if self.n == 0:
            self.mean = batch_mean
            self.S = batch_S
        else:
            total_n = self.n + batch_n
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_n / total_n
            self.S = self.S + batch_S + delta ** 2 * self.n * batch_n / total_n
            self.mean = new_mean

        self.n += batch_n
        self.std = np.sqrt(self.S / self.n)

    def normalize(self, x):
        if isinstance(x, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=torch.float32, device=x.device)
            std = torch.tensor(self.std, dtype=torch.float32, device=x.device)
            return (x - mean) / (std + 1e-8)  # 避免除零
        else:
            return (x - self.mean) / (self.std + 1e-8)  # 避免除零
          
        
class DataProcessor:
    def __init__(self, file_path=None):
        self.records = []  # 存储所有待写入的记录
        self.current_data = OrderedDict()  # 当前正在处理的数据
        self.file_name = (file_path if file_path else "") + "output.xlsx"

    def process_input(self, data, name, prefix=""):
        # 按照既定的逻辑处理输入，将结果存储在current_data中
        data = np.squeeze(data)
        if data.size == 1:
            self.current_data[f'{prefix}{name}'] = data.item()
        elif data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
            stats = {
                'max': np.max(data),
                'min': np.min(data),
                'mean': np.mean(data),
                'std': np.std(data)
            }
            for stat_name, value in stats.items():
                self.current_data[f'{prefix}{name}_{stat_name}'] = value
        elif data.ndim == 2:
            stats = {
                'max': np.max(data),
                'min': np.min(data),
                'mean': np.mean(data),
                'std': np.std(data)
            }
            for stat_name, value in stats.items():
                self.current_data[f'{prefix}{name}_{stat_name}'] = value

    def new_record(self):
        # 将当前数据加入到记录列表并准备新的数据
        self.records.append(copy.deepcopy(self.current_data))
        self.current_data = OrderedDict()  # 重置当前数据

    def write_to_excel(self):
        # 将累积的数据写入Excel文件
        new_data_df = pd.DataFrame(self.records)
        if not os.path.isfile(self.file_name):
            new_data_df.to_excel(self.file_name, index=False)
        else:
            with pd.ExcelWriter(self.file_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                existing_df = pd.read_excel(self.file_name)
                combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
                combined_df.to_excel(writer, index=False)
        
        self.records.clear()  # 清空记录，准备下次累积

    def reset_data(self):
        self.current_data.clear()  # 明确方法清空当前数据

        
        

