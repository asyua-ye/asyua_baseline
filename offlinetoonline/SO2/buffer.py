import numpy as np
import torch
import h5py
from utils.tool import RunningMeanStd




class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, hp):
        
        self.max_size = int(hp.max_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = hp.batch
        self.rewards_norm = RunningMeanStd(1)
        self.R = torch.zeros(1).to(self.device)
        self.ptr = 0
        self.size = 0
        self.discount = hp.discount
        self.train_per_epoch = hp.train_per_epoch
        self.reward_scale_enable = hp.reward_scale_enable
        
        self.states = np.zeros((self.max_size,state_dim))
        self.actions = np.zeros((self.max_size, action_dim))
        self.next_states = np.zeros((self.max_size,state_dim))
        self.rewards = np.zeros((self.max_size, 1))
        self.masks = np.zeros((self.max_size, 1))
        
        
    def scale(self,x,x_norm,mask):
        
        self.R = mask * self.R
        self.R = self.discount * self.R + x
        x_norm.update(self.R)
        std = torch.tensor(self.rewards_norm.std, dtype=torch.float32, device=x.device)
        x = x / (std + 1e-8)
        
        return x
    
    
    def push(self, state, action, next_state, reward, mask):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        
        if self.reward_scale_enable:
            reward = self.scale(reward,self.rewards_norm,mask)
        
        self.rewards[self.ptr] = reward
        self.masks[self.ptr] = mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        

    def sample(self):
        total_samples = self.batch_size * self.train_per_epoch
        inds = np.random.randint(0, self.size, size=total_samples).reshape(self.train_per_epoch, self.batch_size)
        
        for ind in inds:
            
            state = self.states[ind, :]
            action = self.actions[ind, :]
            next_state = self.next_states[ind,:]
            reward = self.rewards[ind, :]
            mask = self.masks[ind, :]
            
            yield (torch.FloatTensor(state).to(self.device), 
                   torch.FloatTensor(action).to(self.device), 
                   torch.FloatTensor(next_state).to(self.device),
                   torch.FloatTensor(reward).to(self.device),
                   torch.FloatTensor(mask).to(self.device))
            
            
    def load_dataset(self, dataset_path):
        """
        参考: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py
        
        
        """
        
        with h5py.File(dataset_path, 'r') as dataset:
            observations = dataset['observations'][:]
            actions = dataset['actions'][:]
            next_observations = dataset['next_observations'][:]
            rewards = dataset['rewards'][:]
            terminals = dataset['terminals'][:]

            num_samples = observations.shape[0]
            assert num_samples <= self.max_size, "Dataset size exceeds replay buffer capacity."

            self.states[:num_samples] = observations
            self.actions[:num_samples] = actions
            self.next_states[:num_samples] = next_observations
            self.rewards[:num_samples] = rewards.reshape(-1, 1)
            self.masks[:num_samples] = 1.0 - terminals.reshape(-1, 1).astype(np.float32)

            self.size = num_samples
            self.ptr = num_samples % self.max_size
            print(f"Loaded {num_samples} samples into replay buffer.")
            
    def __len__(self):
        return self.size
