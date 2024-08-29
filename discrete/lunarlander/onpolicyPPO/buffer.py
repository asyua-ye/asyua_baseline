import numpy as np
import torch
from utils.tool import RunningMeanStd
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class RolloutBuffer(object):
    def __init__(self, state_dim, hp):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gae_tau = hp.gae
        self.num_steps = hp.num_steps
        self.num_processes = hp.num_processes
        self.discount = hp.discount
        self.num_steps = hp.num_steps
        self.num_processes = hp.num_processes
        self.reward_scale_enable = hp.reward_scale_enable
        self.device = device
        
        self.states = torch.zeros((self.num_steps+1, self.num_processes,) + state_dim).to(device)
        self.rewards = torch.zeros(self.num_steps, self.num_processes, 1).to(device)
        self.actions = torch.zeros(self.num_steps, self.num_processes, 1).to(device, torch.long)
        self.masks = torch.ones(self.num_steps+1 , self.num_processes, 1).to(device)
        self.exp = torch.zeros(self.num_steps, self.num_processes, 1).to(device)
        self.values = torch.zeros(self.num_steps, self.num_processes, 1).to(device)
        self.action_log_probs = torch.zeros(self.num_steps, self.num_processes, 1).to(device)
        self.returns = torch.zeros(self.num_steps, self.num_processes, 1).to(device)
        
        
        
        self.step = 0
        
        self.rewards_norm = RunningMeanStd(1)
        self.R = torch.zeros(self.num_processes,1).to(device)
        
    def scale(self,x,x_norm,mask):
        
        self.R = mask * self.R
        self.R = self.discount * self.R + x
        x_norm.update(self.R)
        std = torch.tensor(self.rewards_norm.std, dtype=torch.float32, device=x.device)
        x = x / (std + 1e-8)
        
        return x

    def insert(self, state, action, reward, mask, log_prob, value):
        
        self.states[self.step].copy_(state)
        self.actions[self.step].copy_(action)
        if self.reward_scale_enable:
            reward = self.scale(reward,self.rewards_norm,mask)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step].copy_(mask)
        self.action_log_probs[self.step].copy_(log_prob)
        self.values[self.step].copy_(value)
        
        self.step = (self.step + 1) % self.num_steps
        
    def lastInsert(self,next_state,next_mask):
        self.states[-1].copy_(next_state)
        self.masks[-1].copy_(next_mask)
        
    def preTrj(self,next_value):
        
        
        gae = 0
        mask = self.masks[-1]
        
        for step in reversed(range(self.rewards.shape[0])):
            delta= self.rewards[step] + next_value * self.discount * mask - self.values[step]
            gae = delta + self.discount * self.gae_tau * mask * gae
            self.returns[step] = gae + self.values[step]
            
            next_value = self.values[step]
            if step!=0:
                mask = self.masks[step+1]
            else:
                mask = torch.ones_like(self.masks[step+1])
            
            
        self.adv = self.returns - self.values
        
        self.adv = (self.adv - torch.mean(self.adv)) / (
            torch.std(self.adv) + 1e-8)
        
        state=self.states[:-1].view(-1,*self.states.size()[2:]).cpu().data.numpy()
        action=self.actions.view(-1, 1).cpu().data.numpy()
        log_prob = self.action_log_probs.view(-1, 1).cpu().data.numpy()
        adv = self.adv.view(-1,1).cpu().data.numpy()
        returns = self.returns.view(-1,1).cpu().data.numpy()
        
        return (state,action,log_prob,adv,returns)


class ReplayBuffer(object):
    def __init__(self, state_dim, hp):
        self.max = hp.buffer_size
        self.numactor = hp.num_processes
        self.rollout = hp.num_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos = 0
        self.rl = 0
        
        self.ppo_update = hp.ppo_update
        self.mini_batch = hp.mini_batch
        
        # 预分配 NumPy 数组
        self.states = np.zeros((self.max, self.numactor * self.rollout,state_dim[0]), dtype=np.float32)
        self.actions = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)
        self.next_states = np.zeros((self.max, self.numactor * self.rollout,state_dim[0]), dtype=np.float32)
        self.log_probs = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)
        self.advs = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)
        self.returns = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)

    def push(self, data):
        state, action, log_prob, adv, ret = data
        
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.advs[self.pos] = adv
        self.returns[self.pos] = ret
        self.log_probs[self.pos] = log_prob
        
        self.pos = (self.pos + 1) % self.max
        self.rl = min(self.rl + 1, self.max)
     
    def PPOsample(self):
        for _ in range(self.ppo_update):
            mini_batch_size = len(self) * self.numactor * self.rollout // self.mini_batch
            sampler = BatchSampler(SubsetRandomSampler(range(len(self) * self.numactor * self.rollout)), mini_batch_size, drop_last=False)
            
            for ind in sampler:
                ind = np.array(ind)
                
                batch_indices = ind // (self.numactor * self.rollout)
                rollout_indices = ind % (self.numactor * self.rollout)
                
                state = self.states[batch_indices, rollout_indices, :]
                action = self.actions[batch_indices, rollout_indices, :]
                log_prob = self.log_probs[batch_indices, rollout_indices, :]
                adv = self.advs[batch_indices, rollout_indices, :]
                ret = self.returns[batch_indices, rollout_indices, :]
                
                yield (torch.tensor(state).to(self.device), 
                    torch.tensor(action).to(self.device), 
                    torch.tensor(log_prob).to(self.device), 
                    torch.tensor(adv).to(self.device),
                    torch.tensor(ret).to(self.device)
                    )

    def __len__(self):
        return self.rl
