import numpy as np
import torch
from utils.tool import RunningMeanStd


class RolloutBuffer(object):
    def __init__(self, state_dim, hp):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.reward_scale_enable = hp.reward_scale_enable
        self.discount = hp.discount
        self.num_steps = hp.num_steps
        self.num_processes = hp.num_processes
        self.nstep = hp.nstep
        
        self.states = torch.zeros((self.num_steps+1, self.num_processes,) + self.state_dim).to(device)
        self.rewards = torch.zeros(self.num_steps, self.num_processes, 1).to(device)
        self.actions = torch.zeros(self.num_steps, self.num_processes, 1).to(device, torch.long)
        self.masks = torch.ones(self.num_steps+1 , self.num_processes, 1).to(device)
        self.exp = torch.zeros(self.num_steps, self.num_processes, 1).to(device)
        
        self.device = device
        self.step = 0
        self.rewards_norm = RunningMeanStd(1)
        self.R = torch.zeros(self.num_processes,1).to(device)
        
        if self.nstep != -1:
            self.exp.fill_(self.nstep)
        
    def scale(self,x,x_norm,mask):
        
        self.R = mask * self.R
        self.R = self.discount * self.R + x
        x_norm.update(self.R)
        std = torch.tensor(self.rewards_norm.std, dtype=torch.float32, device=x.device)
        x = x / (std + 1e-8)
        
        return x

    def insert(self, state, action, reward, mask):
        
        self.states[self.step].copy_(state)
        self.actions[self.step].copy_(action)
        if self.reward_scale_enable:
            reward = self.scale(reward,self.rewards_norm,mask)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step].copy_(mask)
        self.step = (self.step + 1) % self.num_steps
        
    def lastInsert(self,next_state,next_mask):
        self.states[-1].copy_(next_state)
        self.masks[-1].copy_(next_mask)
        
    def preTrj(self):
        
        temp = torch.zeros(self.num_processes, 1, device=self.device)
        nstep = torch.zeros(self.num_processes, 1, device=self.device)
        if self.nstep == -1:
            for i in reversed(range(self.rewards.size(0))):
                temp = temp + self.rewards[i]
                nstep += 1
                self.exp[i] = nstep.clone()
                self.rewards[i] = temp.clone()
                temp *= self.masks[i]
                nstep = torch.where(temp.sum(dim=-1, keepdim=True) == 0, torch.zeros_like(nstep), nstep)
                temp = temp * self.discount
        else:
            for i in reversed(range(self.rewards.size(0))):
                temp = temp + self.rewards[i]
                nstep += 1
                self.rewards[i] = temp.clone()
                temp *= self.masks[i]
                reset_condition = (temp.sum(dim=-1, keepdim=True) == 0) | (nstep == self.nstep)
                nstep = torch.where(reset_condition, torch.zeros_like(nstep), nstep)
                temp = torch.where(reset_condition, torch.zeros_like(temp), temp)
                temp = temp * self.discount
                
        
        state = self.states[:-1].reshape(-1, *self.states.shape[2:]).cpu().numpy()
        action = self.actions.reshape(-1, 1).cpu().numpy()
        next_state = self.states[1:].reshape(-1, *self.states.shape[2:]).cpu().numpy()
        reward = self.rewards.reshape(-1, 1).cpu().numpy()
        mask = self.masks[1:].reshape(-1, 1).cpu().numpy()
        exp = self.exp.reshape(-1, 1).cpu().numpy()
        
        return (state, action, next_state, reward, mask, exp)


class priorReplayBuffer(object):
    def __init__(self, state_dim, hp):
        self.max = hp.buffer_size
        self.numactor = hp.num_processes
        self.rollout = hp.num_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos = 0
        self.rl = 0
        
        self.batch = hp.batch
        self.epoch_per_train = hp.epoch_per_train
        self.alpha = hp.alpha
        self.beta = hp.beta
        self.beta_increment = hp.beta_increment
        self.epsilon = hp.epsilon
        
        # 预分配 NumPy 数组
        self.states = np.zeros((self.max, self.numactor * self.rollout,state_dim[0]), dtype=np.float32)
        self.actions = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)
        self.next_states = np.zeros((self.max, self.numactor * self.rollout,state_dim[0]), dtype=np.float32)
        self.rewards = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)
        self.masks = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)
        self.exps = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)
        self.weights = torch.zeros((self.max*self.numactor * self.rollout,1),dtype=torch.float64).to(self.device)
        self.max_priority = 1.0

    def push(self, data):
        state, action, next_state, reward, mask, exp = data
        
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.next_states[self.pos] = next_state
        self.rewards[self.pos] = reward
        self.masks[self.pos] = mask
        self.exps[self.pos] = exp
        
        self.pos = (self.pos + 1) % self.max
        self.rl = min(self.rl + 1, self.max)
        
        weights = torch.full((mask.shape[0] * mask.shape[1], 1), (self.max_priority + self.epsilon)**self.alpha).to(self.device)
        if self.pos==0:
            self.weights[self.pos * self.numactor * self.rollout : (self.pos + 1) * self.numactor * self.rollout] = weights
        else:
            self.weights[(self.pos - 1) * self.numactor * self.rollout : self.pos * self.numactor * self.rollout] = weights
        
     
    def sample(self):
        for _ in range(self.epoch_per_train):
            weights_slice = self.weights[:self.rl * self.numactor * self.rollout].flatten()
            csum = torch.cumsum(weights_slice.double(), 0)
            val = torch.clamp(
                    torch.rand(self.batch, device=self.device) * csum[-1].item(),
                    max=csum[-1].item() - torch.finfo(csum.dtype).eps
                )
            ind = torch.searchsorted(csum, val).cpu().numpy()
            
            
            batch_indices = ind // (self.numactor * self.rollout)
            rollout_indices = ind % (self.numactor * self.rollout)
            

            state = self.states[batch_indices, rollout_indices, :]
            action = self.actions[batch_indices, rollout_indices, :]
            next_state = self.next_states[batch_indices, rollout_indices, :]
            reward = self.rewards[batch_indices, rollout_indices, :]
            mask = self.masks[batch_indices, rollout_indices, :]
            exp = self.exps[batch_indices, rollout_indices, :]

            weight = self.weights[ind].view(-1,1).to(self.device)


            p = weight / weight.sum()
            importance_weight = (1 / (len(self.weights) * p)) ** self.beta
            importance_weight /= importance_weight.max()
            
            self.beta = min(1., self.beta + self.beta_increment)

            yield (
                torch.FloatTensor(state).to(self.device),
                torch.FloatTensor(action).to(self.device),
                torch.FloatTensor(next_state).to(self.device),
                torch.FloatTensor(reward).to(self.device),
                torch.FloatTensor(mask).to(self.device),
                torch.FloatTensor(exp).to(self.device),
            ), ind, importance_weight
            
            
    def update_batch(self, idxs, errors):
        priorities = errors + self.epsilon
        self.max_priority = max(self.max_priority, torch.max(priorities).item())
        scaled_priorities = (priorities ** self.alpha).view(-1,1).to(torch.float64)
        self.weights[idxs] = scaled_priorities

    def __len__(self):
        return self.rl
