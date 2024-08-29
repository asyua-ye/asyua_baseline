import numpy as np
import torch
from utils.tool import RunningMeanStd


class RolloutBuffer(object):
    def __init__(self, state_dim, hp):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_steps = hp.num_steps
        self.num_processes = hp.num_processes
        self.nstep = hp.nstep
        self.discount = hp.discount
        self.num_steps = hp.num_steps
        self.device = device
        self.step = 0
        self.reward_scale_enable = hp.reward_scale_enable
        self.hidden_state = hp.hidden_state
        l = hp.num_layer + 1*hp.bidirectional
        
        self.states = torch.zeros((self.num_steps+1, self.num_processes,self.nstep,) + state_dim).to(device)
        self.rewards = torch.zeros(self.num_steps, self.num_processes, 1).to(device)
        self.actions = torch.zeros(self.num_steps, self.num_processes, 1).to(device, torch.long)
        self.masks = torch.ones(self.num_steps+1 , self.num_processes, 1).to(device)
        self.hidden = torch.zeros(self.num_steps, self.num_processes, l,self.hidden_state).to(device)
        self.cell = torch.zeros(self.num_steps, self.num_processes, l,self.hidden_state).to(device)
        
        self.rewards_norm = RunningMeanStd(1)
        self.R = torch.zeros(self.num_processes,1).to(device)
        
    def scale(self,x,x_norm,mask):
        
        self.R = mask * self.R
        self.R = self.discount * self.R + x
        x_norm.update(self.R)
        std = torch.tensor(self.rewards_norm.std, dtype=torch.float32, device=x.device)
        x = x / (std + 1e-8)
        
        return x

    def insert(self, state, action, reward, mask, hidden, cell):
        
        self.states[self.step].copy_(state)
        self.actions[self.step].copy_(action)
        if self.reward_scale_enable:
            reward = self.scale(reward,self.rewards_norm,mask)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step].copy_(mask)
        self.hidden[self.step].copy_(hidden)
        self.cell[self.step].copy_(cell)
        
        self.step = (self.step + 1) % self.num_steps
        
    def lastInsert(self,next_state,next_mask):
        self.states[-1].copy_(next_state)
        self.masks[-1].copy_(next_mask)
        
    def preTrj(self):
        
        state=self.states[:-1].view(-1,*self.states.size()[2:]).cpu().data.numpy()
        action=self.actions.view(-1, 1).cpu().data.numpy()
        next_state=self.states[1:].view(-1,*self.states.size()[2:]).cpu().data.numpy()
        reward = self.rewards.view(-1, 1).cpu().data.numpy()
        mask = self.masks[1:].view(-1,1).cpu().data.numpy()
        
        
        return (state,action,next_state,reward,mask)


class ReplayBuffer(object):
    def __init__(self, state_dim, hp):
        self.max = hp.buffer_size
        self.numactor = hp.num_processes
        self.rollout = hp.num_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos = 0
        self.rl = 0
        
        self.batch = hp.batch
        self.epoch_per_train = hp.epoch_per_train
        self.nstep = hp.nstep
        # 预分配 NumPy 数组
        self.states = np.zeros(((self.max, self.numactor * self.rollout,self.nstep,)+state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)
        self.next_states = np.zeros(((self.max, self.numactor * self.rollout,self.nstep,)+state_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)
        self.masks = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)

    def push(self, data):
        state, action, next_state, reward, mask = data
        
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.next_states[self.pos] = next_state
        self.rewards[self.pos] = reward
        self.masks[self.pos] = mask

        self.pos = (self.pos + 1) % self.max
        self.rl = min(self.rl + 1, self.max)
     
    def sample(self):
        total_samples = self.batch * self.epoch_per_train
        inds = np.random.randint(0, self.rl * self.numactor * self.rollout, size=total_samples).reshape(self.epoch_per_train, self.batch)
        
        for ind in inds:
            batch_indices = ind // (self.numactor * self.rollout)
            rollout_indices = ind % (self.numactor * self.rollout)
            
            state = self.states[batch_indices, rollout_indices, :]
            action = self.actions[batch_indices, rollout_indices, :]
            next_state = self.next_states[batch_indices, rollout_indices,:]
            reward = self.rewards[batch_indices, rollout_indices, :]
            mask = self.masks[batch_indices, rollout_indices, :]
            
            yield (torch.tensor(state).to(self.device), 
                   torch.tensor(action).to(self.device), 
                   torch.tensor(next_state).to(self.device),
                   torch.tensor(reward).to(self.device),
                   torch.tensor(mask).to(self.device))

    def __len__(self):
        return self.rl
