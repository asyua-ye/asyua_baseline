import numpy as np
import torch
from utils.tool import RunningMeanStd


class RolloutBuffer(object):
    def __init__(self, num_steps, num_processes, state_dim, discount, nstep, reward_scale_enable=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = torch.zeros((num_steps+1, num_processes,) + state_dim).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.actions = torch.zeros(num_steps, num_processes, 1).to(device, torch.long)
        self.masks = torch.ones(num_steps+1, num_processes, 1).to(device)
        self.exp = torch.zeros(num_steps, num_processes, 1).to(device)
        
        self.discount = discount
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.device = device
        self.step = 0
        self.reward_scale_enable = reward_scale_enable
        self.rewards_norm = RunningMeanStd(1)
        self.R = torch.zeros(num_processes, 1).to(device)
        self.nstep = nstep
        if self.nstep != -1:
            self.exp.fill_(self.nstep)
    
    def scale(self, x, x_norm, mask):
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
            reward = self.scale(reward, self.rewards_norm, mask)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step].copy_(mask)
        self.step = (self.step + 1) % self.num_steps
    
    def lastInsert(self, next_state, next_mask):
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
        

        episode_ends = []
        for proc in range(self.num_processes):
            proc_ends = torch.where(self.masks[1:, proc, 0] == 0)[0]
            if len(proc_ends) == 0:
                proc_ends = torch.tensor([self.num_steps], device=self.device)
            else:
                proc_ends = torch.cat([proc_ends, torch.tensor([self.num_steps], device=self.device)])
            episode_ends.append(proc_ends)

        next_state_indices = torch.zeros(self.num_steps, self.num_processes, dtype=torch.long, device=self.device)

        for proc in range(self.num_processes):
            episode_start = 0
            for episode_end in episode_ends[proc]:
                if self.nstep == -1:
                    next_state_indices[episode_start:episode_end, proc] = episode_end
                else:
                    steps = torch.arange(episode_start, episode_end, device=self.device)
                    next_steps = torch.clamp(steps + self.nstep, max=episode_end)
                    next_state_indices[episode_start:episode_end, proc] = next_steps
                episode_start = episode_end

        next_state = self.states[next_state_indices, torch.arange(self.num_processes)].reshape(-1, *self.states.shape[2:]).cpu().numpy()
        
        
        # next_state = self.states[1:].reshape(-1, *self.states.shape[2:]).cpu().numpy()
        
        
        state = self.states[:-1].reshape(-1, *self.states.shape[2:]).cpu().numpy()
        action = self.actions.reshape(-1, 1).cpu().numpy()
        reward = self.rewards.reshape(-1, 1).cpu().numpy()
        mask = self.masks[1:].reshape(-1, 1).cpu().numpy()
        exp = self.exp.reshape(-1, 1).cpu().numpy()
        
        return (state, action, next_state, reward, mask, exp)


class ReplayBuffer(object):
    def __init__(self, maxs, num_processes, num_steps, state_dim, batch, epoch_per_train):
        self.max = maxs
        self.numactor = num_processes
        self.rollout = num_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos = 0
        self.rl = 0
        
        self.batch = batch
        self.epoch_per_train = epoch_per_train
        
        # 预分配 NumPy 数组
        self.states = np.zeros((self.max, self.numactor * self.rollout,state_dim[0]), dtype=np.float32)
        self.actions = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)
        self.next_states = np.zeros((self.max, self.numactor * self.rollout,state_dim[0]), dtype=np.float32)
        self.rewards = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)
        self.masks = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)
        self.exps = np.zeros((self.max, self.numactor * self.rollout,1), dtype=np.float32)

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
            exp = self.exps[batch_indices, rollout_indices, :]
            
            yield (torch.tensor(state).to(self.device), 
                   torch.tensor(action).to(self.device), 
                   torch.tensor(next_state).to(self.device),
                   torch.tensor(reward).to(self.device),
                   torch.tensor(mask).to(self.device),
                   torch.tensor(exp).to(self.device))

    def __len__(self):
        return self.rl
