import numpy as np
import torch
from utils.tool import RunningMeanStd


class RolloutBuffer(object):
    def __init__(self, num_steps, num_processes, state_dim, discount, reward_scale_enable=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = torch.zeros((num_steps+1, num_processes,) + state_dim).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.actions = torch.zeros(num_steps, num_processes, 1).to(device, torch.long)
        self.masks = torch.ones(num_steps+1 , num_processes, 1).to(device)
        self.exp = torch.zeros(num_steps, num_processes, 1).to(device)
        
        self.discount = discount
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.device = device
        self.step = 0
        self.reward_scale_enable = reward_scale_enable
        self.rewards_norm = RunningMeanStd(1)
        self.R = torch.zeros(num_processes,1).to(device)
        
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
        
        state=self.states[:-1].view(-1,*self.states.size()[2:]).cpu().data.numpy()
        action=self.actions.view(-1, 1).cpu().data.numpy()
        next_state=self.states[1:].view(-1,*self.states.size()[2:]).cpu().data.numpy()
        reward = self.rewards.view(-1, 1).cpu().data.numpy()
        mask = self.masks[1:].view(-1,1).cpu().data.numpy()
        
        return (state,action,next_state,reward,mask)


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
    
    


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, indices, changes):
        parent_indices = np.where(indices % 2 == 0, (indices - 2)//2, (indices-1)//2)
        while not np.all(parent_indices <= 0):
            np.add.at(self.tree, parent_indices[parent_indices>0], changes[parent_indices>0])
            parent_indices = np.where(parent_indices % 2 == 0, (parent_indices - 2)//2, (parent_indices-1)//2)
        self.tree[0] += sum(changes) 
    def total(self):
        return self.tree[0]
    
    def validate(self):
        """
        验证 SumTree 的每个节点的和是否都计算正确。

        返回：
            bool: 如果所有节点的和都正确，则返回 True，否则返回 False。
        """
        parent_indices = np.arange(self.capacity - 1)

        left_child_indices = 2 * parent_indices + 1
        right_child_indices = left_child_indices + 1

        sums_correct = np.isclose(
            self.tree[parent_indices],
            self.tree[left_child_indices] + self.tree[right_child_indices],
            rtol=1e-5, atol=1e-8
        )

        return np.all(sums_correct)

    def add_batch(self, priorities, data):
        assert len(priorities) == len(data), "Priorities and data must have the same length"
        batch_size = len(priorities)
        indices = np.arange(self.write, self.write + batch_size) % self.capacity
        tree_indices = indices + self.capacity - 1

        self.data[indices] = data
        changes = priorities - self.tree[tree_indices]
        self.tree[tree_indices] = priorities
        
        self._propagate(tree_indices,changes)

        self.write = (self.write + batch_size) % self.capacity
        self.n_entries = min(self.n_entries + batch_size, self.capacity)
        assert self.validate(), "err,sumtree wrong,maybe add maybe update"

    def update_batch(self, indices, priorities):
        tree_indices = indices + self.capacity - 1
        unique_tree_indices, unique_indices = np.unique(tree_indices, return_index=True)
        changes = priorities[unique_indices] - self.tree[unique_tree_indices]
        self.tree[unique_tree_indices] = priorities[unique_indices]
        self._propagate(unique_tree_indices,changes)

    def get_batch(self, s_values):
        ## 这里应该有问题，因为用了sumtree，训练不出来
        
        indices = np.zeros(s_values.shape[0], dtype=np.int32)
        
        def retrieve(s_vals):
            idx = np.zeros_like(s_vals, dtype=np.int32)
            curr_s = s_vals.copy()
            mask = np.ones_like(idx, dtype=bool)
            
            while np.any(mask):
                left = 2 * idx + 1
                right = left + 1
                
                is_leaf = left >= len(self.tree)
                mask = np.logical_and(mask, ~is_leaf)
                
                if not np.any(mask):
                    break
                
                go_left = curr_s[mask] <= self.tree[left[mask]]
                idx[mask] = np.where(go_left, left[mask], right[mask])
                curr_s[mask] = np.where(go_left, curr_s[mask], curr_s[mask] - self.tree[left[mask]])
            
            return idx
        
        indices = retrieve(s_values)
        # Handle out-of-range indices
        out_of_range = indices >= (self.n_entries + self.capacity - 1)
        indices[out_of_range] = np.random.randint(self.capacity - 1, self.capacity - 1 + self.n_entries, size=np.sum(out_of_range))
        
        data_indices = indices - self.capacity + 1
        data_indices = np.clip(data_indices, 0, self.n_entries - 1)
        priorities = self.tree[indices]
        data = self.data[data_indices]
        
        return data_indices, priorities, data


class PrioritizedReplayBuffer:
    def __init__(self, capacity, batch_size,epoch_per_train, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.batch = batch_size
        self.epoch_per_train = epoch_per_train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def push(self, experiences, priorities=None):
        
        states, actions, next_states, rewards, masks = experiences
        num_experiences = len(states)

        if priorities is None:
            priorities = np.full(num_experiences, self.max_priority)
        else:
            priorities = np.array(priorities)
            self.max_priority = max(self.max_priority, np.max(priorities))

        scaled_priorities = np.power(priorities + self.epsilon, self.alpha)
        data = list(zip(states, actions, next_states, rewards, masks))
        self.tree.add_batch(scaled_priorities, data)

    def sample(self):
        for _ in range(self.epoch_per_train):
            segment = self.tree.total() / self.batch
            
            s = np.random.uniform(segment * np.arange(self.batch), segment * (np.arange(self.batch) + 1))

            idxs, priorities, batch_data = self.tree.get_batch(s)
            idxs = np.array(idxs)
            priorities = np.array(priorities, dtype=np.float64)

            self.beta = min(1., self.beta + self.beta_increment)

            sampling_probabilities = priorities / self.tree.total()
            is_weight = np.power(self.tree.n_entries * sampling_probabilities + 1e-6, -self.beta)
            is_weight /= is_weight.max()

            # 使用向量化操作转换数据类型
            states, actions, next_states, rewards, masks = map(np.array, zip(*batch_data))
            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).long().to(self.device)
            next_states = torch.from_numpy(next_states).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            masks = torch.from_numpy(masks).float().to(self.device)
            is_weight = torch.from_numpy(is_weight).float().to(self.device)

            yield (states, actions, next_states, rewards, masks), idxs, is_weight

    def update_batch(self, idxs, errors):
        errors = np.array(errors, dtype=np.float64)
        priorities = errors + self.epsilon
        self.max_priority = max(self.max_priority, np.max(priorities))
        scaled_priorities = np.power(priorities, self.alpha)
        self.tree.update_batch(idxs, scaled_priorities)



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
        self.weights = torch.zeros((self.max*self.numactor * self.rollout,1),dtype=torch.float64).to(self.device)
        self.max_priority = 1.0

    def push(self, data):
        state, action, next_state, reward, mask = data
        
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.next_states[self.pos] = next_state
        self.rewards[self.pos] = reward
        self.masks[self.pos] = mask
        
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
            # ind = torch.multinomial(weights_slice, self.batch, replacement=True).cpu().numpy()
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
            ), ind, importance_weight
            
            
    def update_batch(self, idxs, errors):
        priorities = errors + self.epsilon
        self.max_priority = max(self.max_priority, torch.max(priorities).item())
        scaled_priorities = (priorities ** self.alpha).view(-1,1).to(torch.float64)
        self.weights[idxs] = scaled_priorities

    def __len__(self):
        return self.rl