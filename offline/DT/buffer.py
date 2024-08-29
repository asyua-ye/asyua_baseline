import numpy as np
import torch
import collections
import pickle
import os
import h5py
import random

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, hp, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.K = hp.K
        self.batch_size = hp.batch_size
        self.pct_traj = hp.pct_traj

    def load_dataset(self, d4rl_dataset_path, dataset_path):
        # 检查数据集文件是否存在
        if not os.path.exists(dataset_path):
            dataset = self.load_d4rl_dataset(d4rl_dataset_path)
            N = dataset['rewards'].shape[0]
            data_ = collections.defaultdict(list)

            use_timeouts = 'timeouts' in dataset

            episode_step = 0
            paths = []
            for i in range(N):
                done_bool = bool(dataset['terminals'][i])
                final_timestep = dataset['timeouts'][i] if use_timeouts else (episode_step == 1000 - 1)
                for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                    data_[k].append(dataset[k][i])
                if done_bool or final_timestep:
                    episode_step = 0
                    episode_data = {k: np.array(v) for k, v in data_.items()}
                    paths.append(episode_data)
                    data_ = collections.defaultdict(list)
                episode_step += 1

            # 保存处理后的数据集
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            with open(dataset_path, 'wb') as f:
                pickle.dump(paths, f)
            trajectories = paths
        else:
            # 加载已保存的数据集
            with open(dataset_path, 'rb') as f:
                trajectories = pickle.load(f)
                
        self.ProTrj(trajectories)

    def ProTrj(self, trajectories):
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        self.states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(self.states, axis=0), np.std(self.states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)
        
        # only train on top pct_traj trajectories (for %BC experiment)
        num_timesteps = max(int(self.pct_traj*num_timesteps), 1)
        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        self.sorted_inds = sorted_inds[-num_trajectories:]

        # used to reweight sampling so we sample according to timesteps instead of trajectories
        self.p_sample = traj_lens[self.sorted_inds] / sum(traj_lens[self.sorted_inds])
        self.trajectories = trajectories
        self.num_trajectories = num_trajectories

    def sample(self):
        batch_size = self.batch_size
        max_len = self.K
        max_ep_len = 1000 # for mujoco
        
        batch_inds = np.random.choice(
            np.arange(self.num_trajectories),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = self.trajectories[int(self.sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, self.state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, self.action_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, self.action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return (s, a, r, d, rtg, timesteps, mask)
    
    
    def load_d4rl_dataset(self, dataset_path):
        with h5py.File(dataset_path, 'r') as dataset:
            data_dict = {
                'observations': dataset['observations'][:],
                'actions': dataset['actions'][:],
                'next_observations': dataset['next_observations'][:],
                'rewards': dataset['rewards'][:],
                'terminals': dataset['terminals'][:],
            }
            if 'timeouts' in dataset:
                data_dict['timeouts'] = dataset['timeouts'][:]
            else:
                data_dict['timeouts'] = None

        return data_dict
        
        
        
        
        
        
        