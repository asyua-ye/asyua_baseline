import numpy as np
import torch
# import gym
import gymnasium as gym
import random
import argparse
import os
import time
import datetime
import DT
import buffer
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, asdict
from utils.tool import DataProcessor,log_and_print,initialize_env,get_normalized_score
from pre_env import ProcessEnv



@dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 64
    buffer_size: int = int(1e6)
    
    # DT
    K: int = 20  
    pct_traj: float = 1.
    vocab_size: int = 1 
    n_layer: int = 3 
    n_head: int = 1 
    n_embd: int = 128 
    dropout: float = 0.1
    max_ep_len: int = 1000
    weight_decay: float = 1e-4
    warmup_steps: int = 10000
    learning_rate: float = 1e-4
    block_size: int = 1024
    
    # RL
    env: str = "Hopper-v2"
    offlineEnv: str = "hopper-medium-v2"
    seed: int = 0
    OFFline: bool = True
    num_steps_per_iter: int = 10000
    max_timesteps: int = 10
    
    # Eval
    num_eval_episodes: int = 100
    d4rl: bool = True
    
    # File
    file_name: str = None
    
    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def save_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for key, value in asdict(self).items():
                f.write(f"{key}: {value}\n")
                
    def update_from_args(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)


def train_offline(RL_agent, replaybuffer, eval_env, start_time, filename, hp):
    
    train = []
    s = time.time()
    d4rl_dataset_path = f'E:/python/rl-work/GitHub/asyua_baseline/offline/dataset/{hp.offlineEnv}.hdf5'
    dataset_path = f'./data/{hp.offlineEnv}.pkl'
    offlinereplaybuffer.load_dataset(d4rl_dataset_path,dataset_path)
    e = time.time()
    data_times = (e-s)
    log_and_print(train, (f"dataset process time:  {data_times:.2f}s\n"))
    env_name = eval_env.spec.id.split('-')[0].lower()  # 获取环境名称的小写版本
    env_targets, scale = initialize_env(env_name)  

    file_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    writer = SummaryWriter(f'./output/{filename}/{file_time}/{file_time}-DT-{hp.max_timesteps}')
    log_and_print(train, (f"begin time:  {file_time}\n"))
    if not os.path.exists(f"./output/{filename}/{file_time}/models/"):
        output_directory = f'./output/{filename}/{file_time}/models/'
        os.makedirs(output_directory)
        hp.save_to_file(os.path.join(output_directory, 'hyperparameters.txt'))
    process = DataProcessor(f'./output/{filename}/{file_time}/')
    
    for t in range(int(hp.max_timesteps)):
        s1 = time.time()
        log_and_print(train, f"T: {t}  begintrain！！")
        RL_agent.training(True)
        for i in range(int(hp.num_steps_per_iter)):
            RL_agent.Offlinetrain(replaybuffer.sample(),process,writer)
            if i < int(hp.num_steps_per_iter) - 1 and RL_agent.learn_step%1000 == 0 :
                process.new_record()
        RL_agent.training(False)
        e = time.time()
        train_time = (e-s1)
        
        for tar in env_targets:
            s = time.time()
            text,total_reward,total_length = maybe_evaluate_and_print(RL_agent, replaybuffer, eval_env, t, start_time, hp, tar, scale)
            e = time.time()
            eval_time = (e-s)
            train.extend(text)
            process.process_input(total_reward,f'{tar}-Returns','eval/')
            process.process_input(total_length,f'{tar}-Length','eval/')
            
        np.savetxt(f"./output/{filename}/{file_time}/train.txt", train, fmt='%s')
        RL_agent.save(f'./output/{filename}/{file_time}/models/')
        process.process_input(train_time,'train_time','time/')
        process.process_input(eval_time,'eval_time(s)','time/')
        e = time.time()
        total_time = (e-s1)
        process.process_input(total_time,'total_time(s)','time/')
        process.process_input(t,'Epoch')
        process.new_record()
        process.write_to_excel()
    eval_env.close()
    file_time1 = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_and_print(train, f"\nend time: {file_time1}")
    np.savetxt(f"./output/{hp.file_name}/{file_time}/train.txt", train, fmt='%s')
        
        
def maybe_evaluate_and_print(RL_agent, replaybuffer, eval_env, t, start_time, hp, tar, scale):

    if RL_agent.Train:
        RL_agent.training(False)
    
    text = []
    log_and_print(text, "---------------------------------------")
    log_and_print(text, f"Evaluation at {t} time steps")
    log_and_print(text, f"Total time passed: {round((time.time() - start_time) / 60., 2)} min(s)")
    device = hp.device
    state_mean = torch.from_numpy(replaybuffer.state_mean).to(device=device)
    state_std = torch.from_numpy(replaybuffer.state_std).to(device=device)
    act_dim = replaybuffer.action_dim
    total_reward = np.zeros(hp.num_eval_episodes)
    total_length = np.zeros(hp.num_eval_episodes)
    tar = tar/scale
    
    for ep in range(hp.num_eval_episodes):
        state, done = eval_env.reset(), False
        state = state[0]  # v4
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        target_return = torch.tensor(tar, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        
        i = 0
        while not done:
            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            
            action = RL_agent.select_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()
            # state, reward, done, _ = eval_env.step(action)  # v2
            state, reward, done, _, _ = eval_env.step(action) # v4
            total_reward[ep] += reward
            total_length[ep] += 1
            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward
            
            pred_return = target_return[0,-1] - (reward/scale)
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (i+1)], dim=1)
            i += 1
            

    log_and_print(text, f"Average total reward over {hp.num_eval_episodes} episodes: {np.mean(total_reward):.3f}")
        
    if hp.d4rl:
        d4rl_score = get_normalized_score(np.array(total_reward),hp.offlineEnv)
        log_and_print(text, f"D4RL score: {d4rl_score.mean():.3f}")
    
    log_and_print(text, "---------------------------------------")
    
    
    return text,total_reward,total_length
    
    
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="Hopper-v4", type=str)
    parser.add_argument("--offlineEnv", default="hopper-medium-v2", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--OFFline", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_timesteps", default=20, type=int)
    
    # File
    parser.add_argument('--file_name', default=None)
    args = parser.parse_args()
    
        

    if args.file_name is None:
        args.file_name = f"{args.env}"
        filename = args.file_name

    if args.OFFline:
        args.file_name = f"{args.file_name}/{args.offlineEnv}"
        filename =  args.file_name
    
    if not os.path.exists(f"./data"):
        os.makedirs(f"./data")
    
    if not os.path.exists(f"./output/{filename}"):
        os.makedirs(f"./output/{filename}")
        
    if not os.path.exists(f"./test/{filename}/model"):
        os.makedirs(f"./test/{filename}/model")
    
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    

    env = ProcessEnv(env)
    eval_env = ProcessEnv(eval_env)
    
    args.offlineEnv = args.offlineEnv.replace('-', '_', 1)
    
    hp = Hyperparameters()
    hp.update_from_args(args)
    
    
    
    print("---------------------------------------")
    print(f"Algorithm: DT, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    
    
    
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU setups
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


    RL_agent = DT.agent(state_dim, action_dim, hp)
    offlinereplaybuffer = buffer.ReplayBuffer(state_dim,action_dim,hp)
    
    
    
    if  args.OFFline:
        start_time = time.time()
        train_offline(RL_agent, offlinereplaybuffer, eval_env, start_time, filename, hp)
        