import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import time
import datetime
import SAC
import torch.nn.functional as F
from buffer import ReplayBuffer
from dataclasses import dataclass, asdict
from pre_env import ProcessEnv
from utils.tool import DataProcessor,log_and_print
from typing import Callable

@dataclass
class Hyperparameters:
    # Generic
    max_size: int = 1e6
    device: torch.device = None
    max_steps: int = 0
    
    
    discount: float = 0.99
    grad: float = 0.5
    lr_decay_enable: bool = False
    state_norm_enable: bool = False
    reward_scale_enable: bool = False 
    eps: float = 1e-5
    batch: int = 256
    epoch_per_train: int = 50
    train_per_epoch: int = 50
    allowTrue: int = 10e3
    eval_freq: int = 10e3
    
       
    # actor
    actor_lr: float = 3e-4
    warmup: float = 0.1
    
    # critic
    critic_lr: float = 3e-4
    
    # SAC
    alpha: float = 0.2
    adaptive_alpha_enable: bool = True
    tau: float = 5e-3
    log_std_min: int = -20
    log_std_max: int = 2
    
    # RL
    env: str = "HalfCheetah-v4"
    seed: int = 0
    test: bool = False

    # Evaluation
    eval_eps: int = 10
    max_epoches: int = 1e6
    total_epoches: int = 0
    
    # File
    file_name: str = None
    
    # retrain 
    retrain: bool = True

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def save_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for key, value in asdict(self).items():
                f.write(f"{key}: {value}\n")
    



def train_online(RL_agent, env, eval_env, replaybuffer, hp):
    
    
    state,_ = env.reset()
    start_time = time.time()
    file_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    writer = SummaryWriter(f'./output/{hp.file_name}/{file_time}/{file_time}-SAC-{hp.total_epoches}')
    train = []
    log_and_print(train, (f"begin time:  {file_time}\n"))
    if not os.path.exists(f"./output/{hp.file_name}/{file_time}/models"):
        output_directory = f'./output/{hp.file_name}/{file_time}/models'
        os.makedirs(output_directory)
        hp.save_to_file(os.path.join(f'./output/{hp.file_name}/{file_time}/models', 'hyperparameters.txt'))
    process = DataProcessor(f'./output/{hp.file_name}/{file_time}/')
    total_eval_reward = np.zeros(hp.eval_eps)
    eval_time = 0
    ep_total_reward = 0
    ep_num = 0
    ep_timesteps = 0
    train_num = 0
    for t in range(int(hp.total_epoches+1)):
        if t >= hp.allowTrue:
            action = RL_agent.select_action(np.array(state))
        else:
            action = env.action_space.sample()
        next_state, reward, ep_finished, _, _ = env.step(action) 
        ep_total_reward += reward
        ep_timesteps += 1
        
        mask = 1.0 - float(ep_finished) 
        replaybuffer.push(state, action, next_state, reward, mask)
        state = next_state
        
        if ep_finished: 
            log_and_print(train,f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")
            state, _ = env.reset()
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1
            
        if t >= hp.allowTrue and t % hp.epoch_per_train == 0:
            train_num += 1 
            s1 = time.time()
            RL_agent.training(True)
            for sample in replaybuffer.sample():
                RL_agent.train(sample,process,writer)
            RL_agent.training(False)
            e = time.time()
            train_time = (e-s1)   
            
            if t % hp.eval_freq == 0:
                s = time.time()
                text,total_eval_reward = maybe_evaluate_and_print(RL_agent, eval_env, train_num, start_time, hp)
                e = time.time()
                eval_time = (e-s)
                train.extend(text)
                writer.add_scalar('eval_reward', np.mean(total_eval_reward), global_step=t)
            e = time.time()
            total_time = (e-s1)
            if RL_agent.learn_step % 1000 == 0:
                np.savetxt(f"./output/{hp.file_name}/{file_time}/train.txt", train, fmt='%s')
                RL_agent.save(f'./output/{hp.file_name}/{file_time}/models/')
                process.process_input(total_eval_reward,'Returns','eval/')
                process.process_input(train_time,'train_time','time/')
                process.process_input(eval_time,'eval_time(s)','time/')
                process.process_input(total_time,'total_time(s)','time/')
                process.process_input(t,'Epoch')
                process.write_to_excel()
                eval_time = 0
                total_eval_reward = np.zeros(hp.eval_eps)
                
    env.close()
    eval_env.close()
    file_time1 = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_and_print(train, f"\nend time: {file_time1}")
    np.savetxt(f"./output/{hp.file_name}/{file_time}/train.txt", train, fmt='%s')
            
            
def toTest(RL_agent, eval_env,file_name, hp):
    RL_agent.load(file_name)
    start_time = time.time()
    file_time = None
    maybe_evaluate_and_print(RL_agent, eval_env, 0, start_time,file_time, hp)
    eval_env.close()
    
    
            
def maybe_evaluate_and_print(RL_agent, eval_env, t, start_time, hp):
    
    text = []
    if RL_agent.Train:
        RL_agent.training(False)
    
    log_and_print(text, "---------------------------------------")
    log_and_print(text, f"Evaluation at {t} time trains")
    log_and_print(text, f"Total time passed: {round((time.time() - start_time) / 60., 2)} min(s)")
    
    total_reward = []
    state,_ = eval_env.reset()
        
    episode_rewards = np.zeros(1, dtype=np.float64)
    final_rewards = np.zeros(1, dtype=np.float64)
    while True:
        action = RL_agent.select_action(np.array(state),True)
        next_state, reward, ep_finished, _, info = eval_env.step(action)
        
        mask = 1. - float(ep_finished)
        state = next_state
        episode_rewards += reward
        final_rewards *= mask
        final_rewards += (1. - mask) * episode_rewards
        episode_rewards *= mask
        if np.any(ep_finished==1):
            total_reward.extend(final_rewards[final_rewards!=0])
            final_rewards[final_rewards!=0] = 0
            state = eval_env.reset()
            state = state[0]
        if len(total_reward)>=hp.eval_eps:
            break
    log_and_print(text, f"Average total reward over {hp.eval_eps} episodes: {np.mean(total_reward):.3f}")
    log_and_print(text, "---------------------------------------")
    return text,total_reward



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="HalfCheetah-v4", type=str)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--retrain", default=False, action=argparse.BooleanOptionalAction)

    # Evaluation
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--total_epoches", default=2e6, type=int)
    
    # File
    parser.add_argument('--file_name', default=None)
    
    args = parser.parse_args()
    
    
    if args.file_name is None:
        args.file_name = f"{args.env}"

    if not os.path.exists(f"./output/{args.file_name}"):
        os.makedirs(f"./output/{args.file_name}")
        
        
    if not os.path.exists(f"./test/{args.file_name}/model"):
        os.makedirs(f"./test/{args.file_name}/model")
        
        
    
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    
    env = ProcessEnv(env)
    eval_env = ProcessEnv(eval_env)

    print("---------------------------------------")
    print(f"Algorithm: SAC, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")




    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU setups

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    
    hp = Hyperparameters(
        env=args.env,
        seed=args.seed,
        test=args.test,
        eval_eps=args.eval_eps,
        total_epoches=args.total_epoches,
        file_name=args.file_name,
        retrain = args.retrain
    )
    if hp.max_epoches > args.total_epoches:
        hp.max_epoches = args.total_epoches
        
    hp.eval_eps = args.eval_eps



    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
        
    RL_agent = SAC.agent(state_dim, action_dim, max_action, hp)
    
    replaybuffer = ReplayBuffer(state_dim, action_dim, hp)

    
    if args.test:
        file_name = f"./test/{args.file_name}/model/"
        toTest(RL_agent, eval_env,file_name, hp)
    else:
        from torch.utils.tensorboard import SummaryWriter
        if args.retrain:
            file_name = f"./test/{args.file_name}/model/"
            RL_agent.load(file_name)
        train_online(RL_agent, env, eval_env, replaybuffer, hp)







