import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import time
import datetime
import PEX
import torch.nn.functional as F
from buffer import ReplayBuffer
from dataclasses import dataclass, asdict
from pre_env import ProcessEnv
from utils.tool import DataProcessor,log_and_print,get_normalized_score
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
    epoch_per_train: int = 1
    train_per_epoch: int = 1
    allowTrue: int = 5e3
    eval_freq: int = 5e3
    OTO_epoches: int = 1e6
       
    # actor
    actor_lr: float = 3e-4
    warmup: float = 0.1
    actor_weight_decay: float = 1e-4
    
    # critic
    critic_lr: float = 3e-4
    
    
    # SAC
    alpha: float = 0.0
    adaptive_alpha_enable: bool = False
    tau: float = 5e-3
    log_std_min: int = -20
    log_std_max: int = 2
    
    
    #IQL
    t: float = 0.7   #(0,1)
    beta: float = 3.0    #[0,+inf)
    adv_clip_max: float = 100.0   
    
    #PEX
    _inv_temperature: float = 10.0
    
    # RL
    env: str = "HalfCheetah-v4"
    seed: int = 0
    test: bool = False
    offlineEnv: str = "hopper-medium-v2"
    d4rl: bool = True
    offline: bool = True
    offlinetoonline: bool = False
    
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
    writer = SummaryWriter(f'./output/{hp.file_name}/{file_time}/{file_time}-PEX-{hp.total_epoches}')
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
    
    
def train_offlineTOonline(RL_agent, env, eval_env, offlineReplayBuffer,onlineReplayBuffer, writer, process, train, start_time, file_time, hp):
    total_eval_reward = np.zeros(hp.eval_eps)
    eval_time = 0
    ep_total_reward = 0
    ep_num = 0
    ep_timesteps = 0
    train_num = 0
    RL_agent.get_offline_Actor()
    state,_ = env.reset()
    offlineReplayBuffer.batch_size = int(hp.batch/2)
    onlineReplayBuffer.batch_size = int(hp.batch/2)
    log_and_print(train,f"begin online training!!")
    for t in range(int(hp.OTO_epoches)):
        action = RL_agent.select_action(np.array(state))
        next_state, reward, ep_finished, _, _ = env.step(action) 
        ep_total_reward += reward
        ep_timesteps += 1
        
        mask = 1.0 - float(ep_finished) 
        onlineReplayBuffer.push(state, action, next_state, reward, mask)
        state = next_state
        
        if ep_finished: 
            log_and_print(train,f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")
            state, _ = env.reset()
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1
        if t % hp.epoch_per_train == 0 and t > hp.allowTrue:
            train_num += 1 
            s1 = time.time()
            RL_agent.training(True)
            for offlinesample, onlinesample in zip(offlineReplayBuffer.sample(), onlineReplayBuffer.sample()):
                state2, action2, next_state2, reward2, mask2 = offlinesample
                state1, action1, next_state1, reward1, mask1 = onlinesample
                combined_state = torch.cat((state2, state1), dim=0)
                combined_action = torch.cat((action2, action1), dim=0)
                combined_next_state = torch.cat((next_state2, next_state1), dim=0)
                combined_reward = torch.cat((reward2, reward1), dim=0)
                combined_mask = torch.cat((mask2, mask1), dim=0)
                sample = (combined_state, combined_action, combined_next_state, combined_reward, combined_mask)
                RL_agent.train(sample, process, writer)
            RL_agent.training(False)
            e = time.time()
            train_time = (e-s1)   
            
            if t % hp.eval_freq == 0:
                s = time.time()
                text,total_eval_reward = maybe_evaluate_and_print(RL_agent, eval_env, train_num, start_time, False, hp)
                e = time.time()
                eval_time = (e-s)
                train.extend(text)
                writer.add_scalar('eval_reward', np.mean(total_eval_reward), global_step=(t+hp.total_epoches))
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
    
    
    
def train_offline(RL_agent, env, eval_env,offlineReplaybuffer,onlineReplaybuffer,hp):
    start_time = time.time()
    file_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    writer = SummaryWriter(f'./output/{hp.file_name}/{file_time}/{file_time}-PEX-{hp.total_epoches}')
    train = []
    log_and_print(train, (f"begin time:  {file_time}\n"))
    if not os.path.exists(f"./output/{hp.file_name}/{file_time}/models"):
        output_directory = f'./output/{hp.file_name}/{file_time}/models'
        os.makedirs(output_directory)
        hp.save_to_file(os.path.join(f'./output/{hp.file_name}/{file_time}/models', 'hyperparameters.txt'))
    process = DataProcessor(f'./output/{hp.file_name}/{file_time}/')
    total_eval_reward = np.zeros(hp.eval_eps)
    eval_time = 0
    
    for t in range(int(hp.total_epoches)):
        s0 = time.time()
        if t % hp.eval_freq == 0:
            s = time.time()
            text,total_eval_reward = maybe_evaluate_and_print(RL_agent, eval_env, t, start_time, True, hp)
            e = time.time()
            eval_time = (e-s)
            train.extend(text)
            writer.add_scalar('eval_reward', np.mean(total_eval_reward), global_step=t)
            
        s1 = time.time()
        RL_agent.training(True)
        for sample in offlineReplaybuffer.sample():
            RL_agent.offlineTrain(sample,process,writer)
        RL_agent.training(False)
        e = time.time()
        train_time = (e-s1)   
        e = time.time()
        total_time = (e-s0)
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
    if hp.offlinetoonline:
        train_offlineTOonline(RL_agent, env, eval_env, offlineReplaybuffer,onlineReplaybuffer, writer, process, train, start_time, file_time, hp)
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
    
    
            
            
def maybe_evaluate_and_print(RL_agent, eval_env, t, start_time, offline, hp):
    
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
        action = RL_agent.select_action(np.array(state),eval=True,offline=offline)
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
    if hp.d4rl:
        d4rl_socre = get_normalized_score(np.array(total_reward),hp.env)
        log_and_print(text, f"D4RL score: {d4rl_socre.mean():.3f}")
    log_and_print(text, "---------------------------------------")
    return text,total_reward



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="Hopper-v4", type=str)
    parser.add_argument("--offlineEnv", default="hopper-medium-v2", type=str)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--retrain", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--offline", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--offlinetoonline", default=True, action=argparse.BooleanOptionalAction)

    # Evaluation
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--total_epoches", default=0, type=int)
    
    # File
    parser.add_argument('--file_name', default=None)
    
    args = parser.parse_args()
    
    
    if args.file_name is None:
        args.file_name = f"{args.env}"
        filename = args.file_name

    if args.offline:
        args.file_name = f"{args.file_name}/{args.offlineEnv}"
        filename =  args.file_name

    if not os.path.exists(f"./output/{args.file_name}"):
        os.makedirs(f"./output/{args.file_name}")
        
        
    if not os.path.exists(f"./test/{args.file_name}/model"):
        os.makedirs(f"./test/{args.file_name}/model")
        
        
    
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    
    env = ProcessEnv(env)
    eval_env = ProcessEnv(eval_env)

    print("---------------------------------------")
    print(f"Algorithm: PEX, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    
    
    args.offlineEnv = args.offlineEnv.replace('-', '_', 1)
    
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
        retrain = args.retrain,
        offlineEnv = args.offlineEnv,
        offlinetoonline=args.offlinetoonline,
        offline=args.offline,
    )
    if hp.max_epoches > args.total_epoches:
        hp.max_epoches = args.total_epoches
        
    hp.eval_eps = args.eval_eps



    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
        
    RL_agent = PEX.agent(state_dim, action_dim, max_action, hp)
    
    offlineReplaybuffer = ReplayBuffer(state_dim, action_dim, hp)
    onlineReplaybuffer = ReplayBuffer(state_dim, action_dim, hp)
    
    if args.test:
        file_name = f"./test/{args.file_name}/model/"
        toTest(RL_agent, eval_env,file_name, hp)
    elif args.offline:
        from torch.utils.tensorboard import SummaryWriter
        dataset_path = f'E:/python/rl-work/GitHub/asyua_baseline/dataset/{args.offlineEnv}.hdf5'
        offlineReplaybuffer.load_dataset(dataset_path)
        train_offline(RL_agent, env, eval_env,offlineReplaybuffer,onlineReplaybuffer,hp)
    else:
        from torch.utils.tensorboard import SummaryWriter
        if args.retrain:
            file_name = f"./test/{args.file_name}/model/"
            RL_agent.load(file_name)
        train_online(RL_agent, env, eval_env, onlineReplaybuffer, hp)







