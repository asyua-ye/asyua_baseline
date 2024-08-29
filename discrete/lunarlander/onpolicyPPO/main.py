import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import time
import datetime
import PPO
from utils.tool import DataProcessor,log_and_print
from buffer import RolloutBuffer,ReplayBuffer
from dataclasses import dataclass, asdict
from pre_env import ProcessEnv
from subproc_vec_env import SubprocVecEnv


@dataclass
class Hyperparameters:
    # Generic
    buffer_size: int = 1
    discount: float = 0.98
    gae: float = 0.99
    grad: float = 0.5
    num_processes: int = 4
    num_steps: int = 2048
    device: torch.device = None
    max_steps: int = 0
    
    lr_decay_enable: bool = False
    state_norm_enable: bool = False
    reward_scale_enable: bool = False 
    eps: float = 1e-5
    
    # actor
    actor_lr: float = 1e-4
    warmup: float = 0.1
    entropy: float = 0.01
    
    # critic
    critic_lr: float = 1e-4
    
    # ppo
    clip: float = 0.20
    ppo_update: int = 40
    mini_batch: int = 32
    value: float = 0.5
    actor: float = 1.0
    
    
    # RL
    env: str = "LunarLander-v2"
    seed: int = 0
    test: bool = False
    epoch_per_train: int = 6000

    # Evaluation
    checkpoint: bool = True
    eval_eps: int = 10
    max_epoches: int = 400
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
    



def train_online(RL_agent, env, eval_env, rollout, replaybuffer, hp):
    
    if hp.num_processes == 1:
        state,_ = env.reset()
        mask = torch.ones((hp.num_processes,1)).to(hp.device)
    else:    
        state,mask = env.reset(),torch.ones((hp.num_processes,1)).to(hp.device)
    
    episode_rewards = np.zeros(hp.num_processes, dtype=np.float64)
    final_rewards = np.zeros(hp.num_processes, dtype=np.float64)
    rounds = np.zeros(hp.num_processes, dtype=np.float64)
    rds = 0
    start_time = time.time()
    file_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    writer = SummaryWriter(f'./output/{hp.file_name}/{file_time}/{file_time}-PPO-{hp.total_epoches}')
    if hp.checkpoint and not os.path.exists(f"./output/{hp.file_name}/{file_time}/checkpoint/models"):
        os.makedirs(f"./output/{hp.file_name}/{file_time}/checkpoint/models")
    train = []
    log_and_print(train, (f"begin time:  {file_time}\n"))
    if not os.path.exists(f"./output/{hp.file_name}/{file_time}/models"):
        output_directory = f'./output/{hp.file_name}/{file_time}/models'
        os.makedirs(output_directory)
        hp.save_to_file(os.path.join(f'./output/{hp.file_name}/{file_time}/models', 'hyperparameters.txt'))
    process = DataProcessor(f'./output/{hp.file_name}/{file_time}/')
    Maxscore = [0.0,0.0]
    eval_fin = 0
    total_eval_reward = 0
    eval_all = 0
    for t in range(int(hp.total_epoches+1)):
        s1 = time.time()
        
        total_reward = []
        fin = 0   
        for step in range(hp.num_steps):
            rounds += 1
            action,log_prob,value = RL_agent.select_action(np.array(state))
            
            if hp.num_processes !=1:
                next_state, reward, ep_finished, info = env.step(action)
                next_mask =  1. - ep_finished.astype(np.float32)
            else:
                action = action[0]
                next_state, reward, ep_finished, _, info = env.step(action)
                next_mask = 1. - float(ep_finished)
                reward = np.array(reward)
                next_state = np.array(next_state)
                next_mask = np.array(next_mask)
                action = np.array(action)
                
            episode_rewards += reward
            final_rewards *= next_mask
            final_rewards += (1. - next_mask) * episode_rewards
            temp = (1. - next_mask) * episode_rewards
            total_reward.extend(temp[temp!=0])
            episode_rewards *= next_mask
            rds += np.sum(ep_finished==1)
            
            
            reward = torch.from_numpy(reward.astype(np.float32)).view(-1, 1).to(hp.device)
            next_mask = torch.from_numpy(next_mask).view(-1, 1).to(hp.device)
            state = torch.from_numpy(state.astype(np.float64)).to(hp.device)
            action = torch.from_numpy(action.astype(np.float64)).view(-1, 1).to(hp.device)
            log_prob = torch.from_numpy(log_prob).to(hp.device)
            value = torch.from_numpy(value).to(hp.device)
                       
            rollout.insert(state, action, reward, mask, log_prob, value)
            
            state = next_state
            mask = next_mask
            if torch.any(mask == 0).item() and np.any(final_rewards != 0):
                non_zero_rewards = final_rewards[final_rewards != 0]
                log_and_print(train, (
                    f"T: {t} Total T: {np.sum(rounds)} Total R: {rds}  mean: {np.mean(non_zero_rewards):.3f} "
                    f"mid: {np.median(non_zero_rewards):.3f} max: {np.max(non_zero_rewards):.3f} "
                    f"min: {np.min(non_zero_rewards):.3f}"
                        ))
                if hp.num_processes != 1:
                    for i, m in enumerate(mask):
                        if m==0:
                            if reward[i] >= 100:
                                fin += 1
                else:
                    if reward>=100:
                        fin += 1
                    state,_ = env.reset()
                    mask = torch.ones((hp.num_processes,1)).to(hp.device)
        if hp.checkpoint:
            rs = len(total_reward)
            if fin > rs:
                fin = rs
            if rs!=0:
                all_fin = (fin + eval_fin) / (rs + eval_all) * 100
                sample_fin = fin / rs * 100
                all_score = np.min(total_reward) + np.min(total_eval_reward)
                sample_score = np.min(total_reward)
            else:
                all_fin = 0
                all_score = 0.0
            
            log_and_print(train, f"total {rs}  This Score：{(sample_fin,sample_score)}")
            log_and_print(train, f"checkpoint  total {rs + eval_all}  This Score：{(all_fin,all_score)} Max Score:{Maxscore}")
            flag = RL_agent.IsCheckpoint((all_fin,all_score),Maxscore)
            if flag:
                RL_agent.save(f"./output/{hp.file_name}/{file_time}/checkpoint/models/")
        e = time.time()
        sample_time = (e-s1)
        log_and_print(train, f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")
        s = time.time()
        rollout.lastInsert(torch.from_numpy(next_state.astype(np.float64)).to(hp.device),next_mask)
        
        next_value = RL_agent.get_value(np.array(next_state))
        next_value = torch.from_numpy(next_value).view(-1,1).to(hp.device)
        
        states,actions,log_prob,advs,returns = rollout.preTrj(next_value)
        data=(np.copy(states), np.copy(actions), np.copy(log_prob),np.copy(advs),np.copy(returns))
        replaybuffer.push(data)
        log_and_print(train, f"T: {t}  R: {rds}   sample end begintrain！！")
        RL_agent.training(True)
        for sample in replaybuffer.PPOsample():
            RL_agent.train(sample,process,writer)
        RL_agent.training(False)
        e = time.time()
        train_time = (e-s)   
        if hp.checkpoint:
            s = time.time()
            text,total_eval_reward,eval_fin = maybe_evaluate_and_print(RL_agent, eval_env, t, start_time, hp)
            e = time.time()
            eval_time = (e-s)
            train.extend(text)
            process.process_input(total_eval_reward,'Returns','eval/')
            eval_all = hp.eval_eps
        np.savetxt(f"./output/{hp.file_name}/{file_time}/train.txt", train, fmt='%s')
        writer.add_scalar('eval_reward', np.mean(total_eval_reward), global_step=t)
        RL_agent.save(f'./output/{hp.file_name}/{file_time}/models/')
        process.process_input(sample_time,'sample_time(s)','time/')
        process.process_input(train_time,'train_time','time/')
        process.process_input(eval_time,'eval_time(s)','time/')
        e = time.time()
        total_time = (e-s1)
        process.process_input(total_time,'total_time(s)','time/')
        process.process_input(t,'Epoch')
        process.write_to_excel()
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
    if hp.checkpoint or hp.test:
        log_and_print(text, "---------------------------------------")
        log_and_print(text, f"Evaluation at {t} time steps")
        log_and_print(text, f"Total time passed: {round((time.time() - start_time) / 60., 2)} min(s)")
        
        total_reward = []
        
        if hp.num_processes == 1:
            state,_ = eval_env.reset()
        else:
            state = eval_env.reset()
            
        episode_rewards = np.zeros(hp.num_processes, dtype=np.float64)
        final_rewards = np.zeros(hp.num_processes, dtype=np.float64)
        fin = 0
        while True:
            action,_,_ = RL_agent.select_action(np.array(state),True)
            if hp.num_processes != 1:
                next_state, reward, done, info = eval_env.step(action)
                mask =  1. - done.astype(np.float32)
            else:
                action = action[0]
                next_state, reward, done, _, info = eval_env.step(action)
                mask = 1. - float(done)
            state = next_state
            episode_rewards += reward
            final_rewards *= mask
            final_rewards += (1. - mask) * episode_rewards
            episode_rewards *= mask
            if np.any(done==1):
                total_reward.extend(final_rewards[final_rewards!=0])
                final_rewards[final_rewards!=0] = 0
                if hp.num_processes != 1:
                    for i, d in enumerate(done):
                        if d == 1:
                            if reward[i] >=100:
                                fin += 1
                else:
                    if reward>=100:
                        fin += 1
                    state = eval_env.reset()
                    state = state[0]
            if len(total_reward)>=hp.eval_eps:
                break
        log_and_print(text, f"level cleared: {fin/hp.eval_eps*100:.2f}% , Average total reward over {hp.eval_eps} episodes: {np.mean(total_reward):.3f}")
        log_and_print(text, "---------------------------------------")
        return text,total_reward,fin



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="LunarLander-v2", type=str)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--retrain", default=False, action=argparse.BooleanOptionalAction)

    # Evaluation
    parser.add_argument("--checkpoint", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_eps", default=2, type=int)
    parser.add_argument("--total_epoches", default=200, type=int)
    
    # File
    parser.add_argument('--file_name', default=None)
    
    args = parser.parse_args()
    
    
    if args.file_name is None:
        args.file_name = f"{args.env}"

    if not os.path.exists(f"./output/{args.file_name}"):
        os.makedirs(f"./output/{args.file_name}")
        
        
    if not os.path.exists(f"./test/{args.file_name}/model"):
        os.makedirs(f"./test/{args.file_name}/model")
        
        
    
    env = gym.make('LunarLander-v2',render_mode="human")
    eval_env = gym.make('LunarLander-v2',render_mode="human")
    
    env = ProcessEnv(env)
    eval_env = ProcessEnv(eval_env)

    print("---------------------------------------")
    print(f"Algorithm: PPO, Env: {args.env}, Seed: {args.seed}")
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
        checkpoint=args.checkpoint,
        eval_eps=args.eval_eps,
        total_epoches=args.total_epoches,
        file_name=args.file_name,
        retrain = args.retrain
    )
    if hp.max_epoches > args.total_epoches:
        hp.max_epoches = args.total_epoches
        
    hp.max_steps = hp.max_epoches * hp.mini_batch * hp.ppo_update
    hp.eval_eps = args.eval_eps * hp.num_processes



    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    
    if not hp.test:
        envs = SubprocVecEnv(hp.num_processes,hp.env) if hp.num_processes > 1 else env
        
        
    eval_envs = SubprocVecEnv(hp.num_processes,hp.env) if hp.num_processes > 1 else eval_env
        
    RL_agent = PPO.agent(state_dim, action_dim, hp)
    
    rollout = RolloutBuffer(state_dim, hp)
    replaybuffer = ReplayBuffer(state_dim,hp)

    
    if args.test:
        file_name = f"./test/{args.file_name}/model/"
        toTest(RL_agent, eval_envs,file_name, hp)
    else:
        from torch.utils.tensorboard import SummaryWriter
        if args.retrain:
            file_name = f"./test/{args.file_name}/model/"
            RL_agent.load(file_name)
        train_online(RL_agent, envs, eval_envs, rollout, replaybuffer, hp)







