import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils.tool import RunningMeanStd
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Normal
from itertools import chain



class weightNet(nn.Module):
    def __init__(self, input,output,hp):
        super(weightNet, self).__init__()
        self.fcs = nn.ModuleList()
        in_size = input
        hidden_sizes = hp.hidden_sizes
        self.activ = hp.Q_activ
        for next_size in hidden_sizes:
            fc = nn.Linear(in_size,next_size)
            self.fcs.append(fc)
            in_size = next_size
            
        self.last_fc = nn.Linear(in_size, output)
        self._initialize_weights()
        
    def forward(self,state,action):
        
        sa = torch.cat([state, action], -1)
        
        for fc in self.fcs:
            sa = fc(sa)
            sa = self.activ(sa)
            
        output = self.last_fc(sa)
        
        return self.activ(output)
    
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name != 'last_fc':
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                else:
                    nn.init.orthogonal_(module.weight, 0.01)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)


class ParallelizedLayerMLP(nn.Module):
    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        b = True
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn((ensemble_size,input_dim, output_dim)), requires_grad=True)
        if b:
            self.b = nn.Parameter(torch.zeros((ensemble_size,1, output_dim)).float(), requires_grad=True)
        self._initialize_weights()
         
    def forward(self, x):
        #x(ensemble_size,batch,state_dim)
        x = x @ self.W
        if self.b is not None:
            x += self.b
        return x
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.b, 0.0)


class Actor_ensemble(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hp):
        super(Actor_ensemble, self).__init__()
        self.input_size = state_dim
        self.output_size = action_dim
        self.ensemble_size = hp.N_Q
        self.log_std_min = hp.log_std_min
        self.log_std_max = hp.log_std_max
        self.max_action = max_action
        
        self.activation = hp.actor_activ
        self.fcs = nn.ModuleList()  # 使用 ModuleList 来存储网络层
        in_size = self.input_size
        hidden_sizes = hp.hidden_sizes
        
        for next_size in hidden_sizes:
            self.fcs.append(ParallelizedLayerMLP(
                ensemble_size=self.ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            ))
            in_size = next_size
            
        self.mean_linear = ParallelizedLayerMLP(
            ensemble_size=self.ensemble_size,
            input_dim=in_size,
            output_dim=self.output_size,
        )
        self.log_std_linear = ParallelizedLayerMLP(
            ensemble_size=self.ensemble_size,
            input_dim=in_size,
            output_dim=self.output_size,
        )
        self.state_norm_enable = hp.state_norm_enable
        self.state_norm = RunningMeanStd(state_dim)
        
    def forward(self, state):
        if self.state_norm_enable:
            state = self.norm(state, self.state_norm)
        sa = state
        dim=len(sa.shape)
        if dim < 3:
        # input is (ensemble_size, batch_size, output_size)
            sa = sa.unsqueeze(0)
            if dim == 1:
                sa = sa.unsqueeze(0)
            sa = sa.repeat(self.ensemble_size, 1, 1)
        h = sa
        for fc in self.fcs:
            h = fc(h)
            h = self.activation(h)
            
        mean_esamble = self.mean_linear(h)
        std_esamble = self.log_std_linear(h).exp()
        
        
        avg_mean = mean_esamble.mean(0).unsqueeze(0)  # (1, 64, 6)
        avg_var = (mean_esamble ** 2 + std_esamble ** 2).mean(0).unsqueeze(0) - avg_mean ** 2
        avg_std = avg_var.sqrt()
        
        avg_mean = avg_mean.squeeze(0)
        avg_std = avg_std.squeeze(0)
        avg_std = torch.clamp(avg_std, np.exp(self.log_std_min), np.exp(self.log_std_max)) 
        

        return avg_mean,avg_std
    
    def getAction(self,state,deterministic=False,with_logprob=True,rsample=True):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        
        
        if deterministic:
            z = mean
        else:
            if rsample:
                z = normal.rsample()
            else:
                z = normal.sample()
        action = torch.tanh(z)
        action = self.max_action*action
        
        if with_logprob:
            log_prob = normal.log_prob(z)
            log_prob -= torch.log(1 - action * action + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            log_prob = None
        return action,log_prob
    
    def norm(self, x, x_norm):
        if self.training:
            x_norm.update(x.detach())
        x = x_norm.normalize(x)
        return x
                    
class Q_ensemble(nn.Module):
    def __init__(self, input_size, output_size, hp):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ensemble_size = hp.N_Q
        self.activation = hp.Q_activ
        self.fcs = nn.ModuleList()
        in_size = input_size
        hidden_sizes = hp.hidden_sizes
        for next_size in hidden_sizes:
            self.fcs.append(ParallelizedLayerMLP(
                ensemble_size=self.ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            ))
            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=self.ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
        )
        self.state_norm_enable = hp.state_norm_enable
        self.state_norm = RunningMeanStd(input_size)
        
    def forward(self, state, action):
        if self.state_norm_enable:
            state = self.norm(state, self.state_norm)
        sa = torch.cat([state, action], -1)
        dim=len(sa.shape)
        if dim < 3:
        # input is (ensemble_size, batch_size, output_size)
            sa = sa.unsqueeze(0)
            if dim == 1:
                sa = sa.unsqueeze(0)
            sa = sa.repeat(self.ensemble_size, 1, 1)
        h = sa
        for fc in self.fcs:
            h = fc(h)
            h = self.activation(h)
        output = self.last_fc(h)
        if dim == 1:
            output = output.squeeze(1)
        return output
    
    def norm(self, x, x_norm):
        if self.training:
            x_norm.update(x.detach())
        x = x_norm.normalize(x)
        return x
    
    def getQ(self,state,action):
        q = self.forward(state,action)
        return q
    
                           
                     
class agent(object):
    def __init__(self,state_dim, action_dim, max_action, hp) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor = Actor_ensemble(state_dim,action_dim,max_action,hp).to(self.device)        
        self.critic1 = Q_ensemble((state_dim+action_dim),1,hp).to(self.device)     
        self.critic2 = Q_ensemble((state_dim+action_dim),1,hp).to(self.device)
        self.weightNet = weightNet((state_dim+action_dim),1,hp).to(self.device)
        self.target_Q1 = copy.deepcopy(self.critic1)   
        self.target_Q2 = copy.deepcopy(self.critic2)   
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
        self.critic_optimizer = torch.optim.Adam(chain(self.critic1.parameters(), self.critic2.parameters()), lr=hp.Q_lr)
        self.weight_optimzier = torch.optim.Adam(self.weightNet.parameters(),lr=hp.Q_lr)
        
        # lr_dacay
        if hp.retrain:
            lambda_lr = lambda step: (  
                0.1  + 0.9 * (step / (hp.warmup * hp.max_steps))  
                if step <= int(hp.warmup * hp.max_steps) else  
                (  
                    0.5 * (1.0 + math.cos(math.pi * ((step - int(hp.warmup * hp.max_steps)) / (hp.max_steps - int(hp.warmup * hp.max_steps)))))
                    if int(hp.warmup * hp.max_steps) < step < int(0.9 * hp.max_steps) else  
                    0.1 
                )  
            )
        else:
            lambda_lr = lambda step: 1.0 - step / hp.max_steps if step < int(0.9 *hp.max_steps) else 0.1
        
        ## 
        self.lr_decay_enable = hp.lr_decay_enable
        self.grad = hp.grad
        self.discount = hp.discount
        
        if self.lr_decay_enable:
            self.scheduler = [LambdaLR(self.actor_optimizer, lr_lambda=lambda_lr),
                          LambdaLR(self.critic_optimizer, lr_lambda=lambda_lr)]
        
        ## SAC
        self.adaptive_alpha = hp.adaptive_alpha_enable
        if self.adaptive_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True ,device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=hp.actor_lr)
        else:
            self.alpha = torch.tensor(hp.alpha)
        self.tau = hp.tau
        
        
        # CQL
        self.num_random = hp.num_random
        self.H = hp.H
        self.min_q_weight = hp.min_q_weight
        self.temp = hp.temp
        self.with_lagrange = hp.with_lagrange
        self.log_alpha_prime = torch.nn.Parameter(torch.zeros(1))
        self.target_action_gap = hp.target_action_gap
        self.alpha_prime_optimizer = torch.optim.Adam([self.log_alpha_prime],lr = hp.alpha_lr)
        self.alpha_min = hp.alpha_min
        self.alpha_max = hp.alpha_max
        if self.target_action_gap < 0.0:
            self.with_lagrange = False
            
        # offline2online
        self.temperature = hp.temperature
        self.N = hp.N_Q
        
        ## checkpoint
        self.Maxscore = (0.0,0.0)
        self.learn_step = 0
        
        self.training(False)
        
        
        
    @torch.no_grad()
    def select_action(self,state,eval=False):
        
        if state.ndim == 1:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        else:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).squeeze().to(self.device)
        
        action,_ = self.actor.getAction(state,deterministic=eval)
        action = action.view(self.action_dim).cpu().data.numpy()
        return action
    
    
    def training(self,train=True):
        if train:
            self.actor.train()
            self.critic1.train()
            self.critic2.train()
            self.Train = True
        else:
            self.actor.eval()
            self.critic1.eval()
            self.critic2.eval()
            self.Train = False
            
    def get_action_prob(self,obs):
        obs_temp = obs.unsqueeze(1).repeat(1, self.num_random, 1).view(obs.shape[0] * self.num_random, obs.shape[1])
        actions,log_p = self.actor.getAction(obs_temp)
        
        return actions,log_p.view(obs.shape[0], self.num_random, 1).repeat(self.N, 1, 1)
        

    def get_value(self,obs,actions):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        
        q1,q2 = self.critic1.getQ(obs_temp,actions),self.critic2.getQ(obs_temp,actions)
        
        return q1.view(obs.shape[0] * self.N, num_repeat, 1),q2.view(obs.shape[0] * self.N, num_repeat, 1)
    
    
    def offlineTrain(self,sample,process,writer):
        
        self.learn_step += 1
        
        state, action,next_state,reward,mask = sample
        
        with torch.no_grad():
            next_action,next_log_prob = self.actor.getAction(next_state)
            target_q1,target_q2 = self.target_Q1.getQ(next_state,next_action),self.target_Q2.getQ(next_state,next_action)
            target_q = torch.min(target_q1,target_q2)
            target_value = target_q - self.alpha * next_log_prob
            next_q_value = reward + mask * self.discount * target_value
        
        # 这里完全可以用QR-DQN的方式更新，这里目前是1对1的更新
        q1,q2 = self.critic1.getQ(state,action),self.critic2.getQ(state,action)
        q1_loss = (0.5 * (q1 - next_q_value)**2).mean()
        q2_loss = (0.5 * (q2 - next_q_value)**2).mean()
        
        
        random_actions = torch.FloatTensor(state.shape[0] * self.num_random, action.shape[-1]).uniform_(-1, 1).to(self.device)
        
        obs = state
        next_obs = next_state
        
        curr_actions, curr_log_p = self.get_action_prob(obs)
        next_actions, next_log_p = self.get_action_prob(next_obs)
        
        
        q1_rand,q2_rand = self.get_value(obs,random_actions)
        q1_curr,q2_curr = self.get_value(obs,curr_actions)
        q1_next,q2_next = self.get_value(next_obs,next_actions)
        
        cat_q1 = torch.cat(
            [q1_rand, q1.view(-1,1).unsqueeze(1),q1_next, q1_curr], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2.view(-1,1).unsqueeze(1),q2_next, q2_curr], 1
        )
        
        
        if self.H:
            random_density = np.log(0.5 ** curr_actions.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next - next_log_p.detach(), q1_curr - curr_log_p.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next - next_log_p.detach(), q2_curr - curr_log_p.detach()], 1
            )
            
        min_q1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_q2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        
        
        min_q1_loss = min_q1_loss - q1.mean() * self.min_q_weight
        min_q2_loss = min_q2_loss - q2.mean() * self.min_q_weight
        
        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=self.alpha_min, max=self.alpha_max).to(self.device)
            min_q1_loss = alpha_prime * (min_q1_loss - self.target_action_gap)
            min_q2_loss = alpha_prime * (min_q2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_q1_loss - min_q2_loss)*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
            
        q1_loss = q1_loss + min_q1_loss
        q2_loss = q2_loss + min_q2_loss
        
        q_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(chain(self.critic1.parameters(), self.critic2.parameters()), self.grad)
        self.critic_optimizer.step()
        
        new_action,new_log_prob = self.actor.getAction(state)
        new_q1,new_q2 = self.critic1.getQ(state,new_action),self.critic2.getQ(state,new_action)
        new_q = torch.min(new_q1,new_q2).mean(0)
        
        actor_loss = (self.alpha * new_log_prob - new_q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad)
        self.actor_optimizer.step()
        
        
        
        with torch.no_grad():
            for (target_param1, param1), (target_param2, param2) in zip(
                                    zip(self.target_Q1.parameters(), self.critic1.parameters()), 
                                    zip(self.target_Q2.parameters(), self.critic2.parameters())):
                target_param1.data.copy_(target_param1.data * (1 - self.tau) + param1.data * self.tau)
                target_param2.data.copy_(target_param2.data * (1 - self.tau) + param2.data * self.tau)
                
        
        
        alpha_loss = torch.tensor(0)
        if self.adaptive_alpha:
            alpha_loss = (-self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        
        if self.lr_decay_enable:
            for scheduler in self.scheduler:
                scheduler.step()
        
        writer.add_scalar('actor_loss', actor_loss.item(), global_step=self.learn_step)
        writer.add_scalar('q_loss', q_loss.item(), global_step=self.learn_step)
        
        if self.learn_step % 1000 ==0:
            process.process_input(self.learn_step, 'learn_step', 'train/')
            process.process_input(actor_loss.item(), 'actor_loss', 'train/')
            process.process_input(q_loss.item(), 'q_loss', 'train/')
            process.process_input(next_q_value.detach().cpu().numpy(), 'next_q_value', 'train/')
            process.process_input(q1_rand.detach().cpu().numpy(), 'q1_rand', 'train/')
            process.process_input(q1_next.detach().cpu().numpy(), 'q1_next', 'train/')
            process.process_input(q1_curr.detach().cpu().numpy(), 'q1_curr', 'train/')
        
        
        
    def train(self,offline_sample,online_sample,sample,process,writer):
        
        self.learn_step += 1
        offlineState,offlineAction,_,_,_ = offline_sample
        onlineState,onlineAction,_,_,_ = online_sample
        state,action,next_state,reward,mask = sample
        
        offline_weight  = self.weightNet(offlineState,offlineAction)
        online_weight  = self.weightNet(onlineState,onlineAction)
        offline_f_star = -torch.log(2.0 / (offline_weight + 1) + 1e-10)
        online_f_prime = torch.log(2.0 * online_weight / (online_weight + 1) + 1e-10)
        weight_loss = (offline_f_star - online_f_prime).mean()
        
        
        self.weight_optimzier.zero_grad()
        weight_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.weightNet.parameters(), self.grad)
        self.weight_optimzier.step()
        
        
        with torch.no_grad():
            weight = self.weightNet(state, action)
            normalized_weight = (weight ** (1 / self.temperature)) / (
                (offline_weight ** (1 / self.temperature)).mean() + 1e-10
            )
            new_priority = normalized_weight.clamp(0.001, 1000)
            new_priority = new_priority.squeeze().detach()
            
            
        with torch.no_grad():
            next_action,next_log_prob = self.actor.getAction(next_state)
            target_q1,target_q2 = self.target_Q1.getQ(next_state,next_action),self.target_Q2.getQ(next_state,next_action)
            target_q = torch.min(target_q1,target_q2)
            target_value = target_q - self.alpha * next_log_prob
            next_q_value = reward + mask * self.discount * target_value
        
        # 这里完全可以用QR-DQN的方式更新，这里目前是1对1的更新
        q1,q2 = self.critic1.getQ(state,action),self.critic2.getQ(state,action)
        q1_loss = (0.5 * (q1 - next_q_value)**2).mean()
        q2_loss = (0.5 * (q2 - next_q_value)**2).mean()
        
        q_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(chain(self.critic1.parameters(), self.critic2.parameters()), self.grad)
        self.critic_optimizer.step()
        
        
        new_action,new_log_prob = self.actor.getAction(state)
        new_q1,new_q2 = self.critic1.getQ(state,new_action),self.critic2.getQ(state,new_action)
        new_q = torch.min(new_q1,new_q2).mean(0)
        
        actor_loss = (self.alpha * new_log_prob - new_q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad)
        self.actor_optimizer.step()
        
        
        
        with torch.no_grad():
            for (target_param1, param1), (target_param2, param2) in zip(
                                    zip(self.target_Q1.parameters(), self.critic1.parameters()), 
                                    zip(self.target_Q2.parameters(), self.critic2.parameters())):
                target_param1.data.copy_(target_param1.data * (1 - self.tau) + param1.data * self.tau)
                target_param2.data.copy_(target_param2.data * (1 - self.tau) + param2.data * self.tau)
                
        alpha_loss = torch.tensor(0)
        if self.adaptive_alpha:
            alpha_loss = (-self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        
        if self.lr_decay_enable:
            for scheduler in self.scheduler:
                scheduler.step()
        
        writer.add_scalar('actor_loss', actor_loss.item(), global_step=self.learn_step)
        writer.add_scalar('q_loss', q_loss.item(), global_step=self.learn_step)
        
        if self.learn_step % 1000 ==0:
            process.process_input(self.learn_step, 'learn_step', 'train/')
            process.process_input(actor_loss.item(), 'actor_loss', 'train/')
            process.process_input(q_loss.item(), 'q_loss', 'train/')
            process.process_input(weight_loss.item(), 'weight_loss', 'train/')
            process.process_input(next_q_value.detach().cpu().numpy(), 'next_q_value', 'train/')
            
        return new_priority
        
    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'running_mean_std_state': {
                'n': self.actor.state_norm.n,
                'mean': self.actor.state_norm.mean,
                'S': self.actor.state_norm.S,
                'std': self.actor.state_norm.std
            }
        }, filename + "_actor")
        
        # 如果需要保存优化器状态，可以取消注释以下代码
        # torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        # torch.save({
        #     'critic_state_dict': self.critic.state_dict(),
        # }, filename + "_critic")
        # torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        
        
    def load(self, filename):
        checkpoint = torch.load(filename + "_actor")
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        
        # 恢复 RunningMeanStd 的状态
        self.actor.state_norm.n = checkpoint['running_mean_std_state']['n']
        self.actor.state_norm.mean = checkpoint['running_mean_std_state']['mean']
        self.actor.state_norm.S = checkpoint['running_mean_std_state']['S']
        self.actor.state_norm.std = checkpoint['running_mean_std_state']['std']
        
        # 如果需要加载优化器状态，可以取消注释以下代码
        # self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        
        
    def IsCheckpoint(self, score, maxscore=None):
        """
        判断当前得分是否为最佳得分。

        Args:
            score (tuple): (fin, score), fin代表完成的关卡数，score表示总体表现情况
            maxscore (list, optional): 当前最高分。如果为None，则使用self.Maxscore

        Returns:
            bool: 是否是最佳得分
        """
        current_max = maxscore if maxscore is not None else self.Maxscore
        
        if score[0] > current_max[0] or (score[0] == current_max[0] and score[1] > current_max[1]):
            if maxscore is None:
                self.Maxscore = score
            else:
                maxscore[:] = score
            return True
        return False