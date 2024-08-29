import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils.tool import RunningMeanStd
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Normal





class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hp):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.state_norm_enable = hp.state_norm_enable
        self.log_std_min = hp.log_std_min
        self.log_std_max = hp.log_std_max
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)
        
        self.state_norm = RunningMeanStd(state_dim)
        self._initialize_weights()
        
    def forward(self, state):
        if self.state_norm_enable:
            state = self.norm(state, self.state_norm)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        mean = self.mean_linear(x)
        logstd = self.log_std_linear(x)
        logstd = torch.clamp(logstd, self.log_std_min, self.log_std_max)
        
        return mean,logstd
    
    def getAction(self, state, deterministic=False, with_logprob=True):
        
        mean,log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        if deterministic:
            z = mean
        else:
            z = normal.rsample()
        
        if with_logprob:
            log_prob = normal.log_prob(z).sum(dim=1, keepdim=True)
            log_prob -= (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(dim=1, keepdim=True)
        else:
            log_prob = None
        
        action = torch.tanh(z)
        action = self.max_action*action
        
        return action,log_prob
    
    def getlogprob(self,state,action):
        
        mean, log_std = self.forward(state)
        old_z = torch.atanh(torch.clamp(action, min=-0.999999, max=0.999999))
        std = log_std.exp()
        normal = Normal(mean, std)
        old_logprob = normal.log_prob(old_z) - torch.log(1 - action.pow(2) + 1e-6)
        old_logprob = old_logprob.sum(-1, keepdim=True)
        
        return old_logprob
            
    def norm(self, x, x_norm):
        if self.training:
            x_norm.update(x.detach())
        x = x_norm.normalize(x)
        return x
    
    def _initialize_weights(self):
        
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name != 'mean_linear' or name != 'log_std_linear':
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                else:
                    nn.init.orthogonal_(module.weight, 0.01)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
                
                
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hp):
        super(Critic, self).__init__()
        
        
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, 256)
        self.q1 = nn.Linear(256, 1)
        
        self.fc2 = nn.Linear(state_dim + action_dim, 256)
        self.fc22 = nn.Linear(256, 256)
        self.fc23 = nn.Linear(256, 256)
        self.fc24 = nn.Linear(256, 256)
        self.q2 = nn.Linear(256, 1)
        
        self.state_norm_enable = hp.state_norm_enable
        self.state_norm = RunningMeanStd(state_dim)
        self._initialize_weights()
        
    def forward(self, state, action):
        if self.state_norm_enable:
            state = self.norm(state, self.state_norm)
        
        sq = torch.cat([state,action],dim=-1)    
        
        x1 = F.relu(self.fc1(sq))
        x1 = F.relu(self.fc11(x1))
        x1 = F.relu(self.fc12(x1))
        x1 = F.relu(self.fc13(x1))
        q1 = self.q1(x1)
        

        x2 = F.relu(self.fc2(sq))
        x2 = F.relu(self.fc22(x2))
        x2 = F.relu(self.fc23(x2))
        x2 = F.relu(self.fc24(x2))
        q2 = self.q2(x2)
        
        return q1,q2
    
    def norm(self, x, x_norm):
        if self.training:
            x_norm.update(x.detach())
        x = x_norm.normalize(x)
        return x
    
    def getQ(self,state,action):
        q1,q2= self.forward(state,action)
        return q1,q2
        
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name != 'q1' or name != 'q2':
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
                     
                
class agent(object):
    def __init__(self,state_dim, action_dim, max_action, hp) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim,action_dim,max_action,hp).to(self.device)        
        self.critic = Critic(state_dim,action_dim,hp).to(self.device)     
        self.target_Q = copy.deepcopy(self.critic)   
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=hp.actor_lr, weight_decay=hp.actor_weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
        
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
        
        
        ## AWAC
        self.awr_weight = hp.awr_weight
        self.beta = hp.beta
        self.weight_loss = hp.weight_loss
        
        
        
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
            self.critic.train()
            self.Train = True
        else:
            self.actor.eval()
            self.critic.eval()
            self.Train = False
            
    def train(self,sample,process,writer):
        
        self.learn_step += 1
        state,action,next_state,reward,mask = sample
        
        q1,q2=self.critic.getQ(state,action)    #  这里如果之后直接更新Q会出问题
        
        
        
        with torch.no_grad():
            next_action,next_log_prob = self.actor.getAction(next_state)
            target_q1,target_q2 = self.target_Q.getQ(next_state,next_action)
            target_q = torch.min(target_q1,target_q2)
            target_value = target_q - self.alpha * next_log_prob
            next_q_value = reward + mask * self.discount * target_value
        
        
        
        q1_loss = (0.5 * (q1 - next_q_value)**2).mean()
        q2_loss = (0.5 * (q2 - next_q_value)**2).mean()
        q_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad)
        self.critic_optimizer.step()
        
        
        
        new_action,new_log_prob = self.actor.getAction(state)
        log_old_prob = self.actor.getlogprob(state,action)
        
        new_q1,new_q2 = self.critic.getQ(state,new_action)
        expected_q_value = torch.min(q1,q2)
        value = torch.min(new_q1,new_q2)
        adv = expected_q_value - value
            
        
        actor_loss = self.alpha * new_log_prob.mean()
        
        if self.weight_loss:
            weights = F.softmax(adv / self.beta, dim=0)  # 归一化的目的是防止梯度爆炸
            actor_loss += self.awr_weight*(-log_old_prob* len(weights) * weights.detach()).mean()
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # adv[adv<0] *= 0.001
            # actor_loss += (-log_old_prob * adv.detach()).mean()
        else:
            actor_loss += self.awr_weight*(-log_old_prob).mean()
        
                
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad)
        self.actor_optimizer.step()
        
        
        
        with torch.no_grad():
            for target_param,param in zip(self.target_Q.parameters(),self.critic.parameters()):
                target_param.data.copy_(
                target_param.data *(1 - self.tau)  + param.data * self.tau
            )
            
            
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
            process.process_input(adv.detach().cpu().numpy(), 'adv', 'train/')
            
            
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