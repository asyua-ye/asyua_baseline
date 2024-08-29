import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils.tool import RunningMeanStd
from torch.optim.lr_scheduler import LambdaLR




class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hp):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.state_norm_enable = hp.state_norm_enable
        
        
        self.fc1_1 = nn.Linear(state_dim[0], 512)
        self.mean_linear = nn.Linear(512, action_dim)
        
        self.state_norm = RunningMeanStd(state_dim)
        self._initialize_weights()
        
    def forward(self, state):
        if self.state_norm_enable:
            state = self.norm(state, self.state_norm)
        x_actor = F.relu(self.fc1_1(state))
        q = self.mean_linear(x_actor)
        
        return q
    
    def getAction(self, state):
        logit = self.forward(state)
        probs = F.softmax(logit, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        action_log_probs = (probs + 1e-8).log()
        
        return action, probs, action_log_probs
    
        
    def norm(self, x, x_norm):
        if self.training:
            x_norm.update(x.detach())
        x = x_norm.normalize(x)
        return x
    
    def _initialize_weights(self):
        
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name == 'mean_linear' :
                    nn.init.orthogonal_(module.weight, 0.01)
                else:
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
                
                
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hp):
        super(Critic, self).__init__()
        
        self.state_norm_enable = hp.state_norm_enable
        self.fc1 = nn.Linear(state_dim[0], 512)
        self.fc2 = nn.Linear(state_dim[0], 512)
        self.q1 = nn.Linear(512, action_dim)
        self.q2 = nn.Linear(512, action_dim) 
        
        self.state_norm = RunningMeanStd(state_dim)
        self._initialize_weights()
        
    def forward(self, state):
        if self.state_norm_enable:
            state = self.norm(state, self.state_norm)
            
        x1 = F.relu(self.fc1(state))
        q1 = self.q1(x1)
        
        x2 = F.relu(self.fc2(state))
        q2 = self.q2(x2)  
        
        return q1, q2  
    
    def norm(self, x, x_norm):
        if self.training:
            x_norm.update(x.detach())
        x = x_norm.normalize(x)
        return x
    
    def getQ(self, state):
        q1, q2 = self.forward(state)
        return q1, q2  
    
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
               
                
                
class agent(object):
    def __init__(self,state_dim, action_dim,hp) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim,action_dim,hp).to(self.device)        
        self.critic = Critic(state_dim,action_dim,hp).to(self.device)     
        self.target_Q = copy.deepcopy(self.critic)   
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr,eps=hp.eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr,eps=hp.eps)
        
        
        ## lr_dacay
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
            
        self.scheduler = [LambdaLR(self.actor_optimizer, lr_lambda=lambda_lr),
                          LambdaLR(self.critic_optimizer, lr_lambda=lambda_lr)]
        
        ## 
        self.lr_decay_enable = hp.lr_decay_enable
        self.grad = hp.grad
        self.discount = hp.discount
        self.update_iteration = hp.update_iteration
        
        
        ## SAC
        self.adaptive_alpha = hp.adaptive_alpha_enable
        if self.adaptive_alpha:
            A = torch.scalar_tensor(action_dim, dtype=torch.float64)
            self.target_entropy = (0.98 * (- torch.log(1 / A)))
            self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device), requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=hp.actor_lr)
            self.target_entropy = self.target_entropy.to(self.device)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = hp.alpha
        self.tau = hp.tau
        
        
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
        
        action,_,_ = self.actor.getAction(state)
        action = action.view(-1).cpu().data.numpy()
        
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
        if self.adaptive_alpha:
            alpha = self.alpha.detach()
        else:
            alpha = torch.tensor(self.alpha)
            
        q1,q2 = self.critic.getQ(state)
        q1 = q1.gather(1,action.long())
        q2 = q2.gather(1,action.long())
        
        with torch.no_grad():
            next_action, next_probs, log_next_prob = self.actor.getAction(next_state)
            target_q1,target_q2 = self.target_Q.getQ(next_state)
            target_q = torch.min(target_q1,target_q2)
            target_value = (next_probs * (target_q - alpha * log_next_prob)).sum(-1,keepdim=True)
            next_q_value = reward + mask * self.discount * target_value
        
        
        q1_loss = (0.5 * (q1 - next_q_value)**2).mean()
        q2_loss = (0.5 * (q2 - next_q_value)**2).mean()
        
        q_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad)
        self.critic_optimizer.step()
        
        
        
        new_action, probs, log_prob = self.actor.getAction(state)
        new_q1,new_q2 = self.critic.getQ(state)
        new_q = torch.min(new_q1,new_q2).detach()
        actor_loss = (probs * (alpha * log_prob - new_q)).sum(-1).mean()
        
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad)
        self.actor_optimizer.step()
        
        
        
        if self.learn_step % self.update_iteration == 0:
            self.target_Q.load_state_dict(self.critic.state_dict())
            
        alpha_loss = torch.tensor(0)
        if self.adaptive_alpha:
            alpha_loss = (probs.detach() * (-self.log_alpha * log_prob.detach() + self.target_entropy)).sum(-1).mean()
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
            process.process_input(q_loss.item(), 'value_loss', 'train/')
            process.process_input(new_q.detach().cpu().numpy(), 'new_q', 'train/')
            process.process_input(log_prob.detach().cpu().numpy(), 'log_prob', 'train/')
            process.process_input(next_q_value.detach().cpu().numpy(), 'next_q_value', 'train/')
            process.process_input(alpha.item(), 'alpha', 'train/')
            process.process_input(alpha_loss.item(), 'alpha_loss', 'train/')
            
            
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