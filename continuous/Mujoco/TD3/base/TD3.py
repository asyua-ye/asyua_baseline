import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils.tool import RunningMeanStd
from torch.optim.lr_scheduler import LambdaLR



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,hp):
        super(Actor, self).__init__()

        self.state_norm_enable = hp.state_norm_enable
        self.max_action = max_action
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.state_norm = RunningMeanStd(state_dim)
        self._initialize_weights()

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))
    
    
    def getAction(self,state):
        action = self.forward(state)
        
        return action
        
    
    def norm(self, x, x_norm):
        if self.training:
            x_norm.update(x.detach())
        x = x_norm.normalize(x)
        return x
    
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name != 'l3':
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hp):
        super(Critic, self).__init__()

        self.state_norm_enable = hp.state_norm_enable
        
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        
        self.state_norm = RunningMeanStd(state_dim)
        self._initialize_weights()
        

    def forward(self, state, action):
        if self.state_norm_enable:
            state = self.norm(state, self.state_norm)
        
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
    def getQ(self, state, action):
        q1,q2 = self.forward(state,action)
        
        return q1,q2
        
    def norm(self, x, x_norm):
        if self.training:
            x_norm.update(x.detach())
        x = x_norm.normalize(x)
        return x
    
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name != 'l3' or name != 'l6': 
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)


               
                
                
class agent(object):
    def __init__(self,state_dim, action_dim, max_action, hp) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        self.actor = Actor(state_dim,action_dim,max_action,hp).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)   
        self.critic = Critic(state_dim,action_dim,hp).to(self.device)     
        self.target_Q = copy.deepcopy(self.critic)   
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr,eps=hp.eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr,eps=hp.eps)
        
        ## 
        self.lr_decay_enable = hp.lr_decay_enable
        self.grad = hp.grad
        self.discount = hp.discount
        self.tau = hp.tau
        
        
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
        
        if self.lr_decay_enable:
            self.scheduler = [LambdaLR(self.actor_optimizer, lr_lambda=lambda_lr),
                            LambdaLR(self.critic_optimizer, lr_lambda=lambda_lr)]
        
        
        
        
        ## TD3
        self.noiseClip = hp.noiseclip
        self.updaeActor = hp.update_actor
        self.actionNoise = hp.actionNoise
        self.exNoise = hp.exNoise
        
        
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
        
        action = self.actor.getAction(state)
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        
        if not eval:
            temp = np.random.normal(0, self.max_action * self.exNoise, size=[state.size(0),self.action_dim])
            action = (action+ temp).clip(-self.max_action, self.max_action)
        
        
        return action[0]
    
    
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
        q1,q2 = self.critic.getQ(state,action)
        
        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.actionNoise
            ).clamp(-self.noiseClip, self.noiseClip)
            target_a=self.target_actor(next_state)
            next_action = (target_a + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.target_Q.getQ(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            next_q_value = reward + mask * self.discount * target_Q
        
        q1_loss = (0.5 * (q1 - next_q_value)**2).mean()
        q2_loss = (0.5 * (q2 - next_q_value)**2).mean()
        q_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad)
        self.critic_optimizer.step()
        
        
        
        actor_loss = torch.zeros(1)
        if self.learn_step % self.updaeActor == 0:
            
            actor_loss = -self.critic.Q1(state, self.actor.getAction(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad)
            self.actor_optimizer.step()
            
            for param, target_param in zip(self.critic.parameters(), self.target_Q.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        
        
        if self.lr_decay_enable:
            for scheduler in self.scheduler:
                scheduler.step()
        
        
        writer.add_scalar('actor_loss', actor_loss.item(), global_step=self.learn_step)
        writer.add_scalar('q_loss', q_loss.item(), global_step=self.learn_step)
        
        if self.learn_step % 1000 ==0:
            process.process_input(self.learn_step, 'learn_step', 'train/')
            process.process_input(actor_loss.item(), 'actor_loss', 'train/')
            process.process_input(q_loss.item(), 'value_loss', 'train/')
            process.process_input(next_q_value.detach().cpu().numpy(), 'next_q_value', 'train/')
            
            
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