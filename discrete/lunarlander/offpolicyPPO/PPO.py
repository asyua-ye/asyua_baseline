import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils.tool import RunningMeanStd
from torch.optim.lr_scheduler import LambdaLR



class Actor(nn.Module):
    def __init__(self, state_dim,action_dim,hp):
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
    
    def getAction(self,state):
        logit = self.forward(state)
        dist = torch.distributions.Categorical(logits=logit)
        log_prob = F.log_softmax(logit,dim=-1)
        action = dist.sample().view(-1, 1)
        action_log_probs = log_prob.gather(1, action.long())
        return action,action_log_probs
    
    def getLogprob(self, state, old_action):
        logit = self.forward(state)
        log_prob = F.log_softmax(logit,dim=-1)
        oldaction_log_probs = log_prob.gather(1, old_action.long())
        distentropy = -(log_prob * log_prob.exp()).sum(-1).mean()
        
        return oldaction_log_probs,distentropy
        
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
    def __init__(self, state_dim,hp):
        super(Critic, self).__init__()
        
        self.state_norm_enable = hp.state_norm_enable
        self.fc1_1 = nn.Linear(state_dim[0], 512)
        self.critic_linear = nn.Linear(512, 1)
        
        self.state_norm = RunningMeanStd(state_dim)
        self._initialize_weights()
        
    def forward(self, state):
        if self.state_norm_enable:
            state = self.norm(state, self.state_norm)
        x = F.relu(self.fc1_1(state))
        v = self.critic_linear(x)
        return v
    
    def norm(self, x, x_norm):
        if self.training:
            x_norm.update(x.detach())
        x = x_norm.normalize(x)
        return x
    
    def getValue(self,state):
        value= self.forward(state)
        return value
        
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name == 'critic_linear' :
                    nn.init.orthogonal_(module.weight, 0.01)
                else:
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
        self.critic = Critic(state_dim,hp).to(self.device)        
        
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
        
        ## ppo
        self.entropy = hp.entropy
        self.value_weight = hp.value
        self.actor_weight = hp.actor
        self.clip = hp.clip
        self.c = hp.c
        
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
        
        action,log_prob = self.actor.getAction(state)
        value = self.critic.getValue(state)
        
        action = action.view(-1).cpu().data.numpy()
        log_prob = log_prob.view(-1,1).cpu().data.numpy()
        value = value.view(-1,1).cpu().data.numpy()
        
        return action,log_prob,value
    
    @torch.no_grad()
    def get_value(self,state):
        
        state = torch.FloatTensor(state.reshape(-1, *state.shape)).squeeze().to(self.device)
        value = self.critic.getValue(state)
        value = value.view(-1).cpu().data.numpy()
        
        return value
    
    
    def training(self,train=True):
        if train:
            self.actor.train()
            self.critic.train()
            self.Train = True
        else:
            self.actor.eval()
            self.critic.eval()
            self.Train = False
            
    def evaluate_actions(self, state,actions):
        
        logprob,dist_entropy = self.actor.getLogprob(state,actions)
        return logprob, dist_entropy
            
    def train(self,sample,process,writer):
        
        self.learn_step += 1
        state,action,old_action_log_probs,advs,returns = sample
        
        action_log_probs, dist_entropy = self.evaluate_actions(state,action)
        
        ratio =  torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advs
        actor_loss = -torch.min(surr1, surr2).sum(dim=-1).mean()
        # actor_loss = -torch.max(torch.min(surr1, surr2),self.c*advs).sum(dim=-1).mean()
        
        actor_loss = self.actor_weight * actor_loss - self.entropy * dist_entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad)
        self.actor_optimizer.step()
        
        values = self.critic.getValue(state)
        
        value_loss = F.mse_loss(returns.mean(-1,keepdim=True), values)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad)
        self.critic_optimizer.step()
        
        if self.lr_decay_enable:
            for scheduler in self.scheduler:
                scheduler.step()
        
        
        writer.add_scalar('actor_loss', actor_loss.item(), global_step=self.learn_step)
        writer.add_scalar('value_loss', value_loss.item(), global_step=self.learn_step)
        
        if self.learn_step % 1000 ==0:
            process.process_input(self.learn_step, 'learn_step', 'train/')
            process.process_input(actor_loss.item(), 'actor_loss', 'train/')
            process.process_input(value_loss.item(), 'value_loss', 'train/')
            process.process_input(dist_entropy.item(), 'dist_entropy', 'train/')
            process.process_input(action_log_probs.detach().cpu().numpy(), 'action_log_probs', 'train/')
            process.process_input(old_action_log_probs.detach().cpu().numpy(), 'old_action_log_probs', 'train/')
            process.process_input(ratio.detach().cpu().numpy(), 'ratio', 'train/')
            process.process_input(surr1.detach().cpu().numpy(), 'surr1', 'train/')
            process.process_input(values.detach().cpu().numpy(), 'values', 'train/')
            process.process_input(returns.detach().cpu().numpy(), 'returns', 'train/')
            
            
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