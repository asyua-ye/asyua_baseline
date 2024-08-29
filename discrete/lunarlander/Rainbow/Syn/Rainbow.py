import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils.tool import RunningMeanStd
from torch.optim.lr_scheduler import LambdaLR



class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight_mu.data.uniform_(-std, std)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-std, std)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())



class Q(nn.Module):
    def __init__(self, state_dim, action_dim, hp):
        super(Q, self).__init__()
        self.action_dim = action_dim
        self.state_norm_enable = hp.state_norm_enable
        self.atoms = hp.atoms
        self.z = torch.linspace(hp.vmin,hp.vmax,self.atoms).view(1,1,self.atoms).to(hp.device)
        
        self.fc1 = nn.Linear(state_dim[0], 128)
        self.noise1 = NoisyLinear(128, 128)
        self.output = NoisyLinear(128, action_dim * self.atoms)
        self.v_linear = NoisyLinear(128, self.atoms)
        
        
        self.state_norm = RunningMeanStd(state_dim)
        
        self._initialize_weights()
    
    def forward(self, state):
        if self.state_norm_enable:
            state = self.norm(state)
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.noise1(x))
        a = self.output(x).view(-1,self.action_dim,self.atoms)
        v = self.v_linear(x).view(-1,1,self.atoms)
        # q = v + a - a.mean(-1,keepdim=True)  # 这里是错的，但是实验居然能跑出来
        q = v + a - a.mean(1,keepdim=True)
        p = F.softmax(q.view(-1,self.atoms),dim=-1).view(-1,self.action_dim,self.atoms)
        
        return p
    
    def getQ(self,state):
        p = self.forward(state)
        q_value = (p * self.z).sum(-1).view(-1,self.action_dim)
        return q_value
    
    def getP(self,state):
        p = self.forward(state)
        return p
    
    def reset_noise(self):
        self.noise1.reset_noise()
        self.output.reset_noise()
        self.v_linear.reset_noise()
    
    def norm(self, x):
        if self.training:
            self.state_norm.update(x.detach())
        return self.state_norm.normalize(x)
    
    def _initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias, 0)
               
                
                
class agent(object):
    def __init__(self,state_dim, action_dim, hp) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.Q = Q(state_dim,action_dim,hp).to(self.device)        
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=hp.q_lr,eps=hp.eps)
        
        
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
        self.scheduler = [LambdaLR(self.Q_optimizer, lr_lambda=lambda_lr),]
        
        
        
        ## C51
        self.vmax = hp.vmax
        self.vmin = hp.vmin
        self.atoms = hp.atoms
        self.delta = (self.vmax-self.vmin)/(self.atoms-1)
        self.z = torch.linspace(self.vmin,self.vmax,self.atoms).view(1,1,self.atoms).to(self.device)
        
        
        # epsilon greedy
        self.epsilon_start = hp.epsilon_start
        self.epsilon_final = hp.epsilon_final
        self.epsilon_decay = hp.epsilon_decay
        self.env_interaction = 0
        epsilon_lambda = lambda frame_idx: self.epsilon_final + (self.epsilon_start
                                                - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)
        self.epsilon_by_frame = epsilon_lambda
        
        ## 
        self.gamma = hp.discount
        self.update_iteration = hp.update_iteration
        self.lr_decay_enable = hp.lr_decay_enable
        self.grad = hp.grad
        
        
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
        
        if not eval:
            self.Q.reset_noise()
        q = self.Q.getQ(state)
        action = q.argmax(-1).view(-1).cpu().data.numpy()
        
        return action
    
    
    def training(self,train=True):
        if train:
            self.Q.train()
            self.Train = True
        else:
            self.Q.eval()
            self.Train = False
            
            
    @torch.no_grad()
    def projection_distribution(self,next_states,rewards,masks,exps):
        
        batch_size = next_states.size(0)
        support = self.z
        
        self.Q_target.reset_noise()
        next_P = self.Q_target.getP(next_states)
        next_q = self.Q.getQ(next_states)
        next_action = next_q.max(-1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_P.size(0), 1, next_P.size(2))
        
        next_dist = next_P.gather(1, next_action).squeeze(1)
        
        reward = rewards.expand_as(next_dist)
        mask   = masks.expand_as(next_dist)
        support = support.squeeze(0).expand_as(next_dist)
        
        
        Tz = reward + mask * (self.gamma ** exps) * support
        Tz = Tz.clamp(min=self.vmin, max=self.vmax)
        b  = (Tz - self.vmin) / self.delta
        l  = b.floor().long()
        u  = b.ceil().long()
        
        offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).long()\
                    .unsqueeze(1).expand(batch_size, self.atoms).to(self.device)    

        proj_dist = torch.zeros(next_dist.size()).to(self.device)    
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        
        
        return proj_dist
            
            
    def train(self,data,process,writer):
        
        self.learn_step += 1
        sample, idxs, is_weight = data
        states,actions,next_states,rewards,masks,exps = sample
        
        self.Q.reset_noise()
        all_p = self.Q.getP(states)
        p = all_p.gather(1, actions.long().unsqueeze(1).expand(all_p.size(0), 1, all_p.size(2))).squeeze(1)
        m = self.projection_distribution(next_states,rewards,masks,exps)
        
        loss_sum = - (m * torch.log(p + 1e-8)).sum(dim=-1,keepdim=True)
        weight = loss_sum.abs().clone().detach()
        loss = (is_weight * loss_sum).mean()
        
        self.Q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), self.grad)
        self.Q_optimizer.step()
        
        if self.lr_decay_enable:
            for scheduler in self.scheduler:
                scheduler.step()
        
        
        ## hard update Q_target
        if self.learn_step % self.update_iteration == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
            
        
        ## record
        writer.add_scalar('loss', loss.item(), global_step=self.learn_step)
        
        if self.learn_step % 1000 == 0:
            process.process_input(self.learn_step, 'learn_step', 'train/')
            process.process_input(loss.item(), 'q_loss', 'train/')
            process.process_input(p.detach().cpu().numpy(), 'p', 'train/')
            process.process_input(m.detach().cpu().numpy(), 'm', 'train/')
            
        return weight,idxs
            
            
    def save(self, filename):
        torch.save({
            'q_state_dict': self.Q.state_dict(),
            'running_mean_std_state': {
                'n': self.Q.state_norm.n,
                'mean': self.Q.state_norm.mean,
                'S': self.Q.state_norm.S,
                'std': self.Q.state_norm.std
            }
        }, filename + "_q")
        
        # 如果需要保存优化器状态，可以取消注释以下代码
        # torch.save(self.Q_optimizer.state_dict(), filename + "_q_optimizer")
        
        
        
    def load(self, filename):
        checkpoint = torch.load(filename + "_q")
        self.Q.load_state_dict(checkpoint['q_state_dict'])
        
        # 恢复 RunningMeanStd 的状态
        self.Q.state_norm.n = checkpoint['running_mean_std_state']['n']
        self.Q.state_norm.mean = checkpoint['running_mean_std_state']['mean']
        self.Q.state_norm.S = checkpoint['running_mean_std_state']['S']
        self.Q.state_norm.std = checkpoint['running_mean_std_state']['std']
        
        # 如果需要加载优化器状态，可以取消注释以下代码
        # self.Q_optimizer.load_state_dict(torch.load(filename + "_q_optimizer"))
        
        
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