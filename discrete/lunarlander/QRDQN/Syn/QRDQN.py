import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils.tool import RunningMeanStd
from torch.optim.lr_scheduler import LambdaLR



class Q(nn.Module):
    def __init__(self, state_dim,action_dim, atoms,state_norm_enable=False):
        super(Q, self).__init__()
        self.action_dim = action_dim
        self.state_norm_enable = state_norm_enable
        self.atoms = atoms
        self.fc1_1 = nn.Linear(state_dim[0], 512)
        self.mean_linear = nn.Linear(512, action_dim * self.atoms)
        
        self.state_norm = RunningMeanStd(state_dim)
        self._initialize_weights()
        
    def forward(self, state):
        if self.state_norm_enable:
            state = self.norm(state, self.state_norm)
        x_actor = F.relu(self.fc1_1(state))
        q = self.mean_linear(x_actor)
        
        return q
    
    def getQ(self,state):
        q_value = self.forward(state)
        q_value = q_value.view(-1,self.action_dim,self.atoms).mean(-1)
        
        return q_value
    
    def getP(self,state):
        
        return self.forward(state).view(-1,self.action_dim,self.atoms)
        
    
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
               
                
                
class agent(object):
    def __init__(self,state_dim, action_dim,hp) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.Q = Q(state_dim,action_dim,hp.atoms,hp.state_norm_enable).to(self.device)        
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
        self.atoms = hp.atoms
        self.k = hp.k
        
        self.cumulative_density = torch.tensor((2.0*np.arange(self.atoms)+1)/(2.0*self.atoms)).view(1,-1).to(self.device)
        
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
            self.env_interaction += 1
            epsilon = self.epsilon_by_frame(self.env_interaction)
            if np.random.random() > epsilon:
                q = self.Q.getQ(state)
                action = q.argmax(-1).view(-1).cpu().data.numpy()
            else:
                action = np.random.randint(0, self.action_dim, size=state.shape[0])
                
        else:
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
    def projection_distribution(self,dist,next_state,reward,mask):
        
        batch_size = next_state.shape[0]
        atoms = self.atoms
        
        next_p = self.Q_target.getP(next_state)
        next_action = next_p.mean(-1).max(1)[1].unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.atoms)
        next_dist = next_p.gather(1,next_action).squeeze(1)
        
        Tz = reward + self.gamma * next_dist * mask
        
        # quant_idx = torch.sort(dist, 1, descending=False)[1].to(self.device)
        # tau_hat = torch.linspace(0.0, 1.0 - 1./atoms, atoms) + 0.5 / atoms
        # tau_hat = tau_hat.unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # batch_idx = torch.arange(batch_size).to(self.device)
        # ## 这里本质上是为了取出排序后的tau，相当于按照51个值的大小顺序获得对应的tau，但是通过下面的操作更有效率的执行
        # tau = tau_hat[:, quant_idx][batch_idx, batch_idx]
        
        ## 论文里没看到用dist的大小选择tau，这里另一种处理
        tau = self.cumulative_density
        
        return tau,Tz
        
        
        
    def train(self,sample,process,writer):
        
        
        def Huber_loss(diff):
            if self.k!= 0:
                K = self.k
                flag=(diff.abs()<=K).float().detach()
                return 0.5*diff.pow(2)*flag+K*(diff.abs()-0.5*K)*(1.0-flag)
            else:
                return diff
        
        
        self.learn_step += 1
        states,actions,next_states,rewards,masks = sample
        
        
        all_p = self.Q.getP(states)
        actions = actions.unsqueeze(1).expand(states.shape[0], 1, self.atoms)
        dist = all_p.gather(1, actions.long()).squeeze(1)
        
        tau,next_dist = self.projection_distribution(dist,next_states,rewards,masks)
        diff = next_dist.t().unsqueeze(-1) - dist.unsqueeze(0)
        huber_loss = Huber_loss(diff)
        loss = ((tau - (diff < 0).float()).abs() * huber_loss)
        loss = loss.transpose(0,1)
        loss = loss.mean(1).sum(-1).mean()
        
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
            process.process_input(dist.mean(-1).detach().cpu().numpy(), 'dist', 'train/')
            process.process_input(next_dist.mean(-1).detach().cpu().numpy(), 'next_dist', 'train/')
            
            
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