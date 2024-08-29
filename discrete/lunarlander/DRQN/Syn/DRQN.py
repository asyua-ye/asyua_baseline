import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils.tool import RunningMeanStd
from torch.optim.lr_scheduler import LambdaLR



class Q(nn.Module):
    def __init__(self, state_dim, action_dim, hp):
        super(Q, self).__init__()
        self.action_dim = action_dim
        self.state_norm_enable = hp.state_norm_enable
        self.bidirectional = hp.bidirectional
        self.hidden_state = hp.hidden_state
        self.num_layer = hp.num_layer
        self.l = hp.num_layer + 1*hp.bidirectional
        
        self.num_dir = 1 if not self.bidirectional else 2
        self.lstm = nn.LSTM(state_dim[0], self.hidden_state, num_layers=self.num_layer, batch_first=True, bidirectional=self.bidirectional)
        self.mean_linear = nn.Linear(self.hidden_state, action_dim)
        

        self.state_norm = RunningMeanStd(state_dim)
        
        self._initialize_weights()
    
    def forward(self, state, hidden=None):
        self.lstm.flatten_parameters()
        # (batch_size, sequence_length, state_dim)
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        if self.state_norm_enable:
            state = self.norm(state, self.state_norm)
        
        if hidden is None:
            batch_size = state.size(0)
            hidden = (torch.zeros(self.l,batch_size, self.lstm.hidden_size).to(state.device),
                    torch.zeros(self.l,batch_size, self.lstm.hidden_size).to(state.device))
        else:
            hidden[0],hidden[1] = hidden[0].transpose(0,1),hidden[1].transpose(0,1)
            
        lstm_out, (hidden, cell) = self.lstm(state, hidden)
        
        # 这里直接选取的最后一层，也可以做平均池化，或者全部传入
        lstm_out_last = lstm_out[:, -1, :]
        
        q = self.mean_linear(lstm_out_last)
        
        hidden,cell = hidden.transpose(0,1),cell.transpose(0,1)
        
        return q, (hidden, cell)
    
    def getQ(self, state, hidden=None):
        q_value, (hidden, cell) = self.forward(state, hidden)
        return q_value,(hidden, cell)
    
    def norm(self, x, x_norm):
        if self.training:
            x_norm.update(x.detach().view(-1, x.size(-1)))
        x = x_norm.normalize(x)
        return x
    
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, nn.init.calculate_gain('relu'))
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
               
                
                
class agent(object):
    def __init__(self,state_dim, action_dim,hp) -> None:
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
        self.nstep = hp.nstep
        self._state = torch.zeros((hp.num_processes,self.nstep,)+self.state_dim).to(self.device)
        self.hidden_state = hp.hidden_state
        self.l = hp.num_layer + 1*hp.bidirectional
        
        
        ## checkpoint
        self.Maxscore = (0.0,0.0)
        self.learn_step = 0
        
        
        self.training(False)
        
    
    def processState(self, state, mask):
        
        if state.ndim == 1:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        else:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).squeeze().to(self.device)
        
        mask = mask.squeeze(-1).bool()        
        self._state = torch.roll(self._state, -1, dims=1) 
        self._state[:, -1, :] = state
        seq_state = self._state.clone()
        self._state[~mask] = torch.zeros((self._state.shape[1:])).to(self.device)

        return seq_state
        
    
    @torch.no_grad()
    def select_action(self,state,mask,eval=False):
        
        state = self.processState(state,mask)
        
        if not eval:
            self.env_interaction += 1
            epsilon = self.epsilon_by_frame(self.env_interaction)
            if np.random.random() > epsilon:
                q,(hidden, cell) = self.Q.getQ(state)
                action = q.argmax(-1).view(-1).cpu().data.numpy()
            else:
                action = np.random.randint(0, self.action_dim, size=state.shape[0])
                hidden,cell = torch.zeros((state.shape[0],self.l,self.hidden_state)),torch.zeros((state.shape[0],self.l,self.hidden_state))
                
        else:
            q,(hidden, cell) = self.Q.getQ(state)
            action = q.argmax(-1).view(-1).cpu().data.numpy()
        
        return action,state,(hidden, cell)
    
    
    def training(self,train=True):
        if train:
            self.Q.train()
            self.Train = True
        else:
            self.Q.eval()
            self.Train = False
            
            
    def train(self,sample,process,writer):
        
        self.learn_step += 1
        states,actions,next_states,rewards,masks = sample
        
        
        all_q,_ = self.Q.getQ(states)
        q_values = all_q.gather(1, actions.long())
        with torch.no_grad():
            next_q_values = self.Q_target(next_states)[0].max(1)[0].unsqueeze(-1)
            target_q_values = rewards + (self.gamma * next_q_values * masks)
            
        loss = F.mse_loss(q_values, target_q_values)
        
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
            process.process_input(q_values.detach().cpu().numpy(), 'q_values', 'train/')
            process.process_input(target_q_values.detach().cpu().numpy(), 'target_q_values', 'train/')
            
            
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