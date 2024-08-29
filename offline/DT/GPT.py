import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from transformers.modeling_utils import Conv1D

class CausalSelfAttention(nn.Module):
    
    def __init__(self,config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        
        # self.c_attn = Conv1D(3 * config.n_embd,config.n_embd)
        # self.c_proj = Conv1D(config.n_embd,config.n_embd)
        
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)
                                               ).view(1,1,config.block_size,config.block_size))
        
        
        
    def forward(self, x, attention_mask):
        x = x.float()
        attention_mask = attention_mask.float()
        
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1))) # flash attention
        att = att.masked_fill(self.bias[:,:,:T,:T]==0,float(-1e4)) # float('-inf')
        att = att + attention_mask
        att = F.softmax(att,dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B,nh,T,T) x (B,nh,T,hs) -> (B,nh,T,hs)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        
        return y


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # self.c_fc = Conv1D(4 * config.n_embd, config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        # self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd,config.n_embd)
        # self.c_proj = Conv1D(config.n_embd,4 *config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        # x = self.relu(x)
        x = self.c_proj(x)
        return self.dropout(x)
        
        
class Block(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self,x,attention_mask):
        x = x + self.attn(self.ln_1(x),attention_mask)
        x = x + self.mlp(self.ln_2(x)) 
        return x
    
    
    
class GPT(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),     
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ) )
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias = False)
        self.drop = nn.Dropout(config.dropout)
        # weight sharing scheme
        # GPT中共享了token编码以及最后一层解码的网络权重
        self.transformer.wte.weight = self.lm_head.weight
        # init params
        # self.apply(self._init_weights)
        
        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
        # if isinstance(module, Conv1D):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 这是避免残差流的标准差积累 
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        
    def forward(self, inputs_embeds,attention_mask):
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
        
        tok_emb = inputs_embeds
        x = tok_emb
        x = self.drop(x)
        
        output_shape = input_shape + (x.size(-1),)
        
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        for block in self.transformer.h:
            x = block(x,attention_mask)
        x = self.transformer.ln_f(x)
        
        x = x.view(*output_shape)
        
        return x