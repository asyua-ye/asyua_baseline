import torch
import torch.nn as nn
from torch.nn import functional as F
from GPT import GPT as GPT2Model
# from trajectory_transformer import GPT2Model


class DT(nn.Module):
    def __init__(self,
            state_dim,
            act_dim,
            hidden_size,
            hp,
            max_length=None,
            max_ep_len=1000,
            action_tanh=True,
            ) -> None:
        
        super(DT, self).__init__()
        
        self.max_length = max_length
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(hp)
        
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.act_dim = act_dim
        
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        x = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds
    
    
    def get_action(self, states, actions, rewards, returns_to_go, timesteps):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask)

        return action_preds[0,-1]
    
    
class agent(object):
    def __init__(self,state_dim, action_dim, hp) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DT = DT(state_dim,action_dim,hp.n_embd,hp,hp.K).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.DT.parameters(),
            lr=hp.learning_rate,
            weight_decay=hp.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/hp.warmup_steps, 1)
        )
        self.learn_step = 0
        
        
    @torch.no_grad()
    def select_action(self, states, actions, rewards, returns_to_go, timesteps):
        action = self.DT.get_action(states, actions, rewards, returns_to_go, timesteps)
        return action
    
    def training(self,train=True):
        if train:
            self.DT.train()
            self.Train = True
        else:
            self.DT.eval()
            self.Train = False
    
    def Offlinetrain(self,sample, process, writer):
        self.learn_step += 1
        states, actions, rewards, dones, rtg, timesteps, attention_mask = sample
        action_target = torch.clone(actions)
        
        state_preds, action_preds, reward_preds = self.DT.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        loss = ((action_preds - action_target)**2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.DT.parameters(), .25)
        self.optimizer.step()
        
        self.scheduler.step()
        
        writer.add_scalar('loss', loss.item(), global_step=self.learn_step)
        
        if self.learn_step%1000 == 0 :
            process.process_input(self.learn_step, 'learn_step', 'train/')
            process.process_input(loss.item(), 'loss', 'train/')
            process.process_input(action_preds.detach().cpu().numpy(), 'action_preds', 'train/')
            process.process_input(action_target.detach().cpu().numpy(), 'action_target', 'train/')
        
        
    
    def save(self, filename):
        torch.save(self.DT.state_dict(), filename + "_DT")
    
    def load(self, filename):
        self.DT.load_state_dict(torch.load(filename + "_DT"))
        self.DT.to(self.device)
    
    
    

        
        
        
        
        
        

