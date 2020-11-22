import io
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from .model import Actor,Critic
from .memory import Memory


class DDPG(object):
    def __init__(self, args_dict):
        self.actor_lr = args_dict['actor_lr']
        self.critic_lr = args_dict['critic_lr']
        self.lr_decay = args_dict['lr_decay']
        self.l2_critic = args_dict['l2_critic']
        self.discount = args_dict['discount']
        self.tau = args_dict['tau']
        self.batch_size = args_dict['batch_size']
        self.buffer_size = int(args_dict['buffer_size'])
        self.args_dict = args_dict
        
    def setup(self):
        self.lr_coef = 1
        actor  = Actor( nb_pos=self.args_dict['nb_pos'],
                        nb_laser=self.args_dict['nb_laser'],
                        nb_actions=self.args_dict['nb_actions'],
                        hidden1 = self.args_dict['hidden1'],
                        hidden2 = self.args_dict['hidden2'] ,
                        layer_norm = self.args_dict['layer_norm'])
        critic = Critic(nb_pos=self.args_dict['nb_pos'],
                        nb_laser=self.args_dict['nb_laser'],
                        nb_actions=self.args_dict['nb_actions'],
                        hidden1 = self.args_dict['hidden1'],
                        hidden2 = self.args_dict['hidden2'] ,
                        layer_norm = self.args_dict['layer_norm'])
        self.actor         = copy.deepcopy(actor)
        self.actor_target  = copy.deepcopy(actor)
        self.critic        = copy.deepcopy(critic)
        self.critic_target = copy.deepcopy(critic)

        
        p_groups = [{'params': [param,],
                    'weight_decay': self.l2_critic if ('weight' in name) and ('LN' not in name) else 0
                    } for name,param in self.critic.named_parameters() ]
        self.critic_optim = Adam(params = p_groups, lr=self.critic_lr, weight_decay = self.l2_critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.memory = Memory(limit=self.buffer_size,
                             action_shape=(self.args_dict['nb_actions'],),
                             observation_shape=[(self.args_dict['nb_pos'],), (self.args_dict['nb_laser'],)])
        
    def cuda(self):
        self.memory.cuda()
        for net in (self.actor, self.actor_target, self.critic, self.critic_target):
            if net is not None:
                net.cuda()
        p_groups = [{'params': [param,],
                    'weight_decay': self.l2_critic if ('weight' in name) and ('LN' not in name) else 0
                    } for name,param in self.critic.named_parameters() ]
        self.critic_optim = Adam(params = p_groups, lr=self.critic_lr, weight_decay = self.l2_critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        
        
        
        
    def update_critic(self):
        batch = self.memory.sample(self.batch_size)
        tensor_obs0 = batch['obs0']
        tensor_obs1 = batch['obs1']
        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target(tensor_obs1[0], tensor_obs1[1],self.actor_target(tensor_obs1[0], tensor_obs1[1]))
            target_q_batch = batch['rewards'] + self.discount*(1-batch['terminals1'])*next_q_values
        # Critic update
        self.critic.zero_grad()
        q_batch = self.critic(tensor_obs0[0],tensor_obs0[1], batch['actions'])
        value_loss = nn.functional.mse_loss(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()
        return value_loss.item()
    
    def pretrain_actor(self, data):
        self.actor.zero_grad()
        action = self.actor(data[0],data[1])
        pretrain_loss = nn.functional.mse_loss(action, data[2])
        pretrain_loss.backward()
        self.actor_optim.step()
        return pretrain_loss.cpu().item()

    def update_actor(self, train_actor = True):
        batch = self.memory.sample(self.batch_size)
        tensor_obs0 = batch['obs0']
        if train_actor:
            # Actor update
            self.actor.zero_grad()
            policy_loss = -self.critic(tensor_obs0[0],tensor_obs0[1],self.actor(tensor_obs0[0],tensor_obs0[1]))
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optim.step()
        else:
            with torch.no_grad():
                policy_loss = -self.critic(tensor_obs0[0],tensor_obs0[1],self.actor(tensor_obs0[0],tensor_obs0[1]))
                policy_loss = policy_loss.mean()
        return policy_loss.item()

    def update_critic_target(self,soft_update = True):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau \
                                    if soft_update else param.data)

    def update_actor_target(self,soft_update = True):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau \
                                    if soft_update else param.data)

    def adjust_actor_lr(self,lr):
        for group in self.actor_optim.param_groups:
            group['lr'] = self.actor_lr * lr

    def apply_lr_decay(self, decay_sacle):
        for (opt,base_lr) in ((self.actor_optim,self.actor_lr),(self.critic_optim,self.critic_lr)):
            for group in opt.param_groups:
                group['lr'] = base_lr * decay_sacle
            
    def load_weights(self, model_dir): 
        actor = torch.load('{}/actor.pkl'.format(model_dir) )
        critic = torch.load('{}/critic.pkl'.format(model_dir))
        self.actor.load_state_dict(actor.state_dict(), strict=True)
        self.actor_target.load_state_dict(actor.state_dict(), strict=True)
        self.critic.load_state_dict(critic.state_dict(), strict=True)
        self.critic_target.load_state_dict(critic.state_dict(), strict=True)
            
    def save_model(self, model_dir):
        torch.save(self.actor ,'{}/actor.pkl'.format(model_dir) )
        torch.save(self.critic,'{}/critic.pkl'.format(model_dir))
            
    def get_actor_buffer(self):
        actor_buffer = io.BytesIO()
        torch.save(self.actor, actor_buffer)
        return actor_buffer