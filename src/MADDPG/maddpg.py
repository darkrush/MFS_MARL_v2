import io
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from .model import Critic, Actor
from .memory import Memory
from .utils import onehot_from_logits,gumbel_softmax

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class MADDPG(object):
    def __init__(self, args_dict,agent_num):
        self.actor_lr = args_dict['actor_lr']
        self.critic_lr = args_dict['critic_lr']
        self.lr_decay = args_dict['lr_decay']
        self.l2_critic = args_dict['l2_critic']
        self.discount = args_dict['discount']
        self.tau = args_dict['tau']
        self.batch_size = args_dict['batch_size']
        #self.batch_size = 5
        self.buffer_size = int(args_dict['buffer_size'])
        self.discrete = args_dict['discrete']
        self.args_dict = args_dict
        self.opti_eps = 1e-5
        self.weight_decay = 0
        self.agent_num = agent_num
        self.max_grad_norm = 20.0
        self.act_noise_std = 0.1
        
    def setup(self):
        self.lr_coef = 1
        n_v = self.args_dict['nb_vel']
        n_p = self.args_dict['nb_phi']
        nb_actions = n_v*2*(2*n_p+1)+1
        self.nb_actions = nb_actions
        actor  = Actor( nb_rpos=self.args_dict['nb_rpos'],
                        nb_laser=self.args_dict['nb_laser'],
                        nb_actions=nb_actions,
                        hidden1 = self.args_dict['hidden1'],
                        hidden2 = self.args_dict['hidden2'] ,
                        layer_norm = self.args_dict['layer_norm'],
                        discrete = self.discrete)
        critic = Critic(nb_rpos = self.args_dict['nb_rpos']*self.agent_num,
                         nb_actions=nb_actions*self.agent_num,
                         hidden1 = self.args_dict['hidden1'],
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
        self.memory = Memory(num_agents=self.agent_num,
                             limit=self.buffer_size,
                             action_shape=(nb_actions,),
                             observation_shape=[(self.args_dict['nb_laser'],),(self.args_dict['nb_rpos'],),])
        
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
        BATCH_SIZE = self.batch_size
        AGENT_NUM = self.agent_num
        RPOS_NUM = self.args_dict['nb_rpos']
        LASER_NUM = self.args_dict['nb_laser']
        ACTIONS_NUM = self.nb_actions
        #train_policy_on_batch
        batch = self.memory.sample(BATCH_SIZE)
        #nagent_obs
        tensor_obs0 = batch['obs0']
        tensor_obs1 = batch['obs1']

        with torch.no_grad():
            # [BATCH_SIZE, AGENT_NUM * RPOS_NUM]
            # ==> [BATCH_SIZE * AGENT_NUM, RPOS_NUM]
            target_rpos = tensor_obs1[1].view(BATCH_SIZE * AGENT_NUM, RPOS_NUM)
            # [BATCH_SIZE, AGENT_NUM * LASER_NUM]
            # ==> [BATCH_SIZE * AGENT_NUM, LASER_NUM]
            target_laser = tensor_obs1[0].view(BATCH_SIZE * AGENT_NUM, LASER_NUM)
            # [BATCH_SIZE * AGENT_NUM, ACTIONS_NUM] because of one_hot
            target_actor_out = self.actor_target(target_rpos,target_laser)
            # [BATCH_SIZE * AGENT_NUM, ACTIONS_NUM]
            onehot_actions = onehot_from_logits(target_actor_out)
            # [BATCH_SIZE * AGENT_NUM, ACTIONS_NUM]
            # ==> [BATCH_SIZE, AGENT_NUM * ACTIONS_NUM]
            cent_nact1 = onehot_actions.view(BATCH_SIZE, AGENT_NUM * ACTIONS_NUM)
            # [BATCH_SIZE, AGENT_NUM * RPOS_NUM]
            cent_rpos1 = tensor_obs1[1]
            # [BATCH_SIZE, 1]
            rewards = batch['rewards'].sum(dim=-1).unsqueeze(-1)
            # [BATCH_SIZE, 1]
            next_step_Q = self.critic_target(cent_rpos1, cent_nact1)
            # [BATCH_SIZE, 1]
            target_Qs = rewards + self.discount * next_step_Q

        self.critic_optim.zero_grad()

        cent_rpos = tensor_obs0[1] # [BATCH_SIZE, AGENT_NUM * RPOS_NUM]
        actions = batch['actions'] # [BATCH_SIZE, AGENT_NUM * ACTIONS_NUM]
        predicted_Qs = self.critic(cent_rpos, actions)
        
        value_loss = nn.functional.mse_loss(predicted_Qs, target_Qs)
        value_loss.backward()

        self.critic_optim.step()
    
        return value_loss.item()

    def update_actor(self, train_actor = True):
        BATCH_SIZE = self.batch_size
        AGENT_NUM = self.agent_num
        RPOS_NUM = self.args_dict['nb_rpos']
        LASER_NUM = self.args_dict['nb_laser']
        ACTIONS_NUM = self.nb_actions

        batch = self.memory.sample(self.batch_size)
        tensor_obs0 = batch['obs0']
        cent_laser = tensor_obs0[0] # [BATCH_SIZE, AGENT_NUM * LASER_NUM]
        cent_rpos = tensor_obs0[1] # [BATCH_SIZE, AGENT_NUM * RPOS_NUM]
        # [BATCH_SIZE, AGENT_NUM * RPOS_NUM]
        # ==> [BATCH_SIZE * AGENT_NUM, RPOS_NUM]
        agent_rpos = cent_rpos.view(BATCH_SIZE * AGENT_NUM, RPOS_NUM)
        # [BATCH_SIZE, AGENT_NUM * LASER_NUM]
        # ==> [BATCH_SIZE * AGENT_NUM, LASER_NUM]
        agent_laser = cent_laser.view(BATCH_SIZE * AGENT_NUM, LASER_NUM)
        if train_actor: # Actor update
            # do not update critic    
            for p in self.critic.parameters():
                p.requires_grad = False
            self.actor.zero_grad()
            # [BATCH_SIZE * AGENT_NUM, ACTIONS_NUM] because of one_hot
            agent_actor_out = self.actor(agent_rpos,agent_laser)
            # [BATCH_SIZE * AGENT_NUM, ACTIONS_NUM]
            onehot_actions = gumbel_softmax(agent_actor_out, hard=True)
            # [BATCH_SIZE * AGENT_NUM, ACTIONS_NUM]
            # ==> [BATCH_SIZE, AGENT_NUM * ACTIONS_NUM]
            cent_actions = onehot_actions.view(BATCH_SIZE, AGENT_NUM * ACTIONS_NUM)
            # [BATCH_SIZE, 1]
            predicted_Qs = self.critic(cent_rpos, cent_actions)
            # [BATCH_SIZE, 1]
            policy_loss = -predicted_Qs.mean()
            policy_loss.backward()
            self.actor_optim.step()
            # resume update critic 
            for p in self.critic.parameters():
                p.requires_grad = True
        else:
            with torch.no_grad():
                # [BATCH_SIZE * AGENT_NUM, ACTIONS_NUM] because of one_hot
                agent_actor_out = self.actor(agent_rpos,agent_laser)
                # [BATCH_SIZE * AGENT_NUM, ACTIONS_NUM]
                onehot_actions = onehot_from_logits(agent_actor_out)
                # [BATCH_SIZE * AGENT_NUM, ACTIONS_NUM]
                # ==> [BATCH_SIZE, AGENT_NUM * ACTIONS_NUM]
                cent_actions = onehot_actions.view(BATCH_SIZE, AGENT_NUM * ACTIONS_NUM)
                # [BATCH_SIZE, 1]
                predicted_Qs = self.critic(cent_rpos, cent_actions)
                # [BATCH_SIZE, 1]
                policy_loss = -predicted_Qs.mean()

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