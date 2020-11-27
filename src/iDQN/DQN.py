import io
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from .model import Qnetwork
from .memory import Memory


class DQN(object):
    def __init__(self, args_dict):
        self.Qnetwork_lr = args_dict['Qnetwork_lr']
        self.lr_decay = args_dict['lr_decay']
        self.l2_Qnetwork = args_dict['l2_Qnetwork']
        self.discount = args_dict['discount']
        self.tau = args_dict['tau']
        self.batch_size = args_dict['batch_size']
        self.buffer_size = int(args_dict['buffer_size'])
        self.args_dict = args_dict
        
    def setup(self):
        self.lr_coef = 1
        n_v = self.args_dict['nb_vel']
        n_p = self.args_dict['nb_phi']
        nb_actions = n_v*2*(2*n_p+1)+1
        Q_network = Qnetwork(nb_pos=self.args_dict['nb_pos'],
                        nb_laser=self.args_dict['nb_laser'],
                        nb_actions=nb_actions,
                        hidden1 = self.args_dict['hidden1'],
                        hidden2 = self.args_dict['hidden2'] ,
                        layer_norm = self.args_dict['layer_norm'])
        self.Qnetwork         = copy.deepcopy(Q_network)
        self.Qnetwork_target  = copy.deepcopy(Q_network)

        p_groups = [{'params': [param,],
                    'weight_decay': self.l2_Qnetwork if ('weight' in name) and ('LN' not in name) else 0
                    } for name,param in self.Qnetwork.named_parameters() ]
        self.Qnetwork_optim = Adam(params = p_groups, lr=self.Qnetwork_lr, weight_decay = self.l2_Qnetwork)
        self.memory = Memory(limit=self.buffer_size,
                             action_shape=(1,),
                             observation_shape=[(self.args_dict['nb_pos'],), (self.args_dict['nb_laser'],)])
        
    def cuda(self):
        self.memory.cuda()
        for net in (self.Qnetwork, self.Qnetwork_target):
            if net is not None:
                net.cuda()
        p_groups = [{'params': [param,],
                    'weight_decay': self.l2_Qnetwork if ('weight' in name) and ('LN' not in name) else 0
                    } for name,param in self.Qnetwork.named_parameters() ]
        self.Qnetwork_optim = Adam(params = p_groups, lr=self.Qnetwork_lr, weight_decay = self.l2_Qnetwork)
        
        
    def update_Qnetwork(self):
        batch = self.memory.sample(self.batch_size)
        tensor_obs0 = batch['obs0']
        tensor_obs1 = batch['obs1']
        tensor_actions = batch['actions']

        with torch.no_grad():
            next_Q_values = self.Qnetwork(tensor_obs1[0], tensor_obs1[1])
            next_target_Q_values = self.Qnetwork_target(tensor_obs1[0], tensor_obs1[1])
            next_Q_value = next_target_Q_values.gather(1, torch.max(next_Q_values, 1)[1].unsqueeze(1))
            target_Q_value = batch['rewards'] + self.discount * next_Q_value * (1 - batch['terminals1'])
        # Qnetwork update
        self.Qnetwork.zero_grad()
        Q_values = self.Qnetwork(tensor_obs0[0], tensor_obs0[1])
        Q_value = Q_values.gather(1, tensor_actions)

        value_loss = nn.functional.mse_loss(Q_value, target_Q_value)
        value_loss.backward()
        self.Qnetwork_optim.step()
        return value_loss.item(),torch.mean(Q_value).item()


    def update_Qnetwork_target(self,soft_update = True):
        for target_param, param in zip(self.Qnetwork_target.parameters(), self.Qnetwork.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau \
                                    if soft_update else param.data)

    def apply_lr_decay(self, decay_sacle):
        for group in self.Qnetwork_optim.param_groups:
            group['lr'] = self.Qnetwork_lr * decay_sacle
            
    def load_weights(self, model_dir): 
        Qnetwork = torch.load('{}/Qnetwork.pkl'.format(model_dir))
        self.Qnetwork.load_state_dict(Qnetwork.state_dict(), strict=True)
        self.Qnetwork_target.load_state_dict(Qnetwork.state_dict(), strict=True)
            
    def save_model(self, model_dir):
        torch.save(self.Qnetwork,'{}/Qnetwork.pkl'.format(model_dir))
            
    def get_Qnetwork_buffer(self):
        Qnetwork_buffer = io.BytesIO()
        torch.save(self.Qnetwork, Qnetwork_buffer)
        return Qnetwork_buffer