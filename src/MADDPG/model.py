
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Actor(nn.Module):
    def __init__(self, nb_rpos,nb_laser, nb_actions, hidden1=128, hidden2=128, init_w=3e-3, layer_norm = True, discrete = False):
        super(Actor, self).__init__()
        self.layer_norm = layer_norm
        self.conv1 = nn.Conv1d(1, 1, kernel_size = 5, stride=2, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(1, 1, kernel_size = 5, stride=2, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.fc2 = nn.Linear(nb_laser//4 + nb_rpos, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions, bias = False)
        self.relu = nn.ReLU()
        self.softsign = nn.Softsign()
        self.discrete = discrete
        if self.layer_norm :
            self.LN1_1 = nn.LayerNorm(hidden1//2)
            self.LN1_2 = nn.LayerNorm(hidden1//4)
            self.LN2 = nn.LayerNorm(hidden2)
        
        
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.conv1.weight.data = fanin_init(self.conv1.weight.data.size())
        self.conv2.weight.data = fanin_init(self.conv2.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, rpos,laser):
        
        out = self.conv1(laser.unsqueeze(1))
        if self.layer_norm :
            out = self.LN1_1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out.squeeze(1)
        if self.layer_norm :
            out = self.LN1_2(out)
        out = self.relu(out)
        out = self.fc2(torch.cat([out,rpos],1))
        if self.layer_norm :
            out = self.LN2(out)
        out = self.relu(out)
        out = self.fc3(out)
        if not self.discrete:
            out = self.softsign(out)
        return out

#no_laser
class Critic(nn.Module):
    def __init__(self, nb_rpos, nb_actions, hidden1=128, hidden2=128, init_w=3e-3, layer_norm = True):
        super(Critic, self).__init__()
        self.layer_norm = layer_norm
        self.fc1 = nn.Linear(nb_rpos+nb_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        if self.layer_norm :
            self.LN1 = nn.LayerNorm(hidden1)
            self.LN2 = nn.LayerNorm(hidden2)
        
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, rpos, action):
        out = self.fc1(torch.cat([rpos,action],1))
        if self.layer_norm :
            out = self.LN1(out)
        out = self.relu(out)
        out = self.fc2(out)
        if self.layer_norm :
            out = self.LN2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
