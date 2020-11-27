
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Qnetwork(nn.Module):
    def __init__(self, nb_pos,nb_laser, nb_actions, hidden1=64, hidden2=64, init_w=3e-3, layer_norm = True):
        super(Qnetwork, self).__init__()
        self.layer_norm = layer_norm
        self.conv1 = nn.Conv1d(1, 1, kernel_size = 5, stride=2, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(1, 1, kernel_size = 5, stride=2, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.fc1 = nn.Linear(nb_laser, hidden1)
        self.fc2 = nn.Linear(hidden1//4+nb_pos, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        if self.layer_norm :
            self.LN1_1 = nn.LayerNorm(hidden1//2)
            self.LN1_2 = nn.LayerNorm(hidden1//4)
            self.LN2 = nn.LayerNorm(hidden2)
        
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.conv1.weight.data = fanin_init(self.conv1.weight.data.size())
        self.conv2.weight.data = fanin_init(self.conv2.weight.data.size())
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, pos,laser):
        out = self.conv1(laser.unsqueeze(1))
        if self.layer_norm :
            out = self.LN1_1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.layer_norm :
            out = self.LN1_2(out)
        out = out.squeeze(1)
        out = self.relu(out)
        out = self.fc2(torch.cat([out,pos],1))
        if self.layer_norm :
            out = self.LN2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out