import torch
import math
import numpy as np
import copy
import random
from .utils import gumbel_softmax,onehot_from_logits
def naive_inference(xt,yt,dist=0.2,min_r=0.65):
    if abs(yt) < dist * 0.5:
        vel = np.sign(xt)
        phi = 0
    else:
        in_min_r = (xt**2+(abs(yt)-min_r)**2)< min_r**2
        vel = -1 if (bool(in_min_r) ^ bool(xt<0)) else 1
        phi = -1 if (bool(in_min_r) ^ bool(yt<0)) else 1
    return vel,phi

class Mix_policy(object):
    def __init__(self,actor,epsilon,search_policy,replace_table,action_number,encoder):
        self.actor = copy.deepcopy(actor)
        self.epsilon = epsilon 
        self.search_policy = search_policy
        self.replace_table = replace_table
        self.action_number = action_number
        self.step = 0
        self.encoder = encoder
        # Check actor is on cuda or not
        self.cuda = next(self.actor.parameters()).is_cuda

    def inference(self,obs_list,state_list):
        with torch.no_grad():
            rpos = torch.Tensor(np.vstack([obs[2] for obs in obs_list]))
            laser_data = torch.Tensor(np.vstack([obs[1] for obs in obs_list]))
            if self.cuda:
                rpos = rpos.cuda()
                laser_data = laser_data.cuda()
            actor_output = self.actor(rpos,laser_data)
            if self.epsilon > 0:
                action_list = gumbel_softmax(actor_output, hard=True).cpu().numpy()
            else: 
                action_list = onehot_from_logits(actor_output).cpu().numpy()
            
        if self.search_policy is not None:
            if self.step<len(self.search_policy):
                for agent_idx in range(len(action_list)):
                    if self.replace_table[self.step][agent_idx] >=0:
                        action_list[agent_idx] = self.encoder(self.search_policy[self.step][agent_idx])
            self.step+=1
        return action_list 


class Agent_Mix_policy(object):
    def __init__(self,actor,epsilon,expert_actor,replace_list,action_number):
        self.actor = copy.deepcopy(actor)   
        self.epsilon = epsilon 
        self.expert_actor = copy.deepcopy(expert_actor)
        self.replace_list = replace_list
        self.action_number = action_number
        self.step = 0
        # Check actor is on cuda or not
        self.cuda = next(self.actor.parameters()).is_cuda
    def inference(self,obs_list,state_list):
        with torch.no_grad():
            rpos = torch.Tensor(np.vstack([obs[2] for obs in obs_list]))
            laser_data = torch.Tensor(np.vstack([obs[1] for obs in obs_list]))
            if self.cuda:
                rpos = rpos.cuda()
                laser_data = laser_data.cuda()

            actor_output = self.actor(rpos,laser_data)
            if self.epsilon > 0:
                action_list = gumbel_softmax(actor_output, hard=True).cpu().numpy()
            else: 
                action_list = onehot_from_logits(actor_output).cpu().numpy()

        if self.replace_list is not None:
            with torch.no_grad():
                actor_output = self.expert_actor(rpos,laser_data)
                if self.epsilon > 0:
                    action_replace = gumbel_softmax(actor_output, hard=True).cpu().numpy()
                else: 
                    action_replace = onehot_from_logits(actor_output).cpu().numpy()

            for agent_idx in self.replace_list:
                if state_list[agent_idx].crash or state_list[agent_idx].reach:
                    self.replace_list.remove(agent_idx)
            for agent_idx in self.replace_list:
                action_list[agent_idx] = action_replace[agent_idx]
        return action_list     