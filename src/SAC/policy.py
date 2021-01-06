import torch
import math
import numpy as np
import copy
import random

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
    def __init__(self,actor,epsilon,search_policy,replace_table,min_dis = 0):
        self.actor = copy.deepcopy(actor)
        self.epsilon = epsilon 
        self.min_dis = min_dis
        self.search_policy = search_policy
        self.replace_table = replace_table
        self.step = 0
        # Check actor is on cuda or not
        self.cuda = next(self.actor.parameters()).is_cuda
    def inference(self,obs_list,state_list):
        with torch.no_grad():
            pos = torch.Tensor(np.vstack([obs[0] for obs in obs_list]))
            laser_data = torch.Tensor(np.vstack([obs[1] for obs in obs_list]))
            if self.cuda:
                pos = pos.cuda()
                laser_data = laser_data.cuda()
            action_rand, _, action_mean = self.actor.sample(pos,laser_data)
        if self.epsilon>0.0:
            action = action_rand.cpu().numpy()
        else:
            action = action_mean.cpu().numpy()
        
            
        action = np.clip(action, -1., 1.)
        action_list = []
        for idx in range(len(obs_list)):
            xt = obs_list[idx][0][0]
            yt = obs_list[idx][0][1]
            if xt**2+yt**2 < self.min_dis**2:
                a = naive_inference(xt,yt)
            else:
                a = [float(action[idx,0]),float(action[idx,1])]
            action_list.append(a)
        if self.search_policy is not None:
            if self.step<len(self.search_policy):
                for agent_idx in range(len(action_list)):
                    if self.replace_table[self.step][agent_idx] >=0:
                        action_list[agent_idx] = self.search_policy[self.step][agent_idx]
            self.step+=1
        return action_list     


class Agent_Mix_policy(object):
    def __init__(self,actor,epsilon,expert_actor,replace_list,min_dis = 0):
        self.actor = copy.deepcopy(actor)   
        self.epsilon = epsilon 
        self.min_dis = min_dis
        self.expert_actor = copy.deepcopy(expert_actor)
        self.replace_list = replace_list
        self.step = 0
        # Check actor is on cuda or not
        self.cuda = next(self.actor.parameters()).is_cuda
    def inference(self,obs_list,state_list):
        with torch.no_grad():
            pos = torch.Tensor(np.vstack([obs[0] for obs in obs_list]))
            laser_data = torch.Tensor(np.vstack([obs[1] for obs in obs_list]))
            if self.cuda:
                pos = pos.cuda()
                laser_data = laser_data.cuda()
            action_rand, _, action_mean = self.actor.sample(pos,laser_data)
        if self.epsilon>0.0:
            action = action_rand.cpu().numpy()
        else:
            action = action_mean.cpu().numpy()
        
        action = np.clip(action, -1., 1.)
        action_list = []
        for idx in range(len(obs_list)):
            xt = obs_list[idx][0][0]
            yt = obs_list[idx][0][1]
            if xt**2+yt**2 < self.min_dis**2:
                a = naive_inference(xt,yt)
            else:
                a = [float(action[idx,0]),float(action[idx,1])]
            action_list.append(a)
        if self.replace_list is not None:
            with torch.no_grad():
                _, _, action_replace = self.expert_actor.sample(pos,laser_data)
                action_replace = action_replace.cpu().numpy()
            action_replace = np.clip(action_replace, -1., 1.)
            for agent_idx in self.replace_list:
                if state_list[agent_idx].crash or state_list[agent_idx].reach:
                    self.replace_list.remove(agent_idx)
            for agent_idx in self.replace_list:
                action_list[agent_idx] = [float(action_replace[agent_idx,0]),float(action_replace[agent_idx,1])]
        return action_list     




class naive_policy(object):
    def __init__(self,max_phi,l,dist):
        self.max_phi = max_phi
        self.l = l
        self.dist = dist
        self.min_r = self.l/np.tan(self.max_phi)
        self.right_o = np.array([self.min_r,0.0])
        self.left_o = np.array([-self.min_r,0.0])
    
    def inference(self,obs_list):
        action_list = []
        for obs in obs_list:
            vel,phi = naive_inference(obs[0][0],obs[0][1])
            action_list.append([vel,phi])
        return action_list

class trace_policy(object):
    def __init__(self,trace,repeta):
        self.trace = trace
        self.step  = 0
        self.repeta = repeta
        self.repeta_state = 0
    def inference(self, obs,state):
        if self.step>=len(self.trace):
            action_list = self.trace[-1]
            for idx in range(len(action_list)):
                action_list[idx] = [0,0]
        else:
            action_list = self.trace[self.step]
        self.repeta_state+=1
        if self.repeta_state== self.repeta:
            self.step += 1
            self.repeta_state = 0
        return action_list