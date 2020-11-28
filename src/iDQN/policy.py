import torch
import math
import numpy as np
import copy
import ctypes
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

class NN_policy(object):
    def __init__(self, Qnetwork, epsilon, action_number):
        self.Qnetwork = copy.deepcopy(Qnetwork)   
        self.epsilon = epsilon 
        self.action_number = action_number
        self.cuda = next(self.Qnetwork.parameters()).is_cuda
    def inference(self,obs_list,state_list):
        with torch.no_grad():
            pos = torch.Tensor(np.vstack([obs[0] for obs in obs_list]))
            laser_data = torch.Tensor(np.vstack([obs[1] for obs in obs_list]))
            if self.cuda:
                pos = pos.cuda()
                laser_data = laser_data.cuda()
            qvalue = self.Qnetwork(pos,laser_data).cpu().numpy()
            action_list = np.argmax(qvalue,1)
        if random.random() <self.epsilon:
            action_list = np.random.randint(self.action_number,size=(action_list.shape[0]))

        return action_list

class Mix_policy(object):
    def __init__(self,Qnetwork,epsilon,search_policy,replace_table,action_number,encoder):
        self.Qnetwork = copy.deepcopy(Qnetwork)
        self.epsilon = epsilon 
        self.search_policy = search_policy
        self.replace_table = replace_table
        self.action_number = action_number
        self.step = 0
        self.encoder = encoder
        # Check Qnetwork is on cuda or not
        self.cuda = next(self.Qnetwork.parameters()).is_cuda
    def inference(self,obs_list,state_list):
        with torch.no_grad():
            pos = torch.Tensor(np.vstack([obs[0] for obs in obs_list]))
            laser_data = torch.Tensor(np.vstack([obs[1] for obs in obs_list]))
            if self.cuda:
                pos = pos.cuda()
                laser_data = laser_data.cuda()
            qvalue = self.Qnetwork(pos,laser_data).cpu().numpy()
            action_list = np.argmax(qvalue,1)
        if random.random() <self.epsilon:
            action_list = np.random.randint(self.action_number,size=(action_list.shape[0]))

        if self.search_policy is not None:
            if self.step<len(self.search_policy):
                for agent_idx in range(len(action_list)):
                    if self.replace_table[self.step][agent_idx] >=0:
                        action_list[agent_idx] = self.encoder(self.search_policy[self.step][agent_idx])
            self.step+=1
        return action_list     


class Agent_Mix_policy(object):
    def __init__(self,Qnetwork,epsilon,expert_Qnetwork,replace_list,action_number):
        self.Qnetwork = copy.deepcopy(Qnetwork)   
        self.epsilon = epsilon 
        self.expert_Qnetwork = copy.deepcopy(expert_Qnetwork)
        self.replace_list = replace_list
        self.action_number = action_number
        self.step = 0
        # Check Qnetwork is on cuda or not
        self.cuda = next(self.Qnetwork.parameters()).is_cuda
    def inference(self,obs_list,state_list):
        with torch.no_grad():
            pos = torch.Tensor(np.vstack([obs[0] for obs in obs_list]))
            laser_data = torch.Tensor(np.vstack([obs[1] for obs in obs_list]))
            if self.cuda:
                pos = pos.cuda()
                laser_data = laser_data.cuda()
            qvalue = self.Qnetwork(pos,laser_data).cpu().numpy()
            action_list = np.argmax(qvalue,1)
        if random.random() <self.epsilon:
            action_list = np.random.randint(self.action_number,size=(action_list.shape[0]))

        if self.replace_list is not None:
            with torch.no_grad():
                qvalue_replace = self.Qnetwork(pos,laser_data).cpu().numpy()
                action_replace = dis2conti(np.argmax(qvalue_replace,1))
            for agent_idx in self.replace_list:
                if state_list[agent_idx].crash or state_list[agent_idx].reach:
                    self.replace_list.remove(agent_idx)
            for agent_idx in self.replace_list:
                action_list[agent_idx] = action_replace[agent_idx]
        return action_list     


class naive_policy(object):
    def __init__(self,max_phi,l,dist, encoder):
        self.max_phi = max_phi
        self.l = l
        self.dist = dist
        self.encoder = encoder
        self.min_r = self.l/np.tan(self.max_phi)
        self.right_o = np.array([self.min_r,0.0])
        self.left_o = np.array([-self.min_r,0.0])
    
    def inference(self,obs_list):
        obs_list = obs_list[0]
        action_list = []
        for obs in obs_list:
            vel,phi = naive_inference(obs[0],obs[1])
            action_list.append(self.encoder([vel,phi]))
        return action_list

class trace_policy(object):
    def __init__(self,trace,repeta, action_number):
        self.trace = trace
        self.action_number = action_number
        self.step  = 0
        self.repeta = repeta
        self.repeta_state = 0
    def inference(self, obs,state):
        if self.step>=len(self.trace):
            action_list = self.trace[-1]
            for idx in range(len(action_list)):
                #(self.action_number -1)//2 is the action with 0 velocity
                action_list[idx] = (self.action_number-1)//2
        else:
            action_list = self.trace[self.step]
        self.repeta_state+=1
        if self.repeta_state== self.repeta:
            self.step += 1
            self.repeta_state = 0
        return action_list