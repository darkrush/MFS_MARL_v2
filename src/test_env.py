import time
from MultiCarSim.Env import MultiCarSim
import numpy as np

class naive_policy(object):
    def __init__(self,max_phi,l,dist):
        self.max_phi = max_phi
        self.l = l
        self.dist = dist*0.1
        self.min_r = self.l/np.tan(self.max_phi)
        self.right_o = np.array([self.min_r,0.0])
        self.left_o = np.array([-self.min_r,0.0])
    
    def inference(self,obs_list,state_list):
        action_list = []
        for obs in obs_list:
            [xt,yt] = obs[0]
            if abs(yt) < self.dist:
                vel = np.sign(xt)
                phi = 0
            else:
                in_min_r = (xt**2+(abs(yt)-self.min_r)**2)< self.min_r**2
                vel = -1 if np.bitwise_xor(in_min_r,xt<0) else 1
                phi = -1 if np.bitwise_xor(in_min_r,yt<0) else 1
            action_list.append([vel,phi])
        return np.array(action_list)

env = MultiCarSim('./scenario/scenario_train.yaml', step_t = 0.3,sim_step = 10)
env.global_agent_prop.N_laser = 4
env.global_agent_prop.true_N = 4
env.time_limit = 50.0
env.agent_number = 4

policy = naive_policy(env.global_agent_prop.K_phi, env.global_agent_prop.L_axis, env.global_agent_prop.R_reach)

env.reset()
env.rollout(policy_call_back=policy.inference, use_gui=True)
traj = env.get_trajectoy()
print(traj)
result = env.get_result()
print(result)