#from  gym import Env
import copy
import numpy as np
import time

from .basic import AgentProp,AgentState
from .paser import parse_senario
from .utils import *

def prefer_vel_list(vel,phi):
    if phi == 0:
        #return [(vel,0),(vel,-1),(vel,1),(-vel,0),(-vel,-1),(-vel,1)]
        return [(vel,0),(vel,-1),(vel,1)]
    else:
        #return [(vel,phi),(vel,-phi),(vel,0),(-vel,-phi),(-vel,phi),(-vel,0)]
        return [(vel,phi),(vel,-phi),(vel,0)]

class path_calculator(object):
    def __init__(self,max_phi,L_axis, reach_dist):
        self.max_phi = max_phi
        self.L_axis = L_axis
        self.reach_dist = reach_dist
    
    def path(self, state):
        min_r = self.L_axis/np.tan(self.max_phi)
        xt = state.target_x - state.x
        yt = state.target_y - state.y
        ct = np.cos(state.theta)
        st = np.sin(state.theta)
        xt,yt = (xt*ct+yt*st,yt*ct-xt*st)
        if abs(yt) < self.reach_dist*0.5:
            vel = int(np.sign(xt))
            phi = 0
        else:
            in_min_r = (xt**2+(abs(yt)-min_r)**2)< min_r**2
            vel = -1 if (bool(in_min_r) ^ bool(xt<0)) else 1
            phi = -1 if (bool(in_min_r) ^ bool(yt<0)) else 1
        return vel,phi

class SearchNode(object):
    def __init__(self, state_list, time, path_call_back , parent_action = None, parent_action_index = None):

        # state list for all agent and simulation time for this node
        self.state_list = copy.deepcopy(state_list)
        self.time = time

        # record the action which the state of this node is simulated by
        self.parent_action = parent_action
        self.parent_action_index = parent_action_index

        # list of prefer_action, which is used to generate new node
        self.prefer_action_list = []

        # list of enable agent in this state (enable and not reach)
        # the element ordered in enable index list called (by enable order) 
        # the element ordered in all index list called (by all order) 
        # For example, enable_list:[T,F,T,T](by all order) , enable_index_list:[0,2,3](by enable order)
        self.enable_index_list = [index for index,state in enumerate(self.state_list) if (state.enable and not state.reach)]

        # handle the root node
        if parent_action is None:
            self.parent_action =[(0,0) for _ in range(len(self.state_list))]
            self.parent_action_index = [0 for _ in range(len(self.state_list))]
        

        # build prefer action list for each enable agent(by enable order) 
        # prefer action list: [[a0, a1, a2],[a0, a1, a2],[a0, a1, a2]]
        for index in self.enable_index_list:
            vel,phi = path_call_back(self.state_list[index])
            agent_prefer_action_list = prefer_vel_list(vel,phi)
            self.prefer_action_list.append(agent_prefer_action_list)
        
        # build action index list for each agent(by enable order) 
        # a_list_foreach_agent: [[0,1,2],[0,1,2],[0,1,2]]
        a_list_foreach_agent = [ [i for i in range(len(agent_prefer_action_list))] for l in self.prefer_action_list]

        # combined action index_list built by product of a_list_foreach_agent, and sorted by number of zero
        # combined_action_index_list: [[0,0,0],[0,0,1],[0,1,0],[1,0,0],....,[2,2,2]]
        self.combined_action_index_list = [i for i in product( *a_list_foreach_agent)]
        self.combined_action_index_list.sort(key = lambda l : l.count(0),reverse = True)

        # do sth to prevent inv action problem in DFS (repeate in S_a -> S_b -> S_a -> S_b....)
        if parent_action is not None and len(self.prefer_action_list)>0:
            inv_action_index = []
            have_inv = True
            for index,state in enumerate(self.state_list):
                if not have_inv : break
                if state.enable and not state.reach:
                    inv_action = (-parent_action[index][0],parent_action[index][1])
                    if have_inv and inv_action in self.prefer_action_list[-1]:
                        inv_action_index.append(self.prefer_action_list[-1].index(inv_action))
                    else:
                        have_inv = False
            if have_inv:
                self.combined_action_index_list.remove(tuple(inv_action_index))

    # pruning combined_action_index_list by agent_index_list_to_prune(usually because they are crashed)
    def pruning(self, agent_index_list_to_prune, action_index_list_to_prune = None):

        # now always pruning last action  defaultly
        if action_index_list_to_prune is None:
            action_index_list_to_prune = self.last_action

        # decode the agent_index_list_to_prune into enable_index order
        # only need to check the agent in enable_agent_idx_list
        enable_agent_idx_list = [self.enable_index_list.index(i) for i in agent_index_list_to_prune if i in self.enable_index_list ]

        # add the combined actions have any difference with the action_index_list_to_prune
        # so that can purne the the combined actions cause crash
        new_action_index_list = []
        for combined_action_index in self.combined_action_index_list:
            any_difference = False
            for idx in enable_agent_idx_list:
                if combined_action_index[idx] != action_index_list_to_prune[idx]:
                    any_difference = True
                    break
            if any_difference :
                new_action_index_list.append(combined_action_index)
        self.combined_action_index_list = new_action_index_list
    
    # pop the first action in combined_action_index_list to simulation (by enable order)
    def pop_next_action_index(self):
        if len(self.combined_action_index_list) > 0:
            self.last_action = self.combined_action_index_list.pop(0)
            return self.last_action
        else:
            return None
    
    # extern the action_index_list(by enable order) into full_action_index_list(by all order)
    def extern_decode_action(self,action_index_list):
        #default action [0,0] for agent not enable
        full_action_list = [[0,0] for _ in self.state_list]
        #default action index -1 for agent not enableate_list](index >=0 means need to replace)
        full_action_index_list = [-1 for _ in self.state_list]
        for enable_idx,all_idx in enumerate(self.enable_index_list):
            full_action_list[all_idx] = self.prefer_action_list[enable_idx][action_index_list[enable_idx]]
            full_action_index_list[all_idx] = action_index_list[enable_idx]

        return full_action_list, full_action_index_list
    
    def __str__(self):
        return 'AL: '+str(len(self.combined_action_index_list))
    
    def __repr__(self):
        return self.__str__()
    
class history(object):
    def __init__(self):
        self.reset_state = None
        self.reset_obs = None

    def reset(self, reset_state, reset_obs):
        self.reset_state = copy.deepcopy(reset_state)
        self.reset_obs = copy.deepcopy(reset_obs)
        self.action_history = []
        self.next_state_history = []
        self.obs_history = []
        self.time_history = []
        self.reach_history = []
        self.crash_history = []
        self.reward_history = []

    def insert(self, action, next_state,obs, time, reach, crash, reward):
        assert self.reset_state is not None
        self.action_history.append(copy.deepcopy(action))
        self.next_state_history.append(copy.deepcopy(next_state))
        self.obs_history.append(copy.deepcopy(obs))
        self.time_history.append(copy.deepcopy(time))
        self.reach_history.append(copy.deepcopy(reach))
        self.crash_history.append(copy.deepcopy(crash))
        self.reward_history.append(copy.deepcopy(reward))

    def delete(self, N):
        assert self.reset_state is not None
        self.action_history = self.action_history[:-N]
        self.next_state_history = self.next_state_history[:-N]
        self.obs_history = self.obs_history[:-N]
        self.time_history = self.time_history[:-N]
        self.reach_history = self.reach_history[:-N]
        self.crash_history = self.crash_history[:-N]
        self.reward_history = self.reward_history[:-N]

    def set_length(self, N):
        assert self.reset_state is not None
        self.action_history = self.action_history[:N+1]
        self.next_state_history = self.next_state_history[:N+1]
        self.obs_history = self.obs_history[:N+1]
        self.time_history = self.time_history[:N+1]
        self.reach_history = self.reach_history[:N+1]
        self.crash_history = self.crash_history[:N+1]
        self.reward_history = self.reward_history[:N+1]

    def length(self):
        return len(self.action_history)

class Action_decoder(object):
    def __init__(self, N_vel, N_phi, one_hot):
        #velocity can be [-N_vel, -N_vel+1, ...-1, 0, +1, ...N_vel] total 2N+1
        self.N_vel = N_vel
        #phi can be [-N_phi, -N_phi+1, ...-1, 0, +1, ...N_phi] total 2N+1
        self.N_phi = N_phi
        self.action_number = 2*N_vel*(2*N_phi+1)+1
        #all action number is 2*N_vel*(2*N_phi+1)+1, because all actions with vel==0 are equally
        self.integer_continue_action_table = []
        self.continue_action_table = []

        self.one_hot = one_hot
        # continue_action_table:
        # [(-1,-1),(-1,0),(-1,+1),(0,0),(+1,+1),(+1,0),(+1,+1)]
        # minus velocity part
        for i_vel in range(self.N_vel*2+1):
            if i_vel == self.N_vel:
                self.integer_continue_action_table.append((0,0))
                self.continue_action_table.append((0.0,0.0))
            else:
                for i_phi in range(self.N_phi*2+1):
                    action = (i_vel-self.N_vel, i_phi-self.N_phi)
                    self.integer_continue_action_table.append(action)
                    continue_action = (i_vel/self.N_vel-1, i_phi/self.N_phi-1)
                    self.continue_action_table.append(continue_action)
        
        
    def decode(self, discrete_action):
        if not self.one_hot:
            #one simple action
            if isinstance(discrete_action,int):
                continue_action = self.continue_action_table[discrete_action]
            #list of int action
            if isinstance(discrete_action,list):
                continue_action = [self.continue_action_table[d_a] for d_a in discrete_action]
            #np array int action
            if isinstance(discrete_action,np.ndarray):
                continue_action = np.array([self.continue_action_table[d_a] for d_a in discrete_action])
        else:
            if isinstance(discrete_action,list):
                #list of onehot ndarray
                if isinstance(discrete_action[0],np.ndarray):
                    assert len(discrete_action[0].shape) == 1
                    continue_action = [self.continue_action_table[d_a.argmax()] for d_a in discrete_action]
                #list of onehot list
                if isinstance(discrete_action[0],list):
                    assert isinstance(discrete_action[0][0], float)
                    continue_action = [self.continue_action_table[d_a.index(max(d_a))] for d_a in discrete_action]
            #2d np array onehot
            elif isinstance(discrete_action, np.ndarray):
                assert len(discrete_action.shape) ==2
                continue_action = np.array([self.continue_action_table[d_a.argmax()] for d_a in discrete_action])
        return continue_action
    def conti2disc(self,action):
        index_vel = round(action[0]*self.N_vel)
        index_phi = round(action[1]*self.N_phi)
        action_index = self.integer_continue_action_table.index((index_vel,index_phi))
        if not self.one_hot:
            return action_index
        else:
            one_hot_result = [0.0]*self.action_number
            one_hot_result[action_index] = 1.0
            return one_hot_result

    def encode(self, continue_action):
        if isinstance(continue_action,np.ndarray):
            if len(continue_action.shape) == 2:
                assert continue_action.shape[1] == 2
                discrete_action = np.array([self.conti2disc(continue_action[idx]) for idx in range(continue_action.shape[0])])
            elif len(continue_action.shape) == 1:
                assert continue_action.shape[0] == 2
                discrete_action = np.array(self.conti2disc(continue_action))
            else:
                print('action for encode shape not correct', continue_action)
        elif isinstance(continue_action,list):
            if isinstance(continue_action[0],float) or isinstance(continue_action[0],int):
                assert len(continue_action)==2
                discrete_action = self.conti2disc(continue_action)
            elif isinstance(continue_action[0],np.ndarray):
                discrete_action =[]
                for action in continue_action:
                    assert action.shape[0]==2
                    discrete_action.append(np.array(self.conti2disc(action)))
            elif isinstance(continue_action[0],list) or isinstance(continue_action[0],tuple):
                discrete_action =[]
                for action in continue_action:
                    assert len(action)==2
                    discrete_action.append(self.conti2disc(action))
        return discrete_action


class MultiCarSim(object):

    def __init__(self, scenario_name, step_t = 0.1, sim_step = 100, discrete = False, one_hot = False):
        senario_dict = parse_senario(scenario_name)
        
        self.global_agent_prop = AgentProp(agent_prop=senario_dict['default_agent'])

        self.agent_number = 0
        for (_,agent_group) in senario_dict['agent_groups'].items():
            for agent_prop in agent_group:
                self.agent_number += 1
        
        self.time_limit = senario_dict['common']['time_limit']
        self.reward_coef = senario_dict['common']['reward_coef']
        self.field_range = senario_dict['common']['field_range']
        if discrete is False:
            self.discrete = False
            self.action_decoder = None
        else:
            assert len(discrete) == 2
            self.discrete = True
            self.action_decoder = Action_decoder(discrete[0],discrete[1], self.one_hot)

        self.one_hot = one_hot
        
        self.step_t = step_t
        self.sim_step = sim_step
        self.sim_t = step_t/sim_step
        
        self.history = history()

        # reset_mode is 'random' or 'init'
        self.reset_mode = senario_dict['common']['reset_mode']
        if self.reset_mode =='random':
            self.ref_state_list = None
        elif self.reset_mode =='init':
            self.ref_state_list = []
            for (_,grop) in senario_dict['agent_groups'].items():
                for agent_prop in grop:
                    agent = AgentProp(agent_prop)
                    state = AgentState()
                    state.x = agent.init_x
                    state.y = agent.init_y
                    state.theta = agent.init_theta
                    state.vel_b = agent.init_vel_b
                    state.movable = agent.init_movable
                    state.phi = agent.init_phi
                    state.target_x = agent.init_target_x
                    state.target_y = agent.init_target_y
                    state.enable = True
                    state.crash = False
                    state.reach = False
                    self.ref_state_list.append(state)

        # assign color for each agent
        self.agent_color_list = None
        self.cam_bound = None
        self.viewer = None

    def reset(self):
        # reset initial state list in random way or in inital way
        if self.reset_mode == 'random':
            temp_state_list = [AgentState() for _ in range(self.agent_number)]
            self.last_state_list, enable_list, enable_tmp = random_reset(field_range=self.field_range,
                                                 state_before_reset=temp_state_list,
                                                 R_safe=self.global_agent_prop.R_safe,
                                                 all_reset=True)
        elif self.reset_mode == 'init':
            self.last_state_list = copy.deepcopy(self.ref_state_list)
        else:
            print('reset_mode must be random or init but %s found'%self.reset_mode)
        
        self.total_time = 0.0
        self.laser_dirty = True
        self._reset_render()
        self.last_obs_list = self.get_obs_list()
        self.history.reset(self.last_state_list,self.last_obs_list)
        return self.last_obs_list
    
    def get_state(self):
        return self.last_state_list

    def set_state(self,state_list,enable_list = None,total_time = None):
        if enable_list is None:
            enable_list = [True for _ in state_list]
        for idx,enable in enumerate(enable_list):
            if enable:
                self.last_state_list[idx] = copy.deepcopy(state_list[idx])
        if total_time is not None:
            self.total_time = total_time
        self.laser_dirty = True

    def step(self, action):
        reward_list = self._step(action)

        done = False
        info = {'reward_list':reward_list, 'time_stop':self.total_time > self.time_limit}
        return self.last_obs_list,sum(reward_list),done,info
    
    def get_obs_list(self):
        obs_list = []
        self._update_laser_state()
        for idx in range(self.agent_number):
            state = self.last_state_list[idx]
            xt = state.target_x - state.x
            yt = state.target_y - state.y
            st = np.sin(state.theta)
            ct = np.cos(state.theta)
            related_x = xt*ct+yt*st
            related_y = yt*ct-xt*st
            pos = np.array([related_x,related_y])
            obs_list.append([pos,self.laser_state_list[idx]])
        return obs_list

    def render(self, mode='human'):
        if self.agent_color_list is None:
            self.agent_color_list = []
            for idx in range(self.agent_number):
                color = hsv2rgb(360.0/self.agent_number*idx,1.0,1.0)
                self.agent_color_list.append(color)
        if self.cam_bound is None:
            center_x = (self.field_range[0]+self.field_range[1])/2.0
            center_y = (self.field_range[2]+self.field_range[3])/2.0
            length_x = (self.field_range[1]-self.field_range[0])
            length_y = (self.field_range[3]-self.field_range[2])
            # 1.2 times of field range for camera range
            self.cam_bound = [center_x - 0.7*length_x, 1.4*length_x, center_y - 0.7*length_y, 1.4*length_y ]
        if self.viewer is None:
            from . import rendering 

            import pyglet
            screen = pyglet.canvas.get_display().get_default_screen()
            max_width = int(screen.width * 0.9) 
            max_height = int(screen.height * 0.9)
            if self.cam_bound[1]/self.cam_bound[3]>max_width/max_height:
                screen_width = max_width
                screen_height  = max_width/(self.cam_bound[1]/self.cam_bound[3])
            else:
                screen_height = max_height
                screen_width  = max_height*(self.cam_bound[1]/self.cam_bound[3])
            self.viewer = rendering.Viewer(int(screen_width),int(screen_height))
            
            self.viewer.set_bounds(self.cam_bound[0],self.cam_bound[0]+self.cam_bound[1],self.cam_bound[2],self.cam_bound[2]+self.cam_bound[3])
        # create rendering geometry
        if self.agent_geom_list is None:
            # import rendering only if we need it (and don't import for headless machines)
            from . import rendering
            self.agent_geom_list = []
            
            for idx in range(self.agent_number):
                agent_geom = {}
                total_xform = rendering.Transform()
                agent_geom['total_xform'] = total_xform
                agent_geom['laser_line'] = []

                agent_color = self.agent_color_list[idx]
                geom = rendering.make_circle(self.global_agent_prop.R_reach)
                geom.set_color(*agent_color)
                xform = rendering.Transform()
                geom.add_attr(xform)
                agent_geom['target_circle']=(geom,xform)

                N = self.global_agent_prop.N_laser
                for idx_laser in range(N):
                    theta_i = idx_laser*np.pi*2/N
                    #d = agent.R_laser
                    d = 1
                    end = (np.cos(theta_i)*d, np.sin(theta_i)*d)
                    geom = rendering.make_line((0, 0),end)
                    geom.set_color(0.0,1.0,0.0,alpha = 0.5)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.add_attr(total_xform)
                    agent_geom['laser_line'].append((geom,xform))
                
                half_l = self.global_agent_prop.L_car/2.0
                half_w = self.global_agent_prop.W_car/2.0
                geom = rendering.make_polygon([[half_l,half_w],[-half_l,half_w],[-half_l,-half_w],[half_l,-half_w]])
                geom.set_color(*agent_color,alpha = 0.4)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['car']=(geom,xform)

                geom = rendering.make_line((0,0),(half_l,0))
                geom.set_color(1.0,0.0,0.0,alpha = 1)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['front_line']=(geom,xform)
                
                geom = rendering.make_line((0,0),(-half_l,0))
                geom.set_color(0.0,0.0,0.0,alpha = 1)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['back_line']=(geom,xform)

                self.agent_geom_list.append(agent_geom)

            self.viewer.geoms = []
            for agent_geom in self.agent_geom_list:
                self.viewer.add_geom(agent_geom['target_circle'][0])
                for geom in agent_geom['laser_line']:
                    self.viewer.add_geom(geom[0])
                self.viewer.add_geom(agent_geom['car'][0])
                self.viewer.add_geom(agent_geom['front_line'][0])
                self.viewer.add_geom(agent_geom['back_line'][0])

        self._update_laser_state()
        for agent_idx,agent_geom in enumerate(self.agent_geom_list):
            agent_state = self.last_state_list[agent_idx]
            for idx,laser_line in enumerate(agent_geom['laser_line']):
                    l = self.laser_state_list[agent_idx][idx]
                    laser_line[1].set_scale(l,l) 
            agent_geom['front_line'][1].set_rotation(agent_state.phi)
            agent_geom['target_circle'][1].set_translation(agent_state.target_x*1.0,agent_state.target_y*1.0)
            agent_geom['total_xform'].set_rotation(agent_state.theta)
            agent_geom['total_xform'].set_translation(agent_state.x*1.0,agent_state.y*1.0)
        time_str = "{0:.2f}s".format(self.total_time)
        return self.viewer.render(time=time_str,return_rgb_array = mode=='rgb_array')

    def rollout(self, policy_call_back, finish_call_back = None, pause_call_back = None,use_gui=False,delay = 0):
        result_flag = None
        while True:
            action = policy_call_back(self.last_obs_list, self.last_state_list)

            reward_list = self._step(action)

            #check whether we should pause rollout
            pause_flag = False
            if pause_call_back is not None:
                pause_flag = pause_call_back(self.history.crash_history[-1])
            # if pause_flag is true, return with result_flag='pause'
            if pause_flag:
                result_flag = 'pause'
                break

            # check whether we should stop one rollout
            finish_flag = False
            if finish_call_back is not None:
                finish_flag = finish_call_back(self.last_state_list)
    
            finish_flag = finish_flag or (self.total_time > self.time_limit)
            if finish_flag:
                result_flag = 'finish'
                break
            
            if use_gui:
                self.render()
            if delay>0:
                time.sleep(delay)

        return result_flag

    def get_trajectoy(self):
        last_obs = self.history.reset_obs
        trajectoy = []
        for idx in range(self.history.length()):
            trans = {}
            trans['obs'] = copy.deepcopy(last_obs)
            trans['action'] = copy.deepcopy(self.history.action_history[idx])
            trans['reward'] = copy.deepcopy(self.history.reward_history[idx])
            trans['obs_next'] = copy.deepcopy(self.history.obs_history[idx])
            trans['done'] = [False]*self.agent_number
            last_obs = trans['obs_next']
            trajectoy.append(trans)        
        return trajectoy

    def get_result(self):
        
        result = {}
        crash_time = np.sum(self.history.crash_history)
        reach_time = np.sum(self.history.reach_history)
        total_reward = np.sum(self.history.reward_history)

        vel_list = []
        for agent_states in self.history.next_state_history:
            for state in agent_states:
                vel_list.append(abs(state.vel_b))

        result['crash_time'] = crash_time
        result['reach_time'] = reach_time
        result['total_reward'] = total_reward
        result['no_potential_reward'] = reach_time*self.reward_coef['reach'] - crash_time*self.reward_coef['crash']
        result['total_time'] = self.history.time_history[-1]
        result['mean_vel'] = sum(vel_list)/len(vel_list)
        return result

    def search_policy(self, multi_step = 2, back_number = 1, use_gui = False):
        # path_calculator by naive policy
        path_calc = path_calculator(self.global_agent_prop.K_phi,
                                    self.global_agent_prop.L_axis,
                                    self.global_agent_prop.R_reach)
        
        # count the total DFS_step and stop searching if reach MAX_STEP
        MAX_STEP = 1e4
        DFS_step = 0
        
        #count the total simulation step and stop searching if reach MAX_STEP
        sim_step = 0

        # time and state for root Node 
        start_time =self.total_time
        begin_state = copy.deepcopy(self.last_state_list)
        search_stack = [SearchNode(begin_state,start_time,path_calc.path),]

        # can not do DFS with root node not clean
        for state in begin_state:
            if not state.enable: continue
            if state.reach or state.crash:
                print('init state for search not clean')
                return None,None, sim_step
        
        # DFS body
        while True:
            # count DFS_step
            DFS_step = DFS_step + 1
            
            # finish if DFS_step beyond MAX_STEP
            if DFS_step >= MAX_STEP:
                DFS_flag = 'MAX'
                break

            # finish if stack is empty (no path is avaliable)
            if len(search_stack) == 0:
                DFS_flag = 'NO_PATH'
                break
            
            # pop and get the next action index of the last node to build new node
            next_action_index = search_stack[-1].pop_next_action_index()

            # pop the last node in the search_stack if no next action index in last node
            if next_action_index is None:
                search_stack.pop()
                continue

            # decode the next action index to real action
            next_action,next_action_index = search_stack[-1].extern_decode_action(next_action_index)

            # apply the action and do simulation to get new state 
            self._apply_action(next_action)
            for _ in range(multi_step):
                for _ in range(self.sim_step):
                    self._integrate_state()
                    self._check_collisions()
                    self._check_reach()
                sim_step += 1
            
            # for debug
            if use_gui:
                self.render()
                
            # get the new state
            new_time  = self.total_time
            new_state = copy.deepcopy(self.last_state_list)

            # check if all agent reach or some agent crash
            have_crash = True in [state.crash and state.enable for state in new_state]

            # if some agent crash
            if have_crash:
                for _ in range(len(search_stack) - max(len(search_stack)-back_number,0) - 1):
                    search_stack.pop()
                    
                crash_index = [idx for idx,state in enumerate(new_state) if state.enable and state.crash]
                search_stack[-1].pruning(crash_index)
                self.set_state(search_stack[-1].state_list,total_time = search_stack[-1].time)
                continue
            
            # if no agent crash, append new SearchNode into search stack
            search_stack.append(SearchNode(new_state,new_time,path_calc.path,next_action,next_action_index))

            # finish if DFS_step beyond MAX_STEP
            all_reach = all([state.reach or not state.enable for state in new_state]) 
            if all_reach:
                DFS_flag = 'ALL_REACH'
                break

        if DFS_flag == 'MAX' or DFS_flag == 'NO_PATH':
            # failed search policy
            action_list = None
            action_index_list = None
        elif DFS_flag == 'ALL_REACH':
            # found avaliable policy
            action_list = []
            action_index_list = []
            for node in search_stack[1:]:
                for _ in  range(multi_step):
                    action_list.append(node.parent_action)
                    action_index_list.append(node.parent_action_index)
        
        return action_list, action_index_list, sim_step

    def _step(self, action):
        old_state = copy.deepcopy(self.last_state_list)
        self._apply_action(action)
        for _ in range(self.sim_step):
            self._integrate_state()
            self._check_collisions()
            self._check_reach()
        new_state_before_reset = copy.deepcopy(self.last_state_list)
        reward_list = self._calc_reward_list(old_state, self.last_state_list)

        if self.reset_mode == 'random':
            new_state_after_reset, enable_list, enable_tmp = random_reset(field_range=self.field_range,
                                                 state_before_reset=new_state_before_reset,
                                                 R_safe=self.global_agent_prop.R_safe,
                                                 all_reset=False)
            # if the state real changed, enable is True
            set_new_state = enable_tmp
        elif self.reset_mode == 'init': 
            change = False 
            for idx,state in enumerate(new_state_before_reset):
                if state.reach and state.enable:
                    change = True
                    state.enable = False
            enable_list = None
            set_new_state = change
            new_state_after_reset = new_state_before_reset
        if set_new_state:
            self.set_state(new_state_after_reset, enable_list=enable_list )
        
        reach_list = [state.reach for state in new_state_before_reset]
        crash_list = [state.crash for state in new_state_before_reset]
        self.last_obs_list = self.get_obs_list()
        self.history.insert(action, self.last_state_list,
                            self.last_obs_list, self.total_time,
                            reach_list, crash_list, reward_list)
        
        return reward_list

    def _apply_action(self, action):
        if self.discrete is True:
            action = self.action_decoder.decode(action)
        else:
            action = np.clip(action, -1.0,1.0)
        # set applied forces
        K_vel = self.global_agent_prop.K_vel
        K_phi = self.global_agent_prop.K_phi
        for idx_a in range(self.agent_number):
            state_a = self.last_state_list[idx_a]
            if state_a.movable:
                state_a.vel_b = action[idx_a][0]*K_vel
                state_a.phi   = action[idx_a][1]*K_phi
            else:
                state_a.vel_b = 0
                state_a.phi   = 0
                
    def _integrate_state(self):
        L_axis = self.global_agent_prop.L_axis
        self.total_time += self.sim_t
        for state in self.last_state_list:
            if state.movable:
                state.x, state.y, state.theta = integrate_state_wrap(state, L_axis, self.sim_t)
        self.laser_dirty = True

    def _check_collisions(self):
        R_safe = self.global_agent_prop.R_safe
        for idx_a in range(self.agent_number):
            state_a = self.last_state_list[idx_a]
            if state_a.crash:
                continue
            for idx_b in range(self.agent_number):
                if not idx_a==idx_b :
                    state_b = self.last_state_list[idx_b]
                    if check_AA_collisions(R_safe, state_a, state_b):
                        state_a.crash = True
                        state_a.movable = False
                        break
        
    def _check_reach(self):
        R_reach = self.global_agent_prop.R_reach
        for idx_a in range(self.agent_number):
            state_a = self.last_state_list[idx_a]
            if check_reach(R_reach, state_a):
                state_a.reach = True
                state_a.movable = False

    def _calc_reward_list(self, old_state, new_state):
        reward_list = []
        for idx in range(self.agent_number):
            os = old_state[idx]
            ns = new_state[idx]
            reach_reward = self.reward_coef['reach'] if ns.reach else 0.0
            crash_reward = self.reward_coef['crash'] if ns.crash else 0.0
            time_penalty = self.reward_coef['time_penalty']
            if not ns.reach or ns.crash:
                old_dis = ((os.x-os.target_x)**2+(os.y-os.target_y)**2)**0.5
                new_dis = ((ns.x-ns.target_x)**2+(ns.y-ns.target_y)**2)**0.5
                potential_reward = (old_dis-new_dis)*self.reward_coef['potential']
            else:
                potential_reward = 0.0
            reward_list.append(reach_reward + crash_reward + time_penalty + potential_reward)

        return reward_list

    def _reset_render(self):
        self.agent_geom_list = None

    def _update_laser_state(self):
        if not self.laser_dirty:
            return
        R_laser = self.global_agent_prop.R_laser
        N_laser = self.global_agent_prop.N_laser
        true_N = self.global_agent_prop.true_N
        L_car = self.global_agent_prop.L_car
        W_car = self.global_agent_prop.W_car
        self.laser_state_list = []
        for idx_a in range(self.agent_number):
            laser_state = np.array([R_laser]*N_laser)
            for idx_b in range(self.agent_number):
                if idx_a == idx_b:
                    continue
                new_laser = laser_agent_agent_wrap(R_laser, N_laser, true_N, L_car, W_car, self.last_state_list[idx_a], self.last_state_list[idx_b])
                #agent_a.laser_agent_agent(agent_b)
                laser_state = np.min(np.vstack([laser_state,new_laser]),axis = 0)
            self.laser_state_list.append(laser_state)
        self.laser_dirty = False
