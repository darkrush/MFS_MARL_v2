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
        self.state_list = copy.deepcopy(state_list)
        self.time = time
        self.parent_action = parent_action
        self.parent_action_index = parent_action_index
        self.prefer_action_list = []
        self.enable_index_list = []
        if parent_action is None:
            self.parent_action =[]
            self.parent_action_index = []
            for state in self.state_list:
                self.parent_action.append([0,0])
                self.parent_action_index.append(0)
        action_index_list = []

        for index,state in enumerate(self.state_list):
            if state.enable and not state.reach:
                self.enable_index_list.append(index)
                vel,phi = path_call_back(state)
                self.prefer_action_list.append(prefer_vel_list(vel,phi))
                action_index_list.append([i for i in range(len(self.prefer_action_list[-1]))])
        self.action_index_list = [i for i in product( *action_index_list)]
        #random.shuffle (self.action_index_list )
        self.action_index_list.sort(key = lambda l : l.count(0),reverse = True)

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
                self.action_index_list.remove(tuple(inv_action_index))

    def pruning(self, agent_id_list, action_index_list = None):
        if action_index_list is None:
            action_index_list = self.last_action
        enable_agent_idx_list = [self.enable_index_list.index(i) for i in agent_id_list if i in self.enable_index_list ]
        new_action_index_list = []
        for action_index in self.action_index_list:
            check = False
            for idx in enable_agent_idx_list:
                if action_index[idx] != action_index_list[idx]:
                    check = True
                    break
            if check :
                new_action_index_list.append(action_index)
        self.action_index_list = new_action_index_list
        
        

    def next_action_index(self):
        if len(self.action_index_list) > 0:
            self.last_action = self.action_index_list.pop(0)
            return self.last_action
        else:
            return None
        
    def decode_action(self,action_index_list):
        action_list = [[0,0] for _ in self.state_list]
        action_index_list_ = [-1 for _ in self.state_list]
        for enable_idx,all_idx in enumerate(self.enable_index_list):
            test_action = self.prefer_action_list[enable_idx][action_index_list[enable_idx]]
            action_list[all_idx][0] = test_action[0]
            action_list[all_idx][1] = test_action[1]
            action_index_list_[all_idx] = action_index_list[enable_idx]
        return action_list,action_index_list_
    

    def __str__(self):
        return 'AL: '+str(len(self.action_index_list))
    
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
    
    def length(self):
        return len(self.action_history)


class MultiCarSim(object):

    def __init__(self, scenario_name, step_t = 0.1, sim_step = 100):
        senario_dict = parse_senario(scenario_name)
        
        self.global_agent_prop = AgentProp(agent_prop=senario_dict['default_agent'])

        self.agent_number = 0
        for (_,agent_group) in senario_dict['agent_groups'].items():
            for agent_prop in agent_group:
                self.agent_number += 1
        
        self.time_limit = senario_dict['common']['time_limit']
        self.reward_coef = senario_dict['common']['reward_coef']
        self.field_range = senario_dict['common']['field_range']
        
        self.step_t = step_t
        self.sim_step = sim_step
        self.sim_t = step_t/sim_step
        
        self.history = history()

        # reset_mode is 'random' or 'init'
        self.reset_mode = senario_dict['common']['reset_mode']
        if self.reset_mode =='random':
            self.ref_state_list = None
        else:
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
                pause_flag = pause_call_back(self.last_state_list)
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
        result['total_time'] = self.history.time_history[-1]
        result['mean_vel'] = sum(vel_list)/len(vel_list)
        return result

    def search_policy(self, multi_step = 2, back_number = 1, use_gui = False):
        path_calc = path_calculator(self.global_agent_prop.K_phi,
                                    self.global_agent_prop.L_axis,
                                    self.global_agent_prop.R_reach)
        search_step = 0
        MAX_STEP = 1e4
        start_time =self.total_time
        begin_state = copy.deepcopy(self.last_state_list)
        for state in begin_state:
            if not state.enable: continue
            if state.reach or state.crash:
                print('init state for search not clean')
                return None,None, search_step
        search_stack = [SearchNode(begin_state,start_time,path_calc.path),]
        search_finish = False
        step = 0

        #DFS body
        while not search_finish:

            print(search_stack)
            step = step + 1
            if step >= MAX_STEP:
                return None, None, search_step

            if len(search_stack) == 0:
                return None, None, search_step
            next_action_index = search_stack[-1].next_action_index()
            if next_action_index is None:
                search_stack.pop()
                continue
            next_action,next_action_index = search_stack[-1].decode_action(next_action_index)

            self._apply_action(next_action)
            for _ in range(multi_step):
                for _ in range(self.sim_step):
                    self._integrate_state()
                    self._check_collisions()
                    self._check_reach()
                search_step += 1
                
            if use_gui:
                self.render()
                

            new_time  = self.total_time
            new_state = copy.deepcopy(self.last_state_list)
            have_crash = False
            all_reach = True
            crash_index = []
            for idx,state in enumerate(new_state):
                if state.enable:
                    if state.crash:
                        crash_index.append(idx)
                    all_reach = all_reach and state.reach
                    have_crash = have_crash or state.crash
            
            if have_crash:
                search_stack = search_stack[:-back_number]
                #for _ in range(len(search_stack) - max(len(search_stack)-back_number,0) - 1):
                #    search_stack.pop()

                search_stack[-1].pruning(crash_index)#,next_action_index)
                self.set_state(search_stack[-1].state_list,total_time = search_stack[-1].time)
                continue
            else:
                search_stack.append(SearchNode(new_state,new_time,path_calc.path,next_action,next_action_index))
                if all_reach:
                    search_finish = True
        action_list = []
        action_index_list = []
        for node in search_stack[1:]:
            for _ in  range(multi_step):
                action_list.append(node.parent_action)
                action_index_list.append(node.parent_action_index)
        return action_list,action_index_list,search_step

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
        clip_action = np.clip(action, -1.0,1.0)
        # set applied forces
        K_vel = self.global_agent_prop.K_vel
        K_phi = self.global_agent_prop.K_phi
        for idx_a in range(self.agent_number):
            state_a = self.last_state_list[idx_a]
            if state_a.movable:
                state_a.vel_b = clip_action[idx_a][0]*K_vel
                state_a.phi   = clip_action[idx_a][1]*K_phi
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
