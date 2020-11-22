import copy
import numpy as np
import numba
from itertools import product

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = np.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    #r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return (r, g, b)


def random_reset(field_range, state_before_reset, R_safe, all_reset = False, retry_time = 40):
    agent_number = len(state_before_reset)
    state_list  = copy.deepcopy(state_before_reset)
    enable_list = [ all_reset|state.crash|state.reach for  state in state_list]
    enable_tmp = True in enable_list

    # list all crashed agent or list all agent if all_reset
    crash_idx_list = []
    for idx ,state in enumerate(state_list):
        if state.crash or all_reset: crash_idx_list.append(idx)

    if len(crash_idx_list)>0:
        # reset agent crash and movable flag
        for idx in crash_idx_list:
            state_list[idx].crash = False
            state_list[idx].movable = True
        for try_time in range(retry_time):
            # random replace the crashed agent
            for idx in crash_idx_list:
                state_list[idx].x = np.random.uniform(field_range[0],field_range[1])
                state_list[idx].y = np.random.uniform(field_range[2],field_range[3])
                state_list[idx].theta = np.random.uniform(0,3.1415926*2)
            

            # check whether the random state_list is no conflict
            no_conflict = True
            for idx_a,idx_b in product(range(agent_number),range(agent_number)):
                if idx_a == idx_b: continue
                state_a = state_list[idx_a]
                state_b = state_list[idx_b]
                agent_dist = ((state_a.x-state_b.x)**2+(state_a.y-state_b.y)**2)**0.5
                no_conflict = agent_dist > 2*R_safe
                # retry if conflict
                if not no_conflict : break
            # stop retrying if no conflict
            if no_conflict: break
        # if not no_conflict: print('failed to place agent with no confiliction')

    # list all reached agent or list all agent if all_reset
    reach_idx_list = []
    for idx ,state in enumerate(state_list):
        if state.reach or all_reset: reach_idx_list.append(idx)

    if len(reach_idx_list)>0:
        for idx in reach_idx_list:
            state_list[idx].reach = False
            state_list[idx].movable = True
        for try_time in range(retry_time):
            # random replace the targets of reached agent
            for idx in reach_idx_list:
                state_list[idx].target_x = np.random.uniform(field_range[0], field_range[1])
                state_list[idx].target_y = np.random.uniform(field_range[2], field_range[3])
                
            # check whether the random state_list is no conflict
            no_conflict = True
            for idx_a,idx_b in product(range(agent_number),range(agent_number)):
                if idx_a == idx_b: continue
                state_a = state_list[idx_a]
                state_b = state_list[idx_b]
                agent_dist = ((state_a.target_x-state_b.target_x)**2+(state_a.target_y-state_b.target_y)**2)**0.5
                no_conflict = agent_dist > 2*R_safe
                # retry if conflict
                if not no_conflict : break
            # stop retrying if no conflict
            if no_conflict: break
        #if not no_conflict: print('failed to place target with no confiliction')
    return state_list, enable_list, enable_tmp

def check_AA_collisions(R_safe, state_a, state_b):
    min_dist_sqr = 4*R_safe**2
    ab_dist_sqr = (state_a.x - state_b.x)**2 + (state_a.y - state_b.y)**2
    return ab_dist_sqr<=min_dist_sqr

def check_reach(R_reach, state_a):
    max_dist_sqr = R_reach**2
    at_dist_sqr = (state_a.x - state_a.target_x)**2 + (state_a.y - state_a.target_y)**2
    return at_dist_sqr<=max_dist_sqr

def integrate_state_wrap(state,L_axis,dt):
    return integrate_state_njit(state.phi,state.vel_b,state.theta,L_axis,state.x,state.y,dt)

@numba.njit()
def integrate_state_njit(_phi,_vb,_theta,_L,_x,_y,dt):
    sth = np.sin(_theta)
    cth = np.cos(_theta)
    _xb = _x - cth*_L/2.0
    _yb = _y - sth*_L/2.0
    tphi = np.tan(_phi)
    _omega = _vb/_L*tphi
    _delta_theta = _omega * dt
    if abs(_phi)>0.00001:
        _rb = _L/tphi
        _delta_tao = _rb*(1-np.cos(_delta_theta))
        _delta_yeta = _rb*np.sin(_delta_theta)
    else:
        _delta_tao = _vb*dt*(_delta_theta/2.0)
        _delta_yeta = _vb*dt*(1-_delta_theta**2/6.0)
    _xb += _delta_yeta*cth - _delta_tao*sth
    _yb += _delta_yeta*sth + _delta_tao*cth
    _theta += _delta_theta
    _theta = (_theta/3.1415926)%2*3.1415926

    nx = _xb + np.cos(_theta)*_L/2.0
    ny = _yb + np.sin(_theta)*_L/2.0
    ntheta = _theta
    return nx,ny,ntheta

def laser_agent_agent_wrap(R_laser,N_laser,true_N,L_car,W_car,a,b):
    return laser_agent_agent_njit(R_laser,N_laser,true_N,a.x,a.y,a.theta,b.x,b.y,b.theta,L_car,W_car)

@numba.njit()
def laser_agent_agent_njit(R_laser,N_laser,true_N,a_x,a_y,a_t,b_x,b_y,b_t,b_L,b_W):

    #l_laser = np.array([R_laser]*N_laser)
    true_l_laser = np.ones((true_N,),dtype=np.float32)*R_laser
    o_pos =  np.array([a_x,a_y])
    oi_pos = np.array([b_x,b_y])
    if np.linalg.norm(o_pos-oi_pos)>R_laser+(b_L**2 + b_W**2)**0.5 / 2.0:
        return np.ones((N_laser,),dtype=np.float32)*R_laser
    theta = a_t
    theta_b = b_t
    cthb= np.cos(theta_b)
    sthb= np.sin(theta_b)
    half_l_shift = np.array([cthb,sthb])*b_L/2.0
    half_w_shift = np.array([-sthb,cthb])*b_W/2.0
    car_points = []
    car_points.append(oi_pos+half_l_shift+half_w_shift-o_pos)
    car_points.append(oi_pos-half_l_shift+half_w_shift-o_pos)
    car_points.append(oi_pos-half_l_shift-half_w_shift-o_pos)
    car_points.append(oi_pos+half_l_shift-half_w_shift-o_pos)
    car_line = [[car_points[i],car_points[(i+1)%len(car_points)]] for i in range(len(car_points))]
    for start_point, end_point in  car_line:
        v_es = start_point-end_point
        tao_es = np.array((v_es[1],-v_es[0]))
        tao_es = tao_es/np.linalg.norm(tao_es)
        if abs(np.dot(start_point,tao_es))>R_laser:
            continue
        
        if start_point[0]*end_point[1]< start_point[1]*end_point[0] :
            start_point,end_point = end_point,start_point
        theta_start = np.arccos(start_point[0]/np.linalg.norm(start_point))
        if start_point[1]<0:
            theta_start = np.pi*2-theta_start
        theta_start-=theta
        theta_end = np.arccos(end_point[0]/np.linalg.norm(end_point))
        if end_point[1]<0:
            theta_end = np.pi*2-theta_end
        theta_end-=theta
        laser_idx_start = theta_start/(2*np.pi/true_N)
        laser_idx_end   =   theta_end/(2*np.pi/true_N)
        if laser_idx_start> laser_idx_end:
            laser_idx_end+=true_N
        if np.floor(laser_idx_end)-np.floor(laser_idx_start)==0:
            continue
        laser_idx_start = np.ceil(laser_idx_start)
        laser_idx_end = np.floor(laser_idx_end)
        for laser_idx in range(laser_idx_start,laser_idx_end+1):
            laser_idx%=true_N
            x1 = start_point[0]
            y1 = start_point[1]
            x2 = end_point[0]
            y2 = end_point[1]
            theta_i = theta+laser_idx*np.pi*2/true_N
            cthi = np.cos(theta_i)
            sthi = np.sin(theta_i)
            temp = (y1-y2)*cthi - (x1-x2)*sthi
            # temp equal zero when collinear
            if abs(temp) <= 1e-10:
                dist = R_laser 
            else:
                dist = (x2*y1-x1*y2)/(temp)
            if dist > 0 and dist < true_l_laser[laser_idx]:
                true_l_laser[laser_idx] = dist
    
    linear_sacle = N_laser//true_N
    if linear_sacle == 1:
        return true_l_laser
    else:
        l_laser = np.ones((N_laser,),dtype=np.float32)*R_laser
        for i in range(true_N):
            for j in range(linear_sacle):
                l_data = true_l_laser[i]
                r_data = true_l_laser[(i+1)%true_N]
                alpha = j/linear_sacle
                tmp = l_data*(1-alpha) + r_data*alpha
                l_laser[i*linear_sacle+j] = tmp
        return l_laser