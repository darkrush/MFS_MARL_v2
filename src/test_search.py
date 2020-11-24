from MultiCarSim.Env import MultiCarSim
from DDPG.policy import trace_policy
import time
def check_done(state_list):
    for state in state_list:
        if state.movable :
            return False
    return True


env = MultiCarSim('./scenario/test_search.yaml', step_t = 0.5,sim_step = 100)
env.reset()

search_env = MultiCarSim('./scenario/test_search.yaml', step_t = 0.5,sim_step = 1)
search_env.reset()

state = env.get_state()
start_time = env.total_time

search_env.set_state(state,total_time = start_time)

[trace,index,search_step] = search_env.search_policy(multi_step=2 , back_number= 2, use_gui= False)
if trace is None :
    print("failed!!!")
    exit()
t_policy = trace_policy(trace,1)

env.rollout(t_policy.inference,finish_call_back=check_done,use_gui=True,delay = 0.0)
print(env.get_result())
traj = env.get_trajectoy()
print(traj[0])
