from MultiCarSim.Env import MultiCarSim


def random_set_state(state):
    
    return state

def run_search(arg_dict):
    
    search_env = MultiCarSim(arg_dict['scenario'], step_t = arg_dict['step_t'],sim_step = arg_dict['sim_step'])

    success_time = 0
    search_step_list = []
    for idx in range(arg_dict['repeat_number'])
        search_env.reset()
        state = search_env.get_state()

        # DO sth to random set state
        new_state = random_set_state(state)

        search_env.set_state(new_state)

        [trace,index,search_step] = search_env.search_policy(multi_step=2 , MAX_STEP=1e4)

        if trace is not None:
            success_time += 1
            search_step_list.append(search_step)
    return success_time, sum(search_step_list)/len(search_step_list)