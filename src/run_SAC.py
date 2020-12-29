import time
from .MultiCarSim import Env
from .SAC.trainer import SAC_trainer
#from .MF_env.paser import  parse_senario
from .utils import process_bar,float2time
import numpy as np
import torch
import random

def get_env(scenario_name, step_t, sim_step, agent_number):
    env = Env.MultiCarSim(scenario_name, step_t, sim_step,agent_number = agent_number)
    return env

def run_SAC(args_dict, run_instance = None):

    # set seed for reproducibility
    manualSeed = args_dict['seed']
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    # if you are suing GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # setup environment instance for trainning, searching policy and evaluation
    train_env = get_env(args_dict['train_env'], step_t=args_dict['step_t'], sim_step=args_dict['train_sim_step'],agent_number=args_dict['nb_agents'])
    search_env = get_env(args_dict['search_env'], step_t=args_dict['step_t'], sim_step=args_dict['search_sim_step'],agent_number=args_dict['nb_agents'])
    eval_env = get_env(args_dict['eval_env'], step_t=args_dict['step_t'], sim_step=args_dict['eval_sim_step'],agent_number=args_dict['nb_agents'])
    for env in [train_env, search_env, eval_env]:
        env.reward_coef['reach'] = args_dict['reach']
        env.reward_coef['crash'] = args_dict['crash']
        env.reward_coef['potential'] = args_dict['potential']
        
    # setup SAC trainer.
    trainer = SAC_trainer(args_dict)
    trainer.setup(train_env, search_env, eval_env)
    if args_dict['cuda']==1:
        trainer.cuda()
    
    # Trainning SAC
    cycle_count = 0
    total_search_sample = 0
    total_train_sample = 0
    total_cycle = args_dict['nb_cycles_per_epoch']*args_dict['nb_epoch']

    # Init process_bar
    PB = process_bar(total_cycle)
    PB.start()
    
    for epoch in range(args_dict['nb_epoch']):
        for cycle in range(args_dict['nb_cycles_per_epoch']):
            # Calculate the epsilon decayed by epoch.
            # Which used for policy search exploration.
            t = cycle_count/total_cycle
            a1 = args_dict['decay_args1']
            a2 = args_dict['decay_args2']
            if args_dict['decay_type'] == 'exp':
                epsilon = a1*np.exp(-t*a2)
            elif args_dict['decay_type'] == 'inverse':
                epsilon = a1/(t+a2)
            elif args_dict['decay_type'] == 'linear':
                epsilon = a1-a2*t
            if epsilon >1.0:
                epsilon = 1.0
            if epsilon <0.0:
                epsilon = 0.0
            # Do training in cycle-way. In each cycle, rollout one trajectory and update critic and actor.
            log_info = trainer.cycle(crash_p = epsilon, epsilon = epsilon, train_actor = epoch>0, no_exploration = args_dict['no_exploration'])

            #Calculate total serach step in trainning
            total_search_sample += log_info['train_search_step']
            log_info['train_total_search_sample'] = total_search_sample

            #Calculate total train step in trainning
            total_train_sample += log_info['train_train_step']
            log_info['train_total_train_sample'] = total_train_sample

            log_info['cycle_count'] = cycle_count

            # Do log recording by wandb.
            if run_instance is not None:
                run_instance.log(log_info)
            
            # process bar take a tik
            process_past, time_total, time_left = PB.tik()
            str_time_total = float2time(time_total)
            str_time_left = float2time(time_left)
            # print log info, epoch, cycle, total_reward, crash_time, reach_time.
            str_log_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +\
                       ' epoch: %d/%d, '%(epoch+1, args_dict['nb_epoch']) +\
                       'cycle: %d/%d, '%(cycle_count+1, total_cycle) +\
                       'process: %d%%, '%(process_past*100) +\
                       'time: %s/%s, '%(str_time_left,str_time_total) +\
                       'train_total_reward: %f, '%(log_info['train_total_reward']) +\
                       'train_crash_time: %f, '%(log_info['train_crash_time']) +\
                       'train_reach_time: %f, '%(log_info['train_reach_time'])
            print(str_log_info)
            
            
            # count the total cycle number 
            cycle_count += 1
            
        # save model
        trainer.save_model(run_instance.dir)

        # eval_task
        result = trainer.eval()
        result['epoch'] = epoch
        if run_instance is not None:
            run_instance.log(result)