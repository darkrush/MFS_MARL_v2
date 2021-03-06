import torch
import random
from copy import deepcopy
from .SAC import SAC
from .policy import Mix_policy, Agent_Mix_policy


class Crash_Checker(object):
    def __init__(self):
        self.mask = []
        self.p = 1
    def set_mask(self,mask):
        self.mask = mask
    def set_p(self,p):
        self.p = p
    def check_crash(self,crash_list):
        if random.random() < self.p:
            for idx,crash in enumerate(crash_list):
                if crash :
                    if idx in self.mask:
                        self.mask.remove(idx)
                        continue
                    else:
                        return True
        return False

def _mean(l):
    return sum(l)/len(l)

class SAC_trainer(object):
    def __init__(self, args_dict):
        self.args_dict = args_dict
        self.nb_epoch = args_dict['nb_epoch']
        self.nb_cycles_per_epoch = args_dict['nb_cycles_per_epoch']
        self.nb_rollout_steps = args_dict['nb_rollout_steps']
        self.nb_train_steps = args_dict['nb_train_steps']
        self.nb_warmup_steps = args_dict['nb_warmup_steps']
        self.train_mode = args_dict['train_mode']
        self.search_method= args_dict['search_method']
        self.back_step = args_dict['back_step']
        self.multi_step = args_dict['multi_step']
        self.expert_file = args_dict['expert_file']
        self.max_step = args_dict['max_step']

        self.agent = SAC(args_dict)


    def setup(self, env_instance, search_env_instance, eval_env_instance):
        self.env = env_instance
        self.search_env = search_env_instance
        self.eval_env = eval_env_instance
        self.agent.setup()
        if self.search_method is 2 :
            self.expert_actor = torch.load(self.expert_file)

    def cuda(self):
        self.agent.cuda()

    def cycle(self, crash_p, epsilon, train_actor,no_exploration = 0):
        self.env.reset()
        self.search_env.reset()
        crash_checker = Crash_Checker()
        crash_checker.set_p(crash_p)
        #get trajection
        search_policy = None
        replace_table = None
        search_state_index= 0
        last_back_index = 0
        search_step = 0
        while True:
            epsilon_explr = 1.0

            if self.search_method is 1 :
                rollout_policy = Mix_policy(self.agent.actor, epsilon_explr,
                                            search_policy, replace_table)
                finish = self.env.rollout(rollout_policy.inference,
                         pause_call_back = crash_checker.check_crash)
            elif self.search_method is 2 :
                rollout_policy = Agent_Mix_policy(self.agent.actor, epsilon_explr,
                                                  self.expert_actor, replace_table)
                finish = self.env.rollout(rollout_policy.inference,
                         pause_call_back = crash_checker.check_crash)
            else:
                rollout_policy = Mix_policy(self.agent.actor, epsilon_explr,
                                            None, None)
                finish = self.env.rollout(rollout_policy.inference)
            if finish is 'finish' :
                break
            elif finish is 'pause':

                crash_list = self.env.history.crash_history[-1]
                # record the length of the trajectory where paused.
                last_back_index= search_state_index

                search_state_index = max(self.env.history.length()-self.back_step-1,0)
                if self.search_method is 1 :
                    if search_state_index > last_back_index and self.env.history.length() >0:
                        backtrack_state = deepcopy(self.env.history.next_state_history[search_state_index])
                        for b_State,crash in zip(backtrack_state,crash_list):
                            b_State.enable = crash
                        self.search_env.set_state(backtrack_state)
                        search_policy,replace_table,search_step = self.search_env.search_policy(multi_step = self.multi_step,back_number = 1,use_gui = False, MAX_STEP=self.max_step)
                    else:
                        search_policy = None
                        replace_table = None

                elif self.search_method is 2:
                    if search_state_index > last_back_index:
                        replace_table = []
                        for idx,crash in enumerate(crash_list):
                            if crash:
                                replace_table.append(idx)
                    else:
                        replace_table = None
                mask = []
                if replace_table is None:
                    for idx,crash in enumerate(self.env.history.crash_history[-1]):
                        if crash:
                            mask.append(idx)
                    crash_checker.set_mask(mask)
                    self.env.history.delete(1)
                    if self.env.history.length() == 0:
                        self.env.set_state(self.env.history.reset_state,total_time = 0)
                    else:
                        self.env.set_state(self.env.history.next_state_history[-1],total_time = self.env.history.time_history[-1])
                    
                else :
                    self.env.history.set_length(search_state_index)
                    self.env.set_state(self.env.history.next_state_history[search_state_index],total_time = self.env.history.time_history[search_state_index])
                    

        # after get trajectory append data into memory
        trajectoy = self.env.get_trajectoy()
        train_sample = len(trajectoy)
        for trans in trajectoy:
            for agent_idx in range(len(trans['obs'])):
                self.agent.memory.append( trans['obs'][agent_idx], 
                                          trans['action'][agent_idx],
                                          trans['reward'][agent_idx],
                                          trans['obs_next'][agent_idx],
                                          trans['done'][agent_idx])

        c1_loss, c2_loss, q1, q2, policy_loss, ent_loss, alpha = \
            self._apply_train(train_actor=train_actor)

        results = self.env.get_result()

        log_info = {'train_total_reward': results['total_reward'],
                    'train_crash_time': results['crash_time'],
                    'train_reach_time': results['reach_time'],
                    'train_no_potential_reward': results['no_potential_reward'],
                    'train_mean_vel': results['mean_vel'],
                    'train_total_time': results['total_time'],
                    'train_c1_loss': c1_loss,
                    'train_c2_loss': c2_loss,
                    'train_q2_value': q1,
                    'train_q2_value': q2,
                    'train_actor_loss': policy_loss,
                    'train_entropy_loss': ent_loss,
                    'train_alpha': alpha,
                    'train_search_step': search_step,
                    'train_train_step': train_sample}
        return log_info
    
    def save_model(self, model_dir):
        self.agent.save_model(model_dir)

    def eval(self):
        self.eval_env.reset()
        rollout_policy = Mix_policy(self.agent.actor,0.0,None,None)
        self.eval_env.rollout(rollout_policy.inference)
        results = self.eval_env.get_result()
        return results

    def _apply_train(self,train_actor = True):
        cl1_list = []
        cl2_list = []
        q1_list = []
        q2_list = []
        policy_loss_list = []
        ent_loss_list = []
        alpha_list = []

        for _ in range(self.nb_train_steps):
            cl1, cl2,q1,q2 = self.agent.update_critic()
            policy_loss, ent_loss, alpha = self.agent.update_actor()
            cl1_list.append(cl1)
            cl2_list.append(cl2)
            q1_list.append(q1)
            q2_list.append(q2)
            policy_loss_list.append(policy_loss)
            ent_loss_list.append(ent_loss)
            alpha_list.append(alpha)
            if self.train_mode == 0:
                self.agent.updata_target(soft_update=True)
        if self.train_mode == 1:
            self.agent.updata_target(soft_update=False)

        return _mean(cl1_list), _mean(cl2_list), \
               _mean(q1_list), _mean(q2_list), \
               _mean(policy_loss_list), _mean(ent_loss_list), _mean(alpha_list)