import copy
import torch
import numpy
from .memory import Memory
from .model import Critic, Actor

class SAC(object):
    def __init__(self, args_dict):
        self.gamma = args_dict['discount']
        self.tau = args_dict['tau']
        self.alpha = args_dict['alpha']
        self.batch_size = args_dict['batch_size']
        self.automatic_entropy_tuning = args_dict['automatic_entropy_tuning']
        self.buffer_size = int(args_dict['buffer_size'])
        self.lr = args_dict['lr']
        self.args_dict = args_dict

    def setup(self):
        self.lr_coef = 1
        self.critic1 = Critic(nb_pos=self.args_dict['nb_pos'],
                        nb_laser=self.args_dict['nb_laser'],
                        nb_actions=self.args_dict['nb_actions'],
                        hidden2 = self.args_dict['hidden2'] ,
                        layer_norm = self.args_dict['layer_norm'])
        self.critic2 =  Critic(nb_pos=self.args_dict['nb_pos'],
                        nb_laser=self.args_dict['nb_laser'],
                        nb_actions=self.args_dict['nb_actions'],
                        hidden2 = self.args_dict['hidden2'] ,
                        layer_norm = self.args_dict['layer_norm'])
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=self.lr )
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=self.lr )
        
        self.critic1_target =  copy.deepcopy(self.critic1)
        self.critic2_target =  copy.deepcopy(self.critic2)

        self.actor = Actor( nb_pos=self.args_dict['nb_pos'],
                            nb_laser=self.args_dict['nb_laser'],
                            nb_actions=self.args_dict['nb_actions'],
                            hidden2 = self.args_dict['hidden2'] ,
                            layer_norm = self.args_dict['layer_norm'])
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr )
        
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning == True:
            self.target_entropy = - self.args_dict['nb_actions']
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr )

        self.memory = Memory(limit=self.buffer_size,
                             action_shape=(self.args_dict['nb_actions'],),
                             observation_shape=[(self.args_dict['nb_pos'],), (self.args_dict['nb_laser'],), (6,)])



    def cuda(self):
        self.memory.cuda()
        for net in (self.critic1, self.critic1_target,
                    self.critic2, self.critic2_target, self.actor):
            if net is not None:
                net.cuda()
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=self.lr )
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=self.lr )
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr )
        if self.automatic_entropy_tuning == True:
            self.target_entropy = - self.args_dict['nb_actions']
            self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda')
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr )


    def select_action(self, state, eval=False):
        if eval == False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        action = torch.clamp(action, min = -1.0, max = 1.0) 
        return action

    def update_critic(self):
        batch = self.memory.sample(self.batch_size)
        state_batch = batch['obs0']
        next_state_batch = batch['obs1']
        action_batch = batch['actions']
        reward_batch = batch['rewards']
        done_batch = batch['terminals1']
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch[0],next_state_batch[1])
            qf1_next_target = self.critic1_target(next_state_batch[0],next_state_batch[1], next_state_action)
            qf2_next_target = self.critic2_target(next_state_batch[0],next_state_batch[1], next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1-done_batch) * self.gamma * (min_qf_next_target)

        q_loss_list = []
        q_value_list = []
        for critic, optim in [[self.critic1, self.critic1_optim],
                              [self.critic2, self.critic2_optim]]:
            q_value = critic(state_batch[0],state_batch[1], action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            q_loss = torch.nn.functional.mse_loss(q_value, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            optim.zero_grad()
            q_loss.backward()
            optim.step()
            q_loss_list.append(q_loss.item())
            q_value_list.append(q_value.mean().item())

        return q_loss_list[0], q_loss_list[1], q_value_list[0],q_value_list[1]

    def update_actor(self):
        batch = self.memory.sample(self.batch_size)
        state_batch = batch['obs0']

        pi, log_pi, _ = self.actor.sample(state_batch[0],state_batch[1])
        qf1_pi = self.critic1(state_batch[0],state_batch[1], pi)
        qf2_pi = self.critic2(state_batch[0],state_batch[1], pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
    
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0

        alpha = self.alpha.item()

        return policy_loss.item(), alpha_loss.item(), alpha

    def updata_target(self, soft_update = True):
        net_pair_list = [[self.critic1_target, self.critic1],
                         [self.critic2_target, self.critic2]]
        for target_net,net in net_pair_list:
            param_zip = zip(target_net.parameters(), net.parameters())
            for target_param, param in param_zip:
                if soft_update:
                    new_param = target_param.data * (1.0 - self.tau) \
                                + param.data * self.tau
                else:
                    new_param = param.data
                target_param.data.copy_(new_param)

    def load_weights(self, model_dir): 
        actor = torch.load('{}/actor.pkl'.format(model_dir) )
        critic1 = torch.load('{}/critic1.pkl'.format(model_dir))
        critic1 = torch.load('{}/critic2.pkl'.format(model_dir))
        self.actor.load_state_dict(actor.state_dict(), strict=True)
        self.critic1.load_state_dict(critic1.state_dict(), strict=True)
        self.critic1_target.load_state_dict(critic1.state_dict(), strict=True)
        self.critic2.load_state_dict(critic2.state_dict(), strict=True)
        self.critic2_target.load_state_dict(critic2.state_dict(), strict=True)


    def save_model(self, model_dir):
        torch.save(self.actor ,'{}/actor.pkl'.format(model_dir))
        torch.save(self.critic1,'{}/critic1.pkl'.format(model_dir))
        torch.save(self.critic2,'{}/critic2.pkl'.format(model_dir))