# MultiCarSim

\_\_init\_\_(self, scenario_name, step_t = 0.1, sim_step = 100)
- scenario_name : scenario.yaml文件路径
- step_t : 每一个step的仿真时长，意味着一个step后total_time增加step_t
- sim_step: 每个step_t被分为sim_step个间隔进行仿真，sim_step越大仿真间隔越细，执行耗时也越长。[v1中使用sim_t=step_t/sim_step表示]

reset(self)[step 和 rollout前需要调用reset]

render(self, mode='human')[v1中没有直接render的接口，而是通过backend的use_gui属性控制，现在修改的和gym的环境一致]

rollout(self, policy_call_back, finish_call_back = None, pause_call_back = None,use_gui=False,delay = 0)
- policy_call_back : rollout中需要使用的策略
- finish_call_back : 判断rollout停止的回调
- pause_call_back : 判断rollout暂停的回调
- use_gui : render的开关[v1中没有，通过backend的use_gui属性控制]
- delay : 控制每一帧的延时，防止仿真过快影响观察，默认为0

get_trajectoy(self) [顺序改为StepNum个Trans，每个Trans为一个dict，包含'obs','action','reward','obs_next','done'。每个元素均为包含AgentNum个元素的list]