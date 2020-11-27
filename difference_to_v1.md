# MultiCarSim

\_\_init\_\_(self, scenario_name, step_t = 0.1, sim_step = 100, , discrete = False, one_hot = False)
- scenario_name : scenario.yaml文件路径
- step_t : 每一个step的仿真时长，意味着一个step后total_time增加step_t
- sim_step: 每个step_t被分为sim_step个间隔进行仿真，sim_step越大仿真间隔越细，执行耗时也越长。[v1中使用sim_t=step_t/sim_step表示]
- discrete: 表示是否使用离散任务，discrete = False 为连续任务，discrete = [N_vel, N_phi]为离散任务，具体设定见后面内容
- one_hot: 表示在离散任务是否使用one_hot向量为输入

reset(self)[step 和 rollout前需要调用reset]

render(self, mode='human')[v1中没有直接render的接口，而是通过backend的use_gui属性控制，现在修改的和gym的环境一致]

rollout(self, policy_call_back, finish_call_back = None, pause_call_back = None,use_gui=False,delay = 0)
- policy_call_back : rollout中需要使用的策略
- finish_call_back : 判断rollout停止的回调
- pause_call_back : 判断rollout暂停的回调
- use_gui : render的开关[v1中没有，通过backend的use_gui属性控制]
- delay : 控制每一帧的延时，防止仿真过快影响观察，默认为0

get_trajectoy(self) [顺序改为StepNum个Trans，每个Trans为一个dict，包含'obs','action','reward','obs_next','done'。每个元素均为包含AgentNum个元素的list]


# discrete environment
构造函数中加入discrete的参数，默认为False
```
MultiCarSim(scenario_name, step_t = 0.1, sim_step = 100, discrete = False):
```
需要离散环境时，设置discrete = [ N\_vel, N\_phi ]，其中N\_vel, N\_phi均为整数

这样环境的动作空间变为0~2 * N\_vel * (2 * N\_phi+1) 共2 * N\_vel * (2 * N\_phi+1)+1 个动作

在N_vel = 1, N_phi = 1时，有7个动作
动作顺序为[(-1,-1),(-1,0),(-1,1),(0,0),(1,-1),(1,0),(1,1)]

生成动作序列的代码见src/MultiCarSim/Env.py 中的 Action_decoder类

加入了