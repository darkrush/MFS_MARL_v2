from src.run_SAC import run_SAC
import wandb

def parse_arg():
    import argparse
    parser = argparse.ArgumentParser(description='SAC on pytorch')
    # Wandb args
    parser.add_argument('--bash-tag', default='0', type=str, help='only for bash')
    parser.add_argument('--name', default='SAC_test', type=str, help='name of this exp')
    parser.add_argument('--group', default='debug', type=str, help='group of this exp')
    parser.add_argument('--tags',nargs='*',  help='tags of this exp')
    parser.add_argument('--job-type', default='debug', type=str, help='jobv type of this exp')
    #Training args
    parser.add_argument('--nb-epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--nb-cycles-per-epoch', default=20, type=int, help='number of cycles per epoch')
    parser.add_argument('--nb-rollout-steps', default=100, type=int, help='number rollout steps')
    parser.add_argument('--nb-train-steps', default=20, type=int, help='number train steps')
    parser.add_argument('--nb-warmup-steps', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--train-mode', default=1, type=int, help='traing mode')
    parser.add_argument('--decay-args1', default=1.0, type=float, help='decay coef')
    parser.add_argument('--decay-args2', default=11.513, type=float, help='decay coef')
    parser.add_argument('--decay-type', default='exp', type=str, help='type of decay, exp, inverse, linear')
    
    
    parser.add_argument('--max-step', default = 10000, type=int, help='max search step')
    parser.add_argument('--search-method', default = 0, type=int, help='1 if do policy search')
    parser.add_argument('--back-step', default = 7, type=int, help='back step for search policy')
    parser.add_argument('--multi-step', default = 2, type=int, help='multi search step for search policy')
    parser.add_argument('--seed', default=0, type=int, help='random_seed')
    parser.add_argument('--expert-file', default='./SAC_expert.pkl', type=str, help='expert actor file dir')
    parser.add_argument('--cuda', default=1, type=int, help='cuda')
    parser.add_argument('--sync-step', default=1, type=int, help='1:Sync step for MARL,0:Async step for RL')
    parser.add_argument('--no-exploration',default=0, type=int, help='1:no e-greedy exploration 0:with e-greedy')
    
    #SAC args
    parser.add_argument('--discount', default=0.99, type=float, help='reward discout')
    parser.add_argument('--tau', default=0.005, type=float, help='moving average for target network')
    parser.add_argument('--alpha', default=0.2, type=float, help='entropy coef')
    parser.add_argument('--batch-size', default=256, type=int, help='minibatch size')
    parser.add_argument('--buffer-size', default=1e5, type=int, help='memory buffer size')
    parser.add_argument('--lr', default=0.003, type=float, help='critic and actor net learning rate')
    parser.add_argument('--automatic-entropy-tuning', default=1, type=int, help='auto tuning entropy, default is true')

    #Model args
    parser.add_argument('--hidden1', default=128, type=int, help='number of hidden1')
    parser.add_argument('--hidden2', default=128, type=int, help='number of hidden2')
    parser.add_argument('--not-LN', dest='layer_norm', action='store_false',help='model without LayerNorm')
    parser.set_defaults(layer_norm=True)
    parser.add_argument('--nb-pos', default=2, type=int, help='number of pos')
    parser.add_argument('--nb-laser', default=128, type=int, help='number of laser')
    parser.add_argument('--nb-actions', default=2, type=int, help='number of actions')
    
    #Env args
    parser.add_argument('--reach', default=10.0, type=float, help='reach reward')
    parser.add_argument('--crash', default=-10.0, type=float, help='crash reward')
    parser.add_argument('--potential', default=1.0, type=float, help='potential coef')
    parser.add_argument('--train-env', default='./src/scenario/scenario_train.yaml', type=str, help='train env file')
    parser.add_argument('--search-env', default='./src/scenario/scenario_search.yaml', type=str, help='search env file')
    parser.add_argument('--eval-env', default='./src/scenario/scenario_train.yaml', type=str, help='eval env file')
    parser.add_argument('--step-t', default=1.0, type=float, help='sim time for each step')
    parser.add_argument('--train-sim-step', default=10, type=int, help='sim delta time in train')
    parser.add_argument('--search-sim-step', default=1, type=int, help='sim delta time in search')
    parser.add_argument('--eval-sim-step', default=10, type=int, help='sim delta time in eval')
    parser.add_argument('--nb-agents', default=0, type=int, help='number of agent, 0 use number in yaml file')

    args = parser.parse_args()
    return args

args = parse_arg()

run = wandb.init(config=args, 
                project="multi-fidelity-sim-v2",
                tags=args.tags,
                name=args.name,
                group=args.group,
                dir='./',
                job_type=args.job_type)

run.config.update(args)
run_SAC(run.config,run)