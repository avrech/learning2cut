# file: default_parser.py
# description: default args for cut_selection_dqn & maxcut experiments

import argparse
import yaml


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# default parser to parse system args, corresponding to config files.
# this parser can be used to read default args from config files, and override specific args for specific purpose.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="L2 regularization")
parser.add_argument("--nstep_learning", type=int, default=1, help="n-step learning")
parser.add_argument("--target_update_interval", type=int, default=1000, help="number of gradient steps between two consequent updates of the target network")
parser.add_argument("--value_aggr", type=str, default="max", help="aggregation function for next state q-values. options: max, mean. tqnet-v2 works with max only. ")
parser.add_argument("--credit_assignment", type=str2bool, nargs='?', const=True, default=False, help="assign credit according to rhs normalized slack")
parser.add_argument("--gamma", type=float, default=0.99, help="reward discount factor")
parser.add_argument("--eps_start", type=float, default=0.9, help="epsilon-greedy start probability for random action")
parser.add_argument("--eps_end", type=float, default=0.05, help="epsilon-greedy end probability for random action")
parser.add_argument("--eps_decay", type=int, default=100000, help="number of environment steps between epsilon-greedy start and end values")
parser.add_argument("--dqn_objective", type=str, default="db_auc", help="metric to optimize in cut_selection_dqn steps. options: db_auc, gap_auc")
parser.add_argument("--reward_func", type=str, default="db_auc", help="options: db_auc, gap_auc, db_aucXslope, db_slopeXdiff")
parser.add_argument("--empty_action_penalty", type=int, default=0, help="additive penalty to empty actions")
parser.add_argument("--select_at_least_one_cut", type=str2bool, nargs='?', const=True, default=False, help="enforce selecting at least one cut every separation round")
parser.add_argument("--update_rule", type=str, default="DQN", help="update rule. options: DQN, DDQN")
parser.add_argument("--discard_bad_experience", type=str2bool, nargs='?', const=True, default=False, help="discard training episodes which terminated before LP_ITERATIONS_LIMIT due to weak cuts")
parser.add_argument("--norm_reward", type=str2bool, nargs='?', const=True, default=False, help="normalize the agent auc by SCIP auc. applicapble when fixing training seed to 223, and fotr tuning envs only")
parser.add_argument("--square_reward", type=str2bool, nargs='?', const=True, default=False, help="square the objective auc areas")
parser.add_argument("--n_step_loss_coef", type=float, default=0.0, help="cut_selection_dqn n_step loss coefficient in the total objective loss")
parser.add_argument("--demonstration_loss_coef", type=float, default=1.0, help="demonstration loss coefficient")
parser.add_argument("--demonstration_large_margin", type=float, default=0.1, help="enforce margin between demonstration and other actions")
parser.add_argument("--dqn_arch", type=str, default="TQNet", help="policy net architecture. options: QNet, TQNet")
parser.add_argument("--tqnet_version", type=str, default="v3", help="transformer version. currently v3 is the only one supported.")
parser.add_argument("--emb_dim", type=int, default=128, help="hidden layers dimensionality")
parser.add_argument("--cut_conv", type=str, default="CutConv", help="between-cut graph convolution")
parser.add_argument("--encoder_cut_conv_layers", type=int, default=1, help="number of gnn layers in encoder")
parser.add_argument("--encoder_lp_conv_layers", type=int, default=1, help="tripartite gnn layers")
parser.add_argument("--decoder_layers", type=int, default=1, help="tqnet decoder layera")
parser.add_argument("--decoder_conv", type=str, default="RemainingCutsConv", help="decoder gnn model")
parser.add_argument("--attention_heads", type=int, default=4, help="GATConv heads")
parser.add_argument("--lp_conv_aggr", type=str, default="mean", help="lp conv aggregation")
parser.add_argument("--cut_conv_aggr", type=str, default="mean", help="betwwen-cuts gnn aggregation")
parser.add_argument("--conditional_q_heads", type=str2bool, nargs='?', const=True, default=False, help="predict tuning q values one by one conditioning on prev preds each time")
parser.add_argument("--seed", type=int, default=259385, help="not in use")
parser.add_argument("--hide_scip_output", type=str2bool, nargs='?', const=True, default=True, help="avoids printing scip messages to the console")
parser.add_argument("--num_episodes", type=int, default=2000000, help="number of episodes to play")
parser.add_argument("--log_interval", type=int, default=20, help="number of model parameters updates between logs")
parser.add_argument("--checkpoint_interval", type=int, default=20, help="number of model param updates between consequent checkpoints")
parser.add_argument("--verbose", type=int, default=1, help="verbosity")
parser.add_argument("--ignore_eval_interval", type=str2bool, nargs='?', const=True, default=False, help="if True, evaluates all validation sets every evaluation round, else, follows data config")
parser.add_argument("--ignore_test_early_stop", type=str2bool, nargs='?', const=True, default=False, help="if True, ignore bad episodes, e.g when branching occured")
parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False, help="debug mode")
parser.add_argument("--debug_cuda", type=str2bool, nargs='?', const=True, default=False, help="enable tracing cuda errors")
parser.add_argument("--debug_events", type=str, nargs='+', default=[], help="print debug messages for scip events listed in debug_events")
parser.add_argument("--overfit", type=str, nargs='+', default=[], help="dataset names to train on")
parser.add_argument("--sanity_check", type=str2bool, nargs='?', const=True, default=False, help="cutting-planes debug mode")
parser.add_argument("--fix_training_scip_seed", type=int, default=0, help="set scip random seed in training episodes. if 0, use random seed")
parser.add_argument("--reset_maxcuts", type=int, default=100000, help="default maxcuts value to avoid enoughcuts in SCIP separationRoundLP")
parser.add_argument("--reset_maxcutsroot", type=int, default=100000, help="default maxcuts value to avoid enoughcuts in SCIP separationRoundLP")
parser.add_argument("--use_general_cuts", type=str2bool, nargs='?', const=True, default=False, help="enable scip default cuts")
parser.add_argument("--aggressive_separation", type=str2bool, nargs='?', const=True, default=None, help="let scip adding more cuts every round")
parser.add_argument("--slack_tol", type=float, default=1e-6, help="round slack variables with absolute value less than slack_tol to zero")
parser.add_argument("--use_cycles", type=str2bool, nargs='?', const=True, default=None, help="use cycle inequalities for MAXCUT models")
parser.add_argument("--chordless_only", type=str2bool, nargs='?', const=True, default=False, help="add chordless cycles only")
parser.add_argument("--simple_cycle_only", type=str2bool, nargs='?', const=True, default=False, help="add simple cycles only")
parser.add_argument("--enable_chordality_check", type=str2bool, nargs='?', const=True, default=False, help="set True to count how many chordless cycles applied, and to filter chordal cycles. WARNING! time consuming!")
parser.add_argument("--record_cycles", type=str2bool, nargs='?', const=True, default=False, help="record cycle trajectories (for debug purpose)")
parser.add_argument("--learner_gpu", type=str2bool, nargs='?', const=True, default=False, help="use gpu on learner side")
parser.add_argument("--worker_gpu", type=str2bool, nargs='?', const=True, default=False, help="use gpu for workers (not recommended - degrade performance)")
parser.add_argument("--tester_gpu", type=str2bool, nargs='?', const=True, default=False, help="use gpu on tester side")
parser.add_argument("--num_workers", type=int, default=3, help="number of Apex workers")
parser.add_argument("--num_learners", type=int, default=1, help="todo - verify redundancy and remove this arg")
parser.add_argument("--param_sync_interval", type=int, default=100, help="number of gradient steps on learner side between consequent updates of workers' model parameter ")
parser.add_argument("--max_num_updates", type=int, default=100000, help="maximal number of model parameter updates")
parser.add_argument("--replay_buffer_capacity", type=int, default=100000, help="replay buffer capacity (in transitions)")
parser.add_argument("--replay_buffer_n_demonstrations", type=int, default=10000, help="how many demonstration transitions to save in replay buffer during training")
parser.add_argument("--replay_buffer_demonstration_priority_bonus", type=float, default=0.0001, help="additive bonus to demonstrations priority")
parser.add_argument("--replay_buffer_minimum_size", type=int, default=10000, help="minimal number of transitions in replay buffer to start learning")
parser.add_argument("--replay_buffer_max_mem", type=int, default=40, help="maximal memory usage in GBytes")
parser.add_argument("--use_per", type=str2bool, nargs='?', const=True, default=False, help="use prioritiezed replay buffer")
parser.add_argument("--priority_alpha", type=float, default=0.6, help="prioritized replay buffer alpha")
parser.add_argument("--priority_beta", type=float, default=0.4, help="prioritized replay buffer beta (exploration temperature. 1 is random?)")
parser.add_argument("--priority_beta_start", type=float, default=0.4, help="initial value for prioritized replay buffer beta")
parser.add_argument("--priority_beta_end", type=float, default=1.0, help="final value for prioritized replay buffer beta")
parser.add_argument("--priority_beta_decay", type=int, default=100000, help="number of model parameter steps for linear decay of beta")
parser.add_argument("--worker_buffer_size", type=int, default=1000, help="todo verify and remove")
parser.add_argument("--local_buffer_size", type=int, default=200, help="number of transitions to store locally on workers side before sending to central buffer")
parser.add_argument('--resume', type=str2bool, nargs='?', const=True, default=False, help='set to load the last training status from checkpoint file')
parser.add_argument("--run_id", type=str, default=None, help='wandb run id for resuming')
parser.add_argument("--project", type=str, default='learning2cut', help='wandb project name')
parser.add_argument("--tags", type=str, nargs='+', default=[], help='wandb tags')
parser.add_argument('--rootdir', type=str, default='results', help='path to save results')
parser.add_argument('--datadir', type=str, default='data', help='path to generate/read data')
parser.add_argument('--data_config', type=str, default='../../data/configs/mvc_data_config.yaml', help='data config file, must match --problem')
parser.add_argument('--problem', type=str, default='MVC', help='options: MVC, MAXCUT')
parser.add_argument('--configfile', type=str, default='configs/exp5.yaml', help='general experiment settings')
parser.add_argument('--scip_env', type=str, default='cut_selection_mdp', help='SCIP environment. options: cut_selection, tuning')
parser.add_argument('--gpu-id', type=int, default=None, help='gpu id to use')
parser.add_argument('--use-gpu', type=str2bool, nargs='?', const=True, default=False, help='use gpu if available')
parser.add_argument('--wandb_offline', type=str2bool, nargs='?', const=True, default=False, help='set to run wandb offline')


def update_hparams(hparams, args):
    # override default hparams with specified system args
    # prioritization: 0 (highest) - specified system args, 1 - yaml, 2 - parser defaults.
    for k, v in vars(args).items():
        if k not in hparams.keys() or parser.get_default(k) != v:
            hparams[k] = v
    return hparams


def get_hparams(args):
    # read data specifications
    with open(args.data_config) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)

    # read default experiment config from yaml
    with open(args.configfile) as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)

    # general hparam dict for all modules
    hparams = {**experiment_config, **data_config}

    # update hparams with system args
    hparams = update_hparams(hparams, args)
    return hparams