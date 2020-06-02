""" DQN
In this experiment cycle inequalities are added on the fly,
and DQN agent select which cuts to apply.
The model optimizes for the dualbound integral.
"""
from utils.scip_models import maxcut_mccormic_model
from separators.mccormick_cycle_separator import MccormickCycleSeparator
import pickle
import os
import torch
from agents.dqn import DQN
from utils.functions import get_normalized_areas
import numpy as np
from tqdm import tqdm


def experiment(hparams):
    # fix random seed for all experiment
    if hparams.get('seed', None) is not None:
        np.random.seed(hparams['seed'])
        torch.manual_seed(hparams['seed'])

    # datasets and baselines
    dataset_paths = {}
    datasets = hparams['datasets']
    for dataset_name, dataset in datasets.items():
        dataset_config = dataset['config']
        dataset_paths[dataset_name] = os.path.join(hparams['datadir'], dataset_name, "barabasi-albert-n{}-m{}-weights-{}-seed{}".format(dataset_config['graph_size'], dataset_config['barabasi_albert_m'], dataset_config['weights'], dataset_config['dataset_generation_seed']))

        # read all graphs with their baselines from disk
        # and find the max lp_iterations_limit across all instances
        dataset['instances'] = []
        lp_iterations_limit = 0
        for filename in tqdm(os.listdir(dataset_paths[dataset_name]), desc=f'Loading {dataset_name}'):
            with open(os.path.join(dataset_paths[dataset_name], filename), 'rb') as f:
                G, baseline = pickle.load(f)
                if baseline['is_optimal']:
                    lp_iterations_limit = max(lp_iterations_limit, baseline['lp_iterations_limit'])
                    dataset['instances'].append((G, baseline))
                else:
                    print(filename, ' is not solved to optimality')
        dataset['lp_iterations_limit'] = lp_iterations_limit
        dataset['num_instances'] = len(dataset['instances'])

    # for the validation and test datasets compute some metrics:
    for dataset_name, dataset in datasets.items():
        if dataset_name == 'trainset':
            continue
        db_auc_list = []
        gap_auc_list = []
        for (_, baseline) in dataset['instances']:
            optimal_value = baseline['optimal_value']
            dualbound = baseline['rootonly_stats']['dualbound']
            gap = baseline['rootonly_stats']['gap']
            lpiter = baseline['rootonly_stats']['lp_iterations']
            db_auc = sum(get_normalized_areas(t=lpiter, ft=dualbound, t_support=dataset['lp_iterations_limit'], reference=optimal_value))
            gap_auc = sum(get_normalized_areas(t=lpiter, ft=gap, t_support=dataset['lp_iterations_limit'], reference=0))
            baseline['db_auc'] = db_auc
            baseline['gap_auc'] = gap_auc
            db_auc_list.append(db_auc)
            gap_auc_list.append(gap_auc)
        # compute stats for the whole dataset
        db_auc_avg = np.mean(db_auc)
        db_auc_std = np.std(db_auc)
        gap_auc_avg = np.mean(gap_auc)
        gap_auc_std = np.std(gap_auc)
        dataset['stats'] = {}
        dataset['stats']['db_auc_avg'] = db_auc_avg
        dataset['stats']['db_auc_std'] = db_auc_std
        dataset['stats']['gap_auc_avg'] = gap_auc_avg
        dataset['stats']['gap_auc_std'] = gap_auc_std

    # training
    trainset = datasets['trainset']
    graph_indices = torch.randperm(trainset['num_instances'])

    # dqn agent
    dqn_agent = DQN(hparams=hparams)
    dqn_agent.train()

    if hparams.get('resume_training', False):
        dqn_agent.load_checkpoint()

    def execute_episode(G, baseline, lp_iterations_limit, dataset_name='trainset', scip_seed=None):
        # create SCIP model for G
        model, x, y = maxcut_mccormic_model(G, use_general_cuts=hparams.get('use_general_cuts', False))  # disable default cuts

        # include cycle inequalities separator with high priority
        cycle_sepa = MccormickCycleSeparator(G=G, x=x, y=y, name='MLCycles', hparams=hparams)
        model.includeSepa(cycle_sepa, 'MLCycles',
                          "Generate cycle inequalities for the MaxCut McCormic formulation",
                          priority=1000000, freq=1)

        # reset dqn_agent to start a new episode
        dqn_agent.init_episode(G, x, y, lp_iterations_limit, cut_generator=cycle_sepa, baseline=baseline, dataset_name=dataset_name)

        # include dqn_agent, setting lower priority than the cycle inequalities separator
        model.includeSepa(dqn_agent, 'DQN', 'Cut selection agent',
                          priority=-100000000, freq=1)

        # set some model parameters, to avoid branching, early stopping etc.
        # termination condition is either optimality or lp_iterations_limit exceeded.
        # since there is no way to limit lp_iterations explicitely,
        # it is enforced implicitely inside the separators.
        model.setLongintParam('limits/nodes', 1)  # solve only at the root node
        model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever

        # set environment random seed
        if scip_seed is None:
            # set random scip seed
            scip_seed = np.random.randint(1000000000)
        model.setBoolParam('randomization/permutevars', True)
        model.setIntParam('randomization/permutationseed', scip_seed)
        model.setIntParam('randomization/randomseedshift', scip_seed)

        if hparams.get('hide_scip_output', True):
            model.hideOutput()

        # gong! run episode
        model.optimize()

        # once episode is done
        dqn_agent.finish_episode()

    for i_episode in range(dqn_agent.i_episode+1, hparams['num_episodes']):
        # sample graph randomly
        graph_idx = graph_indices[i_episode % len(graph_indices)]
        G, baseline = trainset['instances'][graph_idx]
        if hparams.get('debug', False):
            filename = os.listdir(dataset_paths['trainset'])[graph_idx]
            filename = os.path.join(dataset_paths['trainset'], filename)
            print(f'instance no. {graph_idx}, filename: {filename}')

        execute_episode(G, baseline, trainset['lp_iterations_limit'])

        if i_episode % hparams.get('backprop_interval', 10) == 0:
            dqn_agent.optimize_model()

        if i_episode % hparams.get('target_update_interval', 1000) == 0:
            dqn_agent.update_target()

        if i_episode % hparams.get('log_interval', 100) == 0:
            dqn_agent.log_stats()

        # evaluate the model on the validation and test sets
        dqn_agent.eval()
        for dataset_name, dataset in datasets.items():
            if dataset_name == 'trainset':
                continue
            if i_episode % dataset['eval_interval'] == 0:
                print('Evaluating ', dataset_name)
                for G, baseline in dataset['instances']:
                    for scip_seed in dataset['scip_seed']:
                        execute_episode(G, baseline, dataset['lp_iterations_limit'], dataset_name=dataset_name, scip_seed=scip_seed)
                dqn_agent.log_stats(save_best=(dataset_name[-8:] == 'validset'))
        dqn_agent.train()

        if i_episode % hparams.get('checkpoint_interval', 100) == 0:
            dqn_agent.save_checkpoint()

        if i_episode % len(graph_indices) == 0:
            graph_indices = torch.randperm(trainset['num_instances'])

    return 0


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='results',
                        help='path to save results')
    parser.add_argument('--datadir', type=str, default='data/dqn',
                        help='path to generate/read data')
    parser.add_argument('--configfile', type=str, default='experiment_config.yaml',
                        help='general experiment settings')
    parser.add_argument('--resume-training', action='store_true',
                        help='set to load the last training status from checkpoint file')
    parser.add_argument('--mixed-debug', action='store_true',
                        help='set for mixed python/c debugging')
    parser.add_argument('--gpu-id', type=int, default=None,
                        help='gpu id to use if available')

    args = parser.parse_args()
    if args.mixed_debug:
        import ptvsd
        port = 3000
        # ptvsd.enable_attach(secret='my_secret', address =('127.0.0.1', port))
        ptvsd.enable_attach(address=('127.0.0.1', port))
        ptvsd.wait_for_attach()

    with open(args.configfile) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        hparams[k] = v
    if hparams.get('debug_cuda', False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    datasets = hparams['datasets']
    for dataset_name, dataset in datasets.items():
        with open(f'{dataset_name}_config.yaml') as f:
            dataset['config'] = yaml.load(f, Loader=yaml.FullLoader)
    # set logdir according to hparams
    relative_logdir = f"lr_{hparams['lr']}-nstep_{hparams['nstep_learning']}-credit_{hparams['credit_assignment']}-gamma_{hparams['gamma']}-obj_{hparams['dqn_objective']}"
    hparams['logdir'] = os.path.join(hparams['logdir'], relative_logdir)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    experiment(hparams)
