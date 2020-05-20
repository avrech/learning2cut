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
        dualbound_integral_list = []
        gap_integral_list = []
        for (_, baseline) in dataset['instances']:
            optimal_value = baseline['optimal_value']
            dualbound = baseline['rootonly_stats']['dualbound']
            gap = baseline['rootonly_stats']['gap']
            lpiter = baseline['rootonly_stats']['lp_iterations']
            dualbound_integral = sum(get_normalized_areas(t=lpiter, ft=dualbound, t_support=dataset['lp_iterations_limit'], reference=optimal_value))
            gap_integral = sum(get_normalized_areas(t=lpiter, ft=gap, t_support=dataset['lp_iterations_limit'], reference=0))
            baseline['dualbound_integral'] = dualbound_integral
            baseline['gap_integral'] = gap_integral
            dualbound_integral_list.append(dualbound_integral)
            gap_integral_list.append(gap_integral)
        # compute stats for the whole dataset
        dualbound_integral_avg = np.mean(dualbound_integral)
        dualbound_integral_std = np.std(dualbound_integral)
        gap_integral_avg = np.mean(gap_integral)
        gap_integral_std = np.std(gap_integral)
        dataset['stats'] = {}
        dataset['stats']['dualbound_integral_avg'] = dualbound_integral_avg
        dataset['stats']['dualbound_integral_std'] = dualbound_integral_std
        dataset['stats']['gap_integral_avg'] = gap_integral_avg
        dataset['stats']['gap_integral_std'] = gap_integral_std

    # training
    trainset = datasets['trainset']
    graph_indices = torch.randperm(trainset['num_instances'])

    # dqn agent
    dqn_hparams = hparams['dqn_hparams']
    dqn_hparams['logdir'] = hparams['logdir']
    dqn_agent = DQN(hparams=dqn_hparams)
    dqn_agent.train()

    if hparams.get('resume_training', False):
        dqn_agent.load_checkpoint()

    def execute_episode(G, baseline, lp_iterations_limit, dataset_name='trainset'):
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

        # set up randomization
        if hparams.get('scip_seed', None) is not None:
            model.setBoolParam('randomization/permutevars', True)
            model.setIntParam('randomization/permutationseed', hparams.get('scip_seed'))
            model.setIntParam('randomization/randomseedshift', hparams.get('scip_seed'))

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
                    execute_episode(G, baseline, dataset['lp_iterations_limit'], dataset_name=dataset_name)
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
    args = parser.parse_args()
    if args.mixed_debug:
        import ptvsd
        port = 3000
        # ptvsd.enable_attach(secret='my_secret', address =('127.0.0.1', port))
        ptvsd.enable_attach(address=('127.0.0.1', port))
        ptvsd.wait_for_attach()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    with open(args.configfile) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        hparams[k] = v
    datasets = hparams['datasets']
    for dataset_name, dataset in datasets.items():
        with open(f'{dataset_name}_config.yaml') as f:
            dataset['config'] = yaml.load(f, Loader=yaml.FullLoader)
    experiment(hparams)
