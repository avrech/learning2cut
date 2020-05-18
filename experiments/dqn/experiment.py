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


def experiment(hparams):
    # read datasets
    dataset_paths = {}
    datasets = {}
    for dataset in ['trainset', 'easy_validset', 'medium_validset', 'hard_validset', 'easy_testset', 'medium_testset', 'hard_testset']:
        dataset_paths[dataset] = os.path.join(hparams['datadir'], dataset, "barabasi-albert-n{}-m{}-weights-{}-seed{}".format(hparams[dataset]['graph_size'], hparams[dataset]['barabasi_albert_m'], hparams[dataset]['weights'], hparams[dataset]['dataset_generation_seed']))
        # read all graphs with their baselines from disk
        datasets[dataset] = []
        for filename in os.listdir(dataset_paths[dataset]):
            with open(os.path.join(dataset_paths[dataset], filename), 'rb') as f:
                datasets[dataset].append(pickle.load(f))
    trainset = datasets['trainset']
    validation_sets = {k: datasets[k] for k in ['easy_validset', 'medium_validset', 'hard_validset']}
    test_sets = {k: datasets[k] for k in ['easy_testset', 'medium_testset', 'hard_testset']}
    graph_indices = torch.randperm(len(trainset))

    # dqn agent
    dqn_agent = DQN(hparams=hparams)
    dqn_agent.train()

    if hparams.get('resume_training', False):
        dqn_agent.load_checkpoint()

    def execute_episode(G, baseline, dataset='trainset'):
        # create SCIP model for G
        model, x, y = maxcut_mccormic_model(G, use_cuts=False)  # disable default cuts

        # set the appropriate lp_iterations_limit
        hparams['lp_iterations_limit'] = hparams[dataset]['lp_iterations_limit']

        # include cycle inequalities separator with high priority
        cycle_sepa = MccormickCycleSeparator(G=G, x=x, y=y, name='MLCycles', hparams=hparams)
        model.includeSepa(cycle_sepa, 'MLCycles',
                          "Generate cycle inequalities for the MaxCut McCormic formulation",
                          priority=1000000, freq=1)

        # reset dqn_agent to start a new episode
        dqn_agent.init_episode(G, x, y, cut_generator=cycle_sepa, baseline=baseline, dataset=dataset)

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

        # model.hideOutput()

        # gong! run episode
        model.optimize()

        # once episode is done
        dqn_agent.finish_episode()

    for i_episode in range(dqn_agent.i_episode+1, hparams['num_episodes']):
        # sample graph randomly
        graph_idx = graph_indices[i_episode % len(graph_indices)]
        G, baseline = trainset[graph_idx]

        execute_episode(G, baseline)

        if i_episode % hparams.get('backprop_interval', 10) == 0:
            dqn_agent.optimize_model()

        if i_episode % hparams.get('target_update_interval', 1000) == 0:
            dqn_agent.update_target()

        if i_episode % hparams.get('log_interval', 100) == 0:
            dqn_agent.log_stats()

        if i_episode % hparams.get('test_interval', 1000) == 0:
            # evaluate the model on the validation and test sets
            dqn_agent.eval()
            for dataset, instances in validation_sets.items():
                for G, baseline in instances:
                    execute_episode(G, baseline, dataset)
                dqn_agent.log_stats(save_best=True)
            for dataset, instances in test_sets.items():
                for G, baseline in instances:
                    execute_episode(G, baseline, dataset)
                dqn_agent.log_stats()
            dqn_agent.train()

        if i_episode % hparams.get('checkpoint_interval', 100) == 0:
            dqn_agent.save_checkpoint()

        if i_episode % len(graph_indices) == 0:
            graph_indices = torch.randperm(len(trainset))

    return 0


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='results',
                        help='path to results root')
    parser.add_argument('--datadir', type=str, default='data/dqn',
                        help='path to generate/read data')
    parser.add_argument('--resume_training', action='store_true',
                        help='path to generate/read data')

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    with open('experiment_config.yaml') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        hparams[k] = v
    for dataset in ['trainset', 'easy_validset', 'medium_validset', 'hard_validset', 'easy_testset', 'medium_testset', 'hard_testset']:
        with open(f'{dataset}_config.yaml') as f:
            hparams[dataset] = yaml.load(f, Loader=yaml.FullLoader)
    experiment(hparams)
