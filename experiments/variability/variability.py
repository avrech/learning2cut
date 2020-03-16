""" Variability
Graph type: Barabasi-Albert
MaxCut formulation: McCormic
Baseline: SCIP defaults

Each graph is solved using different scip_seed,
and SCIP statistics are collected.

All results are written to experiment_results.pkl file
and should be post-processed using experiments/analyze_experiment_results.py

utils/analyze_experiment_results.py can generate tensorboard hparams,
and a csv file summarizing the statistics in a table (useful for latex).

"""
from tqdm import tqdm
from pathlib import Path
import networkx as nx
from ray import tune
from ray.tune import track
from argparse import ArgumentParser
import numpy as np
import yaml
from datetime import datetime
from utils.scip_models import maxcut_mccormic_model, get_separator_cuts_applied
from separators.mccormic_cycle_separator import MccormicCycleSeparator
import pickle
import os
from torch.utils.tensorboard import SummaryWriter

NOW = str(datetime.now())[:-7].replace(' ', '.').replace(':', '-').replace('.', '/')
sweep_parser = ArgumentParser()
sweep_parser.add_argument('--config-file', type=str, default='variability-config.yaml',
                          help='path to hyper-parameters config yaml file')
sweep_parser.add_argument('--log-dir', type=str, default='./results/tmp/' + NOW,
                          help='path to results root')
sweep_parser.add_argument('--data-dir', type=str, default='./data',
                          help='path to generate/read data')
sweep_parser.add_argument('--tensorboard', action='store_true',
                          help='log to tensorboard')
sweep_args = sweep_parser.parse_args()

# load sweep configuration
with open(sweep_args.config_file) as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)

# generate tune config for the sweep hparams
tune_config = dict()
for hp, config in sweep_config['sweep'].items():
    tune_config[hp] = {'grid': tune.grid_search(config.get('values')),
                       'grid_range': tune.grid_search(np.arange(config.get('range', 1))),
                       'choice': tune.choice(config.get('values')),
                       'randint': tune.randint(config.get('min'), config.get('max')),
                       'uniform': tune.sample_from(lambda spec: np.random.uniform(config.get('min'), config.get('max')))
                       }.get(config['search'])

n = sweep_config['constants']["graph_size"]
m = sweep_config['constants']["barabasi_albert_m"]
weights = sweep_config['constants']["weights"]
dataset_generation_seed = sweep_config['constants']["dataset_generation_seed"]

DATA_ROOT_DIR = os.path.join(sweep_args.data_dir, "barabasi-albert-n{}-m{}-weights-{}-seed{}".format(n, m, weights, dataset_generation_seed))
if not os.path.isdir(DATA_ROOT_DIR):
    os.makedirs(DATA_ROOT_DIR)

for graph_idx in tqdm(range(sweep_config['sweep']['graph_idx']['range'])):
    filepath = os.path.join(DATA_ROOT_DIR, "graph_idx_{}.pkl".format(graph_idx))
    if not os.path.exists(filepath):
        # generate the graph and save in a pickle file
        # set randomization for reproducing dataset
        barabasi_albert_seed = (1 + 223 * graph_idx) * dataset_generation_seed
        np.random.seed(barabasi_albert_seed)
        G = nx.barabasi_albert_graph(n, m, seed=barabasi_albert_seed)
        if weights == 'ones':
            w = 1
        elif weights == 'uniform01':
            w = {e: np.random.uniform() for e in G.edges}
        elif weights == 'normal':
            w = {e: np.random.normal() for e in G.edges}
        nx.set_edge_attributes(G, w, name='weight')
        with open(filepath, 'wb') as f:
            pickle.dump(G, f)


def experiment(config):
    # set the current sweep trial parameters
    for k, v in sweep_config['constants'].items():
        config[k] = v

    if not config['use_cycle_cuts']:
        # run this configuration only once with all hparams get their first choice
        for k, v in sweep_config['sweep'].items():
            if k != 'use_cycle_cuts' and k != 'graph_id' and config[k] != v['values'][0]:
                print('!!!!!!!!!!!!!!!!!!!!! SKIPPING DUPLICATED not use_cycle_cuts !!!!!!!!!!!!!!!!!!!!!!1')
                return

    # generate graph
    graph_idx = config['graph_idx']
    filepath = os.path.join(DATA_ROOT_DIR, "graph_idx_{}.pkl".format(graph_idx))
    with open(filepath, 'rb') as f:
        G = pickle.load(f)

    scip_seed = config['scip_seed']
    model, x, y = maxcut_mccormic_model(G)
    sepa = MccormicCycleSeparator(G=G, x=x, y=y, hparams=config)
    model.includeSepa(sepa, 'McCormicCycles',
                      "Generate cycle inequalities for the MaxCut McCormic formulation",
                      priority=1000000, freq=1)

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', scip_seed)
    model.setIntParam('randomization/randomseedshift', scip_seed)
    # set time limit
    model.setRealParam('limits/time', config['time_limit_sec'])
    model.optimize()

    cycles_sepa_time = sepa.time_spent / model.getSolvingTime() if config['use_cycle_cuts'] else 0
    if config['use_cycle_cuts']:
        cycle_cuts, cycle_cuts_applied = get_separator_cuts_applied(model, 'McCormicCycles')
    else:
        cycle_cuts, cycle_cuts_applied = 0, 0

    # Statistics
    stats = {}
    stats['cycle_cuts'] = cycle_cuts
    stats['cycle_cuts_applied'] = cycle_cuts_applied
    stats['total_cuts_applied'] = model.getNCutsApplied()
    stats['cycles_sepa_time'] = cycles_sepa_time
    stats['solving_time'] = model.getSolvingTime()
    stats['processed_nodes'] = model.getNNodes()
    stats['gap'] = model.getGap()
    stats['LP_rounds'] = model.getNLPs()

    # set log-dir for tensorboard logging of the specific trial
    log_dir = tune.track.trial_dir()

    # save stats to pkl
    experiment_results_filepath = os.path.join(log_dir, 'experiment_results.pkl')
    experiment_results = {}
    experiment_results['stats'] = stats
    experiment_results['config'] = config
    experiment_results['filepath'] = filepath
    experiment_results['sweep_config'] = sweep_config
    experiment_results['description'] = 'Variability'
    with open(experiment_results_filepath, 'wb') as f:
        pickle.dump(experiment_results, f)
        print('Saved experiment results to: ' + experiment_results_filepath)

    if sweep_args.tensorboard:
        writer = SummaryWriter(log_dir)
        writer.add_hparams(hparam_dict=config, metric_dict=stats)
        writer.close()


# initialize global tracker for all experiments
track.init()

# run sweep
analysis = tune.run(experiment,
                    config=tune_config,
                    resources_per_trial={'cpu': 1, 'gpu': 0},
                    local_dir=sweep_args.log_dir,
                    trial_name_creator=None,
                    max_failures=1  # TODO learn how to recover from checkpoints
                    )


# for path in Path(sweep_args.log_dir).rglob('experiment_results.pkl'):
#     print(path.name)
#
# summary = []
# for path in Path(sweep_args.log_dir).rglob('experiment_results.pkl'):
#     abspath = path.absolute()
#     with open(abspath, 'rb') as f:
#         res = pickle.load(f)
#         summary.append(res)
#
#
# with open(os.path.join(sweep_args.log_dir, 'summary.pkl'), 'wb') as f:
#     pickle.dump(summary, f)
# print('Saved experiment summary to: ', os.path.join(sweep_args.log_dir, 'summary.pkl'))
