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
from importlib import import_module
# from tqdm import tqdm
# import networkx as nx
from ray import tune
from ray.tune import track
from argparse import ArgumentParser
import numpy as np
import yaml
from datetime import datetime
# import pickle
import os
# from experiments.variability.experiment import experiment
NOW = str(datetime.now())[:-7].replace(' ', '.').replace(':', '-').replace('.', '/')
parser = ArgumentParser()
parser.add_argument('--experiment', type=str, default='variability',
                    help='experiment dir')
parser.add_argument('--log-dir', type=str, default='variability/results/tmp/' + NOW,
                    help='path to results root')
parser.add_argument('--data-dir', type=str, default='variability/data',
                    help='path to generate/read data')
parser.add_argument('--tensorboard', action='store_true',
                    help='log to tensorboard')
args = parser.parse_args()

# load sweep configuration
with open(os.path.join(args.experiment, 'config.yaml')) as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)

# generate tune config for the sweep hparams
tune_config = dict()
for hp, config in sweep_config['sweep'].items():
    tune_config[hp] = {'grid': tune.grid_search(config.get('values')),
                       'grid_range': tune.grid_search(list(range(config.get('range', 2)))),
                       'choice': tune.choice(config.get('values')),
                       'randint': tune.randint(config.get('min'), config.get('max')),
                       'uniform': tune.sample_from(lambda spec: np.random.uniform(config.get('min'), config.get('max')))
                       }.get(config['search'])
# add the sweep_config as parameter for global management
tune_config['sweep_config'] = tune.grid_search([sweep_config])
# # dataset generation
# n = sweep_config['constants']["graph_size"]
# m = sweep_config['constants']["barabasi_albert_m"]
# weights = sweep_config['constants']["weights"]
# dataset_generation_seed = sweep_config['constants']["dataset_generation_seed"]
#
# data_abspath = os.path.join(sweep_args.data_dir, "barabasi-albert-n{}-m{}-weights-{}-seed{}".format(n, m, weights, dataset_generation_seed))
# if not os.path.isdir(data_abspath):
#     os.makedirs(data_abspath)
# data_abspath = os.path.abspath(data_abspath)
#
# for graph_idx in tqdm(range(sweep_config['sweep']['graph_idx']['range'])):
#     filepath = os.path.join(data_abspath, "graph_idx_{}.pkl".format(graph_idx))
#     if not os.path.exists(filepath):
#         # generate the graph and save in a pickle file
#         # set randomization for reproducing dataset
#         barabasi_albert_seed = (1 + 223 * graph_idx) * dataset_generation_seed
#         np.random.seed(barabasi_albert_seed)
#         G = nx.barabasi_albert_graph(n, m, seed=barabasi_albert_seed)
#         if weights == 'ones':
#             w = 1
#         elif weights == 'uniform01':
#             w = {e: np.random.uniform() for e in G.edges}
#         elif weights == 'normal':
#             w = {e: np.random.normal() for e in G.edges}
#         nx.set_edge_attributes(G, w, name='weight')
#         with open(filepath, 'wb') as f:
#             pickle.dump(G, f)
data_generator = import_module('experiments.' + args.experiment + '.data_generator')
data_abspath = data_generator.generate_data(sweep_config, args.data_dir)
tune_config['data_abspath'] = tune.grid_search([data_abspath])


# initialize global tracker for all experiments
experiment = import_module('experiments.' + args.experiment + '.experiment')
track.init()

# run experiment
analysis = tune.run(experiment.experiment,
                    config=tune_config,
                    resources_per_trial={'cpu': 1, 'gpu': 0},
                    local_dir=args.log_dir,
                    trial_name_creator=None,
                    max_failures=1  # TODO learn how to recover from checkpoints
                    )
