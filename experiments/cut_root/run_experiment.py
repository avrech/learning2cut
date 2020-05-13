""" run_experiment
Launch multiple experiment configurations in parallel on distributed resources.
Requires a folder in ./ containing experiment.py, data_generator,py and config_fixed_max_rounds.yaml
See example in ./variability
"""
from importlib import import_module
from ray import tune
from ray.tune import track
from argparse import ArgumentParser
import numpy as np
import yaml
from datetime import datetime
import os
NOW = str(datetime.now())[:-7].replace(' ', '.').replace(':', '-').replace('.', '/')
parser = ArgumentParser()
parser.add_argument('--experiment', type=str, default='cut_root',
                    help='experiment dir')
parser.add_argument('--configfile', type=str, default='cut_root/experts_config.yaml',
                    help='relative path to config file to generate configs for ray.tune.run')
parser.add_argument('--logdir', type=str, default='cut_root/results/maxcutsapplied2000/' + NOW,
                    help='path to results root')
parser.add_argument('--datadir', type=str, default='cut_root/data',
                    help='path to generate/read data')
parser.add_argument('--solvegraphs', action='store_true',
                    help='whether to solve the graphs to optimality when generating data or not.')
parser.add_argument('--mp', type=str, default='ray',
                    help='multiprocessing package: ray, mp')

args = parser.parse_args()

# load sweep configuration
with open(args.configfile) as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)

# dataset generation
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)
data_generator = import_module('experiments.' + args.experiment + '.data_generator')
data_abspath = data_generator.generate_data(sweep_config, args.datadir, solve_maxcut=args.solvegraphs, time_limit=600)

# generate tune config for the sweep hparams
tune_search_space = dict()
for hp, config in sweep_config['sweep'].items():
    tune_search_space[hp] = {'grid': tune.grid_search(config.get('values')),
                       'grid_range': tune.grid_search(list(range(config.get('range', 2)))),
                       'choice': tune.choice(config.get('values')),
                       'randint': tune.randint(config.get('min'), config.get('max')),
                       'uniform': tune.sample_from(lambda spec: np.random.uniform(config.get('min'), config.get('max')))
                             }.get(config['search'])

# add the sweep_config and data_abspath as constant parameters for global experiment management
tune_search_space['sweep_config'] = tune.grid_search([sweep_config])
tune_search_space['data_abspath'] = tune.grid_search([data_abspath])

# initialize global tracker for all experiments
experiment = import_module('experiments.' + args.experiment + '.experiment')
track.init()

# run experiment
if args.mp == 'ray':
    analysis = tune.run(experiment.experiment,
                        config=tune_search_space,
                        resources_per_trial={'cpu': 1, 'gpu': 0},
                        local_dir=args.logdir,
                        trial_name_creator=None,
                        max_failures=1  # TODO learn how to recover from checkpoints
                        )

elif args.mp == 'mp':
    from itertools import product
    from multiprocessing import Pool
    from experiments.imitation.experiment import experiment as func
    # fix some hparams ranges according to taskid:
    search_space_size = np.prod([d['range'] if k =='graph_idx' else len(d['values']) for k, d in sweep_config['sweep'].items()])
    sweep_keys = list(sweep_config['sweep'].keys())
    sweep_vals = [np.arange(sweep_config['sweep'][k]['range']).tolist()
                     if k =='graph_idx' else
                     sweep_config['sweep'][k]['values']
                     for k in sweep_keys]

    products = list(product(*sweep_vals))
    n_tasks = len(products)
    configs = []
    for combination in products:
        cfg = {k: v for k, v in zip(sweep_keys, combination)}
        cfg['sweep_config'] = sweep_config
        cfg['data_abspath'] = data_abspath
        configs.append(cfg)

    # time_limit_minutes = max(int(np.ceil(1.5*search_space_size/n_tasks/(args.cpus_per_task-1)) + 2), 16)

    with Pool() as p:
        res = p.map_async(func, configs)
        res.wait()
        print(f'multiprocessing finished {"successfully" if res.successful() else "with errors"}')

print('finished')
