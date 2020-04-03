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
parser.add_argument('--config-file', type=str, default='config_fixed_max_cuts.yaml',
                    help='config file to generate configs for ray.tune.run')
parser.add_argument('--log-dir', type=str, default='cut_root/results/fixed_maxcutsroot/' + NOW,
                    help='path to results root')
parser.add_argument('--data-dir', type=str, default='cut_root/data',
                    help='path to generate/read data')
parser.add_argument('--tensorboard', action='store_true',
                    help='log to tensorboard')
args = parser.parse_args()

# load sweep configuration
config_file = os.path.basename(args.config_file)
with open(os.path.join(args.experiment, args.config_file)) as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)

# dataset generation
data_generator = import_module('experiments.' + args.experiment + '.data_generator')
data_abspath = data_generator.generate_data(sweep_config, args.data_dir, solve_maxcut=False, time_limit=600)

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
analysis = tune.run(experiment.experiment,
                    config=tune_search_space,
                    resources_per_trial={'cpu': 1, 'gpu': 0},
                    local_dir=args.log_dir,
                    trial_name_creator=None,
                    max_failures=1  # TODO learn how to recover from checkpoints
                    )
