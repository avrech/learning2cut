""" run_experiment
Launch multiple experiment configurations in parallel on distributed resources.
Requires a folder in ./ containing experiment.py, data_generator,py and config.yaml
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

# dataset generation
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
