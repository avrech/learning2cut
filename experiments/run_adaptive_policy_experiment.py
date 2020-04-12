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
import os, pickle
from experiments.cut_root.experiment import experiment
from experiments.cut_root.analyze_results import analyze_results

NOW = str(datetime.now())[:-7].replace(' ', '.').replace(':', '-').replace('.', '/')
parser = ArgumentParser()
parser.add_argument('--experiment', type=str, default='cut_root',
                    help='experiment dir')
parser.add_argument('--config-file', type=str, default='cut_root/adaptive_policy_config.yaml',
                    help='relative path to config file to generate configs for ray.tune.run')
parser.add_argument('--log-dir', type=str, default='cut_root/results/adaptive_policy/' + NOW,
                    help='path to results root')
parser.add_argument('--data-dir', type=str, default='cut_root/data',
                    help='path to generate/read data')

args = parser.parse_args()

# load sweep configuration
with open(args.config_file) as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)

# dataset generation
data_generator = import_module('experiments.' + args.experiment + '.data_generator')
data_abspath = data_generator.generate_data(sweep_config, args.data_dir, solve_maxcut=True, time_limit=600)

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
track.init()

# run experiment:
# initialize starting policies:
os.makedirs(args.log_dir)
starting_policies_abspath = os.path.abspath(os.path.join(args.log_dir, 'starting_policies.pkl'))
tune_search_space['starting_policies_abspath'] = tune.grid_search([starting_policies_abspath])

with open(starting_policies_abspath, 'wb') as f:
    pickle.dump([], f)

# run n policy iterations,
# in each iteration k, load k-1 starting policies from args.experiment,
# run exhaustive search for the best k'th policy - N LP rounds search,
# and for the rest use default cut selection.
# Then when all experiments ended, find the best policy for the i'th iteration and append to starting policies.
iter_logdir = ''
for k_iter in range(sweep_config['constants']['n_policy_iterations']):
    # run exhaustive search
    iter_logdir = os.path.join(args.log_dir, 'iter{}results'.format(k_iter))
    iter_analysisdir = os.path.join(args.log_dir, 'iter{}analysis'.format(k_iter))
    tune.run(experiment,
             config=tune_search_space,
             resources_per_trial={'cpu': 1, 'gpu': 0},
             local_dir=iter_logdir,
             trial_name_creator=None,
             max_failures=1  # TODO learn how to recover from checkpoints
             )
    # analyze the results and find the best one. complete missing experiments if any.
    # assuming we test only 1 graph:
    analysis = list(analyze_results(rootdir=iter_logdir, dstdir=iter_analysisdir).values())[0]
    while analysis.get('complete_experiment_commandline', None) is not None:
        print('completing experiments...')
        os.system(analysis['complete_experiment_commandline'])
        analysis = list(analyze_results(rootdir=iter_logdir, dstdir=iter_analysisdir).values())[0]
    best_policy = analysis['best_policy'][0]

    # append best policy to starting policies
    with open(starting_policies_abspath, 'rb') as f:
        starting_policies = pickle.load(f)
    starting_policies.append(best_policy)
    with open(starting_policies_abspath, 'wb') as f:
        pickle.dump(starting_policies, f)

# finally create a tensorboard from the best adaptive policy only:
analyze_results(rootdir=iter_logdir, dstdir=os.path.join(args.log_dir, 'final_analysis'), tensorboard=True)


