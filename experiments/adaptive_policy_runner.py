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
from itertools import product

NOW = str(datetime.now())[:-7].replace(' ', '.').replace(':', '-').replace('.', '/')
parser = ArgumentParser()
parser.add_argument('--experiment', type=str, default='cut_root',
                    help='experiment dir')
parser.add_argument('--config_file', type=str, default='cut_root/adaptive_policy_config.yaml',
                    help='relative path to config file to generate configs for ray.tune.run')
parser.add_argument('--log_dir', type=str, default='cut_root/results/adaptive_policy/' + NOW,
                    help='path to results root')
parser.add_argument('--data_dir', type=str, default='cut_root/data',
                    help='path to generate/read data')
parser.add_argument('--taskid', type=int,
                    help='serial number to choose maxcutsroot and objparalfac')
parser.add_argument('--product_keys', nargs='+', default=[],
                    help='list of hparam keys on which to product')
parser.add_argument('--auto', action='store_true',
                    help='run again automatically after each iteration completed')

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

# fix some hparams ranges according to taskid:
if len(args.product_keys) > 0:
    product_lists = [sweep_config['sweep'][k]['values'] for k in args.product_keys]
    products = list(product(*product_lists))
    task_values = products[args.taskid]
    for idx, k in enumerate(args.product_keys):
        tune_search_space[k] = tune.grid_search([task_values[idx]])

# add the sweep_config and data_abspath as constant parameters for global experiment management
tune_search_space['sweep_config'] = tune.grid_search([sweep_config])
tune_search_space['data_abspath'] = tune.grid_search([data_abspath])

# initialize global tracker for all experiments
track.init(experiment_dir=args.log_dir)

# run experiment:
# initialize starting policies:
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
starting_policies_abspath = os.path.abspath(os.path.join(args.log_dir, 'starting_policies.pkl'))
tune_search_space['starting_policies_abspath'] = tune.grid_search([starting_policies_abspath])
if not os.path.exists(starting_policies_abspath):
    with open(starting_policies_abspath, 'wb') as f:
        pickle.dump([], f)

# run n policy iterations,
# in each iteration k, load k-1 starting policies from args.experiment,
# run exhaustive search for the best k'th policy - N LP rounds search,
# and for the rest use default cut selection.
# Then when all experiments ended, find the best policy for the i'th iteration and append to starting policies.
iter_logdir = ''
for k_iter in range(sweep_config['constants'].get('n_policy_iterations',1)):
    # recovering from checkpoints:
    # skip iteration if completed in previous runs
    print('loading starting policies from: ', starting_policies_abspath)
    with open(starting_policies_abspath, 'rb') as f:
        starting_policies = pickle.load(f)
    if len(starting_policies) > k_iter:
        print('iteration completed - continue')
        continue
    print('################ RUNNING ITERATION {} ################'.format(k_iter))

    # run exhaustive search
    iter_logdir = os.path.join(args.log_dir, 'iter{}results'.format(k_iter))
    if not os.path.exists(iter_logdir):
        os.makedirs(iter_logdir)

    tune.run(experiment,
             config=tune_search_space,
             resources_per_trial={'cpu': 1, 'gpu': 0},
             local_dir=iter_logdir,
             trial_name_creator=None,
             max_failures=1,  # TODO learn how to recover from checkpoints
             verbose=0)

    if args.auto:
        # write DONE to a file named "task-id-finished" in iter_logdir
        with open(os.path.join(iter_logdir, 'task-{}-finished'.format(args.taskid)), 'w') as f:
            f.writelines('DONE')
        # override job output file and clean for the next iteration restarting
        with open(os.path.join(args.log_dir, 'iter{}-cfg{}.out'.format(k_iter, args.taskid)), 'w') as f:
            f.writelines('RESTARTING ITERATION {}'.format(k_iter+1))

        # check if all tasks finished and launch run_server.py again
        # the number of tasks is always len(products) (defined in run_server.py)
        all_finished = True
        for taskid in range(len(products)):
            if not os.path.exists(os.path.join(iter_logdir, 'task-{}-finished'.format(taskid))):
                all_finished = False
                break
        if all_finished:
            with open(os.path.join(args.log_dir, 'cmd.txt'), 'r') as f:
                cmd = f.readline()
            print('EXECUTING THE NEXT POLICY ITERATION')
            print(cmd)
            os.system(cmd)

