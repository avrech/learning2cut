""" run_experiment
Launch multiple experiment configurations in parallel on distributed resources.
Requires a folder in ./ containing experiment.py, data_generator,py and config_fixed_max_rounds.yaml
See example in ./variability
"""
from tqdm import tqdm
from ray import tune
from ray.tune import track
from argparse import ArgumentParser
import numpy as np
import yaml
from datetime import datetime
import os, pickle
from experiments.cutrootnode.experiment import experiment
from experiments.cutrootnode.data_generator import generate_data
from itertools import product
import time
from experiments.cutrootnode.analyze_results import analyze_results
from pathlib import Path


NOW = str(datetime.now())[:-7].replace(' ', '.').replace(':', '-').replace('.', '/')
parser = ArgumentParser()
parser.add_argument('--config_file', type=str, default='adaptive_policy_config.yaml',
                    help='relative path to config file to generate configs for ray.tune.run')
parser.add_argument('--log_dir', type=str, default='results/adaptive_policy/' + NOW,
                    help='path to results root')
parser.add_argument('--data_dir', type=str, default='data',
                    help='path to generate/read data')
parser.add_argument('--taskid', type=int,
                    help='serial number to choose maxcutsroot and objparalfac')
parser.add_argument('--product_keys', nargs='+', default=[],
                    help='list of hparam keys on which to product')
parser.add_argument('--controller', action='store_true',
                    help='control the adaptive policy runners, analyze all results and update starting_policies.pkl')
parser.add_argument('--auto', action='store_true',
                    help='automatically analyze experiments and run the next iteration')

args = parser.parse_args()

# load sweep configuration
with open(args.config_file) as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)

# dataset generation
data_abspath = generate_data(sweep_config, args.data_dir, solve_maxcut=True, time_limit=600)

# generate tune config for the sweep hparams
tune_search_space = dict()
search_space_dim = []
for hp, config in sweep_config['sweep'].items():
    tune_search_space[hp] = {'grid': tune.grid_search(config.get('values')),
                       'grid_range': tune.grid_search(list(range(config.get('range', 2)))),
                       'choice': tune.choice(config.get('values')),
                       'randint': tune.randint(config.get('min'), config.get('max')),
                       'uniform': tune.sample_from(lambda spec: np.random.uniform(config.get('min'), config.get('max')))
                             }.get(config['search'])
    if config['search'] == 'grid':
        search_space_dim.append(len(config['values']))
    elif config['search'] == 'grid_range':
        search_space_dim.append(config['range'])
    else:
        raise NotImplementedError

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

search_space_size = np.prod(search_space_dim)

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
# in each iteration k, load k-1 starting policies,
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
    try:
        tune.run(experiment,
                 config=tune_search_space,
                 resources_per_trial={'cpu': 1, 'gpu': 0},
                 local_dir=iter_logdir,
                 trial_name_creator=None,
                 max_failures=1,  # TODO learn how to recover from checkpoints
                 verbose=0)
    except Exception as e:
        print(e)

    # the following doesn't really work on the cluster. permission denied.
    if args.auto and args.controller:  # todo
        # check if iteration completed but not analyzed
        iter_analysisdir = os.path.join(args.log_dir, 'iter{}analysis'.format(k_iter))
        print('################ CHECKING ITERATION {} ################'.format(k_iter))
        # check if all workers finished successfully all their jobs
        iteration_completed = False
        while not iteration_completed:
            n_finished = 0
            for path in tqdm(Path(iter_logdir).rglob(args.filepattern), desc='Checking finished trials'):
                n_finished += 1
            if n_finished == search_space_size:
                iteration_completed = True
            else:
                time.sleep(10)
            # todo iteration timeout

        # all trials have been completed. analyze and start the next iteration

        analyses = analyze_results(rootdir=iter_logdir, dstdir=iter_analysisdir,
                                   starting_policies_abspath=starting_policies_abspath)
        if len(analyses) > 0:
            analysis = list(analyses.values())[0]
            best_policy = analysis['best_policy'][0]
            # append best policy to starting policies
            with open(starting_policies_abspath, 'rb') as f:
                starting_policies = pickle.load(f)
            starting_policies.append(best_policy)
            with open(starting_policies_abspath, 'wb') as f:
                pickle.dump(starting_policies, f)
            print('iteration analyzed')

            print('################ PREPARING ITERATION {} ################'.format(k_iter+1))
            # create a list of completed trials for from previos checkpoints for recovering from failures.
            print('loading checkpoints from ', iter_logdir)
            checkpoint = []
            iter_logdir = os.path.join(args.log_dir, 'iter{}results'.format(k_iter+1))
            if not os.path.exists(iter_logdir):
                os.makedirs(iter_logdir)
            with open(os.path.join(iter_logdir, 'checkpoint.pkl'), 'wb') as f:
                pickle.dump(checkpoint, f)

            # continue to the next iteration
            continue
        else:
            raise Exception('Analysis failed')

    elif args.auto and not args.controller:
        # wait for starting_policies update
        while len(starting_policies) == k_iter:
            time.sleep(10)
            with open(starting_policies_abspath, 'rb') as f:
                starting_policies = pickle.load(f)
        # todo timeout
        # go to the next iteration
        continue

    # else, exit when iteration finished
    print('Iteration completed successfully. Terminating.')
    exit(0)

