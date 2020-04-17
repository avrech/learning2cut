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
import os, pickle, time
from experiments.cut_root.analyze_results import analyze_results
from tqdm import tqdm
from pathlib import Path

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
parser.add_argument('--cpus-per-task', type=int, default=32,
                    help='Graham - 32, Niagara - 40')

args = parser.parse_args()

def submit_job(config_file, jobname):
    # SLURM COMMAND
    commandline = 'sbatch ' + \
        ' --time=00::00' + \
        ' --account=def-alodi' + \
        ' --output=graham-output/%j.out' + \
        ' --mem-per-cpu=8G' + \
        ' --cpus-per-task=1' + \
        ' --mem=0' + \
        ' --mail-user=avrech@campus.tecnion.ac.il' + \
        ' --mail-type=BEGIN' + \
        ' --mail-type=END' + \
        ' --mail-type=FAIL' + \
        ' --nodes=1' + \
        ' --job-name={}'.format(jobname) + \
        ' --ntasks-per-node=1' + \
        ' --cpus-per-task='.format(args.cpus_per_task) + \
        ' python adaptive_policy_runner.py --experiment {} --log_dir {} --config-file {} --data-dir {}'.format(
            os.path.abspath(args.experiment),
            os.path.abspath(args.log_dir),
            config_file,
            os.path.abspath(args.data_dir)
        )
    os.system(commandline)

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
for k_iter in range(sweep_config['constants']['n_policy_iterations']):
    # recovering from checkpoints:
    # skip iteration if completed in previous runs
    print('loading stating policies from: ', starting_policies_abspath)
    with open(starting_policies_abspath, 'rb') as f:
        starting_policies = pickle.load(f)
    if len(starting_policies) > k_iter:
        print('iteration completed and analyzed - continue')
        continue

    # check if iteration completed but not analyzed
    iter_analysisdir = os.path.join(args.log_dir, 'iter{}analysis'.format(k_iter))
    iter_logdir = os.path.join(args.log_dir, 'iter{}results'.format(k_iter))
    if not os.path.exists(iter_logdir):
        os.makedirs(iter_logdir)
    print('################ CHECKING ITERATION {} ################'.format(k_iter))
    analyses = analyze_results(rootdir=iter_logdir, dstdir=iter_analysisdir)
    if len(analyses) > 0:
        analysis = list(analyses.values())[0]
        if analysis.get('complete_experiment_commandline', None) is not None:
            print('iteration has not been completed - now completing')
        else:
            best_policy = analysis['best_policy'][0]
            # append best policy to starting policies
            with open(starting_policies_abspath, 'rb') as f:
                starting_policies = pickle.load(f)
            starting_policies.append(best_policy)
            with open(starting_policies_abspath, 'wb') as f:
                pickle.dump(starting_policies, f)
            print('iteration analyzed - continue')
            continue
    else:
        print('iteration has not been completed - now completing')

    print('################ RUNNING ITERATION {} ################'.format(k_iter))
    # run exhaustive search
    # create a list of completed trials for from previos checkpoints for recovering from failures.
    print('loading checkpoints from ', iter_logdir)
    checkpoint = []
    for path in tqdm(Path(iter_logdir).rglob('experiment_results.pkl'), desc='Loading files'):
        with open(path, 'rb') as f:
            res = pickle.load(f)
            checkpoint.append(res)
    with open(os.path.join(iter_logdir, 'checkpoint.pkl'), 'wb') as f:
        pickle.dump(checkpoint, f)

    # break the search space into 5 smaller jobs, according to objparalfac
    # hardcoded config files are predefined.
    # submit 5 experiments each one execute one config file.
    # after all jobs complete, continue to the next iteration.
    for cfgidx in range(5):
        config_file = os.path.abspath('cut_root/adaptive_policy_config{}.yaml'.format(cfgidx))
        jobname = 'runner{}'.format(cfgidx)
        submit_job(config_file, jobname)
        time.sleep(1)

    print('submitted jobs')
    print('run again when all jobs completed')

# finally create a tensorboard from the best adaptive policy only:
analyze_results(rootdir=iter_logdir, dstdir=os.path.join(args.log_dir, 'final_analysis'), tensorboard=True)
print('finished adaptive policy search. congrats!')

