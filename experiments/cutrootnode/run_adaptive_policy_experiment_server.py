""" run_experiment
Launch multiple experiment configurations in parallel on distributed resources.
Requires a folder in ./ containing experiment.py, data_generator,py and config_fixed_max_rounds.yaml
See example in ./variability
"""
from importlib import import_module
from argparse import ArgumentParser
import numpy as np
import yaml
from datetime import datetime
import os, pickle, time, sys
from experiments.cutrootnode.analyze_results import analyze_results
from tqdm import tqdm
from pathlib import Path
from itertools import product
import argunparse
from experiments.cutrootnode.data_generator import generate_data

NOW = str(datetime.now())[:-7].replace(' ', '.').replace(':', '-').replace('.', '/')
parser = ArgumentParser()
parser.add_argument('--config_file', type=str, default='adaptive_policy_config.yaml',
                    help='relative path to config file to generate configs for ray.tune.run')
parser.add_argument('--log_dir', type=str, default='results/adaptive_policy',
                    help='path to results root')
parser.add_argument('--data_dir', type=str, default='data',
                    help='path to generate/read data')
parser.add_argument('--cpus_per_task', type=int, default=40,
                    help='Graham - 32, Niagara - 40')
parser.add_argument('--product_keys', nargs='+', default=['intsupportfac', 'maxcutsroot'],
                    help='list of hparam keys on which to product')
# TODO: that doesn't work. use jobs dependency options in slurm.
parser.add_argument('--auto', action='store_true',
                    help='run again automatically after each iteration completed')

args = parser.parse_args()
unparser = argunparse.ArgumentUnparser()
kwargs = vars(args)
prefix = f'python {sys.argv[0]} '
arg_string = unparser.unparse(**kwargs)
cmd_string = prefix + arg_string

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
with open(os.path.join(args.log_dir, 'cmd.txt'), 'w') as f:
    f.writelines(cmd_string + "\n")


def submit_job(jobname, taskid, time_limit_minutes):
    # CREATE SBATCH FILE
    job_file = os.path.join(args.log_dir, jobname + '.sh')
    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines('#SBATCH --time=00:{}:00\n'.format(time_limit_minutes))
        fh.writelines('#SBATCH --account=def-alodi\n')
        fh.writelines('#SBATCH --output={}/{}.out\n'.format(args.log_dir,jobname))
        fh.writelines('#SBATCH --mem=0\n')
        fh.writelines('#SBATCH --mail-user=avrech@campus.technion.ac.il\n')
        fh.writelines('#SBATCH --mail-type=END\n')
        fh.writelines('#SBATCH --mail-type=FAIL\n')
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines('#SBATCH --job-name={}\n'.format(jobname))
        fh.writelines('#SBATCH --ntasks-per-node=1\n')
        fh.writelines('#SBATCH --cpus-per-task={}\n'.format(args.cpus_per_task))
        fh.writelines('module load python\n')
        fh.writelines('source $HOME/server_bashrc\n')
        fh.writelines('source $HOME/venv/bin/activate\n')
        fh.writelines('python adaptive_policy_runner.py --log_dir {} --config_file {} --data_dir {} --taskid {} {} --product_keys {}\n'.format(
            args.log_dir,
            args.config_file,
            args.data_dir,
            taskid,
            '--auto' if args.auto else '',
            ' '.join(args.product_keys)
        ))

    os.system("sbatch {}".format(job_file))


# load sweep configuration
with open(args.config_file) as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)

# dataset generation
data_abspath = generate_data(sweep_config, args.data_dir, solve_maxcut=True, time_limit=600)

# # generate tune config for the sweep hparams
# tune_search_space = dict()
# for hp, config in sweep_config['sweep'].items():
#     tune_search_space[hp] = {'grid': tune.grid_search(config.get('values')),
#                        'grid_range': tune.grid_search(list(range(config.get('range', 2)))),
#                        'choice': tune.choice(config.get('values')),
#                        'randint': tune.randint(config.get('min'), config.get('max')),
#                        'uniform': tune.sample_from(lambda spec: np.random.uniform(config.get('min'), config.get('max')))
#                              }.get(config['search'])
#
# # add the sweep_config and data_abspath as constant parameters for global experiment management
# tune_search_space['sweep_config'] = tune.grid_search([sweep_config])
# tune_search_space['data_abspath'] = tune.grid_search([data_abspath])

# # initialize global tracker for all experiments
# track.init()

# run experiment:
# initialize starting policies:
#starting_policies_abspath = os.path.abspath(os.path.join(args.log_dir, 'starting_policies.pkl'))
starting_policies_abspath = os.path.join(args.log_dir, 'starting_policies.pkl')

# tune_search_space['starting_policies_abspath'] = tune.grid_search([starting_policies_abspath])
if not os.path.exists(starting_policies_abspath):
    with open(starting_policies_abspath, 'wb') as f:
        pickle.dump([], f)

# fix some hparams ranges according to taskid:
search_space_size = np.prod([d['range'] if k =='graph_idx' else len(d['values']) for k, d in sweep_config['sweep'].items()])
product_lists = [sweep_config['sweep'][k]['values'] for k in args.product_keys]
products = list(product(*product_lists))
n_tasks = len(products)
time_limit_minutes = max(int(np.ceil(1.5*search_space_size/n_tasks/(args.cpus_per_task-1)) + 2), 16)
assert 60 > time_limit_minutes > 0

# run n policy iterations, parallelizing on n_tasks, each task on a separated node.
# in each iteration k, load k-1 starting policies,
# run exhaustive search for the best k'th policy - N LP rounds search,
# and for the rest use default cut selection.
# Then when all experiments ended, find the best policy for the i'th iteration and append to starting policies.
iter_logdir = ''
for k_iter in range(sweep_config['constants'].get('n_policy_iterations', 1)):
    # recovering from checkpoints:
    # skip iteration if completed in previous runs
    print('loading starting policies from: ', starting_policies_abspath)
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
    analyses = analyze_results(rootdir=iter_logdir, dstdir=iter_analysisdir, starting_policies_abspath=starting_policies_abspath)
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
    print('submitting jobs:')
    for taskid in range(n_tasks):
        jobname = 'iter{}-cfg{}'.format(k_iter, taskid)
        submit_job(jobname, taskid, time_limit_minutes)
        time.sleep(1)

    print('submitted jobs - run again in {} minutes after all jobs completed'.format(time_limit_minutes))
    exit(0)

# run the final adaptive policy in a clean directory and save the experiment results
config_file = os.path.abspath('cutrootnode/final_adaptive_policy_config.yaml')
print('To generate clean final results run:')
print('python adaptive_policy_runner.py --log-dir {} --config-file {} --data-dir {}\n'.format(
    os.path.abspath(os.path.join(args.log_dir, 'final_adaptive_policy')),
    config_file,
    os.path.abspath(args.data_dir)
))
# analyze_results(rootdir=iter_logdir, dstdir=os.path.join(args.log_dir, 'final_analysis'), tensorboard=True)
print('finished adaptive policy search. congrats!')

'"python /home/avrech/learning2cut/experiments/run_adaptive_policy_experiment_server.py --experiment=cutrootnode --config_file=cutrootnode/adaptive_policy_config.yaml --log_dir=cutrootnode/results/cutsbudget1000/adaptive_policy --data_dir=cutrootnode/data --cpus_per_task=40 --product_keys="[\'intsupportfac\', \'maxcutsroot\']" --auto"'
