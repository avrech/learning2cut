from utils.scip_models import mvc_model, CSBaselineSepa, set_aggresive_separation, CSResetSepa, maxcut_mccormic_model
from pathlib import Path
import numpy as np
import pyarrow as pa
from utils.functions import get_normalized_areas
from tqdm import tqdm
import pickle
from argparse import ArgumentParser
import os
import wandb
import random
import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
import hyperopt

parser = ArgumentParser()
parser.add_argument('--num_trials', type=int, default=1000, help='number of instances to solve in each trial')
parser.add_argument('--batch_size', type=int, default=100, help='number of instances to solve in each trial')
parser.add_argument('--trainset_size', type=int, default=100, help='trainset size')
parser.add_argument('--problem', type=str, default='maxcut', help='mvc | maxcut')
parser.add_argument('--local', action='store_true', help='run local wandb sweep controller')
parser.add_argument('--rootdir', type=str, default='results/wandb_sweep', help='rootdir to store results')
parser.add_argument('--datadir', type=str, default='/home/avrech/learning2cut/data', help='data path')
args = parser.parse_args()
# os.environ['WANDB_API_KEY'] = 'd1e669477d060991ed92264313cade12a7995b3d'
# os.environ['WANDB_MODE'] = 'dryrun'
SEEDS = [52, 176, 223]  # [46, 72, 101]
ROOTDIR = args.rootdir
# WANDB_RUN_DIR = wandb.util.generate_id()


def eval_db_auc_avg(config):
    # wandb.init(dir=f'{args.rootdir}/{WANDB_RUN_DIR}',
    #            config=config,
    #            project='learning2cut',
    #            tags=['hyperopt']
    #            )
    # load trainset
    print(f'loading training data from: {args.datadir}/{args.problem.upper()}')
    with open(f'{args.datadir}/{args.problem.upper()}/data.pkl', 'rb') as f:
        data = pickle.load(f)
    trainset = [d['instances'] for k, d in data.items() if 'train' in k][0][:args.trainset_size]
    # randomize batch_size samples to solve
    trial_instances = random.sample(trainset, args.batch_size)
    del data, trainset
    baseline = 'scip_tuned'
    db_auc_list = []

    for g, info in trial_instances:
        for seed in SEEDS:
            if args.problem == 'mvc':
                model, _ = mvc_model(g)
                lp_iterations_limit = 1500
            elif args.problem == 'maxcut':
                model, _, _ = maxcut_mccormic_model(g)
                lp_iterations_limit = 5000
            else:
                raise ValueError
            set_aggresive_separation(model)
            sepa_params = {'lp_iterations_limit': lp_iterations_limit,
                           'policy': 'tuned',
                           'reset_maxcuts': 100,
                           'reset_maxcutsroot': 100,
                           }
            sepa_params.update(config)
            sepa = CSBaselineSepa(hparams=sepa_params)
            model.includeSepa(sepa, '#CS_baseline', baseline, priority=-100000000, freq=1)
            reset_sepa = CSResetSepa(hparams=sepa_params)
            model.includeSepa(reset_sepa, '#CS_reset', f'reset maxcuts params', priority=99999999, freq=1)
            model.setBoolParam("misc/allowdualreds", 0)
            model.setLongintParam('limits/nodes', 1)  # solve only at the root node
            model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever
            model.setIntParam('branching/random/priority', 10000000)
            model.setBoolParam('randomization/permutevars', True)
            model.setIntParam('randomization/permutationseed', seed)
            model.setIntParam('randomization/randomseedshift', seed)
            model.hideOutput(True)
            model.optimize()
            sepa.update_stats()
            stats = sepa.stats
            db_auc = sum(get_normalized_areas(t=stats['lp_iterations'], ft=stats['dualbound'],
                                              t_support=lp_iterations_limit, reference=info['optimal_value']))
            db_auc_list.append(db_auc)
    db_auc_avg = np.mean(db_auc_list)
    # wandb.log({'db_auc_avg': db_auc_avg})
    print('finished')
    tune.report(db_auc_avg=db_auc_avg)
    return db_auc_avg

search_space = {
    'objparalfac': hyperopt.hp.choice('objparalfac', [0.1, 0.5, 1]),
    'dircutoffdistfac': hyperopt.hp.choice('dircutoffdistfac', [0.1, 0.5, 1]),
    'efficacyfac': hyperopt.hp.choice('efficacyfac', [0.1, 0.5, 1]),
    'intsupportfac': hyperopt.hp.choice('intsupportfac', [0.1, 0.5, 1]),
    'maxcutsroot': hyperopt.hp.choice('maxcutsroot', [5, 15, 2000]),
    'minorthoroot': hyperopt.hp.choice('minorthoroot', [0.5, 0.9, 1]),
}
tune_search_space = {
    'objparalfac': tune.choice([0.1, 0.5, 1]),
    'dircutoffdistfac': tune.choice([0.1, 0.5, 1]),
    'efficacyfac': tune.choice([0.1, 0.5, 1]),
    'intsupportfac': tune.choice([0.1, 0.5, 1]),
    'maxcutsroot': tune.choice([5, 15, 2000]),
    'minorthoroot': tune.choice([0.5, 0.9, 1]),
}

algo = HyperOptSearch(
    space=search_space,
    points_to_evaluate=[{
        # index of default params in the search space
        'objparalfac': 0,
        'dircutoffdistfac': 1,
        'efficacyfac': 2,
        'intsupportfac':  0,
        'maxcutsroot': 2,
        'minorthoroot': 1,
    }],
    mode='max',
    metric='db_auc_avg'
)

ray.init()
analysis = tune.run(
    eval_db_auc_avg,
    local_dir=args.rootdir,
    search_alg=algo,
    config=tune_search_space,
    num_samples=args.num_trials
)
print("Best parameters found: ", analysis.get_best_config('db_auc_avg'))


