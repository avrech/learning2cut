import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import os
from datetime import datetime
from shutil import copy
import pickle
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import operator
NOW = str(datetime.now())[:-7].replace(' ', '.').replace(':', '-').replace('.', '/')
parser = ArgumentParser()
parser.add_argument('--rootdir', type=str, default='variability/results/tmp', help='path to experiment results root dir')
parser.add_argument('--dstdir', type=str, default='variability/analysis/' + NOW, help='path to store tables, tensorboard etc.')
parser.add_argument('--pattern', type=str, default='experiment_results.pkl', help='pattern of pickle files')
parser.add_argument('--tensorboard', action='store_true', help='generate tensorboard folder in <dstdir>/tb')
parser.add_argument('--generate-experts', action='store_true', help='save experts configs to <dstdir>/experts')


args = parser.parse_args()

if not os.path.exists(args.dstdir):
    os.makedirs(args.dstdir)

summary = []
for path in tqdm(Path(args.rootdir).rglob(args.pattern), desc='Loading files'):
    with open(path, 'rb') as f:
        res = pickle.load(f)
        # set some defaults if necessary
        if 'forcecut' not in res['config'].keys():
            res['config']['forcecut'] = False
        summary.append(res)

results = {}  # stats of cycle inequalities
baseline = {}  # stats of scip defaults
datasets = {}  # metadata for grouping results

# parse the experiment result files
# The structure of results is:
# results[dataset][config][stat_key][graph_idx][seed]

for s in tqdm(summary, desc='Parsing files'):
    dataset = s['config']['data_abspath'].split('/')[-1]  # the name of the dataset
    if dataset not in datasets.keys():
        print('Adding dataset ', dataset)
        datasets[dataset] = {}
        datasets[dataset]['config_keys'] = [k for k in s['config'].keys() if k != 'scip_seed' and k != 'graph_idx' and k != 'sweep_config' and k != 'data_abspath']
        # store these two to ensure that all the experiments completed successfully.
        datasets[dataset]['scip_seeds'] = s['config']['sweep_config']['sweep']['scip_seed']['values']
        datasets[dataset]['graph_idx_range'] = list(range(s['config']['sweep_config']['sweep']['graph_idx']['range']))
        datasets[dataset]['missing_experiments'] = []
        datasets[dataset]['sweep_config'] = s['config']['sweep_config']
        datasets[dataset]['configs'] = {}
        datasets[dataset]['experiment'] = s['experiment']


        results[dataset] = {}
        baseline[dataset] = {}

    config = tuple([s['config'][k] for k in datasets[dataset]['config_keys']])
    if config not in datasets[dataset]['configs'].keys():
        datasets[dataset]['configs'][config] = s['config']
        if s['config']['use_cycle_cuts']:
            results[dataset][config] = {stat_key: {graph_idx: {}
                                                   for graph_idx in range(s['config']['sweep_config']['sweep']['graph_idx']['range'])}
                                        for stat_key in s['stats'].keys()}
        else:
            baseline[dataset][config] = {stat_key: {graph_idx: {}
                                                    for graph_idx in range(s['config']['sweep_config']['sweep']['graph_idx']['range'])}
                                         for stat_key in s['stats'].keys()}

    # select the appropriate dictionary
    graph_idx = s['config']['graph_idx']
    scip_seed = s['config']['scip_seed']
    dictionary = results if s['config']['use_cycle_cuts'] else baseline

    # insert stats into dictionary:
    for stat_key, value in s['stats'].items():
        dictionary[dataset][config][stat_key][graph_idx][scip_seed] = value


# process the results
# if any experiment is missing, generate its configuration and append to missing_experiments
# results[dataset][config][stat_key][graph_idx][seed]
for dataset in datasets.keys():
    bsl = baseline[dataset]
    res = results[dataset]
    # TODO: for all configs, compute mean and std across all seeds within graphs,
    # and also std of stds across all graphs
    for dictionary in [bsl, res]:
        for config, stats in dictionary.items():
            missing_graph_and_seed = []
            for stat_key, graph_dict in stats.items():
                all_values = []
                all_stds = []
                for graph_idx in datasets[dataset]['graph_idx_range']:
                    values = []
                    for scip_seed in datasets[dataset]['scip_seeds']:
                        if scip_seed not in graph_dict[graph_idx].keys():
                            if (graph_idx, scip_seed) not in missing_graph_and_seed:
                                experiment_config = datasets[dataset]['configs'][config].copy()
                                experiment_config['graph_idx'] = graph_idx
                                experiment_config['scip_seed'] = scip_seed
                                datasets[dataset]['missing_experiments'].append(experiment_config)
                                missing_graph_and_seed.append((graph_idx, scip_seed))
                            continue
                        values.append(graph_dict[graph_idx][scip_seed])
                        all_values.append(graph_dict[graph_idx][scip_seed])
                    if len(values) > 0:
                        graph_dict[graph_idx]['mean'] = np.mean(values)
                        graph_dict[graph_idx]['std'] = np.std(values)
                        all_stds.append(np.std(values))
                # compute the mean and std of stds across all graphs.
                if len(all_stds) > 0:
                    graph_dict['mean'] = np.mean(all_values)
                    graph_dict['std'] = np.std(all_stds)

    # gap = res['gap']
    # solving_time = res['solving_time']
    #
    # # compute mean solving time and std across seeds
    # solving_time_avg = []
    # solving_time_std = []
    # for g in solving_time:
    #     avg = []
    #     std = []
    #     for hp_config in g:
    #         avg.append(np.mean(hp_config))
    #         std.append(np.std(hp_config))
    #     solving_time_avg.append(avg)
    #     solving_time_std.append(std)
    # gap_avg = []
    # for g in gap:
    #     avg = []
    #     for hp_config in g:
    #         avg.append(np.mean(hp_config))
    #     gap_avg.append(avg)

    # 2. for each dataset and graph find the best hparams
    # according to gap/solving_time avg across all seeds

    # lists of best performance for each graph_idx
    best_config = []
    best_avg_time = []
    best_avg_gap = []
    best_std_time = []
    best_std_gap = []
    bsl_avg_time = []
    bsl_avg_gap = []
    bsl_std_time = []
    bsl_std_gap = []

    # results[dataset][config][stat_key][graph_idx][seed]
    for graph_idx in datasets[dataset]['graph_idx_range']:
        configs = []
        avg_time = []
        std_time = []
        avg_gap = []
        std_gap = []
        for config, stats in res.items():
            if stats['gap'][graph_idx].get('mean', None) is not None:
                configs.append(config)
                avg_gap.append(stats['gap'][graph_idx]['mean'])
                std_gap.append(stats['gap'][graph_idx]['std'])
                avg_time.append(stats['solving_time'][graph_idx]['mean'])
                std_time.append(stats['solving_time'][graph_idx]['std'])

        # sort tuples of (gap, time, hp_config_idx) according to gap, and then according to time.
        gap_and_time = list(zip(avg_gap, avg_time, std_gap, std_time, configs))
        if len(gap_and_time) > 0:
            gap_and_time.sort(key=operator.itemgetter(0, 1))
            best = gap_and_time[0]
            best_avg_gap.append(best[0])
            best_avg_time.append(best[1])
            best_std_gap.append(best[2])
            best_std_time.append(best[3])
            best_config.append(best[4])
        else:
            best_avg_gap.append('-')
            best_avg_time.append('-')
            best_std_gap.append('-')
            best_std_time.append('-')
            best_config.append(None)
        if len(bsl) > 0:
            bsl_avg_gap.append(list(bsl.values())[0]['gap'][graph_idx].get('mean', '-'))
            bsl_avg_time.append(list(bsl.values())[0]['solving_time'][graph_idx].get('mean', '-'))
            bsl_std_gap.append(list(bsl.values())[0]['gap'][graph_idx].get('std', '-'))
            bsl_std_time.append(list(bsl.values())[0]['solving_time'][graph_idx].get('std', '-'))
        else:
            bsl_avg_gap.append('-')
            bsl_avg_time.append('-')
            bsl_std_gap.append('-')
            bsl_std_time.append('-')
    # compute the average gap and solving time for scip baseline:
    time_limit = datasets[dataset]['sweep_config']['constants']['time_limit_sec']
    # best_avg_gap = np.array(best_avg_gap)
    # best_avg_time = np.array(best_avg_time)
    # bsl_avg_gap = np.array(bsl_avg_gap)
    # bsl_avg_time = np.array(bsl_avg_time)

    # best_config = np.array(best_config)

    # # compute number of wins,
    # # i.e the fraction of seeds on which cycle inequalities improved
    # best_config_times = np.array([t[best_config[graph_id]] for graph_id, t in enumerate(solving_time)]).clip(max=time_limit)
    # best_config_gaps = np.array([g[best_config[graph_id]] for graph_id, g in enumerate(gap)])
    # strictly_better_gap = best_config_gaps < bsl_gaps
    # strictly_better_time = best_config_times < bsl_times
    #
    # wins = np.logical_or(strictly_better_gap, strictly_better_time)
    # wins_rate = np.mean(wins, axis=-1)
    # wins_rate_str = ['{}/{}'.format(np.sum(w), len(w)) for w in wins]
    speedup_avg = []
    for t1, t2 in zip(bsl_avg_time, best_avg_time):
        if t1 != '-' and t2 != '-':
            speedup_avg.append(t1 / t2)
        else:
            speedup_avg.append('-')
    # speedup_avg = bsl_avg_time / best_avg_time

    # 3. for each dataset find the best hparams
    # according to gap/solving_time avg across all graphs and seeds

    # 5. write the summary into pandas DataFrame
    d = {#'Wins': wins_rate_str,
         'Speedup avg': speedup_avg + [np.mean(speedup_avg) if '-' not in speedup_avg else '-'],
         'Baseline time': bsl_avg_time + [np.mean(bsl_avg_time) if '-' not in bsl_avg_time else '-'],
         'Baseline time-std': bsl_std_time + [np.mean(bsl_std_time) if '-' not in bsl_std_time else '-'],
         'Best time': best_avg_time + [np.mean(best_avg_time)],
         'Best time-std': best_std_time + [np.mean(best_std_time)],
         # 'Wins rate': wins_rate,
         'Baseline gap': bsl_avg_gap + [np.mean(bsl_avg_gap) if '-' not in bsl_avg_gap else '-'],
         'Baseline gap-std': bsl_std_gap + [np.mean(bsl_std_gap) if '-' not in bsl_std_gap else '-'],
         'Best gap': best_avg_gap + [np.mean(best_avg_gap)],
         'Best gap-std': best_std_gap + [np.mean(best_std_gap)],
         }
    for k in datasets[dataset]['configs'][best_config[0]].keys():
        d[k] = []

    for graph_idx, bc in enumerate(best_config):
        best_hparams = datasets[dataset]['configs'][bc]
        for k, v in best_hparams.items():
            d[k].append(v)
    # append empty row for the avg row
    for k, v in best_hparams.items():
        d[k].append('-')

    df = pd.DataFrame(data=d, index=list(range(len(speedup_avg))) + ['avg'])
    df.to_csv(os.path.join(args.dstdir, 'tables', dataset + '_results.csv'), float_format='%.3f')
    print('Experiment summary saved to {}'.format(
        os.path.join(args.dstdir, 'tables', dataset + '_results.csv')))
    # latex_str = df.to_latex(float_format='%.3f')
    # print(latex_str)
    if len(datasets[dataset]['missing_experiments']) > 0:
        missing_experiments_file = os.path.join(args.dstdir, 'missing_experiments', dataset + '_missing_experiments.pkl')
        with open(missing_experiments_file, 'wb') as f:
            pickle.dump(datasets[dataset]['missing_experiments'], f)
        print('WARNING: missing experiments saved to {}'.format(missing_experiments_file))
        print('To complete experiments, run the following command inside experiments/ folder:')
        print('python complete_experiment.py --experiment {} --config-file {} --log-dir {}'.format(
            datasets[dataset]['experiment'],
            os.path.abspath(missing_experiments_file),
            os.path.abspath(args.rootdir)))
print('finish')

