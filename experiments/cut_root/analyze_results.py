import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import os
from datetime import datetime
import pickle
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import operator
NOW = str(datetime.now())[:-7].replace(' ', '.').replace(':', '-').replace('.', '/')
parser = ArgumentParser()
parser.add_argument('--rootdir', type=str, default='results/', help='path to experiment results root dir')
parser.add_argument('--dstdir', type=str, default='analysis/' + NOW, help='path to store tables, tensorboard etc.')
parser.add_argument('--pattern', type=str, default='experiment_results.pkl', help='pattern of pickle files')
parser.add_argument('--tensorboard', action='store_true', help='generate tensorboard folder in <dstdir>/tb')
parser.add_argument('--tb-k-best', type=int, help='generate tensorboard for the k best configs (and baseline)')
parser.add_argument('--generate-experts', action='store_true', help='save experts configs to <dstdir>/experts')


args = parser.parse_args()

if not os.path.exists(args.dstdir):
    os.makedirs(args.dstdir)

summary = []
for path in tqdm(Path(args.rootdir).rglob(args.pattern), desc='Loading files'):
    with open(path, 'rb') as f:
        res = pickle.load(f)
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
        if s['config']['max_per_root'] > 0:
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
    dictionary = results if s['config']['max_per_root'] > 0 else baseline

    # insert stats into dictionary:
    for stat_key, value in s['stats'].items():
        dictionary[dataset][config][stat_key][graph_idx][scip_seed] = value


# process the results
# if any experiment is missing, generate its configuration and append to missing_experiments
for dataset in datasets.keys():
    bsl = baseline[dataset]
    res = results[dataset]
    # TODO: for all configs, compute mean and std across all seeds within graphs,
    # and also std of stds across all graphs
    for dictionary in [bsl, res]:
        for config, stats in tqdm(dictionary.items(), desc='Analyzing'):
            # compute the integral of dual_bound w.r.t lp_iterations
            # report missing seeds/graphs
            missing_graph_and_seed = []
            dualbounds = stats['dualbound']
            lp_iterations = stats['lp_iterations']
            all_values = []
            all_stds = []
            stats['dualbound_integral'] = {}
            for graph_idx in datasets[dataset]['graph_idx_range']:
                values = []
                stats['dualbound_integral'][graph_idx] = {}
                for scip_seed in datasets[dataset]['scip_seeds']:
                    if scip_seed not in dualbounds[graph_idx].keys():
                        if (graph_idx, scip_seed) not in missing_graph_and_seed:
                            experiment_config = datasets[dataset]['configs'][config].copy()
                            experiment_config['graph_idx'] = graph_idx
                            experiment_config['scip_seed'] = scip_seed
                            datasets[dataset]['missing_experiments'].append(experiment_config)
                            missing_graph_and_seed.append((graph_idx, scip_seed))
                        continue
                    # compute integral:
                    dualbound = np.array(dualbounds[graph_idx][scip_seed])
                    lp_iter = np.array(lp_iterations[graph_idx][scip_seed])
                    integral = np.sum(dualbound * lp_iter)
                    stats['dualbound_integral'][graph_idx][scip_seed] = integral
                    values.append(integral)
                    all_values.append(integral)
                if len(values) > 0:
                    # compute the std of integral across seeds, and store in stats
                    stats['dualbound_integral'][graph_idx]['avg'] = np.mean(values)
                    stats['dualbound_integral'][graph_idx]['std'] = np.std(values)
                    all_stds.append(np.std(values))
            # compute the avg and std of stds across all graphs.
            if len(all_stds) > 0:
                stats['dualbound_integral']['avg'] = np.mean(all_values)
                stats['dualbound_integral']['std'] = np.std(all_stds)



            # for stat_key, graph_dict in stats.items():
            #     all_values = []
            #     all_stds = []
            #     for graph_idx in datasets[dataset]['graph_idx_range']:
            #         values = []
            #         for scip_seed in datasets[dataset]['scip_seeds']:
            #             if scip_seed not in graph_dict[graph_idx].keys():
            #                 if (graph_idx, scip_seed) not in missing_graph_and_seed:
            #                     experiment_config = datasets[dataset]['configs'][config].copy()
            #                     experiment_config['graph_idx'] = graph_idx
            #                     experiment_config['scip_seed'] = scip_seed
            #                     datasets[dataset]['missing_experiments'].append(experiment_config)
            #                     missing_graph_and_seed.append((graph_idx, scip_seed))
            #                 continue
            #             # TODO - consider to stop here, and analyze after.
            #             values.append(graph_dict[graph_idx][scip_seed])
            #             all_values.append(graph_dict[graph_idx][scip_seed])
            #         if len(values) > 0:
            #             graph_dict[graph_idx]['mean'] = np.mean(values)
            #             graph_dict[graph_idx]['std'] = np.std(values)
            #             all_stds.append(np.std(values))
            #     # compute the mean and std of stds across all graphs.
            #     if len(all_stds) > 0:
            #         graph_dict['mean'] = np.mean(all_values)
            #         graph_dict['std'] = np.std(all_stds)

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

    # list of k-best performance for each graph_idx
    best_config = []
    k_best_configs = []
    best_dualbound_int_avg = []
    best_dualbound_int_std = []
    baseline_dualbound_int_avg = []
    baseline_dualbound_int_std = []

    # results[dataset][config][stat_key][graph_idx][seed]
    for graph_idx in datasets[dataset]['graph_idx_range']:
        configs = []
        dualbound_int_avg = []
        dualbound_int_std = []
        for config, stats in res.items():
            if stats['dualbound_integral'][graph_idx].get('avg', None) is not None:
                configs.append(config)
                dualbound_int_avg.append(stats['dualbound_integral'][graph_idx]['avg'])
                dualbound_int_std.append(stats['dualbound_integral'][graph_idx]['std'])


        # sort tuples of (dualbound_int_avg, dualbound_int_std, config)
        # according to gap, and then according to dualbound_int_avg
        avg_std_config = list(zip(dualbound_int_avg, dualbound_int_std, configs))
        if len(avg_std_config) > 0:
            avg_std_config.sort(key=operator.itemgetter(0))
            best = avg_std_config[0]
            best_dualbound_int_avg.append(best[0])
            best_dualbound_int_std.append(best[1])
            best_config.append(best[2])
            k_best_configs.append([cfg[2] for cfg in avg_std_config[:args.tb_k_best]])
        else:
            best_dualbound_int_avg.append('-')
            best_dualbound_int_std.append('-')
            best_config.append(None)
            k_best_configs.append(None)

        # TODO process baseline and find missing experiments
        if len(bsl) > 0:
            baseline_dualbound_int_avg.append(list(bsl.values())[0]['dualbound_integral'][graph_idx].get('avg', '-'))
            baseline_dualbound_int_std.append(list(bsl.values())[0]['dualbound_integral'][graph_idx].get('std', '-'))
        else:
            baseline_dualbound_int_avg.append('-')
            baseline_dualbound_int_std.append('-')

    # compute the average gap and solving time for scip baseline:
    time_limit = datasets[dataset]['sweep_config']['constants']['time_limit_sec']
    integral_ratio = []
    for best_integral, baseline_integral in zip(best_dualbound_int_avg, baseline_dualbound_int_avg):
        if best_integral != '-' and baseline_integral != '-':
            integral_ratio.append(best_integral / baseline_integral)
        else:
            integral_ratio.append('-')

    # 3. for each dataset find the best hparams
    # according to gap/solving_time avg across all graphs and seeds

    # 5. write the summary into pandas DataFrame
    d = {'Integral Ratio': integral_ratio + [np.mean(integral_ratio) if '-' not in integral_ratio else '-'],
         'Best avg': best_dualbound_int_avg + [np.mean(best_dualbound_int_avg) if '-' not in best_dualbound_int_avg else '-'],
         'Best std': best_dualbound_int_std + [np.mean(best_dualbound_int_std) if '-' not in best_dualbound_int_std else '-'],
         'Baseline avg': baseline_dualbound_int_avg + [np.mean(baseline_dualbound_int_avg) if '-' not in baseline_dualbound_int_avg else '-'],
         'Baseline std': baseline_dualbound_int_std + [np.mean(baseline_dualbound_int_std) if '-' not in baseline_dualbound_int_std else '-'],
         # 'Baseline time': bsl_avg_time + [np.mean(bsl_avg_time) if '-' not in bsl_avg_time else '-'],
         # 'Baseline time-std': bsl_std_time + [np.mean(bsl_std_time) if '-' not in bsl_std_time else '-'],
         # 'Best time': best_avg_time + [np.mean(best_avg_time)],
         # 'Best time-std': best_std_time + [np.mean(best_std_time)],
         # # 'Wins rate': wins_rate,
         # 'Baseline gap': bsl_avg_gap + [np.mean(bsl_avg_gap) if '-' not in bsl_avg_gap else '-'],
         # 'Baseline gap-std': bsl_std_gap + [np.mean(bsl_std_gap) if '-' not in bsl_std_gap else '-'],
         # 'Best gap': best_avg_gap + [np.mean(best_avg_gap)],
         # 'Best gap-std': best_std_gap + [np.mean(best_std_gap)],
         }
    for k in datasets[dataset]['configs'][best_config[0]]['sweep_config']['sweep'].keys():
        d[k] = []

    for graph_idx, bc in enumerate(best_config):
        best_hparams = datasets[dataset]['configs'][bc]['sweep_config']['sweep']
        for k, v in best_hparams.items():
            d[k].append(v)
    # append empty row for the avg row
    for k, v in best_hparams.items():
        d[k].append('-')

    tables_dir = os.path.join(args.dstdir, 'tables')
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)
    csv_file = os.path.join(tables_dir, dataset + '_results.csv')
    df = pd.DataFrame(data=d, index=list(range(len(integral_ratio))) + ['avg'])
    df.to_csv(csv_file, float_format='%.3f')
    print('Experiment summary saved to {}'.format(csv_file))
    # latex_str = df.to_latex(float_format='%.3f')
    # print(latex_str)
    missing_experiments_dir = os.path.join(args.dstdir, 'missing_experiments')
    if not os.path.exists(missing_experiments_dir):
        os.makedirs(missing_experiments_dir)
    if len(datasets[dataset]['missing_experiments']) > 0:
        missing_experiments_file = os.path.join(missing_experiments_dir, dataset + '_missing_experiments.pkl')
        with open(missing_experiments_file, 'wb') as f:
            pickle.dump(datasets[dataset]['missing_experiments'], f)
        print('WARNING: missing experiments saved to {}'.format(missing_experiments_file))
        print('Statistics might not be accurate.')
        print('To complete experiments, run the following command inside experiments/ folder:')
        print('python complete_experiment.py --experiment {} --config-file {} --log-dir {}'.format(
            datasets[dataset]['experiment'],
            os.path.abspath(missing_experiments_file),
            os.path.abspath(args.rootdir)))

    # Generate tensorboard for the K-best configs
    if args.tensorboard:
        tensorboard_dir = os.path.join(args.dstdir, 'tensorboard', dataset)
        writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, 'hparams'))
        # Generate hparams tab for the k-best-on-average configs, and in addition for the baseline.
        # The hparams specify for each graph and seed some more stats.
        for graph_idx, kbcfgs in enumerate(k_best_configs):
            for config in kbcfgs:
                stats = res[config]
                hparams = datasets[dataset]['configs'][config].copy()
                hparams.pop('data_abspath', None)
                hparams.pop('sweep_config', None)
                hparams['graph_idx'] = graph_idx
                metric_lists = {k: [] for k in stats.keys()}
                # plot hparams for each seed
                for scip_seed in stats['dualbound'][graph_idx].keys():
                    hparams['scip_seed'] = scip_seed
                    metrics = {k: v[graph_idx][scip_seed][-1] for k, v in stats.items() if k != 'dualbound_integral'}
                    metrics['dualbound_integral'] = stats['dualbound_integral'][graph_idx][scip_seed]
                    metrics['cycles_sepa_time'] = metrics['cycles_sepa_time'] / metrics['solving_time']
                    for k, v in metrics.items():
                        metric_lists[k].append(v)
                    for k in metric_lists.keys():
                        metrics[k+'_std'] = 0
                    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)
                # plot hparams for each graph averaged across seeds
                hparams['scip_seed'] = 'avg'
                metrics = {}
                for k, v_list in metric_lists.items():
                    metrics[k] = np.mean(v_list)
                    metrics[k+'_std'] = np.std(v_list)
                writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

        # add hparams for the baseline
        for graph_idx in datasets[dataset]['graph_idx_range']:
            for config, stats in bsl.items():
                hparams = datasets[dataset]['configs'][config].copy()
                hparams.pop('data_abspath', None)
                hparams.pop('sweep_config', None)
                hparams['graph_idx'] = graph_idx
                metric_lists = {k: [] for k in stats.keys()}
                # plot hparams for each seed
                for scip_seed in stats['dualbound'][graph_idx].keys():
                    hparams['scip_seed'] = scip_seed
                    metrics = {k: v[graph_idx][scip_seed][-1] for k, v in stats.items() if k != 'dualbound_integral'}
                    metrics['dualbound_integral'] = stats['dualbound_integral'][graph_idx][scip_seed]
                    metrics['cycles_sepa_time'] = metrics['cycles_sepa_time'] / metrics['solving_time']
                    for k, v in metrics.items():
                        metric_lists[k].append(v)
                    for k in metric_lists.keys():
                        metrics[k+'_std'] = 0
                    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)
                # plot hparams for each graph averaged across seeds
                hparams['scip_seed'] = 'avg'
                metrics = {}
                for k, v_list in metric_lists.items():
                    metrics[k] = np.mean(v_list)
                    metrics[k+'_std'] = np.std(v_list)
                writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)
        writer.close()

        # add plots of metrics vs time for the best config
        # each graph plot separately.
        for graph_idx, config in enumerate(best_config):
            stats = res[config]
            for scip_seed, db in stats['dualbound'][graph_idx].items():
                writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, 'scalars', 'best',
                                                            'graph_idx{}'.format(graph_idx), 'scip_seed{}'.format(scip_seed)))
                for lp_round in range(len(db)):
                    records = {k: v[graph_idx][scip_seed][lp_round] for k, v in stats.items() if k != 'dualbound_integral'}
                    # dualbound vs. lp iterations
                    writer.add_scalar(tag='Dualbound_vs_LP_Iterations/g{}'.format(graph_idx, scip_seed),
                                      scalar_value=records['dualbound'],
                                      global_step=records['lp_iterations'],
                                      walltime=records['solving_time'])
                    # dualbound vs. cycles applied
                    writer.add_scalar(tag='Dualbound_vs_Cycles_Applied/g{}'.format(graph_idx, scip_seed),
                                      scalar_value=records['dualbound'],
                                      global_step=records['cycle_ncuts_applied'],
                                      walltime=records['solving_time'])
                    # dualbound vs. total cuts applied
                    writer.add_scalar(tag='Dualbound_vs_Total_Cuts_Applied/g{}'.format(graph_idx, scip_seed),
                                      scalar_value=records['dualbound'],
                                      global_step=records['total_ncuts_applied'],
                                      walltime=records['solving_time'])
                writer.close()

        # add plots of metrics vs time for the baseline
        for config, stats in bsl.items():
            for graph_idx in stats['dualbound'].keys():
                for scip_seed, db in stats['dualbound'][graph_idx].items():
                    writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, 'scalars', 'baseline',
                                                                'graph_idx{}'.format(graph_idx),
                                                                'scip_seed{}'.format(scip_seed)))

                    for lp_round in range(len(db)):
                        records = {k: v[graph_idx][scip_seed][lp_round] for k, v in stats.items() if
                                   k != 'dualbound_integral'}
                        # dualbound vs. lp iterations
                        writer.add_scalar(tag='Dualbound_vs_LP_Iterations/g{}'.format(graph_idx, scip_seed),
                                          scalar_value=records['dualbound'],
                                          global_step=records['lp_iterations'],
                                          walltime=records['solving_time'])
                        # dualbound vs. total cuts applied
                        writer.add_scalar(tag='Dualbound_vs_Total_Cuts_Applied/g{}'.format(graph_idx, scip_seed),
                                          scalar_value=records['dualbound'],
                                          global_step=records['total_ncuts_applied'],
                                          walltime=records['solving_time'])

                    writer.close()

        print('Tensorboard events written to ' + tensorboard_dir)
        print('To open tensorboard tab on web browser, run in terminal the following command:')
        print('tensorboard --logdir ' + os.path.abspath(tensorboard_dir))

print('finish')

