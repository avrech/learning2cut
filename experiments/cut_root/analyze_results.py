import networkx as nx
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
import matplotlib.pyplot as plt

def analyze_results(rootdir='results', dstdir='analysis', filepattern='experiment_results.pkl', tensorboard=False, tb_k_best=1, csv=False, final_adaptive=False, plot=False):
    # make directory where to save analysis files - tables, tensorboard etc.
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)

    # load all results files stored in the rootdir
    summary = []
    for path in tqdm(Path(rootdir).rglob(filepattern), desc='Loading files'):
        with open(path, 'rb') as f:
            res = pickle.load(f)
            summary.append(res)


    def str_hparams(hparams_dict):
        """ Serialize predefined key-value pairs into a string,
        useful to define tensorboard logdirs,
        such that configs can be identified and filtered on tensorboard scalars tab
        :param hparams_dict: a dictionary of hparam, value pairs.
        :returns s: a string consists of acronyms of keys and their corresponding values.
        """
        short_keys = {
            'policy': 'plc',
            # MCCORMIC_CYCLE_SEPARATOR PARAMETERS
            'max_per_node': 'mpnd',
            'max_per_round': 'mprd',
            'criterion': 'crt',
            'max_per_root': 'mprt',
            'forcecut': 'fct',
            # SCIP SEPARATING PARAMETERS
            'objparalfac': 'opl',
            'dircutoffdistfac': 'dcd',
            'efficacyfac': 'eff',
            'intsupportfac': 'isp',
            'maxcutsroot': 'mcr',
        }
        s = 'cfg'
        for k, sk in short_keys.items():
            v = hparams_dict.get(k, None)
            if v is not None:
                s += '-{}={}'.format(sk, v)
        return s

    ##### PARSING LOG FILES #####
    # parse the experiment result files
    results = {}  # stats of cycle inequalities policies
    baselines = {}  # stats of some predefined baselines
    datasets = {}  # metadata for grouping/parsing results
    analysis = {}  # return dict containing some important info
    # statistics are stored in results/baselines dictionaries in the following hierarchy
    # results[<dataset str>][<config tuple>][<stat_key str>][<graph_idx int>][<seed int>]
    for s in tqdm(summary, desc='Parsing files'):
        dataset = s['config']['data_abspath'].split('/')[-1]  # the name of the dataset
        if dataset not in datasets.keys():
            print('Adding dataset ', dataset)
            analysis[dataset] = {}
            datasets[dataset] = {}
            datasets[dataset]['config_keys'] = [k for k in s['config'].keys() if k != 'scip_seed' and k != 'graph_idx' and k != 'sweep_config' and k != 'data_abspath' and k != 'starting_policies_abspath']
            # store these two to ensure that all the experiments completed successfully.
            datasets[dataset]['scip_seeds'] = set(s['config']['sweep_config']['sweep']['scip_seed']['values'])
            datasets[dataset]['graph_idx_range'] = list(range(s['config']['sweep_config']['sweep']['graph_idx']['range']))
            datasets[dataset]['missing_experiments'] = []
            datasets[dataset]['sweep_config'] = s['config']['sweep_config']
            datasets[dataset]['configs'] = {}
            datasets[dataset]['experiment'] = s['experiment']
            datasets[dataset]['optimal_values'] = {}
            datasets[dataset]['max_lp_iterations'] = {graph_idx: {}
                                                      for graph_idx in
                                                      range(s['config']['sweep_config']['sweep']['graph_idx']['range'])}
            datasets[dataset]['best_dualbound'] = {graph_idx: {}
                                                   for graph_idx in
                                                   range(s['config']['sweep_config']['sweep']['graph_idx']['range'])}
            results[dataset] = {}
            baselines[dataset] = {}
        # for a case some scip seed was missing when the dictionary was created
        datasets[dataset]['scip_seeds'].update(s['config']['sweep_config']['sweep']['scip_seed']['values'])

        graph_idx = s['config']['graph_idx']
        scip_seed = s['config']['scip_seed']

        # read and update the instance optimal value (MAXCUT)
        if graph_idx not in datasets[dataset]['optimal_values'].keys():
            # read the annotated graph and update its optimal value if any
            filepath = os.path.join(s['config']['data_abspath'], 'graph_idx_{}.pkl'.format(graph_idx))
            # filepath = 'data/barabasi-albert-n50-m10-weights-normal-seed100/graph_idx_0.pkl'
            with open(filepath, 'rb') as f:
                G = pickle.load(f)
            cut = nx.get_edge_attributes(G, 'cut')
            if len(cut) > 0:
                weight = nx.get_edge_attributes(G, 'weight')
                datasets[dataset]['optimal_values'][graph_idx] = sum([weight[e] for e in G.edges if cut[e]])
            else:
                datasets[dataset]['optimal_values'][graph_idx] = 0  # default

        # set baselines policy string for plots legend:
        if s['config']['maxcutsroot'] == 2000 and \
                s['config']['maxcutsroot'] == 2000 and \
                s['config']['intsupportfac'] == 0.1 and \
                s['config']['efficacyfac'] == 1 and \
                s['config']['dircutoffdistfac'] == 0.5 and \
                s['config']['objparalfac'] == 0.1 and \
                not s['config']['forcecut'] and \
                s['config']['policy'] != 'adaptive':
            s['config']['policy'] = 'default_cut_selection'
        elif s['config']['policy'] == 'baseline' and s['config']['max_per_root'] == 0:
            s['config']['policy'] = 'no_cycles'
        elif s['config']['policy'] == 'baseline' and s['config']['max_per_round'] > 0:
            s['config']['policy'] = 'force{}{}'.format(s['config']['max_per_round'], s['config']['criterion'])
        elif s['config']['policy'] == 'baseline' and s['config']['max_per_round'] == -1:
            s['config']['policy'] = 'force_all_cycles'

        # create a hashable config identifier
        config = tuple([s['config'][k] for k in datasets[dataset]['config_keys']])

        # create skeleton for storing stats collected from experiments with config
        if config not in datasets[dataset]['configs'].keys():
            datasets[dataset]['configs'][config] = s['config']
            if s['config']['policy'] == 'expert' or (s['config']['policy'] == 'adaptive' and not final_adaptive):
                results[dataset][config] = {stat_key: {graph_idx: {}
                                                       for graph_idx in range(s['config']['sweep_config']['sweep']['graph_idx']['range'])}
                                            for stat_key in s['stats'].keys()}
            else:
                baselines[dataset][config] = {stat_key: {graph_idx: {}
                                                         for graph_idx in range(s['config']['sweep_config']['sweep']['graph_idx']['range'])}
                                              for stat_key in s['stats'].keys()}

        # now store the experiment results in the appropriate dictionary
        dictionary = results if s['config']['policy'] == 'expert' or (s['config']['policy'] == 'adaptive' and not final_adaptive) else baselines
        for stat_key, value in s['stats'].items():
            dictionary[dataset][config][stat_key][graph_idx][scip_seed] = value

    ##### PROCESSING RESULTS #####
    # if an experiment is missing, generate its configuration and append to missing_experiments
    # the script will generate a configuration file, and command line to run in order to
    # accomplish all the missing experiments
    for dataset in datasets.keys():
        bsl = baselines[dataset]
        res = results[dataset]
        max_lp_iterations = datasets[dataset]['max_lp_iterations']
        best_dualbound = datasets[dataset]['best_dualbound']

        ###########################################################################
        # 1. find missing experiments, and by the way,
        # store the best_dualbound and max_lp_iterations for each graph and seed
        ###########################################################################
        for dictionary in [bsl, res]:
            for config, stats in tqdm(dictionary.items(), desc='Detecting missing experiments'):
                # compute the integral of dual_bound w.r.t lp_iterations
                # report missing seeds/graphs
                missing_graph_and_seed = []
                dualbounds = stats['dualbound']
                lp_iterations = stats['lp_iterations']
                for graph_idx in datasets[dataset]['graph_idx_range']:
                    for scip_seed in datasets[dataset]['scip_seeds']:
                        if scip_seed not in dualbounds[graph_idx].keys():
                            if (graph_idx, scip_seed) not in missing_graph_and_seed:
                                experiment_config = datasets[dataset]['configs'][config].copy()
                                experiment_config['graph_idx'] = graph_idx
                                experiment_config['scip_seed'] = scip_seed
                                datasets[dataset]['missing_experiments'].append(experiment_config)
                                missing_graph_and_seed.append((graph_idx, scip_seed))
                            continue
                        # find the best dualbound achieved and the maximal lp_iterations
                        max_lp_iterations[graph_idx][scip_seed] = max(max_lp_iterations[graph_idx].get(scip_seed, 0),
                                                                      lp_iterations[graph_idx][scip_seed][-1])
                        best_dualbound[graph_idx][scip_seed] = min(best_dualbound[graph_idx].get(scip_seed, 0),
                                                                   dualbounds[graph_idx][scip_seed][-1])


        ###############################################################################################
        # 2. for each config, graph and seed, compute the dualbound integral w.r.t the lp_iterations.
        # then, compute the mean and std across all seeds within graphs,
        # and also std of stds across all graphs
        ###############################################################################################
        for dictionary in [bsl, res]:
            for config, stats in tqdm(dictionary.items(), desc='Computing dualbound integral'):
                dualbounds = stats['dualbound']
                lp_iterations = stats['lp_iterations']
                all_values = []  # all dualbound integrals to compute overall average
                all_stds = []  # all graph-wise integral std to compute std of stds
                stats['dualbound_integral'] = {}
                for graph_idx in datasets[dataset]['graph_idx_range']:
                    values = []  # dualbound integrals of the current graph to compute average and std across seeds
                    stats['dualbound_integral'][graph_idx] = {}
                    for scip_seed in dualbounds[graph_idx].keys():
                        # the integral support is [0, max_lp_iterations]
                        # TODO: check if extension is saved to the source object.
                        support_end = max_lp_iterations[graph_idx][scip_seed]
                        dualbound = dualbounds[graph_idx][scip_seed]
                        lp_iter = lp_iterations[graph_idx][scip_seed]
                        if lp_iter[-1] < support_end:
                            lp_iter.append(support_end)
                            # dualbound.append(dualbound[-1])
                            # extend all stats with their last value (constant extension)
                            for k, s in stats.items():
                                if k not in ['dualbound_integral', 'lp_iterations']:
                                    s[graph_idx][scip_seed].append(s[graph_idx][scip_seed][-1])
                        dualbound = np.array(dualbound)
                        # compute the lp iterations executed at each round to compute the dualbound_integral by Riemann sum
                        lp_iter_intervals = np.array(lp_iter)
                        lp_iter_intervals[1:] -= lp_iter_intervals[:-1]

                        integral = np.sum(dualbound * lp_iter_intervals)
                        stats['dualbound_integral'][graph_idx][scip_seed] = integral
                        values.append(integral)
                        all_values.append(integral)
                    if len(values) > 0:
                        # compute the average and std of the integral across seeds, and store in stats
                        stats['dualbound_integral'][graph_idx]['avg'] = np.mean(values)
                        stats['dualbound_integral'][graph_idx]['std'] = np.std(values)
                        all_stds.append(np.std(values))  # store std to compute at last the std of stds across all graphs

                # compute the avg and std of stds across all graphs
                if len(all_stds) > 0:
                    stats['dualbound_integral']['avg'] = np.mean(all_values)
                    stats['dualbound_integral']['std'] = np.std(all_stds)


        ##########################################################################################
        # 3. find the k-best experts for each graph according to the averaged dualbound integral.
        # in addition, find the best baseline.
        ##########################################################################################

        # list of k-best experts for each graph_idx
        best_config = []
        k_best_configs = []
        best_dualbound_integral_avg = []
        best_dualbound_integral_std = []

        # results[dataset][config][stat_key][graph_idx][seed]
        for graph_idx in tqdm(datasets[dataset]['graph_idx_range'], desc='Finding experts'):
            # insert the metrics into a long list
            configs = []
            dualbound_int_avg = []
            dualbound_int_std = []
            for config, stats in res.items():
                if stats['dualbound_integral'][graph_idx].get('avg', None) is not None:
                    configs.append(config)
                    dualbound_int_avg.append(stats['dualbound_integral'][graph_idx]['avg'])
                    dualbound_int_std.append(stats['dualbound_integral'][graph_idx]['std'])

            # sort tuples of (dualbound_int_avg, dualbound_int_std, config)
            # according to the dualbound_int_avg.
            avg_std_config = list(zip(dualbound_int_avg, dualbound_int_std, configs))
            if len(avg_std_config) > 0:
                avg_std_config.sort(key=operator.itemgetter(0))
                best = avg_std_config[0]
                best_dualbound_integral_avg.append(best[0])
                best_dualbound_integral_std.append(best[1])
                best_config.append(best[2])
                k_best_configs.append([cfg[2] for cfg in avg_std_config[:tb_k_best]])
            else:
                best_dualbound_integral_avg.append('-')
                best_dualbound_integral_std.append('-')
                # best_config.append(None)
                # k_best_configs.append(None)


        ######################################################################################
        # 5. write the summary into pandas.DataFrame
        ######################################################################################
        if csv:
            # collect the relevant baseline stats for the table
            baselines_table = {}
            for config, stats in bsl.items():
                bsl_str = 'nocycles' if datasets[dataset]['configs'][config]['max_per_root'] == 0 else '{}{}'.format(
                    datasets[dataset]['configs'][config]['max_per_round'], datasets[dataset]['configs'][config]['criterion'])
                baseline_dualbound_integral_avg = []
                baseline_dualbound_integral_std = []
                baseline_dualbound = []
                for graph_idx in datasets[dataset]['graph_idx_range']:
                    baseline_dualbound_integral_avg.append(stats['dualbound_integral'][graph_idx].get('avg', '-'))
                    baseline_dualbound_integral_std.append(stats['dualbound_integral'][graph_idx].get('std', '-'))
                    baseline_dualbound.append(np.mean([db[-1] for db in stats['dualbound'][graph_idx].values()]))
                baseline_dualbound_integral_avg.append(np.mean(baseline_dualbound_integral_avg))  # avg across graphs
                baseline_dualbound_integral_std.append(np.mean(baseline_dualbound_integral_std))  # std of stds across graphs
                baseline_dualbound.append(np.mean(baseline_dualbound))  # avg across graphs
                baselines_table[config] = {bsl_str + ' integral avg': baseline_dualbound_integral_avg,
                                           bsl_str + ' integral std': baseline_dualbound_integral_std,
                                           bsl_str + ' dualbound': baseline_dualbound}

            optimal_values = [datasets[dataset]['optimal_values'][graph_idx] for graph_idx in datasets[dataset]['graph_idx_range']]
            optimal_values.append(np.mean(optimal_values))
            best_dualbound_avg = []
            for graph_idx in datasets[dataset]['graph_idx_range']:
                # append the average across seeds
                best_dualbound_avg.append(np.mean([db[-1] for db in res[best_config[graph_idx]]['dualbound'][graph_idx].values()]))
            best_dualbound_avg.append(np.mean(best_dualbound_avg))  # append the average across graphs

            # create a table dictionary: keys - columns name, values - columns values.
            # row indices will be the graph_idx + 'avg' as the bottomline
            table_dict = {
                'Best integral avg': best_dualbound_integral_avg + [np.mean(best_dualbound_integral_avg) if '-' not in best_dualbound_integral_avg else '-'],
                'Best integral std': best_dualbound_integral_std + [np.mean(best_dualbound_integral_std) if '-' not in best_dualbound_integral_std else '-'],
                'Best dualbound avg': best_dualbound_avg,
            }
            for d in baselines_table.values():
                for k, v in d.items():
                    table_dict[k] = v

            for k in datasets[dataset]['configs'][best_config[0]]['sweep_config']['sweep'].keys():
                table_dict[k] = []

            for graph_idx, bc in enumerate(best_config):
                best_hparams = datasets[dataset]['configs'][bc]['sweep_config']['sweep']
                for k, v in best_hparams.items():
                    table_dict[k].append(v)
            # append empty row for the 'avg' row
            for k, v in best_hparams.items():
                table_dict[k].append('-')

            tables_dir = os.path.join(dstdir, 'tables')
            if not os.path.exists(tables_dir):
                os.makedirs(tables_dir)
            csv_file = os.path.join(tables_dir, dataset + '_results.csv')
            df = pd.DataFrame(data=table_dict, index=datasets[dataset]['graph_idx_range'] + ['avg'])
            df.to_csv(csv_file, float_format='%.3f')
            print('Experiment summary saved to {}'.format(csv_file))
            # latex_str = df.to_latex(float_format='%.3f')
            # print(latex_str)

        ######################################################################################
        # 6. report on missing experiments and create a terminal command to accomplish them
        ######################################################################################
        missing_experiments_dir = os.path.join(dstdir, 'missing_experiments')
        if not os.path.exists(missing_experiments_dir):
            os.makedirs(missing_experiments_dir)
        if len(datasets[dataset]['missing_experiments']) > 0:
            missing_experiments_file = os.path.join(missing_experiments_dir, dataset + '_missing_experiments.pkl')
            with open(missing_experiments_file, 'wb') as f:
                pickle.dump(datasets[dataset]['missing_experiments'], f)
            print('WARNING: missing experiments saved to {}'.format(missing_experiments_file))
            print('Statistics might not be accurate.')
            print('To complete experiments, run the following command inside experiments/ folder:')
            complete_experiment_commandline = 'python complete_experiment.py --experiment {} --config-file {} --log-dir {}'.format(
                datasets[dataset]['experiment'],
                os.path.abspath(missing_experiments_file),
                os.path.abspath(rootdir))
            print(complete_experiment_commandline)
        else:
            complete_experiment_commandline = None

        ######################################################################################
        # 7. generate tensorboard
        ######################################################################################
        # Generate tensorboard for the K-best configs
        if tensorboard:
            tensorboard_dir = os.path.join(dstdir, 'tensorboard', dataset)
            # Generate hparams tab for the k-best-on-average configs, and in addition for the baseline.
            # The hparams specify for each graph and seed some more stats.
            for graph_idx, config_list in tqdm(enumerate(k_best_configs), desc='Generating tensorboard hparams'):
                for place, config in enumerate(config_list):
                    stats = res[config]
                    hparams = datasets[dataset]['configs'][config].copy()
                    writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, 'expert{}'.format(place+1), str_hparams(hparams)))
                    hparams.pop('data_abspath', None)
                    hparams.pop('sweep_config', None)
                    hparams['graph_idx'] = graph_idx
                    metric_lists = {k: [] for k in stats.keys()}
                    # plot hparams for each seed
                    for scip_seed in stats['dualbound'][graph_idx].keys():
                        hparams['scip_seed'] = scip_seed
                        metrics = {k: v[graph_idx][scip_seed][-1] for k, v in stats.items() if k != 'dualbound_integral'}
                        dualbound = np.array(stats['dualbound'][graph_idx][scip_seed])
                        lp_iter_intervals = np.array(stats['lp_iterations'][graph_idx][scip_seed])
                        # compute the lp iterations executed at each round to compute the dualbound_integral by Riemann sum
                        lp_iter_intervals[1:] -= lp_iter_intervals[:-1]
                        metrics['dualbound_integral'] = np.sum(dualbound * lp_iter_intervals)
                        metrics['cycles_sepa_time'] = metrics['cycles_sepa_time'] / metrics['solving_time']
                        # if extended, take the actual lp_iterations done
                        if dualbound[-1] == dualbound[-2]:
                            metrics['lp_iterations'] = stats['lp_iterations'][graph_idx][scip_seed][-2]
                        # override the gap according to the optimal value of the graph
                        optimal_dualbound = datasets[dataset]['optimal_values'][graph_idx]
                        metrics['gap'] = (metrics['dualbound'] - optimal_dualbound) / optimal_dualbound
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

            # add hparams for the baseline
            for config, stats in bsl.items():
                hparams = datasets[dataset]['configs'][config].copy()
                hparams.pop('data_abspath', None)
                hparams.pop('sweep_config', None)
                writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, 'baselines', str_hparams(hparams)))
                for graph_idx in datasets[dataset]['graph_idx_range']:
                    hparams['graph_idx'] = graph_idx
                    metric_lists = {k: [] for k in stats.keys()}
                    # plot hparams for each seed
                    for scip_seed in datasets[dataset]['scip_seeds']:  #stats['dualbound'][graph_idx].keys():
                        hparams['scip_seed'] = scip_seed
                        metrics = {k: v[graph_idx][scip_seed][-1] for k, v in stats.items() if k != 'dualbound_integral'}
                        dualbound = np.array(stats['dualbound'][graph_idx][scip_seed])
                        lp_iter_intervals = np.array(stats['lp_iterations'][graph_idx][scip_seed])
                        # compute the lp iterations executed at each round to compute the dualbound_integral by Riemann sum
                        lp_iter_intervals[1:] -= lp_iter_intervals[:-1]
                        metrics['dualbound_integral'] = np.sum(dualbound * lp_iter_intervals)
                        metrics['cycles_sepa_time'] = metrics['cycles_sepa_time'] / metrics['solving_time']
                        # if extended, take the actual lp_iterations done
                        if dualbound[-1] == dualbound[-2]:
                            metrics['lp_iterations'] = stats['lp_iterations'][graph_idx][scip_seed][-2]
                        # override the gap according to the optimal value of the graph
                        optimal_dualbound = datasets[dataset]['optimal_values'][graph_idx]
                        metrics['gap'] = (metrics['dualbound'] - optimal_dualbound) / optimal_dualbound

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

            ####################################################
            # add plots for the best config
            # each graph plot separately.
            ####################################################
            for graph_idx, config_list in tqdm(enumerate(k_best_configs), desc='Generating tensorboard scalars'):
                for place, config in enumerate(config_list):
                    stats = res[config]
                    hparams = datasets[dataset]['configs'][config]
                    for scip_seed, db in stats['dualbound'][graph_idx].items():
                        writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, 'expert{}'.format(place+1), str_hparams(hparams),
                                                                    'g{}-seed{}'.format(graph_idx, scip_seed)))
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
                            writer.add_scalar(tag='Dualbound_vs_LP_Rounds/g{}'.format(graph_idx),
                                              scalar_value=records['dualbound'],
                                              global_step=records['lp_rounds'],
                                              walltime=records['solving_time'])

                            writer.add_scalar(tag='Cycles_Applied_vs_LP_round/g{}'.format(graph_idx),
                                              scalar_value=records['cycle_ncuts_applied'],
                                              global_step=records['lp_rounds'],
                                              walltime=records['solving_time'])
                            writer.add_scalar(tag='Cycles_Generated_vs_LP_round/g{}'.format(graph_idx),
                                              scalar_value=records['cycle_ncuts'],
                                              global_step=records['lp_rounds'],
                                              walltime=records['solving_time'])

                        writer.close()

            # add plots of metrics vs time for the baseline
            for bsl_idx, (config, stats) in enumerate(bsl.items()):
                hparams = datasets[dataset]['configs'][config]
                for graph_idx in stats['dualbound'].keys():
                    for scip_seed, db in stats['dualbound'][graph_idx].items():
                        writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, 'baselines', str_hparams(hparams),
                                                                    'g{}-seed{}'.format(graph_idx, scip_seed)))

                        for lp_round in range(len(db)):
                            records = {k: v[graph_idx][scip_seed][lp_round] for k, v in stats.items() if
                                       k != 'dualbound_integral'}
                            # dualbound vs. lp iterations
                            writer.add_scalar(tag='Dualbound_vs_LP_Iterations/g{}'.format(graph_idx),
                                              scalar_value=records['dualbound'],
                                              global_step=records['lp_iterations'],
                                              walltime=records['solving_time'])
                            writer.add_scalar(tag='Dualbound_vs_LP_Rounds/g{}'.format(graph_idx),
                                              scalar_value=records['dualbound'],
                                              global_step=records['lp_rounds'],
                                              walltime=records['solving_time'])

                            # dualbound vs. cycles applied
                            if 'cycle_ncuts_applied' in records.keys():
                                writer.add_scalar(tag='Dualbound_vs_Cycles_Applied/g{}'.format(graph_idx),
                                                  scalar_value=records['dualbound'],
                                                  global_step=records['cycle_ncuts_applied'],
                                                  walltime=records['solving_time'])
                                writer.add_scalar(tag='Cycles_Applied_vs_LP_round/g{}'.format(graph_idx),
                                                  scalar_value=records['cycle_ncuts_applied'],
                                                  global_step=records['lp_rounds'],
                                                  walltime=records['solving_time'])
                                writer.add_scalar(tag='Cycles_Generated_vs_LP_round/g{}'.format(graph_idx),
                                                  scalar_value=records['cycle_ncuts'],
                                                  global_step=records['lp_rounds'],
                                                  walltime=records['solving_time'])

                            # dualbound vs. total cuts applied
                            writer.add_scalar(tag='Dualbound_vs_Total_Cuts_Applied/g{}'.format(graph_idx),
                                              scalar_value=records['dualbound'],
                                              global_step=records['total_ncuts_applied'],
                                              walltime=records['solving_time'])

                        writer.close()

            # plot the optimal value as constant on the dualbound plots
            for graph_idx in datasets[dataset]['graph_idx_range']:
                writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, 'optimal-g{}'.format(graph_idx)))
                v = datasets[dataset]['optimal_values'][graph_idx]
                support_end = max(list(max_lp_iterations[graph_idx].values()))
                # dualbound vs. lp iterations
                writer.add_scalar(tag='Dualbound_vs_LP_Iterations/g{}'.format(graph_idx),
                                  scalar_value=v, global_step=0, walltime=0)
                writer.add_scalar(tag='Dualbound_vs_LP_Iterations/g{}'.format(graph_idx),
                                  scalar_value=v, global_step=support_end, walltime=400)
                # dualbound vs. cycles applied
                writer.add_scalar(tag='Dualbound_vs_Cycles_Applied/g{}'.format(graph_idx),
                                  scalar_value=v, global_step=0, walltime=0)
                writer.add_scalar(tag='Dualbound_vs_Cycles_Applied/g{}'.format(graph_idx),
                                  scalar_value=v, global_step=2000, walltime=400)
                # dualbound vs. total cuts applied
                writer.add_scalar(tag='Dualbound_vs_Total_Cuts_Applied/g{}'.format(graph_idx),
                                  scalar_value=v, global_step=0, walltime=0)
                writer.add_scalar(tag='Dualbound_vs_Total_Cuts_Applied/g{}'.format(graph_idx),
                                  scalar_value=v, global_step=2000, walltime=400)
                writer.close()
            print('Tensorboard events written to ' + tensorboard_dir)
            print('To open tensorboard tab on web browser, run in terminal the following command:')
            tensorboard_commandline = 'tensorboard --logdir ' + os.path.abspath(tensorboard_dir) + ' --port 6007'
            print('tensorboard --logdir ' + os.path.abspath(tensorboard_dir) + ' --port 6007')
        else:
            tensorboard_commandline = None

        if plot:
            def plot_y_vs_x(y, x, records, hparams={}, fignum=1, xstr=None, ystr=None, title=None, label=None, style=None):
                plt.figure(fignum)
                xstr = ' '.join([s[0].upper() + s[1:] for s in x.split('_')]) if xstr is None else xstr
                ystr = ' '.join([s[0].upper() + s[1:] for s in y.split('_')]) if ystr is None else ystr
                title = ystr + ' vs. ' + xstr if title is None else title
                label = hparams['policy'] if label is None else label
                style = {'default_cut_selection': '--',
                         'expert': '-',
                         'adaptive': '-',
                         'force10most_effective': '-.'}.get(hparams['policy'], ':') if style is None else style
                plt.plot(records[x], records[y], style, label=label)
                plt.title(title)
                plt.xlabel(xstr)
                plt.ylabel(ystr)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2, borderaxespad=0.)

            fig_filenames = {1: 'dualbound_vs_lp_rounds.png',
                             2: 'dualbound_vs_lp_iterations.png',
                             3: 'dualbound_vs_solving_time.png'}
            figcnt = 4
            ####################################################
            # add plots for the best config
            # each graph plot separately.
            ####################################################
            for graph_idx, config_list in tqdm(enumerate(k_best_configs), desc='Generating plots'):
                for place, config in enumerate(config_list):
                    stats = res[config]
                    hparams = datasets[dataset]['configs'][config]
                    for scip_seed, db in stats['dualbound'][graph_idx].items():
                        records = {k: v[graph_idx][scip_seed] for k, v in stats.items() if k != 'dualbound_integral'}
                        plot_y_vs_x('dualbound', 'lp_rounds', records, hparams, 1)
                        plot_y_vs_x('dualbound', 'lp_iterations', records, hparams, 2)
                        plot_y_vs_x('dualbound', 'solving_time', records, hparams, 3)
                        plot_y_vs_x('cycle_ncuts_applied', 'lp_rounds', records, hparams, figcnt, ystr='# Cuts', title=hparams['policy'], label='cuts applied', style='-')
                        plot_y_vs_x('cycle_ncuts', 'lp_rounds', records, hparams, figcnt, ystr='# Cuts', title=hparams['policy'], label='cuts generated', style='-')
                        fig_filenames[figcnt] = '{}-g{}-scipseed{}.png'.format(hparams['policy'], graph_idx, scip_seed)
                        figcnt += 1

            # add plots of metrics vs time for the baseline
            for bsl_idx, (config, stats) in enumerate(bsl.items()):
                hparams = datasets[dataset]['configs'][config]
                for graph_idx in stats['dualbound'].keys():
                    for scip_seed, db in stats['dualbound'][graph_idx].items():
                        records = {k: v[graph_idx][scip_seed] for k, v in stats.items() if k != 'dualbound_integral'}
                        plot_y_vs_x('dualbound', 'lp_rounds', records, hparams, 1)
                        plot_y_vs_x('dualbound', 'lp_iterations', records, hparams, 2)
                        plot_y_vs_x('dualbound', 'solving_time', records, hparams, 3)
                        plot_y_vs_x('cycle_ncuts_applied', 'lp_rounds', records, hparams, figcnt, ystr='# Cuts', title=hparams['policy'], label='cuts applied', style='-')
                        plot_y_vs_x('cycle_ncuts', 'lp_rounds', records, hparams, figcnt, ystr='# Cuts', title=hparams['policy'], label='cuts generated', style='-')
                        fig_filenames[figcnt] = '{}-g{}-scipseed{}.png'.format(hparams['policy'], graph_idx, scip_seed)
                        figcnt += 1

            # plot the optimal value as constant on the dualbound plots
            for graph_idx in datasets[dataset]['graph_idx_range']:
                v = datasets[dataset]['optimal_values'][graph_idx]
                support_end = max(list(max_lp_iterations[graph_idx].values()))
                records = {'dualbound': [v, v], 'lp_iterations': [0, support_end], 'solving_time': [0, 90], 'lp_rounds': [0, 100]}
                plot_y_vs_x('dualbound', 'lp_rounds', records, {}, 1, label='optimal', style='-k')
                plot_y_vs_x('dualbound', 'lp_iterations', records, {}, 2, label='optimal', style='-k')
                plot_y_vs_x('dualbound', 'solving_time', records, {}, 3, label='optimal', style='-k')

            # save all figures
            for fignum, filename in fig_filenames.items():
                plots_dir = os.path.join(dstdir, 'plots')
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)
                filepath = os.path.join(plots_dir, filename)
                plt.figure(fignum)
                plt.savefig(filepath, bbox_inches='tight')
            print('Saved all plots to: ', plots_dir)

        # return analysis
        best_policy = [datasets[dataset]['configs'][bc] for bc in best_config]
        analysis[dataset]['best_policy'] = best_policy
        analysis[dataset]['complete_experiment_commandline'] = complete_experiment_commandline
        analysis[dataset]['tensorboard_commandline'] = tensorboard_commandline

    print('finished analysis!')
    return analysis

if __name__ == '__main__':
    NOW = str(datetime.now())[:-7].replace(' ', '.').replace(':', '-').replace('.', '/')
    parser = ArgumentParser()
    parser.add_argument('--rootdir', type=str, default='results/', help='path to experiment results root dir')
    parser.add_argument('--dstdir', type=str, default='analysis/' + NOW, help='path to store tables, tensorboard etc.')
    parser.add_argument('--filepattern', type=str, default='experiment_results.pkl', help='pattern of pickle files')
    parser.add_argument('--tensorboard', action='store_true', help='generate tensorboard folder in <dstdir>/tb')
    parser.add_argument('--csv', action='store_true', help='print csv table to <dstdir>/tables')

    parser.add_argument('--tb-k-best', type=int, help='generate tensorboard for the k best configs (and baseline)',
                        default=1)
    parser.add_argument('--support-partition', type=int,
                        help='number of support partitions to compute the dualbound integral', default=4)
    parser.add_argument('--generate-experts', action='store_true', help='save experts configs to <dstdir>/experts')
    parser.add_argument('--final-adaptive', action='store_true', help='include "adaptive" policy with baselines')
    parser.add_argument('--plot', action='store_true', help='generates matplotlib figures')

    args = parser.parse_args()
    analyze_results(rootdir=args.rootdir, dstdir=args.dstdir, filepattern=args.filepattern,
                    tensorboard=args.tensorboard, tb_k_best=args.tb_k_best, csv=args.csv, final_adaptive=args.final_adaptive, plot=args.plot)
