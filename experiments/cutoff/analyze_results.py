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

def analyze_results(rootdir='results', dstdir='analysis', filepattern='experiment_results.pkl',
                    tensorboard=False, tb_k_best=1, csv=False, final_adaptive=False, plot=False,
                    starting_policies_abspath=None, avg=False):
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
            tensorboard_dir = os.path.join(dstdir, 'tensorboard')
            # Generate hparams tab for the k-best-on-average configs, and in addition for the baseline.
            # The hparams specify for each graph and seed some more stats.
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
                                              scalar_value=records['cuts_applied'],
                                              global_step=records['lp_rounds'],
                                              walltime=records['solving_time'])
                            writer.add_scalar(tag='Cycles_Applied_Normalized_vs_LP_round/g{}'.format(graph_idx),
                                              scalar_value=records['cuts_applied_normalized'],
                                              global_step=records['lp_rounds'],
                                              walltime=records['solving_time'])
                            writer.add_scalar(tag='Cycles_Generated_vs_LP_round/g{}'.format(graph_idx),
                                              scalar_value=records['cuts_generated'],
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
                                                  scalar_value=records['cuts_applied'],
                                                  global_step=records['lp_rounds'],
                                                  walltime=records['solving_time'])
                                writer.add_scalar(tag='Cycles_Applied_Normalized_vs_LP_round/g{}'.format(graph_idx),
                                                  scalar_value=records['cuts_applied_normalized'],
                                                  global_step=records['lp_rounds'],
                                                  walltime=records['solving_time'])
                                writer.add_scalar(tag='Cycles_Generated_vs_LP_round/g{}'.format(graph_idx),
                                                  scalar_value=records['cuts_generated'],
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
            def plot_y_vs_x(y, x, records, hparams={}, fignum=1, subplot=0, xstr=None, ystr=None, title=None, label=None, style=None, ncol=1, ylim=None):
                plt.figure(fignum)
                if subplot > 0:
                    plt.subplot(subplot)
                xstr = ' '.join([s[0].upper() + s[1:] for s in x.split('_')]) if xstr is None else xstr
                ystr = ' '.join([s[0].upper() + s[1:] for s in y.split('_')]) if ystr is None else ystr
                title = ystr + ' vs. ' + xstr if title is None else title
                label = hparams['policy'] if label is None else label
                style = {'default_cut_selection': '--',
                         'expert': '-',
                         'adaptive': '-',
                         'force10strong': '-.'}.get(hparams['policy'], ':') if style is None else style
                plt.plot(records[x], records[y], style, label=label)
                plt.title(title)
                plt.xlabel(xstr)
                plt.ylabel(ystr)
                if ylim is not None:
                    plt.ylim(ylim)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=ncol, borderaxespad=0.)

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
                        plot_y_vs_x('cuts_applied', 'lp_rounds', records, hparams, figcnt, ystr='# Cuts', title=hparams['policy'], label='cuts applied', style='-')
                        plot_y_vs_x('cuts_generated', 'lp_rounds', records, hparams, figcnt, ystr='# Cuts', title=hparams['policy'], label='cuts generated', style='-')
                        plot_y_vs_x('maxcutsroot', 'lp_rounds', records, hparams, figcnt, ystr='# Cuts', title=hparams['policy'], label='maxcutsroot', style='--k', ylim=[0, hparams['graph_size']])
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
                        plot_y_vs_x('cuts_applied', 'lp_rounds', records, hparams, figcnt, ystr='# Cuts', title=hparams['policy'], label='cuts applied', style='-')
                        plot_y_vs_x('cuts_generated', 'lp_rounds', records, hparams, figcnt, ystr='# Cuts', title=hparams['policy'], label='cuts generated', style='-')
                        plot_y_vs_x('maxcutsroot', 'lp_rounds', records, hparams, figcnt, ystr='# Cuts', title=hparams['policy'], label='maxcutsroot', style='--k', ylim=[0, hparams['graph_size']])
                        fig_filenames[figcnt] = '{}-g{}-scipseed{}.png'.format(hparams['policy'], graph_idx, scip_seed)
                        figcnt += 1

            # plot the optimal value as constant on the dualbound plots
            for graph_idx in datasets[dataset]['graph_idx_range']:
                v = datasets[dataset]['optimal_values'][graph_idx]
                support_end = max(list(max_lp_iterations[graph_idx].values()))
                records = {'dualbound': [v, v], 'lp_iterations': [0, support_end], 'solving_time': [0, 90], 'lp_rounds': [0, 1000]}
                plot_y_vs_x('dualbound', 'lp_rounds', records, {}, 1, label='optimal objective', style='-k', ncol=2)
                plot_y_vs_x('dualbound', 'lp_iterations', records, {}, 2, label='optimal objective', style='-k', ncol=2)
                plot_y_vs_x('dualbound', 'solving_time', records, {}, 3, label='optimal objective', style='-k', ncol=2)

            # save all figures
            for fignum, filename in fig_filenames.items():
                plots_dir = os.path.join(dstdir, 'plots')
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)
                filepath = os.path.join(plots_dir, filename)
                plt.figure(fignum)
                plt.tight_layout()
                plt.savefig(filepath, bbox_inches='tight')
            print('Saved all plots to: ', plots_dir)

        # return analysis
        best_policy = [datasets[dataset]['configs'][bc] for bc in best_config]
        analysis[dataset]['best_policy'] = best_policy
        analysis[dataset]['complete_experiment_commandline'] = complete_experiment_commandline
        analysis[dataset]['tensorboard_commandline'] = tensorboard_commandline
        analysis[dataset]['dualbound_integral_average'] = dbi_avg_dict if avg else None

    print('finished analysis!')
    return analysis


def average_dualbound_integral(rootdir, dstdir, n_iter):
    """
    search for iter<>results inside rootdir.
    at each subfolder run analysis, and store the dual bound integral average.
    accumulate results in a table, and print everything to a csv.
    :param rootdir:
    :param dstdir:
    :param n_iter:
    :return:
    """
    results = None
    for idx in range(n_iter):
        print('##### ANALYZING ITERATION {} #####'.format(idx))
        iterdir = os.path.join(rootdir, 'iter{}results'.format(idx))
        dbi_avg = list(analyze_results(rootdir=iterdir, avg=True).values())[0]['dualbound_integral_average']
        if results is None:
            for k, values in dbi_avg.items():
                results[k] = {}
                for v in values.keys():
                    results[k][v] = []
        for k, values in dbi_avg.items():
            for val, dbi in values.items():
                results[k][val].append(dbi)
    # print each key in results to a separated file dstdir/<k>.csv
    for param, d in results.items():
        csv_file = os.path.join(dstdir, param+'.csv')
        df = pd.DataFrame(data=d)
        df.to_csv(csv_file, float_format='%.2f')
    print('Saved all csv files to {}'.format(dstdir))


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
    parser.add_argument('--starting-policies-abspath', type=str, default='', help='pattern of pickle files')
    args = parser.parse_args()
    if args.starting_policies_abspath == '':
        args.starting_policies_abspath = os.path.join(args.rootdir, 'starting_policies.pkl')
    analyze_results(rootdir=args.rootdir, dstdir=args.dstdir, filepattern=args.filepattern,
                    tensorboard=args.tensorboard, tb_k_best=args.tb_k_best, csv=args.csv,
                    final_adaptive=args.final_adaptive, plot=args.plot, starting_policies_abspath=args.starting_policies_abspath)
