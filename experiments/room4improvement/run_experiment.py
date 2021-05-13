from utils.scip_models import mvc_model, CSBaselineSepa, set_aggresive_separation, CSResetSepa, maxcut_mccormic_model
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from utils.functions import get_normalized_areas, truncate
from tqdm import tqdm
import pickle
import pandas as pd
from argparse import ArgumentParser
import os
parser = ArgumentParser()
parser.add_argument('--rootdir', type=str, default='results', help='rootdir to store results')
args = parser.parse_args()
np.random.seed(777)
ROOTDIR = args.rootdir
if not os.path.isdir(ROOTDIR):
    os.makedirs(ROOTDIR)


print('############### generating data ###############')
# todo make it distributed
if not os.path.exists(os.path.join(ROOTDIR, 'data.pkl')):
    data = {'mvc': {}, 'maxcut': {}}
    # mvc
    graph_sizes = [60, 100, 150]
    densities = [0.25, 0.15, 0.1]
    for gs, density in tqdm(zip(graph_sizes, densities), desc='generating graphs for MVC'):
        g = nx.erdos_renyi_graph(n=gs, p=density, directed=False)
        nx.set_node_attributes(g, {i: np.random.random() for i in g.nodes}, 'c')
        model, _ = mvc_model(g, use_random_branching=False, allow_restarts=True, use_heuristics=True)
        # model.hideOutput(True)
        model.optimize()
        assert model.getGap() == 0
        stats = {}
        stats['time'] = model.getSolvingTime()
        stats['lp_iterations'] = model.getNLPIterations()
        stats['nodes'] = model.getNNodes()
        stats['applied'] = model.getNCutsApplied()
        stats['lp_rounds'] = model.getNLPs()
        stats['optval'] = model.getObjVal()
        data['mvc'][gs] = (g, stats)

    # maxcut
    graph_sizes = [40, 70, 100]
    ms = [15, 15, 15]
    for gs, m in tqdm(zip(graph_sizes, ms), desc='generating graphs for MAXCUT'):
        g = nx.barabasi_albert_graph(n=gs, m=m)
        nx.set_edge_attributes(g, {e: np.random.random() for e in g.edges}, 'weight')
        model, _, _ = maxcut_mccormic_model(g, use_random_branching=False, allow_restarts=True, use_cycles=False)
        # model.hideOutput(True)
        model.optimize()
        assert model.getGap() == 0
        stats = {}
        stats['time'] = model.getSolvingTime()
        stats['lp_iterations'] = model.getNLPIterations()
        stats['nodes'] = model.getNNodes()
        stats['applied'] = model.getNCutsApplied()
        stats['lp_rounds'] = model.getNLPs()
        stats['optval'] = model.getObjVal()
        data['maxcut'][gs] = (g, stats)


    print(f'saving data to: {ROOTDIR}/data.pkl')
    with open(f'{ROOTDIR}/data.pkl', 'wb') as f:
        pickle.dump(data, f)
else:
    print(f'loading data from: {ROOTDIR}/data.pkl')
    with open(f'{ROOTDIR}/data.pkl', 'rb') as f:
        data = pickle.load(f)


if not os.path.exists(f'{ROOTDIR}/scip_tuned_best_config.pkl'):
    print('run scip tuned baseline first, then re-run again.')
    exit(0)

with open(f'{ROOTDIR}/scip_tuned_best_config.pkl', 'rb') as f:
    scip_tuned_best_config = pickle.load(f)

if not os.path.exists(f'{ROOTDIR}/scip_adaptive_params.pkl'):
    print('run scip adaptive baseline first, then re-run again.')
    exit(0)

with open(f'{ROOTDIR}/scip_adaptive_params.pkl', 'rb') as f:
    scip_adaptive_params = pickle.load(f)


# todo run here also tuned and adaptive policies
print('############### run all baselines on local machine to compare solving time ###############')
# run default, 15-random, 15-most-violated and all-cuts baselines
SEEDS = [46, 72, 101]
problems = ['mvc', 'maxcut']
baselines = ['default', '15_random', '15_most_violated', 'all_cuts', 'tuned', 'adaptive']
if not os.path.exists(f'{ROOTDIR}/all_baselines_results.pkl'):
    results = {p: {b: {} for b in baselines} for p in problems}
    for problem, graphs in data.items():
        for baseline in tqdm(baselines, desc='run simple baselines'):
            graphs = data[problem]
            for graph_size, (g, info) in graphs.items():
                results[problem][baseline][graph_size] = {}
                for seed in SEEDS:
                    if problem == 'mvc':
                        model, _ = mvc_model(g)
                        lp_iterations_limit = 1500
                    elif problem == 'maxcut':
                        model, _, _ = maxcut_mccormic_model(g)
                        lp_iterations_limit = {40: 5000, 70: 7000, 100: 10000}.get(graph_size)
                    else:
                        raise ValueError
                    set_aggresive_separation(model)
                    sepa_params = {'lp_iterations_limit': lp_iterations_limit,
                                   'policy': baseline,
                                   'reset_maxcuts': 100,
                                   'reset_maxcutsroot': 100}
                    if baseline == 'tuned':
                        # set tuned params
                        tuned_params = scip_tuned_best_config[problem][graph_size][seed]
                        sepa_params.update(tuned_params)

                    if baseline == 'adaptive':
                        # set adaptive param lists
                        adapted_param_list = scip_adaptive_params[problem][graph_size][seed]
                        adapted_params = {
                            'objparalfac': {},
                            'dircutoffdistfac': {},
                            'efficacyfac': {},
                            'intsupportfac': {},
                            'maxcutsroot': {},
                            'minorthoroot': {}
                        }
                        for round_idx, kvlist in enumerate(adapted_param_list):
                            param_dict = {k: v for k, v in kvlist}
                            for k in adapted_params.keys():
                                adapted_params[k][round_idx] = param_dict[k]
                        sepa_params.update(adapted_params)
                        sepa_params['default_separating_params'] = scip_tuned_best_config[problem][graph_size][seed]

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
                    model.setBoolParam('randomization/permutevars', True)
                    model.setIntParam('randomization/permutationseed', seed)
                    model.setIntParam('randomization/randomseedshift', seed)
                    model.hideOutput(True)
                    model.optimize()
                    sepa.update_stats()
                    stats = sepa.stats
                    stats['db_auc'] = sum(get_normalized_areas(t=stats['lp_iterations'], ft=stats['dualbound'], t_support=lp_iterations_limit, reference=info['optval']))
                    results[problem][baseline][graph_size][seed] = stats
    with open(f'{ROOTDIR}/all_baselines_results.pkl', 'wb') as f:
        pickle.dump(results, f)
else:
    with open(f'{ROOTDIR}/all_baselines_results.pkl', 'rb') as f:
        results = pickle.load(f)


print('############### analyzing results ###############')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

adaptive_params_dict = {}
for problem, baselines in results.items():
    columns = [gs for gs in data[problem].keys()]
    summary = {baseline: [] for baseline in baselines.keys()}
    for baseline, baseline_results in baselines.items():
        for graph_size, seeds in baseline_results.items():
            # for MVC shorten the support to 100 lp iters.
            if problem == 'mvc':
                lpiter_support = 700
                db_aucs = []
                db_auc_imps = []
                for seed, stats in seeds.items():
                    lpiter, db = truncate(t=stats['lp_iterations'], ft=stats['dualbound'], support=lpiter_support, interpolate=True)
                    db_auc = sum(get_normalized_areas(t=lpiter, ft=db, t_support=lpiter_support, reference=data[problem][graph_size][1]['optval']))
                    db_aucs.append(db_auc)
                    default_lpiter, default_db = truncate(t=baselines['default'][graph_size][seed]['lp_iterations'], ft=baselines['default'][graph_size][seed]['dualbound'], support=lpiter_support, interpolate=True)
                    db_auc_imps.append(db_auc / sum(get_normalized_areas(t=default_lpiter, ft=default_db, t_support=lpiter_support, reference=data[problem][graph_size][1]['optval'])))
                db_aucs = np.array(db_aucs)
                db_auc_imps = np.array(db_auc_imps)
            else:
                db_aucs = np.array([stats['db_auc'] for stats in seeds.values()])
                db_auc_imps = db_aucs / np.array([baselines['default'][graph_size][seed]['db_auc'] for seed in seeds.keys()])
            summary[baseline].append('{:.4f}{}{:.4f}({:.3f})'.format(db_aucs.mean(), u"\u00B1", db_aucs.std(), db_auc_imps.mean()))
    df = pd.DataFrame.from_dict(summary, orient='index', columns=columns)
    print(f'{"#"*40} {problem} {"#"*40}')
    print(df)
    csvfile = f'{ROOTDIR}/{problem}_baselines.csv'
    df.to_csv(csvfile)
    print(f'saved {problem} csv to: {csvfile}')

    # print adaptive and tuned params to csv
    adaptive_params_dict[problem] = {}
    for graph_size, seeds in baselines['adaptive'].items():
        adaptive_params_dict[problem][graph_size] = {}
        for seed in seeds.keys():
            adapted_param_list = scip_adaptive_params[problem][graph_size][seed]
            adapted_params = {
                'objparalfac': {},
                'dircutoffdistfac': {},
                'efficacyfac': {},
                'intsupportfac': {},
                'maxcutsroot': {},
                'minorthoroot': {}
            }
            for round_idx, kvlist in enumerate(adapted_param_list):
                param_dict = {k: v for k, v in kvlist}
                for k in adapted_params.keys():
                    adapted_params[k][round_idx] = param_dict[k]
            adaptive_params_dict[problem][graph_size][seed] = adapted_params
        columns = ['objparalfac', 'dircutoffdistfac', 'efficacyfac', 'intsupportfac', 'maxcutsroot', 'minorthoroot']
        summary = {}
        for round_idx in range(len(adapted_param_list)):
            row = []
            for col in columns:
                for seed in seeds.keys():
                    row.append(adaptive_params_dict[problem][graph_size][seed][col][round_idx])
            summary[round_idx] = row
        # append last row for the tuned params
        tuned_params_row = []
        for col in columns:
            for seed in seeds.keys():
                tuned_params_row.append(scip_tuned_best_config[problem][graph_size][seed][col])
        summary['tuned'] = tuned_params_row
        columns *= 3
        df = pd.DataFrame.from_dict(summary, orient='index', columns=columns)
        csvfile = f'{ROOTDIR}/{problem}_{graph_size}_adaptive_tuned_params.csv'
        df.to_csv(csvfile)
        print(f'saved {problem}-{graph_size} adaptive and tuned params table to: {csvfile}')


colors = {'default': 'b', '15_random': 'gray', '15_most_violated': 'purple', 'all_cuts': 'orange', 'tuned': 'r', 'adaptive': 'g'}
for problem in results.keys():
    fig1, axes1 = plt.subplots(3, 3, sharex='col')  # dual bound vs. lp iterations
    fig2, axes2 = plt.subplots(3, 3, sharex='col')  # dual bound vs. solving time

    for col, graph_size in enumerate(data[problem].keys()):
        fig3, axes3 = plt.subplots(3, 1)  # ncuts_applied vs. round idx
        for row, seed in enumerate(SEEDS):
            for baseline in results[problem].keys():
                if (baseline in ['default', 'all_cuts'] and col == 0) or baseline in ['15_random', '15_most_violated'] and col == 1 or (baseline in ['tuned', 'adaptive'] and col == 2):
                    label = baseline
                else:
                    label = None
                stats = results[problem][baseline][graph_size][seed]
                axes1[row, col].plot(stats['lp_iterations'], stats['dualbound'], colors[baseline], label=label)
                axes2[row, col].plot(stats['solving_time'], stats['dualbound'], colors[baseline], label=label)
                if baseline in ['default', 'tuned', 'adaptive']:
                    ncuts_applied_cumsum = np.array(stats['ncuts_applied'])
                    ncuts_applied = ncuts_applied_cumsum[1:] - ncuts_applied_cumsum[:-1]
                    maxcutsroot = {'default': 2000,
                                   'tuned': scip_tuned_best_config[problem][graph_size][seed]['maxcutsroot'],
                                   'adaptive': adaptive_params_dict[problem][graph_size][seed]['maxcutsroot']}.get(baseline)
                    if baseline == 'adaptive':
                        pad_len = len(ncuts_applied) - len(maxcutsroot)
                        if pad_len > 0:
                            maxcutsroot = np.array(list(maxcutsroot.values()) + [scip_tuned_best_config[problem][graph_size][seed]['maxcutsroot']]*pad_len)
                            assert len(maxcutsroot) == len(ncuts_applied)
                        else:
                            maxcutsroot = np.array(list(maxcutsroot.values()))[:len(ncuts_applied)]

                    maxcutsroot_active = np.nonzero(ncuts_applied == maxcutsroot)[0]
                    # plt.plot(xs, ys, '-gS', markevery=markers_on)
                    lpiters = stats['lp_iterations'][1:]
                    axes3[row].plot(lpiters, ncuts_applied, '-D', markevery=maxcutsroot_active.tolist(), label=baseline)

        for row, seed in enumerate(SEEDS):
            axes3[row].set_ylabel(f'Seed={seed}')
        axes3[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3, borderaxespad=0.).get_frame().set_linewidth(0.0) # fancybox=True, shadow=True,
        fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig3.savefig(f'{ROOTDIR}/{problem}_{graph_size}_napplied.png')

    for row, gs in enumerate(SEEDS):
        axes1[row, 0].set_ylabel(f'Seed={seed}')
        axes2[row, 0].set_ylabel(f'Seed={seed}')
    for col, gs in enumerate(data[problem].keys()):
        axes1[0, col].set_title(f'Size={gs}')
        axes2[0, col].set_title(f'Size={gs}')
        axes1[2, col].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=1, borderaxespad=0.).get_frame().set_linewidth(0.0) # fancybox=True, shadow=True,
        axes2[2, col].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=1, borderaxespad=0.).get_frame().set_linewidth(0.0) # fancybox=True, shadow=True,
    # axes1[2, 1].set_xlabel('LP Iterations')
    # axes2[2, 1].set_xlabel('Solving Time')
    # fig1.suptitle(f'{problem.upper()} Dual Bound vs. LP Iterations')
    # fig2.suptitle(f'{problem.upper()} Dual Bound vs. Solving Time')
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(f'{ROOTDIR}/{problem}_db_vs_lpiters.png')
    fig2.savefig(f'{ROOTDIR}/{problem}_db_vs_soltime.png')


print('finished')