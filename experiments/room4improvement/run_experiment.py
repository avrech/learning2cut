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
import yaml
parser = ArgumentParser()
parser.add_argument('--rootdir', type=str, default='results/large_action_space', help='rootdir to store results')
parser.add_argument('--datadir', type=str, default='../../data', help='rootdir to store results')
parser.add_argument('--configfile', type=str, default='configs/large_action_space.yaml', help='path to config yaml file')
args = parser.parse_args()
np.random.seed(777)
ROOTDIR = args.rootdir
DATADIR = args.datadir
if not os.path.isdir(ROOTDIR):
    os.makedirs(ROOTDIR)
with open(args.configfile) as f:
    action_space = yaml.load(f, Loader=yaml.FullLoader)

print('############### loading data ###############')
if not os.path.exists(os.path.join(ROOTDIR, 'data.pkl')):
    # take data from cut_selection_dqn MVC and MAXCUT datasets.
    data = {'mvc': {}, 'maxcut': {}}
    for problem in data.keys():
        if not os.path.exists(os.path.join(DATADIR, problem.upper(), 'data.pkl')):
            print(f'Generate {problem} data using learning2cut/data/generate_data.py, then run again')
        with open(f'{DATADIR}/{problem.upper()}/data.pkl', 'rb') as f:
            problem_data = pickle.load(f)
        # take the first instance from each validation set
        for dataset_name, dataset in problem_data.items():
            if 'valid' in dataset_name:
                G, stats = dataset['instances'][0]
                graph_size = G.number_of_nodes()
                data[problem][graph_size] = (G, stats)

        # # mvc
        # graph_sizes = [60, 100, 150]
        # densities = [0.25, 0.15, 0.1]
        # for gs, density in tqdm(zip(graph_sizes, densities), desc='generating graphs for MVC'):
        #     g = nx.erdos_renyi_graph(n=gs, p=density, directed=False)
        #     nx.set_node_attributes(g, {i: np.random.random() for i in g.nodes}, 'c')
        #     model, _ = mvc_model(g, use_random_branching=False, allow_restarts=True, use_heuristics=True)
        #     model.hideOutput(True)
        #     model.optimize()
        #     assert model.getGap() == 0
        #     stats = {}
        #     stats['time'] = model.getSolvingTime()
        #     stats['lp_iterations'] = model.getNLPIterations()
        #     stats['nodes'] = model.getNNodes()
        #     stats['applied'] = model.getNCutsApplied()
        #     stats['lp_rounds'] = model.getNLPs()
        #     stats['optimal_value'] = model.getObjVal()
        #     data['mvc'][gs] = (g, stats)
        #
        # # maxcut
        # graph_sizes = [40, 70, 100]
        # ms = [15, 15, 15]
        # for gs, m in tqdm(zip(graph_sizes, ms), desc='generating graphs for MAXCUT'):
        #     g = nx.barabasi_albert_graph(n=gs, m=m)
        #     nx.set_edge_attributes(g, {e: np.random.random() for e in g.edges}, 'weight')
        #     model, _, _ = maxcut_mccormic_model(g, use_random_branching=False, allow_restarts=True, use_cycles=False)
        #     model.hideOutput(True)
        #     model.optimize()
        #     assert model.getGap() == 0
        #     stats = {}
        #     stats['time'] = model.getSolvingTime()
        #     stats['lp_iterations'] = model.getNLPIterations()
        #     stats['nodes'] = model.getNNodes()
        #     stats['applied'] = model.getNCutsApplied()
        #     stats['lp_rounds'] = model.getNLPs()
        #     stats['optimal_value'] = model.getObjVal()
        #     data['maxcut'][gs] = (g, stats)
    print(f'saving data to: {ROOTDIR}/data.pkl')
    with open(f'{ROOTDIR}/data.pkl', 'wb') as f:
        pickle.dump(data, f)
else:
    print(f'loading data from: {ROOTDIR}/data.pkl')
    with open(f'{ROOTDIR}/data.pkl', 'rb') as f:
        data = pickle.load(f)


if not os.path.exists(f'{ROOTDIR}/scip_tuned_best_config.pkl'):
    print('run run_scip_tuned.py first, then run again.')
    exit(0)

with open(f'{ROOTDIR}/scip_tuned_best_config.pkl', 'rb') as f:
    scip_tuned_best_config = pickle.load(f)

if not os.path.exists(f'{ROOTDIR}/scip_adaptive_params.pkl'):
    print('run run_scip_adaptive.py first, then run again.')
    exit(0)

with open(f'{ROOTDIR}/scip_adaptive_params.pkl', 'rb') as f:
    scip_adaptive_params = pickle.load(f)

if not os.path.exists(f'{ROOTDIR}/scip_tuned_avg_best_config.pkl'):
    print('run run_scip_tuned_avg.py first, then run again.')
    exit(0)

with open(f'{ROOTDIR}/scip_tuned_avg_best_config.pkl', 'rb') as f:
    scip_tuned_avg_best_config = pickle.load(f)

if not os.path.exists(f'{ROOTDIR}/scip_overfit_avg_best_config.pkl'):
    print('run run_scip_tuned_avg.py first, then run again.')
    exit(0)

with open(f'{ROOTDIR}/scip_overfit_avg_best_config.pkl', 'rb') as f:
    scip_overfit_avg_best_config = pickle.load(f)

# todo run here also tuned and adaptive policies
print('############### run all baselines on local machine to compare solving time ###############')
# run default, 15-random, 15-most-violated and all-cuts baselines
SEEDS = [52, 176, 223]  # [46, 72, 101]
problems = ['mvc', 'maxcut']
baselines = ['default', '15_random', '15_most_violated', 'all_cuts', 'overfit_avg', 'tuned_avg', 'tuned', 'adaptive']
if os.path.exists(f'{ROOTDIR}/all_baselines_results.pkl'):
    with open(f'{ROOTDIR}/all_baselines_results.pkl', 'rb') as f:
        room4imp_all_baselines_results = pickle.load(f)
# else:
#     results = {p: {b: {} for b in baselines} for p in problems}
for p in problems:
    if p not in room4imp_all_baselines_results.keys():
        room4imp_all_baselines_results[p] = {}
    for b in baselines:
        if b not in room4imp_all_baselines_results[p].keys():
            room4imp_all_baselines_results[p][b] = {}

for problem, graphs in data.items():
    for baseline in tqdm(baselines, desc='run simple baselines'):
        graphs = data[problem]
        for (graph_size, (g, info)), lp_iterations_limit in zip(graphs.items(), [5000, 7000, 10000]):
            if graph_size not in room4imp_all_baselines_results[problem][baseline].keys():
                room4imp_all_baselines_results[problem][baseline][graph_size] = {}
            for seed in SEEDS:
                if seed in room4imp_all_baselines_results[problem][baseline][graph_size].keys():
                    continue
                if problem == 'mvc':
                    model, _ = mvc_model(g)
                    lp_iterations_limit = 1500
                elif problem == 'maxcut':
                    model, _, _ = maxcut_mccormic_model(g)
                    # lp_iterations_limit = {40: 5000, 70: 7000, 100: 10000}.get(graph_size)
                else:
                    raise ValueError
                set_aggresive_separation(model)
                sepa_params = {'lp_iterations_limit': lp_iterations_limit,
                               'policy': baseline,
                               'reset_maxcuts': 100,
                               'reset_maxcutsroot': 100,
                               'cut_stats': True}
                if baseline == 'tuned':
                    # set tuned params
                    tuned_params = scip_tuned_best_config[problem][graph_size][seed]
                    sepa_params.update(tuned_params)

                if baseline == 'tuned_avg':
                    # set tuned_avg params
                    tuned_avg_params = scip_tuned_avg_best_config[problem][graph_size]
                    sepa_params.update(tuned_avg_params)
                    sepa_params['policy'] = 'tuned'

                if baseline == 'overfit_avg':
                    # set tuned_avg params
                    overfit_avg_params = scip_overfit_avg_best_config[problem][graph_size]
                    sepa_params.update(overfit_avg_params)
                    sepa_params['policy'] = 'tuned'

                if baseline == 'adaptive':
                    # set adaptive param lists
                    adapted_param_list = scip_adaptive_params[problem][graph_size][seed]
                    # adapted_params = {
                    #     'objparalfac': {},
                    #     'dircutoffdistfac': {},
                    #     'efficacyfac': {},
                    #     'intsupportfac': {},
                    #     'maxcutsroot': {},
                    #     'minorthoroot': {}
                    # }
                    adapted_params = {k: {} for k in action_space.keys()}
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
                stats['db_auc'] = sum(get_normalized_areas(t=stats['lp_iterations'], ft=stats['dualbound'], t_support=lp_iterations_limit, reference=info['optimal_value']))
                room4imp_all_baselines_results[problem][baseline][graph_size][seed] = stats

with open(f'{ROOTDIR}/all_baselines_results.pkl', 'wb') as f:
    pickle.dump(room4imp_all_baselines_results, f)
# with open(f'{ROOTDIR}/all_baselines_results.pkl', 'rb') as f:
#     results = pickle.load(f)


print('############### analyzing results ###############')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

adaptive_params_dict = {}
for problem, baselines in room4imp_all_baselines_results.items():
    columns = [gs for gs in data[problem].keys()]
    summary = {baseline: [] for baseline in baselines.keys()}
    for baseline, baseline_results in baselines.items():
        for graph_size, seeds in baseline_results.items():
            # for MVC shorten the support to 100 lp iters.
            if problem == 'mvc':
                lpiter_support = 1500
                db_aucs = []
                db_auc_imps = []
                for seed, stats in seeds.items():
                    lpiter, db = truncate(t=stats['lp_iterations'], ft=stats['dualbound'], support=lpiter_support, interpolate=True)
                    db_auc = sum(get_normalized_areas(t=lpiter, ft=db, t_support=lpiter_support, reference=data[problem][graph_size][1]['optimal_value']))
                    db_aucs.append(db_auc)
                    default_lpiter, default_db = truncate(t=baselines['default'][graph_size][seed]['lp_iterations'], ft=baselines['default'][graph_size][seed]['dualbound'], support=lpiter_support, interpolate=True)
                    db_auc_imps.append(db_auc / sum(get_normalized_areas(t=default_lpiter, ft=default_db, t_support=lpiter_support, reference=data[problem][graph_size][1]['optimal_value'])))
                db_aucs = np.array(db_aucs)
                db_auc_imps = np.array(db_auc_imps)
            else:
                db_aucs = np.array([stats['db_auc'] for stats in seeds.values()])
                db_auc_imps = db_aucs / np.array([baselines['default'][graph_size][seed]['db_auc'] for seed in seeds.keys()])
            # convert improvement ratio to percents
            db_auc_imps = (db_auc_imps - 1)*100
            summary[baseline].append('{:.1f}{}{:.1f}% ({:.3f})'.format(db_auc_imps.mean(), u"\u00B1", db_auc_imps.std(), db_aucs.mean()))
    df = pd.DataFrame.from_dict(summary, orient='index', columns=columns)
    print(f'{"#"*40} {problem} {"#"*40}')
    print(df)
    csvfile = f'{ROOTDIR}/{problem}_baselines.csv'
    df.to_csv(csvfile)
    df.to_latex(f'{ROOTDIR}/{problem}-room4imp-baseline-results.tex',
                column_format='lccc',
                caption=f'{problem.upper()} Dual Bound AUC for All Baselines',
                label=f'tab:{problem}-room-for-improvemnt-results',
                )

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
        summary = {'Seed': [seed for seed in seeds.keys() for _ in range(len(seeds.keys()))]}
        for round_idx in range(len(adapted_param_list)):
            row = []
            for col in columns:
                for seed in seeds.keys():
                    row.append(adaptive_params_dict[problem][graph_size][seed][col][round_idx])
            summary[round_idx] = row
        # append a row for the tuned params
        tuned_params_row = []
        for col in columns:
            for seed in seeds.keys():
                tuned_params_row.append(scip_tuned_best_config[problem][graph_size][seed][col])
        summary['tuned'] = tuned_params_row
        # append a row for the overfit avg params
        overfit_avg_params_row = []
        scip_overfit_avg_best_config[problem][graph_size] = overfit_avg_param_dict = {k:v for k, v in scip_overfit_avg_best_config[problem][graph_size]}
        for col in columns:
            for seed in seeds.keys():
                overfit_avg_params_row.append(overfit_avg_param_dict[col])
        summary['overfit_avg'] = overfit_avg_params_row
        # append a row for the tuned avg params
        tuned_avg_params_row = []
        scip_tuned_avg_best_config[problem][graph_size] = tuned_avg_param_dict = {k:v for k, v in scip_tuned_avg_best_config[problem][graph_size]}
        for col in columns:
            for seed in seeds.keys():
                tuned_avg_params_row.append(tuned_avg_param_dict[col])
        summary['tuned_avg'] = tuned_avg_params_row

        columns = [col for col in columns for _ in range(3)]
        df = pd.DataFrame.from_dict(summary, orient='index', columns=columns)
        csvfile = f'{ROOTDIR}/{problem}_{graph_size}_adaptive_tuned_params.csv'
        df.to_csv(csvfile)
        print(f'saved {problem}-{graph_size} adaptive and tuned params table to: {csvfile}')


colors = {'default': 'b', '15_random': 'gray', '15_most_violated': 'purple', 'all_cuts': 'orange', 'overfit_avg': 'k', 'tuned_avg': 'pink', 'tuned': 'r', 'adaptive': 'g'}
for problem in room4imp_all_baselines_results.keys():
    fig1, axes1 = plt.subplots(3, 3, sharex='col')  # dual bound vs. lp iterations
    fig2, axes2 = plt.subplots(3, 3, sharex='col')  # dual bound vs. solving time

    for col, graph_size in enumerate(data[problem].keys()):
        fig3, axes3 = plt.subplots(3, 1)  # ncuts_applied vs. round idx
        fig4, axes4 = plt.subplots(3, 1)  # ncuts_applied fraction vs. round idx
        fig5, axes5 = plt.subplots(3, 1)  # avg minortho vs. round idx
        fig6, axes6 = plt.subplots(3, 1)  # avg efficacy vs. round idx
        fig7, axes7 = plt.subplots(3, 1)  # scip cuts coverage vs. round idx
        fig8, axes8 = plt.subplots(3, 1)  # scip cuts frac vs. round idx
        fig9, axes9 = plt.subplots(3, 1)  # jaccard similarity vs. round idx
        for row, seed in enumerate(SEEDS):
            for baseline in room4imp_all_baselines_results[problem].keys():
                if (baseline in ['default', 'all_cuts'] and col == 0) or baseline in ['15_random', '15_most_violated', 'overfit_avg'] and col == 1 or (baseline in ['tuned_avg', 'tuned', 'adaptive'] and col == 2):
                    label = baseline
                else:
                    label = None
                stats = room4imp_all_baselines_results[problem][baseline][graph_size][seed]
                axes1[row, col].plot(stats['lp_iterations'], stats['dualbound'], colors[baseline], label=label)
                axes2[row, col].plot(stats['solving_time'], stats['dualbound'], colors[baseline], label=label)
                if baseline in ['default', 'tuned', 'adaptive', 'tuned_avg', 'overfit_avg']:
                    ncuts_generated = np.array(stats['ncuts'])[1:]
                    ncuts_applied_cumsum = np.array(stats['ncuts_applied'])
                    ncuts_applied = ncuts_applied_cumsum[1:] - ncuts_applied_cumsum[:-1]
                    maxcutsroot = {'default': 2000,
                                   'tuned': scip_tuned_best_config[problem][graph_size][seed]['maxcutsroot'],
                                   'tuned_avg': scip_tuned_avg_best_config[problem][graph_size]['maxcutsroot'],
                                   'overfit_avg': scip_overfit_avg_best_config[problem][graph_size]['maxcutsroot'],
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
                    axes3[row].step(lpiters, ncuts_applied, '-D', markevery=maxcutsroot_active.tolist(), label=baseline)
                    axes4[row].step(lpiters, ncuts_applied/ncuts_generated, '-D', markevery=maxcutsroot_active.tolist(), label=baseline)
                    # plot avg min orthogonality w.r.t the selected group and avg efficacy
                    sel_minortho_avg = np.array(stats['selected_minortho_avg'], dtype=np.float)
                    sel_minortho_std = np.array(stats['selected_minortho_std'], dtype=np.float)
                    sel_efficacy_avg = np.array(stats['selected_efficacy_avg'], dtype=np.float)
                    sel_efficacy_std = np.array(stats['selected_efficacy_std'], dtype=np.float)
                    x_axis = np.arange(len(sel_minortho_avg))
                    axes5[row].plot(x_axis, sel_minortho_avg, lw=1, label=baseline, color=colors[baseline])
                    axes5[row].fill_between(x_axis, sel_minortho_avg + sel_minortho_std, sel_minortho_avg - sel_minortho_std, facecolor=colors[baseline], alpha=0.4)
                    axes6[row].plot(x_axis, sel_efficacy_avg, lw=1, label=baseline, color=colors[baseline])
                    axes6[row].fill_between(x_axis, sel_efficacy_avg + sel_efficacy_std, sel_efficacy_avg - sel_efficacy_std, facecolor=colors[baseline], alpha=0.4)

                    # plot cutting plane selection similarity metrics
                    if baseline != 'default':
                        scip_cuts = stats['scip_selected_cuts']
                        policy_cuts = stats['policy_selected_cuts'][1:]
                        intersections = np.array([len(set(sc).intersection(pc)) for sc, pc in zip(scip_cuts, policy_cuts)], dtype=np.float)
                        scip_ncuts_applied = np.array([len(sc) for sc in scip_cuts])
                        policy_ncuts_applied = np.array([len(pc) for pc in policy_cuts])
                        jaccard_similarity = intersections / (scip_ncuts_applied + policy_ncuts_applied - intersections)
                        scip_cuts_coverage = intersections / scip_ncuts_applied
                        scip_frac = intersections / policy_ncuts_applied
                        axes7[row].step(lpiters, scip_cuts_coverage, label=baseline)
                        axes8[row].step(lpiters, scip_frac, label=baseline)
                        axes9[row].step(lpiters, jaccard_similarity, label=baseline)
                    # dis_minortho_avg = np.array(stats['discarded_minortho_avg'], dtype=np.float)
                    # dis_minortho_std = np.array(stats['discarded_minortho_std'], dtype=np.float)
                    # dis_efficacy_avg = np.array(stats['discarded_efficacy_avg'], dtype=np.float)
                    # dis_efficacy_std = np.array(stats['discarded_efficacy_std'], dtype=np.float)
                    # dis_minortho_avg[dis_minortho_avg == None] = 0
                    # dis_minortho_std[dis_minortho_std == None] = 0
                    # dis_efficacy_avg[dis_efficacy_avg == None] = 0
                    # dis_efficacy_std[dis_efficacy_std == None] = 0
                    # axes5[row].plot(x_axis, dis_minortho_avg, lw=2, label=baseline, color=colors[baseline])
                    # axes5[row].fill_between(x_axis, dis_minortho_avg + dis_minortho_std, dis_minortho_avg - dis_minortho_std, facecolor=colors[baseline], alpha=0.5)
                    # axes6[row].plot(x_axis, dis_efficacy_avg, lw=2, label=baseline, color=colors[baseline])
                    # axes6[row].fill_between(x_axis, dis_efficacy_avg + dis_efficacy_std, dis_efficacy_avg - dis_efficacy_std, facecolor=colors[baseline], alpha=0.5)

        for row, seed in enumerate(SEEDS):
            axes3[row].set_ylabel(f'Seed={seed}')
            axes4[row].set_ylabel(f'Seed={seed}')
            axes5[row].set_ylabel(f'Seed={seed}')
            axes5[row].grid()
            axes6[row].set_ylabel(f'Seed={seed}')
            axes6[row].grid()
            axes7[row].set_ylabel(f'Seed={seed}')
            axes8[row].set_ylabel(f'Seed={seed}')
            axes9[row].set_ylabel(f'Seed={seed}')

        axes3[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3, borderaxespad=0.).get_frame().set_linewidth(0.0) # fancybox=True, shadow=True,
        axes4[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3, borderaxespad=0.).get_frame().set_linewidth(0.0) # fancybox=True, shadow=True,
        axes5[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3, borderaxespad=0.).get_frame().set_linewidth(0.0) # fancybox=True, shadow=True,
        axes6[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3, borderaxespad=0.).get_frame().set_linewidth(0.0) # fancybox=True, shadow=True,
        axes7[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3, borderaxespad=0.).get_frame().set_linewidth(0.0) # fancybox=True, shadow=True,
        axes8[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3, borderaxespad=0.).get_frame().set_linewidth(0.0) # fancybox=True, shadow=True,
        axes9[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3, borderaxespad=0.).get_frame().set_linewidth(0.0) # fancybox=True, shadow=True,
        fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig3.savefig(f'{ROOTDIR}/{problem}_{graph_size}_napplied.png')
        fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig4.savefig(f'{ROOTDIR}/{problem}_{graph_size}_frac_applied.png')
        fig5.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig5.savefig(f'{ROOTDIR}/{problem}_{graph_size}_minortho.png')
        fig6.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig6.savefig(f'{ROOTDIR}/{problem}_{graph_size}_efficacy.png')
        fig7.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig7.savefig(f'{ROOTDIR}/{problem}_{graph_size}_scip_cuts_coverage.png')
        fig8.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig8.savefig(f'{ROOTDIR}/{problem}_{graph_size}_scip_cuts_frac.png')
        fig9.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig9.savefig(f'{ROOTDIR}/{problem}_{graph_size}_cuts_jaccard_similarity.png')
        for fg in [fig3, fig4, fig5, fig6, fig7, fig8, fig9]:
            plt.close(fg)

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
    plt.close(fig1)
    plt.close(fig2)

print('finished root node analysis')
