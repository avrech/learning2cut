from utils.scip_models import mvc_model, CSBaselineSepa, set_aggresive_separation, CSResetSepa, maxcut_mccormic_model
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from utils.functions import get_normalized_areas
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

for problem, baselines in results.items():
    columns = [gs for gs in data[problem].keys()]
    summary = {baseline: [] for baseline in baselines.keys()}
    for baseline in baselines:
        for graph_size, seeds in baseline.items():
            db_aucs = np.array([stats['db_auc'] for stats in seeds.values()])
            summary[baseline].append('{:.4f}{}{:.4f}'.format(db_aucs.mean(), u"\u00B1", db_aucs.std()))
    df = pd.DataFrame.from_dict(summary, orient='index', columns=columns)
    print(f'{"#"*70} {problem} {"#"*70}')
    print(df)
    csvfile = f'{ROOTDIR}/{problem}_baselines.csv'
    df.to_csv(csvfile)
    print(f'saved {problem} csv to: {csvfile}')

    # fig, ax = plt.subplots(1)
    # discounted_rewards = last_training_episode_stats['discounted_rewards']
    # selected_q_avg = np.array(last_training_episode_stats['selected_q_avg'])
    # selected_q_std = np.array(last_training_episode_stats['selected_q_std'])
    # bootstrapped_returns = last_training_episode_stats['bootstrapped_returns']
    # x_axis = np.arange(len(discounted_rewards))
    # ax.plot(x_axis, discounted_rewards, lw=2, label='discounted rewards', color='blue')
    # ax.plot(x_axis, bootstrapped_returns, lw=2, label='bootstrapped reward', color='green')
    # ax.plot(x_axis, selected_q_avg, lw=2, label='Q', color='red')
    # ax.fill_between(x_axis, selected_q_avg + selected_q_std, selected_q_avg - selected_q_std, facecolor='red',
    #                 alpha=0.5)
    # ax.legend()
    # ax.set_xlabel('step')
    # ax.grid()
