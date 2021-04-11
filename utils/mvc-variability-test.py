from utils.scip_models import mvc_model, CSBaselineSepa, set_aggresive_separation, CSResetSepa
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from utils.functions import get_normalized_areas
from tqdm import tqdm
import pickle
import pandas as pd

import os
if not os.path.isdir('variability_results'):
    os.makedirs('variability_results')

# randomize graphs
graph_sizes = [100, 150, 200]
if False:
    graphs = {size: [nx.barabasi_albert_graph(n=size, m=10, seed=223) for _ in range(10)] for size in graph_sizes}
    for glist in graphs.values():
        for g in glist:
            nx.set_node_attributes(g, {i: np.random.random() for i in g.nodes}, 'c')
        for g1, g2 in combinations(glist, 2):
            assert not nx.is_isomorphic(g1, g2, node_match=lambda v1, v2: v1['c'] == v2['c'])

    with open('variability_results/mvc-variability-graphs.pkl', 'wb') as f:
        pickle.dump(graphs, f)

with open('variability_results/mvc-variability-graphs.pkl', 'rb') as f:
    graphs = pickle.load(f)
seeds = [46, 72, 101]
lp_iterations_limit = 2000

if False:
    cut_aggr_results = {k: {size: [] for size in graph_sizes} for k in ['default', 'aggressive']}
    for cut_aggressivness in ['aggressive', 'default']:
        for size, glist in graphs.items():
            # if size == 100:
            #     continue
            for G in tqdm(glist, desc=f'solving size={size} policy={cut_aggressivness}'):
                model, _ = mvc_model(G, use_general_cuts=True, use_cut_pool=True, use_random_branching=False)
                # model.hideOutput(True)
                # sepa = CSBaselineSepa(hparams={'lp_iterations_limit': -1})
                # model.includeSepa(sepa, '#CS_baseline', 'do-nothing', priority=-100000000, freq=1)

                model.optimize()
                optval = model.getObjVal()
                assert model.getGap() == 0

                for seed in seeds:
                    model, x = mvc_model(G, use_general_cuts=True, use_cut_pool=True)
                    sepa = CSBaselineSepa(hparams={'lp_iterations_limit': lp_iterations_limit})
                    model.includeSepa(sepa, '#CS_baseline', 'do-nothing', priority=-100000000, freq=1)
                    model.setBoolParam("misc/allowdualreds", 0)
                    if cut_aggressivness == 'aggressive':
                        set_aggresive_separation(model)  # todo debug
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
                    db, gap, lpiter = stats['dualbound'], stats['gap'], stats['lp_iterations']
                    ncuts, napplied_cumsum = np.array(stats['ncuts'][1:]), np.array(stats['ncuts_applied'])
                    db_auc = sum(get_normalized_areas(t=lpiter, ft=db, t_support=lp_iterations_limit, reference=optval))
                    gap_auc = sum(get_normalized_areas(t=lpiter, ft=gap, t_support=lp_iterations_limit, reference=0))
                    napplied_round = napplied_cumsum[1:] - napplied_cumsum[:-1]
                    applied_avail = napplied_round / ncuts
                    stats['db_auc'] = db_auc
                    stats['gap_auc'] = gap_auc
                    stats['optval'] = optval
                    stats['napplied/round'] = np.mean(napplied_round)
                    stats['applied/avail'] = np.mean(applied_avail)
                    stats['napplied_std'] = np.std(napplied_round)
                    stats['applied/avail_std'] = np.std(applied_avail)


                    cut_aggr_results[cut_aggressivness][size].append(stats)

    with open(f'variability_results/mvc-variability-results-lpiter{lp_iterations_limit}.pkl', 'wb') as f:
        pickle.dump(cut_aggr_results, f)

with open(f'variability_results/mvc-variability-results-lpiter{lp_iterations_limit}.pkl', 'rb') as f:
    cut_aggr_results = pickle.load(f)

# validate correctness of records:

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
dfs = {}
for size in graph_sizes:
    summary = {}
    columns = ['db_auc', 'gap_auc', 'gap', 'time', 'rounds', 'total ncuts', 'napplied/round', 'applied/avail']
    for cut_aggressivness in ['default', 'aggressive']:
        db_auc_list = np.array([s['db_auc'] for s in cut_aggr_results[cut_aggressivness][size]])
        gap_auc_list = np.array([s['gap_auc'] for s in cut_aggr_results[cut_aggressivness][size]])
        gap = np.array([s['gap'][-1] for s in cut_aggr_results[cut_aggressivness][size]])
        time = np.array([s['solving_time'][-1] for s in cut_aggr_results[cut_aggressivness][size]])
        ncuts = np.array([sum(s['ncuts']) for s in cut_aggr_results[cut_aggressivness][size]])
        napplied_round = np.array([s['napplied/round'] for s in cut_aggr_results[cut_aggressivness][size]])
        applied_avail = np.array([s['applied/avail'] for s in cut_aggr_results[cut_aggressivness][size]])
        napplied_std = np.array([s['napplied_std'] for s in cut_aggr_results[cut_aggressivness][size]])
        applied_avail_std = np.array([s['applied/avail_std'] for s in cut_aggr_results[cut_aggressivness][size]])
        nrounds = np.array([s['lp_rounds'][-1] for s in cut_aggr_results[cut_aggressivness][size]])
        summary[cut_aggressivness] = ['{:.4f}{}{:.4f}'.format(arr.mean(), u"\u00B1", arr.std()) for arr in [db_auc_list, gap_auc_list, gap, time, nrounds, ncuts]]
        summary[cut_aggressivness] += ['{:.4f}{}{:.4f}'.format(napplied_round.mean(), u"\u00B1", napplied_std.mean())]
        summary[cut_aggressivness] += ['{:.4f}{}{:.4f}'.format(applied_avail.mean(), u"\u00B1", applied_avail_std.mean())]
    df = pd.DataFrame.from_dict(summary, orient='index', columns=columns)
    print('#'*70 + f' SIZE {size} ' + '#'*70)
    print(df)
    dfs[size] = df
    df.to_csv(f'mvc-variability-{size}.csv')

with open(f'variability_results/mvc-variability-dfs-lpiter{lp_iterations_limit}.pkl', 'wb') as f:
    pickle.dump(dfs, f)

print('########## test baselines with aggressive cuts ##########')
if False:
    baselines_results = {k: {size: [] for size in graph_sizes if size > 100} for k in ['15_random', '15_most_violated']}
    for baseline in ['15_most_violated', '15_random']:
        for size, glist in graphs.items():
            if size == 100:
                continue
            hparams = {'lp_iterations_limit': lp_iterations_limit,
                       'criterion': baseline,
                       'reset_maxcuts': 100000,
                       'reset_maxcutsroot': 100000}
            for idx, G in enumerate(tqdm(glist, desc=f'solving size={size} baseline={baseline}')):
                optval = cut_aggr_results['aggressive'][size][idx*3]['optval']

                for seed in seeds:
                    model, x = mvc_model(G)
                    sepa = CSBaselineSepa(hparams=hparams)
                    model.includeSepa(sepa, '#CS_baseline', f'enforce baseline {baseline}', priority=-100000000, freq=1)
                    reset_sepa = CSResetSepa(hparams=hparams)
                    model.includeSepa(reset_sepa, '#CS_reset', f'reset maxcuts params', priority=99999999, freq=1)
                    model.setBoolParam("misc/allowdualreds", 0)
                    set_aggresive_separation(model)  # todo debug
                    model.setLongintParam('limits/nodes', 1)  # solve only at the root node
                    model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever
                    model.setIntParam('branching/random/priority', 10000000)
                    model.setBoolParam('randomization/permutevars', True)
                    model.setIntParam('randomization/permutationseed', seed)
                    model.setIntParam('randomization/randomseedshift', seed)
                    # model.hideOutput(True)
                    model.optimize()
                    sepa.update_stats()
                    stats = sepa.stats
                    db, gap, lpiter = stats['dualbound'], stats['gap'], stats['lp_iterations']
                    ncuts, napplied_cumsum = np.array(stats['ncuts'][1:]), np.array(stats['ncuts_applied'])
                    db_auc = sum(get_normalized_areas(t=lpiter, ft=db, t_support=lp_iterations_limit, reference=optval))
                    gap_auc = sum(get_normalized_areas(t=lpiter, ft=gap, t_support=lp_iterations_limit, reference=0))
                    napplied_round = napplied_cumsum[1:] - napplied_cumsum[:-1]
                    applied_avail = napplied_round / ncuts
                    stats['db_auc'] = db_auc
                    stats['gap_auc'] = gap_auc
                    stats['optval'] = optval
                    stats['napplied/round'] = np.mean(napplied_round)
                    stats['applied/avail'] = np.mean(applied_avail)
                    stats['napplied_std'] = np.std(napplied_round)
                    stats['applied/avail_std'] = np.std(applied_avail)
                    baselines_results[baseline][size].append(stats)

    baselines_results['default'] = cut_aggr_results['aggressive']
    with open(f'variability_results/mvc-variability-baselines-results-lpiter{lp_iterations_limit}.pkl', 'wb') as f:
        pickle.dump(baselines_results, f)

with open(f'variability_results/mvc-variability-baselines-results-lpiter{lp_iterations_limit}.pkl', 'rb') as f:
    baselines_results = pickle.load(f)
baselines_results['default'] = cut_aggr_results['aggressive']
# validate correctness of records:

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
dfs = {}
for size in [150, 200]:
    summary = {}
    columns = ['db_auc', 'gap_auc', 'gap', 'time', 'rounds', 'total ncuts', 'napplied/round', 'applied/avail']
    for baseline in ['default', '15_random', '15_most_violated']:
        db_auc_list = np.array([s['db_auc'] for s in baselines_results[baseline][size]])
        gap_auc_list = np.array([s['gap_auc'] for s in baselines_results[baseline][size]])
        gap = np.array([s['gap'][-1] for s in baselines_results[baseline][size]])
        time = np.array([s['solving_time'][-1] for s in baselines_results[baseline][size]])
        ncuts = np.array([sum(s['ncuts']) for s in baselines_results[baseline][size]])
        napplied_round = np.array([s['napplied/round'] for s in baselines_results[baseline][size]])
        applied_avail = np.array([s['applied/avail'] for s in baselines_results[baseline][size]])
        napplied_std = np.array([s['napplied_std'] for s in baselines_results[baseline][size]])
        applied_avail_std = np.array([s['applied/avail_std'] for s in baselines_results[baseline][size]])
        nrounds = np.array([s['lp_rounds'][-1] for s in baselines_results[baseline][size]])
        summary[baseline] = ['{:.4f}{}{:.4f}'.format(arr.mean(), u"\u00B1", arr.std()) for arr in [db_auc_list, gap_auc_list, gap, time, nrounds, ncuts]]
        summary[baseline] += ['{:.4f}{}{:.4f}'.format(napplied_round.mean(), u"\u00B1", napplied_std.mean())]
        summary[baseline] += ['{:.4f}{}{:.4f}'.format(applied_avail.mean(), u"\u00B1", applied_avail_std.mean())]
    df = pd.DataFrame.from_dict(summary, orient='index', columns=columns)
    print('#'*70 + f' SIZE {size} ' + '#'*70)
    print(df)
    dfs[size] = df
    df.to_csv(f'mvc-variability-baselines-{size}.csv')

with open(f'variability_results/mvc-variability-baselines-df-lpiter{lp_iterations_limit}.pkl', 'wb') as f:
    pickle.dump(dfs, f)

if True:
    G = graphs[200][0]
    fig, axes = plt.subplots(2, 2)

    for bsl in ['default', '15_random', '15_most_violated']:
        # model, x = mvc_model(G, use_general_cuts=True)
        # sepa = BaselineSepa(hparams={'lp_iterations_limit': 1500})
        # model.includeSepa(sepa, '#CS_baseline', 'do-nothing', priority=-100000000, freq=1)
        # model.setBoolParam("misc/allowdualreds", 0)
        #
        # if aggresive:
        #     set_aggresive_separation(model)  # todo debug
        # model.setLongintParam('limits/nodes', 1)  # solve only at the root node
        # model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever
        # model.setIntParam('branching/random/priority', 10000000)  # add cuts forever
        #
        # model.setBoolParam('randomization/permutevars', True)
        # model.setIntParam('randomization/permutationseed', 223)
        # model.setIntParam('randomization/randomseedshift', 223)
        #
        # model.optimize()
        #
        # sepa.update_stats()
        # # plt.figure()
        # # nx.spring_layout(G)
        # # nx.draw(G, labels={i: model.getVal(x[i]) for i in G.nodes}, with_labels=True, node_color=['gray' if model.getVal(x[i]) == 0 else 'blue' for i in G.nodes])
        # # plt.savefig('test-sol.png')
        color = {'default': 'b', '15_random': 'g', '15_most_violated': 'r'}.get(bsl)
        stats = baselines_results[bsl][200][0]
        axes[0,0].plot(stats['lp_iterations'], stats['dualbound'], color, label=bsl)
        axes[0,1].plot(stats['lp_iterations'], stats['gap'], color, label=bsl)
        axes[1,0].plot(stats['lp_iterations'], stats['solving_time'], color, label=bsl)
        axes[1,1].plot(stats['solving_time'], stats['gap'], color, label=bsl)

    # axes[0, 0].set_title('default')
    # axes[2, 0].set_xlabel('lp iter')
    # axes[2, 1].set_xlabel('lp iter')
    # axes[0, 1].set_title('aggressive')
    axes[0,0].set_ylabel('db')
    axes[0,1].set_ylabel('gap')
    axes[1,0].set_ylabel('time')
    axes[1,1].set_ylabel('gap')
    axes[0,0].set_xlabel('lp iterations')
    axes[0,1].set_xlabel('lp iterations')
    axes[1,0].set_xlabel('lp iterations')
    axes[1,1].set_xlabel('time')
    axes[0, 0].set_title('db vs. lp iterations')
    axes[0, 1].set_title('gap vs. lp iterations')
    axes[1, 0].set_title('time vs. lp iterations')
    axes[1, 1].set_title('gap vs. time')
    axes[0,1].legend()

    plt.tight_layout()
    fig.savefig(f'test-aggressive-baselines-lpiter{lp_iterations_limit}.png')

    # todo - add graphs of all instances to wandb and generate smoothed graph.
print('finish')