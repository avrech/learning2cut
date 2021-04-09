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
if not os.path.isdir('sweet_spot_results'):
    os.makedirs('sweet_spot_results')

# randomize graphs
graph_sizes = [50, 100, 150, 200]
seeds = [46, 72, 101]
ms = {
    50: list(range(5, 50, 5)),
    100: list(range(5, 70, 5)),
    150: list(range(5, 100, 5)),
    200: list(range(5, 100, 5)),
}


# MVC with branch and cut:
if True:
    results = {gs: {'xticks': [],
                    'time': [],
                    'lp_iterations': [],
                    'nodes': [],
                    'applied': [],
                    'lp_rounds': [],
                    }
               for gs in graph_sizes}
    for gs in graph_sizes:
        for m in tqdm(ms[gs], desc=f'sweeping on graph size {gs}'):
            stats = {
                'time': [],
                'lp_iterations': [],
                'nodes': [],
                'applied': [],
                'lp_rounds': [],
            }
            g = nx.barabasi_albert_graph(n=gs, m=m)
            nx.set_node_attributes(g, {i: np.random.random() for i in g.nodes}, 'c')
            for seed in seeds:
                model, _ = mvc_model(g, use_random_branching=False)
                model.setBoolParam('randomization/permutevars', True)
                model.setIntParam('randomization/permutationseed', seed)
                model.setIntParam('randomization/randomseedshift', seed)
                model.hideOutput(True)
                model.optimize()
                assert model.getGap() == 0
                stats['time'].append(model.getSolvingTime())
                stats['lp_iterations'].append(model.getNLPIterations())
                stats['nodes'].append(model.getNNodes())
                stats['applied'].append(model.getNCutsApplied())
                stats['lp_rounds'].append(model.getNLPs())
            results[gs]['xticks'].append('{}({:.2f})'.format(m, nx.density(g)))
            for k, vs in stats.items():
                results[gs][k].append(np.mean(vs))

    with open(f'sweet_spot_results/mvc-sweet-spot-results-bnc.pkl', 'wb') as f:
        pickle.dump(results, f)

with open(f'sweet_spot_results/mvc-sweet-spot-results-bnc.pkl', 'rb') as f:
    results = pickle.load(f)

# plot for each graph size:
# metric vs. density for metric in stats.keys
fig = plt.figure(figsize=(16, 10))
axes = fig.subplots(5, 4)
plt.suptitle('Default Branch & Cut')
for col, gs in enumerate(graph_sizes):
    x_labels = results[gs]['xticks']
    for row, (k, vals) in enumerate([(kk, v) for kk, v in results[gs].items() if kk != 'xticks']):
        ax = axes[row, col]
        # ax.plot(np.arange(len(vals)), vals)
        ax.plot(x_labels, vals)
        if col == 0:
            ax.set_ylabel(k)
        ax.get_xaxis().set_visible(False)

    axes[4, col].get_xaxis().set_visible(True)
    # axes[4, col].set_xticks(list(range(len(x_labels))), minor=True)
    # axes[4, col].set_xticklabels(x_labels, minor=True, rotation=45)
    fig.autofmt_xdate(rotation=45)
    axes[4, col].set_xlabel('m(denisty)')
    axes[0, col].set_title(f'size={gs}')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
plt.savefig('sweet_spot_results/mvc-sweet-spot-bnc.png')

################ root node #################
lp_iterations_limit = 1500

# MVC with branch and cut:
if True:
    results = {gs: {'xticks': [],
                    'time': [],
                    'lp_iterations': [],
                    'gap': [],
                    'applied': [],
                    'lp_rounds': [],
                    }
               for gs in graph_sizes}
    for gs in graph_sizes:
        for m in tqdm(ms[gs], desc=f'sweeping on graph size {gs}'):
            stats = {
                'time': [],
                'lp_iterations': [],
                'gap': [],
                'applied': [],
                'lp_rounds': [],
            }
            g = nx.barabasi_albert_graph(n=gs, m=m)
            nx.set_node_attributes(g, {i: np.random.random() for i in g.nodes}, 'c')
            for seed in seeds:
                model, _ = mvc_model(g)
                set_aggresive_separation(model)
                sepa = CSBaselineSepa(hparams={'lp_iterations_limit': lp_iterations_limit})
                model.includeSepa(sepa, '#CS_baseline', 'do-nothing', priority=-100000000, freq=1)
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
                stats['time'].append(model.getSolvingTime())
                stats['lp_iterations'].append(model.getNLPIterations())
                stats['gap'].append(model.getGap())
                stats['applied'].append(model.getNCutsApplied())
                stats['lp_rounds'].append(model.getNLPs())
            results[gs]['xticks'].append('{}({:.2f})'.format(m, nx.density(g)))
            for k, vs in stats.items():
                results[gs][k].append(np.mean(vs))

    with open(f'sweet_spot_results/mvc-sweet-spot-results-lpiter{lp_iterations_limit}.pkl', 'wb') as f:
        pickle.dump(results, f)

with open(f'sweet_spot_results/mvc-sweet-spot-results-lpiter{lp_iterations_limit}.pkl', 'rb') as f:
    results = pickle.load(f)

# plot for each graph size:
# metric vs. density for metric in stats.keys
fig = plt.figure(figsize=(16, 10))
axes = fig.subplots(5, 4)
plt.suptitle(f'Default Root Node (LP Iter Limit = {lp_iterations_limit}')
for col, gs in enumerate(graph_sizes):
    x_labels = results[gs]['xticks']
    for row, (k, vals) in enumerate([(kk, v) for kk, v in results[gs].items() if kk != 'xticks']):
        ax = axes[row, col]
        # ax.plot(np.arange(len(vals)), vals)
        ax.plot(x_labels, vals)
        if col == 0:
            ax.set_ylabel(k)
        ax.get_xaxis().set_visible(False)

    axes[4, col].get_xaxis().set_visible(True)
    # axes[4, col].set_xticks(list(range(len(x_labels))), minor=True)
    # axes[4, col].set_xticklabels(x_labels, minor=True, rotation=45)
    fig.autofmt_xdate(rotation=45)
    axes[4, col].set_xlabel('m(denisty)')
    axes[0, col].set_title(f'size={gs}')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
plt.savefig(f'sweet_spot_results/mvc-sweet-spot-lpiter{lp_iterations_limit}.png')

print('finished')
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
# dfs = {}
# summary = {}
# columns = ['db_auc', 'gap_auc', 'gap', 'time', 'rounds', 'total ncuts', 'total applied', 'napplied/round', 'applied/avail']
# for size in graph_sizes:
#     db_auc_list = np.array([s['db_auc'] for s in results[size]])
#     gap_auc_list = np.array([s['gap_auc'] for s in results[size]])
#     gap = np.array([s['gap'][-1] for s in results[size]])
#     time = np.array([s['solving_time'][-1] for s in results[size]])
#     ncuts = np.array([sum(s['ncuts']) for s in results[size]])
#     napplied = np.array([s['ncuts_applied'][-1] for s in results[size]])
#     napplied_round = np.array([s['napplied/round'] for s in results[size]])
#     applied_avail = np.array([s['applied/avail'] for s in results[size]])
#     napplied_std = np.array([s['napplied_std'] for s in results[size]])
#     applied_avail_std = np.array([s['applied/avail_std'] for s in results[size]])
#     nrounds = np.array([s['lp_rounds'][-1] for s in results[size]])
#     summary[size] = ['{:.4f}{}{:.4f}'.format(arr.mean(), u"\u00B1", arr.std()) for arr in [db_auc_list, gap_auc_list, gap, time, nrounds, ncuts]]
#     summary[size] += ['{:.4f}{}{:.4f}'.format(napplied_round.mean(), u"\u00B1", napplied_std.mean())]
#     summary[size] += ['{:.4f}{}{:.4f}'.format(applied_avail.mean(), u"\u00B1", applied_avail_std.mean())]
# df = pd.DataFrame.from_dict(summary, orient='index', columns=columns)
# print(df)
# df.to_csv(f'mvc-sweet-spot-{size}.csv')

# if True:
#     G = ba_graphs[200][0]
#     fig, axes = plt.subplots(2, 2)
#
#     for bsl in ['default', '15_random', '15_most_violated']:
#         color = {'default': 'b', '15_random': 'g', '15_most_violated': 'r'}.get(bsl)
#         stats = baselines_results[bsl][200][0]
#         axes[0,0].plot(stats['lp_iterations'], stats['dualbound'], color, label=bsl)
#         axes[0,1].plot(stats['lp_iterations'], stats['gap'], color, label=bsl)
#         axes[1,0].plot(stats['lp_iterations'], stats['solving_time'], color, label=bsl)
#         axes[1,1].plot(stats['solving_time'], stats['gap'], color, label=bsl)
#
#     # axes[0, 0].set_title('default')
#     # axes[2, 0].set_xlabel('lp iter')
#     # axes[2, 1].set_xlabel('lp iter')
#     # axes[0, 1].set_title('aggressive')
#     axes[0,0].set_ylabel('db')
#     axes[0,1].set_ylabel('gap')
#     axes[1,0].set_ylabel('time')
#     axes[1,1].set_ylabel('gap')
#     axes[0,0].set_xlabel('lp iterations')
#     axes[0,1].set_xlabel('lp iterations')
#     axes[1,0].set_xlabel('lp iterations')
#     axes[1,1].set_xlabel('time')
#     axes[0, 0].set_title('db vs. lp iterations')
#     axes[0, 1].set_title('gap vs. lp iterations')
#     axes[1, 0].set_title('time vs. lp iterations')
#     axes[1, 1].set_title('gap vs. time')
#     axes[0,1].legend()
#
#     plt.tight_layout()
#     fig.savefig(f'test-aggressive-baselines-lpiter{lp_iterations_limit}.png')
#
#     # todo - add graphs of all instances to wandb and generate smoothed graph.
# print('finish')