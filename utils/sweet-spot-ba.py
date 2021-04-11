from utils.scip_models import mvc_model, CSBaselineSepa, set_aggresive_separation, CSResetSepa, maxcut_mccormic_model
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
graph_sizes = [60, 100, 150] # 200
seeds = [46, 72, 101]
ms = {
    60: list(range(5, 40, 5)),
    100: list(range(5, 50, 5)),
    150: list(range(5, 70, 5)),
    # 200: list(range(5, 100, 5)),
}


# MVC with branch and cut:
if False:
    results = {gs: {'density': [],
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
            results[gs]['density'].append('{}({:.2f})'.format(m, nx.density(g)))
            for k, vs in stats.items():
                results[gs][k].append(np.mean(vs))

    with open(f'sweet_spot_results/mvc-ba-sweet-spot-results-bnc.pkl', 'wb') as f:
        pickle.dump(results, f)

with open(f'sweet_spot_results/mvc-ba-sweet-spot-results-bnc.pkl', 'rb') as f:
    results = pickle.load(f)

# plot for each graph size:
# metric vs. density for metric in stats.keys
fig = plt.figure(figsize=(16, 10))
axes = fig.subplots(5, 3)
plt.suptitle('MVC Branch & Cut')
for col, gs in enumerate(graph_sizes):
    x_labels = results[gs]['density']
    for row, (k, vals) in enumerate([(kk, v) for kk, v in results[gs].items() if kk != 'density']):
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
plt.savefig('sweet_spot_results/mvc-ba-sweet-spot-bnc.png')

################ root node #################
lp_iterations_limit = 1500

# MVC in root node:
if False:
    results = {gs: {'density': [],
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
                sepa.update_stats()
                stats['time'].append(model.getSolvingTime())
                stats['lp_iterations'].append(model.getNLPIterations())
                stats['gap'].append(model.getGap())
                stats['applied'].append(model.getNCutsApplied())
                stats['lp_rounds'].append(model.getNLPs())
            results[gs]['density'].append('{}({:.2f})'.format(m, nx.density(g)))
            for k, vs in stats.items():
                results[gs][k].append(np.mean(vs))

    with open(f'sweet_spot_results/mvc-ba-sweet-spot-results-lpiter{lp_iterations_limit}.pkl', 'wb') as f:
        pickle.dump(results, f)

with open(f'sweet_spot_results/mvc-ba-sweet-spot-results-lpiter{lp_iterations_limit}.pkl', 'rb') as f:
    results = pickle.load(f)

# plot for each graph size:
# metric vs. density for metric in stats.keys
fig = plt.figure(figsize=(16, 10))
axes = fig.subplots(5, 3)
plt.suptitle(f'MVC BA Root Node (LP Iter Limit = {lp_iterations_limit}')
for col, gs in enumerate(graph_sizes):
    x_labels = results[gs]['density']
    for row, (k, vals) in enumerate([(kk, v) for kk, v in results[gs].items() if kk != 'density']):
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
plt.savefig(f'sweet_spot_results/mvc-ba-sweet-spot-lpiter{lp_iterations_limit}.png')

print('finished')



print('#################################################')
print('############# MAXCUT ############################')
print('#################################################')

# randomize graphs
graph_sizes = [40, 70, 100]
seeds = [46, 72, 101]
ms = {
    40: list(range(5, 40, 5)),
    70: list(range(5, 55, 5)),
    100: list(range(5, 80, 5)),
}


# MVC with branch and cut:
# if True:
#     results = {gs: {'xticks': [],
#                     'time': [],
#                     'lp_iterations': [],
#                     'nodes': [],
#                     'applied': [],
#                     'lp_rounds': [],
#                     }
#                for gs in graph_sizes}
#     for gs in graph_sizes:
#         for m in tqdm(ms[gs], desc=f'sweeping on graph size {gs}'):
#             stats = {
#                 'time': [],
#                 'lp_iterations': [],
#                 'nodes': [],
#                 'applied': [],
#                 'lp_rounds': [],
#             }
#             g = nx.barabasi_albert_graph(n=gs, m=m)
#             nx.set_node_attributes(g, {i: np.random.random() for i in g.nodes}, 'c')
#             for seed in seeds:
#                 model, _ = mvc_model(g, use_random_branching=False)
#                 model.setBoolParam('randomization/permutevars', True)
#                 model.setIntParam('randomization/permutationseed', seed)
#                 model.setIntParam('randomization/randomseedshift', seed)
#                 model.hideOutput(True)
#                 model.optimize()
#                 assert model.getGap() == 0
#                 stats['time'].append(model.getSolvingTime())
#                 stats['lp_iterations'].append(model.getNLPIterations())
#                 stats['nodes'].append(model.getNNodes())
#                 stats['applied'].append(model.getNCutsApplied())
#                 stats['lp_rounds'].append(model.getNLPs())
#             results[gs]['xticks'].append('{}({:.2f})'.format(m, nx.density(g)))
#             for k, vs in stats.items():
#                 results[gs][k].append(np.mean(vs))
#
#     with open(f'sweet_spot_results/mvc-ba-sweet-spot-results-bnc.pkl', 'wb') as f:
#         pickle.dump(results, f)
#
# with open(f'sweet_spot_results/mvc-ba-sweet-spot-results-bnc.pkl', 'rb') as f:
#     results = pickle.load(f)
#
# # plot for each graph size:
# # metric vs. density for metric in stats.keys
# fig = plt.figure(figsize=(16, 10))
# axes = fig.subplots(5, 4)
# plt.suptitle('MVC Branch & Cut')
# for col, gs in enumerate(graph_sizes):
#     x_labels = results[gs]['xticks']
#     for row, (k, vals) in enumerate([(kk, v) for kk, v in results[gs].items() if kk != 'xticks']):
#         ax = axes[row, col]
#         # ax.plot(np.arange(len(vals)), vals)
#         ax.plot(x_labels, vals)
#         if col == 0:
#             ax.set_ylabel(k)
#         ax.get_xaxis().set_visible(False)
#
#     axes[4, col].get_xaxis().set_visible(True)
#     # axes[4, col].set_xticks(list(range(len(x_labels))), minor=True)
#     # axes[4, col].set_xticklabels(x_labels, minor=True, rotation=45)
#     fig.autofmt_xdate(rotation=45)
#     axes[4, col].set_xlabel('m(denisty)')
#     axes[0, col].set_title(f'size={gs}')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# # plt.show()
# plt.savefig('sweet_spot_results/mvc-ba-sweet-spot-bnc.png')

################ root node #################
lp_iterations_limit = 5000

# MAXCUT in root node:
if False:
    results = {gs: {'density': [],
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
            nx.set_node_attributes(g, {e: np.random.random() for e in g.edges}, 'weight')
            for seed in seeds:
                model, _, _ = maxcut_mccormic_model(g)
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
            results[gs]['density'].append('{}({:.2f})'.format(m, nx.density(g)))
            for k, vs in stats.items():
                results[gs][k].append(np.mean(vs))

    with open(f'sweet_spot_results/maxcut-ba-sweet-spot-results-lpiter{lp_iterations_limit}.pkl', 'wb') as f:
        pickle.dump(results, f)

with open(f'sweet_spot_results/maxcut-ba-sweet-spot-results-lpiter{lp_iterations_limit}.pkl', 'rb') as f:
    results = pickle.load(f)

# plot for each graph size:
# metric vs. density for metric in stats.keys
fig = plt.figure(figsize=(16, 10))
axes = fig.subplots(5, 3)
plt.suptitle(f'MAXCUT BA Root Node (LP Iter Limit = {lp_iterations_limit})')
for col, gs in enumerate(graph_sizes):
    x_labels = results[gs]['density']
    for row, (k, vals) in enumerate([(kk, v) for kk, v in results[gs].items() if kk != 'density']):
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
plt.savefig(f'sweet_spot_results/maxcut-ba-sweet-spot-lpiter{lp_iterations_limit}.png')

# plot dual bound graphs of each graph size to see if plateau was reached:
full_res = {}
if False:
    for gs, m in zip(graph_sizes, [20,35,50]):
        g = nx.barabasi_albert_graph(n=gs, m=m)
        nx.set_node_attributes(g, {e: np.random.random() for e in g.edges}, 'weight')
        for seed in seeds:
            model, _, _ = maxcut_mccormic_model(g)
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
            sepa.update_stats()
            full_res[gs] = sepa.stats
            full_res[gs]['density'] = '{}({:.2f})'.format(m, nx.density(g))
    with open(f'sweet_spot_results/maxcut-ba-full-res-lpiter{lp_iterations_limit}.pkl', 'wb') as f:
        pickle.dump(full_res, f)

with open(f'sweet_spot_results/maxcut-ba-full-res-lpiter{lp_iterations_limit}.pkl', 'rb') as f:
    full_res = pickle.load(f)

fig = plt.figure(figsize=(16,10))
axes = fig.subplots(3, 4)
for row, (color, (gs, stats)) in enumerate(zip(['g','b', 'r'], full_res.items())):
    stats = full_res[gs]
    axes[row, 0].plot(stats['lp_iterations'], stats['dualbound'], color, label=f'size{gs}-density{stats["density"]}')
    axes[row, 1].plot(stats['lp_iterations'], stats['gap'], color, label=f'size{gs}-density{stats["density"]}')
    axes[row, 2].plot(stats['lp_iterations'], stats['solving_time'], color, label=f'size{gs}-density{stats["density"]}')
    axes[row, 3].plot(stats['solving_time'], stats['gap'], color, label=f'size{gs}-density{stats["density"]}')

    # axes[0, 0].set_title('default')
    # axes[2, 0].set_xlabel('lp iter')
    # axes[2, 1].set_xlabel('lp iter')
    # axes[0, 1].set_title('aggressive')
# for row in range(3):
axes[0, 0].set_ylabel(f'size={graph_sizes[0]}')
axes[1, 0].set_ylabel(f'size={graph_sizes[1]}')
axes[2, 0].set_ylabel(f'size={graph_sizes[2]}')
axes[2, 0].set_xlabel('lp iterations')
axes[2, 1].set_xlabel('lp iterations')
axes[2, 2].set_xlabel('lp iterations')
axes[2, 3].set_xlabel('time')
axes[0, 0].set_title('db vs. lp iterations')
axes[0, 1].set_title('gap vs. lp iterations')
axes[0, 2].set_title('time vs. lp iterations')
axes[0, 3].set_title('gap vs. time')
axes[0, 3].legend()
axes[1, 3].legend()
axes[2, 3].legend()
plt.suptitle(f'MAXCUT BA (lp iter limit = {lp_iterations_limit})')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(f'sweet_spot_results/maxcut-ba-curves-lpiter-{lp_iterations_limit}.png')


print('finished')
