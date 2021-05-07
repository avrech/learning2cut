from utils.scip_models import mvc_model, CSBaselineSepa, set_aggresive_separation, CSResetSepa, maxcut_mccormic_model
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from utils.functions import get_normalized_areas
from tqdm import tqdm
import pickle
import pandas as pd

ROOTDIR = 'results'
import os
if not os.path.isdir(ROOTDIR):
    os.makedirs(ROOTDIR)

print('############### generating data ###############')

if not os.path.exists(os.path.join(ROOTDIR, 'data.pkl')):
    data = {'mvc': {}, 'maxcut': {}}
    # mvc
    graph_sizes = [60, 100, 150]
    densities = [0.25, 0.15, 0.1]
    for gs, density in tqdm(zip(graph_sizes, densities), desc='generating graphs for MVC'):
        g = nx.erdos_renyi_graph(n=gs, p=density, directed=False)
        nx.set_node_attributes(g, {i: np.random.random() for i in g.nodes}, 'c')
        model, _ = mvc_model(g, use_random_branching=False, allow_restarts=True)
        model.hideOutput(True)
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
    ms = [15, 20, 40]
    for gs, m in tqdm(zip(graph_sizes, ms), desc='generating graphs for MAXCUT'):
        g = nx.barabasi_albert_graph(n=gs, m=m)
        nx.set_edge_attributes(g, {e: np.random.random() for e in g.edges}, 'weight')
        model, _, _ = maxcut_mccormic_model(g, use_random_branching=False, allow_restarts=True)
        model.hideOutput(True)
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



print('############### running simple baselines ###############')
# run default, 15-random, 15-most-violated and all-cuts baselines
seeds = [46, 72, 101]
problems = ['mvc', 'maxcut']
simple_baselines = ['default', '15_random', '15_most_violated', 'all_cuts']
if not os.path.exists(f'{ROOTDIR}/results.pkl'):
    results = {p: {b: {} for b in simple_baselines} for p in problems}
    for problem, graphs in data.items():
        for baseline in tqdm(simple_baselines, desc='run simple baselines'):
            graphs = data[problem]
            for graph_size, (g, info) in graphs.items():
                results[problem][baseline][graph_size] = {}
                for seed in seeds:
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
    with open(f'{ROOTDIR}/results.pkl', 'wb') as f:
        pickle.dump(results, f)
else:
    with open(f'{ROOTDIR}/results.pkl', 'rb') as f:
        results = pickle.load(f)

if not os.path.exists(f'{ROOTDIR}/scip_tuned_results.pkl'):
    print('run scip tuned baseline first, then re-run again.')
    exit(0)

with open(f'{ROOTDIR}/scip_tuned_results.pkl', 'rb') as f:
    scip_tuned_results = pickle.load(f)

if not os.path.exists(f'{ROOTDIR}/scip_adaptive.pkl'):
    print('run scip adaptive baseline first, then re-run again.')
    exit(0)

with open(f'{ROOTDIR}/scip_adaptive_results.pkl', 'rb') as f:
    scip_adaptive_results = pickle.load(f)

print('############### analyzing results ###############')

# todo - combine all results, print stats, and plot curves for each problem and graph
