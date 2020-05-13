from tqdm import tqdm
import networkx as nx
import pickle
import os
import numpy as np
from utils.scip_models import maxcut_mccormic_model
from separators.mccormick_cycle_separator import MccormickCycleSeparator


def generate_data(sweep_config, data_dir, solve_maxcut=False, time_limit=60, use_cycles=True):
    """
    Generate networkx.barabasi_albert_graph(n,m,seed) according to the values
    specified in sweep_config, and set edge weights as well.
    Save each graph in a graph_idx<idx>.pkl file in data_dir.
    If solve_maxcut, solve the instances using cycle_inequalities baseline (for acceleration),
    and store a binary edge attribute 'cut' = 1 if edge is cut, and 0 otherwise.
    :param sweep_config:
    :param data_dir:
    :param solve_maxcut: Solve the maxcut instance, and store the solution in edge attributes 'cut'.
    :param time_limit: time limit for solving the instances in seconds. If limit exceeded, instance is discarded
    :return: None
    """
    # dataset generation
    n = sweep_config['constants']["graph_size"]
    m = sweep_config['constants']["barabasi_albert_m"]
    weights = sweep_config['constants']["weights"]
    dataset_generation_seed = sweep_config['constants']["dataset_generation_seed"]

    data_abspath = os.path.join(data_dir, "barabasi-albert-n{}-m{}-weights-{}-seed{}".format(n, m, weights, dataset_generation_seed))
    if not os.path.isdir(data_abspath):
        os.makedirs(data_abspath)
    data_abspath = os.path.abspath(data_abspath)

    for graph_idx in tqdm(range(sweep_config['sweep']['graph_idx']['range'])):
        filepath = os.path.join(data_abspath, "graph_idx_{}.pkl".format(graph_idx))
        if not os.path.exists(filepath):
            # generate the graph and save in a pickle file
            # set randomization for reproducing dataset
            barabasi_albert_seed = (1 + 223 * graph_idx) * dataset_generation_seed
            np.random.seed(barabasi_albert_seed)
            G = nx.barabasi_albert_graph(n, m, seed=barabasi_albert_seed)
            if weights == 'ones':
                w = 1
            elif weights == 'uniform01':
                w = {e: np.random.uniform() for e in G.edges}
            elif weights == 'normal':
                w = {e: np.random.normal() for e in G.edges}
            nx.set_edge_attributes(G, w, name='weight')
            if solve_maxcut:
                model, x, y = maxcut_mccormic_model(G)
                # model.setRealParam('limits/time', 1000 * 1)
                """ Define a controller and appropriate callback to add user's cuts """
                hparams = {'max_per_root': 500,
                           'max_per_node': 100,
                           'max_per_round': -1,
                           'criterion': 'most_violated_cycle',
                           'cuts_budget': 1000000}
                if use_cycles:
                    ci_cut = MccormickCycleSeparator(G=G, x=x, y=y, hparams=hparams)
                    model.includeSepa(ci_cut, "MLCycles",
                                      "Generate cycle inequalities for MaxCut using McCormic variables exchange",
                                      priority=1000000,
                                      freq=1)
                model.setRealParam('limits/time', time_limit)
                model.optimize()
                x_values = np.zeros(G.number_of_nodes())
                sol = model.getBestSol()
                for i in range(G.number_of_nodes()):
                    x_values[i] = model.getSolVal(sol, x[i])
                if model.getGap() > 0:
                    print('WARNING: graph no.{} not solved!')
                cut = {(i, j): int(x_values[i] != x_values[j]) for (i, j) in G.edges}
                nx.set_edge_attributes(G, cut, name='cut')


            with open(filepath, 'wb') as f:
                pickle.dump(G, f)

    return data_abspath
