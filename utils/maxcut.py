from tqdm import tqdm
import networkx as nx
import pickle
import os
import numpy as np
from utils.scip_models import maxcut_mccormic_model
from separators.mccormick_cycle_separator import MccormickCycleSeparator
from ray import tune
from ray.tune import track
from itertools import product
from multiprocessing import Pool


def run_solver(config):
    config = config['config']
    G = config['G']
    filepath = config['filepath']
    use_cycles = config['use_cycles']
    time_limit = config['time_limit']

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
    x_values = {}
    y_values = {}
    sol = model.getBestSol()
    for i in G.nodes:
        x_values[i] = model.getSolVal(sol, x[i])
    for ij in G.edges:
        y_values[ij] = model.getSolVal(sol, y[ij])
    if model.getGap() > 0:
        print('WARNING: graph no.{} not solved!')
    cut = {(i, j): int(x_values[i] != x_values[j]) for (i, j) in G.edges}
    nx.set_edge_attributes(G, cut, name='cut')
    # store the solver solution
    nx.set_edge_attributes(G, y_values, name='y')
    nx.set_node_attributes(G, x_values, name='x')
    with open(filepath, 'wb') as f:
        pickle.dump(G, f)


def generate_data(sweep_config, data_dir, solve_maxcut=False, time_limit=60, use_cycles=False, mp='ray'):
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
    if "graph_size" in sweep_config['sweep'].keys():
        n_list = sweep_config['sweep']["graph_size"]['values']
    else:
        n_list = [sweep_config['constants']["graph_size"]]
    if "barabasi_albert_m" in sweep_config['sweep'].keys():
        m_list = sweep_config['sweep']["barabasi_albert_m"]['values']
    else:
        m_list = [sweep_config['constants']["barabasi_albert_m"]]
    if "weights" in sweep_config['sweep'].keys():
        weights_list = sweep_config['sweep']["weights"]['values']
    else:
        weights_list = [sweep_config['constants']["weights"]]
    if "dataset_generation_seed" in sweep_config['sweep'].keys():
        dataset_generation_seed_list = sweep_config['sweep']["dataset_generation_seed"]['values']
    else:
        dataset_generation_seed_list = [sweep_config['constants']["dataset_generation_seed"]]

    # configs for parallel solving if solve_maxcut == True
    configs = []
    paths = {}
    for n in n_list:
        paths[n] = {}
        for m in m_list:
            paths[n][m] = {}
            for weights in weights_list:
                paths[n][m][weights] = {}
                for dataset_generation_seed in dataset_generation_seed_list:
                    data_abspath = os.path.join(data_dir, "barabasi-albert-n{}-m{}-weights-{}-seed{}".format(n, m, weights, dataset_generation_seed))
                    if not os.path.isdir(data_abspath):
                        os.makedirs(data_abspath)
                    data_abspath = os.path.abspath(data_abspath)
                    paths[n][m][weights][dataset_generation_seed] = data_abspath

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
                                # store in list and solve later
                                configs.append({'G': G, 'filepath': filepath, 'time_limit': time_limit, 'use_cycles': use_cycles})
                            else:
                                # store as is and continue
                                with open(filepath, 'wb') as f:
                                    pickle.dump(G, f)
    if solve_maxcut:
        if mp == 'ray':
            # run solver in parallel and solve all graphs
            tune_search_space = {'config': tune.grid_search(configs)}
            track.init()
            analysis = tune.run(run_solver,
                                config=tune_search_space,
                                resources_per_trial={'cpu': 1, 'gpu': 0},
                                local_dir='tmpdir',
                                trial_name_creator=None,
                                max_failures=1  # TODO learn how to recover from checkpoints
                                )
        elif mp == 'mp':

            with Pool() as p:
                res = p.map_async(run_solver, configs)
                res.wait()
                print(f'multiprocessing finished {"successfully" if res.successful() else "with errors"}')
        else:
            print('solving graphs sequentially')
            for cfg in configs:
                run_solver({'config': cfg})
            print('finished')

    return paths


