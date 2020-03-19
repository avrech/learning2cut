from tqdm import tqdm
import networkx as nx
import pickle
import os
import numpy as np


def generate_data(sweep_config, data_dir):
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
            with open(filepath, 'wb') as f:
                pickle.dump(G, f)

    return data_abspath
