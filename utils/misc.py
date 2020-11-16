import networkx as nx
from tqdm import tqdm
import os
from argparse import ArgumentParser
from itertools import product
import pickle


def test_isomorphism(datadir, remove=False):
    """ reads all graphs in datadir, and prints out files with isomorphism """
    filenames = os.listdir(datadir)
    nfiles = len(filenames)
    edge_match_fn = lambda e1, e2: e1['weight'] == e2['weight']
    n_isomorphic = 0
    graphs = {}
    for file in tqdm(filenames, desc='loading graphs'):
        with open(os.path.join(datadir, file), 'rb') as f:
            graphs[file], _ = pickle.load(f)

    for f1 in tqdm(filenames, desc='testing isomorphism'):
        if len(graphs) == 0:
            break
        if f1 not in graphs.keys():
            continue
        g1 = graphs.pop(f1)
        for f2, g2 in list(graphs.items()):
            if nx.is_isomorphic(g1, g2, edge_match=edge_match_fn):
                n_isomorphic += 1
                graphs.pop(f2)
                if remove:
                    os.remove(os.path.join(datadir, f2))
    print('number of isomorphic pairs: ', n_isomorphic)


if __name__ == '__main__':
    # datadir = '/home/avrech/learning2cut/experiments/dqn/data/maxcut/validset25/barabasi-albert-n25-m10-weights-normal-seed36/'
    datadir = '/home/avrech/learning2cut/experiments/dqn/data/maxcut/trainset25/barabasi-albert-n25-m10-weights-normal-seed46'
    test_isomorphism(datadir)
    print('finished')