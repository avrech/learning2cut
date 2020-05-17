"""
Graph type: Barabasi-Albert
MaxCut formulation: McCormic
Solver: SCIP + cycles + default cut selection
Store optimal_dualbound as baseline
Store average of: lp_iterations, optimal_dualbound, initial_dualbound, initial_gap
Each graph is stored with its baseline in graph_<worker_id>_<idx>.pkl
"""

from utils.scip_models import maxcut_mccormic_model
from separators.mccormick_cycle_separator import MccormickCycleSeparator
import pickle
import os
import numpy as np
from tqdm import tqdm
import networkx as nx


def generate_dataset(config):
    ngraphs = config['ngraphs']
    nworkers = config['nworkers']
    workerid = config['workerid']
    if config['workerid'] == config['nworkers'] - 1:
        worker_ngraphs = int(ngraphs - (nworkers-1)*np.floor(ngraphs / nworkers))
    else:
        worker_ngraphs = int(np.floor(ngraphs / nworkers))

    n = config["graph_size"]
    m = config["barabasi_albert_m"]
    weights = config["weights"]
    dataset_generation_seed = config["dataset_generation_seed"]

    dataset_dir = os.path.join(config['datadir'],
                               config['dataset'],
                               f"barabasi-albert-n{n}-m{m}-weights-{weights}-seed{dataset_generation_seed}")
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    for graph_idx in range(worker_ngraphs):
        filepath = os.path.join(dataset_dir, f"graph_{workerid}_{graph_idx}.pkl")
        if not os.path.exists(filepath):
            # generate random graph
            # shift the random seed to generate diversity among workers/graphs
            barabasi_albert_seed = ((1 + workerid) * 223 * graph_idx) * dataset_generation_seed
            np.random.seed(barabasi_albert_seed)
            G = nx.barabasi_albert_graph(n, m, seed=barabasi_albert_seed)
            if weights == 'ones':
                w = 1
            elif weights == 'uniform01':
                w = {e: np.random.uniform() for e in G.edges}
            elif weights == 'normal':
                w = {e: np.random.normal() for e in G.edges}
            nx.set_edge_attributes(G, w, name='weight')

            # create a scip model
            if config['solver'] == 'scip':
                model, x, y = maxcut_mccormic_model(G)
                model.setRealParam('limits/time', config['time_limit_sec'])
                sepa_hparams = {
                    'max_per_root': 500,
                    'max_per_node': 100,
                    'max_per_round': -1,
                    'cuts_budget': 1000000,
                    'max_cuts_applied_node': 20,
                    'max_cuts_applied_root': 20,
                    'record': True,
                }
                sepa = MccormickCycleSeparator(G=G, x=x, y=y, hparams=sepa_hparams)
                model.includeSepa(sepa, "MLCycles",
                                  "Generate cycle inequalities for MaxCut using McCormic variables exchange",
                                  priority=1000000,
                                  freq=1)

                model.optimize()
                sepa.finish_experiment()

                x_values = {}
                y_values = {}
                sol = model.getBestSol()
                for i in G.nodes:
                    x_values[i] = model.getSolVal(sol, x[i])
                for e in G.edges:
                    y_values[e] = model.getSolVal(sol, y[e])
                if model.getGap() > 0:
                    print('WARNING: graph no.{} not solved!')
                cut = {(i, j): int(x_values[i] != x_values[j]) for (i, j) in G.edges}
                nx.set_edge_attributes(G, cut, name='cut')
                nx.set_edge_attributes(G, y_values, name='y')
                nx.set_node_attributes(G, x_values, name='x')
                baseline = {'optimal_value': model.getObjVal()}
                with open(filepath, 'wb') as f:
                    pickle.dump((G, baseline), f)

            if config['solver'] == 'gurobi':
                from utils.gurobi_models import maxcut_mccormic_model as gurobi_model
                model, x, y = gurobi_model(G)
                model.optimize()
                x_values = {}
                y_values = {}

                for i in G.nodes:
                    x_values[i] = x[i].X
                for e in G.edges:
                    y_values[e] = y[e].X

                cut = {(i, j): int(x_values[i] != x_values[j]) for (i, j) in G.edges}
                nx.set_edge_attributes(G, cut, name='cut')
                nx.set_edge_attributes(G, y_values, name='y')
                nx.set_node_attributes(G, x_values, name='x')
                baseline = {'optimal_value': model.getObjective().getValue()}
                with open(filepath, 'wb') as f:
                    pickle.dump((G, baseline), f)
            print('saved graph to ', filepath)


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/dqn',
                        help='path to generate/read data')
    parser.add_argument('--configfile', type=str, default='trainset_config.yaml',
                        help='path to config file')
    parser.add_argument('--workerid', type=int, default=0,
                        help='worker id')
    parser.add_argument('--nworkers', type=int, default=1,
                        help='total number of workers')
    args = parser.parse_args()

    with open(args.configfile) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # config = dataset_config['constants']
    # for k, v in dataset_config['sweep'].items():
    #     if k == 'graph_idx':
    #         config[k] = args.graphidx
    #     else:
    #         config[k] = v['values'][0]
    for k, v in vars(args).items():
        config[k] = v

    generate_dataset(config)

