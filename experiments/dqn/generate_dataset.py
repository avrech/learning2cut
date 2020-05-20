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
import networkx as nx
from tqdm import tqdm
from ray import tune


def generate_dataset(config):
    ngraphs = config['ngraphs']
    nworkers = config['nworkers']
    workerid = config['workerid']
    save_all_stats = config.get('save_all_stats', False)

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

    for graph_idx in tqdm(range(worker_ngraphs), desc=f'Worker {workerid}'):
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

            if config['solver'] == 'scip':
                sepa_hparams = {
                    'max_per_root': 1000000,
                    'max_per_node': 1000000,
                    'max_per_round': -1,
                    'cuts_budget': 1000000,
                    'max_cuts_applied_node': 1000000,
                    'max_cuts_applied_root': 1000000,
                    'record': True,
                }
                # solve with B&C and the default cut selection
                bnc_model, x, y = maxcut_mccormic_model(G, use_general_cuts=False)
                bnc_sepa = MccormickCycleSeparator(G=G, x=x, y=y, hparams=sepa_hparams)
                bnc_model.includeSepa(bnc_sepa, "MLCycles",
                                  "Generate cycle inequalities for MaxCut using McCormic variables exchange",
                                  priority=1000000,
                                  freq=1)
                # set randomization seed

                if config.get('scip_seed', None) is not None:
                    bnc_model.setBoolParam('randomization/permutevars', True)
                    bnc_model.setIntParam('randomization/permutationseed', config.get('scip_seed'))
                    bnc_model.setIntParam('randomization/randomseedshift', config.get('scip_seed'))
                bnc_model.setRealParam('limits/time', config['time_limit_sec'])
                bnc_model.hideOutput(quiet=config.get('quiet', False))
                bnc_model.optimize()
                bnc_sepa.finish_experiment()

                # now solve without branching
                rootonly_model, rootonly_x, rootonly_y = maxcut_mccormic_model(G, use_general_cuts=False)
                rootonly_sepa = MccormickCycleSeparator(G=G, x=rootonly_x, y=rootonly_y, hparams=sepa_hparams)
                rootonly_model.includeSepa(
                    rootonly_sepa, "MLCycles",
                    "Generate cycle inequalities for MaxCut using McCormic variables exchange",
                    priority=1000000,
                    freq=1
                )
                rootonly_model.setRealParam('limits/time', config['time_limit_sec'])
                rootonly_model.setLongintParam('limits/nodes', 1)  # solve only at the root node
                # rootonly_model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever

                # set up randomization
                if config.get('scip_seed', None) is not None:
                    rootonly_model.setBoolParam('randomization/permutevars', True)
                    rootonly_model.setIntParam('randomization/permutationseed', config.get('scip_seed'))
                    rootonly_model.setIntParam('randomization/randomseedshift', config.get('scip_seed'))
                rootonly_model.hideOutput(quiet=config.get('quiet', False))
                rootonly_model.optimize()
                rootonly_sepa.finish_experiment()

                x_values = {}
                y_values = {}
                sol = bnc_model.getBestSol()
                for i in G.nodes:
                    x_values[i] = bnc_model.getSolVal(sol, x[i])
                for e in G.edges:
                    y_values[e] = bnc_model.getSolVal(sol, y[e])

                if bnc_model.getGap() > 0:
                    print('WARNING: {} not solved to optimality!'.format(filepath))
                    is_optimal = False
                else:
                    is_optimal = True

                cut = {(i, j): int(x_values[i] != x_values[j]) for (i, j) in G.edges}
                nx.set_edge_attributes(G, cut, name='cut')
                nx.set_edge_attributes(G, y_values, name='y')
                nx.set_node_attributes(G, x_values, name='x')
                baseline = {'optimal_value': bnc_model.getObjVal(),
                            'is_optimal': is_optimal,
                            'lp_iterations_limit': rootonly_model.getNLPIterations()
                            }
                if save_all_stats:
                    baseline['bnc_stats'] = bnc_sepa.stats
                    baseline['rootonly_stats'] = rootonly_sepa.stats
                with open(filepath, 'wb') as f:
                    pickle.dump((G, baseline), f)

            if config['solver'] == 'gurobi':
                from utils.gurobi_models import maxcut_mccormic_model as gurobi_model
                bnc_model, x, y = gurobi_model(G)
                bnc_model.optimize()
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
                baseline = {'optimal_value': bnc_model.getObjective().getValue()}
                with open(filepath, 'wb') as f:
                    pickle.dump((G, baseline), f)
            if not config.get('quiet', False):
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
    parser.add_argument('--mp', type=str, default='none',
                        help='use ray [ray] or multiprocessing [mp] with nworkers')
    parser.add_argument('--quiet', action='store_true',
                        help='hide scip solving messages')
    args = parser.parse_args()

    with open(args.configfile) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in vars(args).items():
        config[k] = v

    if args.mp == 'mp':
        from multiprocessing import Pool
        configs = []
        for workerid in range(args.nworkers):
            cfg = {k: v for k, v in config.items()}
            cfg['workerid'] = workerid
            configs.append(cfg)

        with Pool() as p:
            res = p.map_async(generate_dataset, configs)
            res.wait()
            print(f'multiprocessing finished {"successfully" if res.successful() else "with errors"}')

    elif args.mp == 'ray':
        from ray.tune import track
        track.init(experiment_dir=args.datadir)
        configs = dict()
        for k, v in config.items():
            configs[k] = tune.grid_search([v])
        configs['workerid'] = tune.grid_search(np.arange(args.nworkers))
        analysis = tune.run(generate_dataset,
                            config=configs,
                            resources_per_trial={'cpu': 1, 'gpu': 0},
                            local_dir=args.datadir,
                            trial_name_creator=None,
                            max_failures=1  # TODO learn how to recover from checkpoints
                            )

    else:
        generate_dataset(config)
    print('finished')

