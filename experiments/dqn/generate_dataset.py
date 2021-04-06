"""
Graph type: Barabasi-Albert
MaxCut formulation: McCormic
Solver: SCIP + cycles + default cut selection
Store optimal_dualbound as baseline
Store average of: lp_iterations, optimal_dualbound, initial_dualbound, initial_gap
Each graph is stored with its baseline in graph_<worker_id>_<idx>.pkl
"""
from utils.scip_models import mvc_model, maxcut_mccormic_model, CSBaselineSepa
import pickle
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
from ray import tune
from copy import deepcopy
from utils.functions import get_normalized_areas


def generate_graphs(configs):
    """
    Generate graphs and ensures no isomorphism
    """
    all_graphs = {i: [] for i in range(101)}
    edge_match_fn = lambda e1, e2: e1['weight'] == e2['weight']
    total_generated = 0
    unique_generated = 0
    for worker_config in configs:
        nworkers = worker_config['nworkers']
        workerid = worker_config['workerid']
        workerid = worker_config['workerid']
        datadir = worker_config['datadir']
        for dataset_config in worker_config['datasets'].values():
            ngraphs = dataset_config['ngraphs']
            if ngraphs >= nworkers:
                if workerid == nworkers - 1:
                    # so assign to the last worker what left to complete ngraphs
                    worker_ngraphs = int(ngraphs - (nworkers - 1) * np.floor(ngraphs / nworkers))
                else:
                    worker_ngraphs = int(np.floor(ngraphs / nworkers))
            else:
                # assign 1 graph to each one of the first ngraphs workers, and terminate the other threads
                if workerid < ngraphs:
                    worker_ngraphs = 1
                else:
                    # there is no work left to do.
                    continue

            nmin, nmax = dataset_config["graph_size"]['min'], dataset_config["graph_size"]['max']
            m = dataset_config["barabasi_albert_m"]
            weights = dataset_config["weights"]
            seed = dataset_config["seed"]
            np.random.seed(seed)

            dataset_dir = os.path.join(datadir,
                                       dataset_config['dataset_name'],
                                       f"barabasi-albert-nmin{nmin}-nmax{nmax}-m{m}-weights-{weights}-seed{seed}")
            if not os.path.isdir(dataset_dir):
                os.makedirs(dataset_dir)

            for graph_idx in tqdm(range(worker_ngraphs), desc=f'Worker {workerid}: generating graphs...'):
                filepath = os.path.join(dataset_dir, f"graph_{workerid}_{graph_idx}.pkl")
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        G, baseline = pickle.load(f)
                    all_graphs[len(G.nodes)].append(G)

                else:
                    # generate random graph
                    while True:
                        n = np.random.randint(nmin, nmax)
                        G = nx.barabasi_albert_graph(n, m) # , seed=seed
                        total_generated += 1
                        if weights == 'ones':
                            w = 1
                        elif weights == 'uniform01':
                            w = {e: np.random.uniform() for e in G.edges}
                        elif weights == 'normal':
                            w = {e: np.random.normal() for e in G.edges}
                        nx.set_edge_attributes(G, w, name='weight')
                        for G2 in all_graphs[n]:
                            if nx.is_isomorphic(G, G2, edge_match=edge_match_fn):
                                continue

                        # add unique graph to dataset
                        unique_generated += 1
                        all_graphs[n].append(G)
                        with open(filepath, 'wb') as f:
                            pickle.dump((G, None), f)
                        break

    print('Total graphs generated: ', total_generated)
    print('Unique graphs generated: ', unique_generated)


def solve_graphs(worker_config):
    """
    Worker thread. Solves graphs assigned to worker according to the specifications in config.
    """
    nworkers = worker_config['nworkers']
    workerid = worker_config['workerid']
    datadir = worker_config['datadir']
    quiet = worker_config.get('quiet', False)
    for dataset_config in worker_config['datasets'].values():

        save_all_stats = dataset_config.get('save_all_stats', False)

        ngraphs = dataset_config['ngraphs']

        if ngraphs >= nworkers:
            if workerid == nworkers - 1:
                # so assign to the last worker what left to complete ngraphs
                worker_ngraphs = int(ngraphs - (nworkers - 1) * np.floor(ngraphs / nworkers))
            else:
                worker_ngraphs = int(np.floor(ngraphs / nworkers))
        else:
            # assign 1 graph to each one of the first ngraphs workers, and terminate the other threads
            if workerid < ngraphs:
                worker_ngraphs = 1
            else:
                # there is no work left to do.
                continue

        nmin, nmax = dataset_config["graph_size"]['min'], dataset_config["graph_size"]['max']
        m = dataset_config["barabasi_albert_m"]
        weights = dataset_config["weights"]
        seed = dataset_config["seed"]
        np.random.seed(seed)
        dataset_dir = os.path.join(datadir,
                                   dataset_config['dataset_name'],
                                   f"barabasi-albert-nmin{nmin}-nmax{nmax}-m{m}-weights-{weights}-seed{seed}")

        for graph_idx in tqdm(range(worker_ngraphs), desc=f'Worker {workerid}: solving graphs...'):
            filepath = os.path.join(dataset_dir, f"graph_{workerid}_{graph_idx}.pkl")
            with open(filepath, 'rb') as f:
                G, info = pickle.load(f)
                if info is not None:
                    # already solved
                    continue

            # if dataset_config['baseline_solver'] == 'scip':
            # todo set problem type in config, and define graph sizes appropriately.
            problems = ['MAXCUT-GP', 'MAXCUT-GP-CYCLES']
            info = {}
            for problem in problems:
                # solve with B&C and the default cut selection
                if 'MAXCUT' in problem:
                    bnc_model, x, _ = maxcut_mccormic_model(G, use_general_cuts=True, use_cycles='CYCLES' in problem)
                elif problem == 'MVC':
                    bnc_model, x = mvc_model(G, use_general_cuts=True)
                    # set arbitrary random seed only for reproducibility and debug - doesn't matter for results
                bnc_seed = 72
                bnc_model.setBoolParam('randomization/permutevars', True)
                bnc_model.setIntParam('randomization/permutationseed', bnc_seed)
                bnc_model.setIntParam('randomization/randomseedshift', bnc_seed)
                bnc_model.setRealParam('limits/time', dataset_config['time_limit_sec'])
                bnc_model.hideOutput(quiet=quiet)
                bnc_sepa = CSBaselineSepa()
                bnc_model.includeSepa(bnc_sepa, 'BSL', 'collect stats', priority=-10000000, freq=1)
                bnc_model.optimize()
                bnc_sepa.update_stats()

                # solve according to all baselines
                # solve also for the default baseline for having training AUC.
                # solve for three baselines 10-random, 10-most-violated, and default.

                # for evaluation we need SCIP default cut selection stats.
                # so now solve without branching, limiting the LP iterations to sufficiently large number
                # such that the solver will reach the plateau
                # solve for all scip seeds provided, and store each seed stats separately
                baselines = ['default', '10_random', '10_most_violated']
                baseline_stats = {k: {} for k in baselines}
                for bsl in baselines:
                    if bsl != 'default' and not save_all_stats:
                        continue

                    for scip_seed in dataset_config['scip_seed']:
                        rootonly_model, _, _ = maxcut_mccormic_model(G, use_general_cuts=False)
                        rootonly_model.setRealParam('limits/time', dataset_config['time_limit_sec'])
                        rootonly_model.setLongintParam('limits/nodes', 1)  # solve only at the root node
                        rootonly_model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever
                        # set up randomization
                        rootonly_model.setBoolParam('randomization/permutevars', True)
                        rootonly_model.setIntParam('randomization/permutationseed', scip_seed)
                        rootonly_model.setIntParam('randomization/randomseedshift', scip_seed)
                        rootonly_model.hideOutput(quiet=quiet)
                        bsl_sepa = CSBaselineSepa(hparams={
                            'lp_iterations_limit': dataset_config['lp_iterations_limit'],
                            'criterion': bsl,
                        })
                        rootonly_model.includeSepa(bsl_sepa, 'BSL', f'enforce cut selection: {bsl}', priority=-1000000, freq=1)
                        rootonly_model.optimize()
                        bsl_sepa.update_stats()
                        assert bsl_sepa.stats['lp_iterations'][-1] <= dataset_config['lp_iterations_limit']
                        # compute some stats
                        db, gap, lp_iter = bsl_sepa.stats['dualbound'], bsl_sepa.stats['gap'], bsl_sepa.stats['lp_iterations']
                        db_auc = sum(get_normalized_areas(t=lp_iter, ft=db, t_support=dataset_config['lp_iterations_limit'], reference=bnc_model.getObjVal()))
                        gap_auc = sum(get_normalized_areas(t=lp_iter, ft=gap, t_support=dataset_config['lp_iterations_limit'], reference=0))
                        stats = {'db_auc': db_auc, 'gap_auc': gap_auc}
                        if save_all_stats:
                            stats.update(bsl_sepa.stats)
                        baseline_stats[bsl][scip_seed] = stats

                # summarize results for G
                # set warning for sub-optimality
                if bnc_model.getGap() > 0:
                    print('WARNING: {} not solved to optimality!'.format(filepath))
                    is_optimal = False
                else:
                    is_optimal = True

                # store the best solution found in G
                x_values = {}
                # y_values = {}
                sol = bnc_model.getBestSol()
                for i in G.nodes:
                    x_values[i] = bnc_model.getSolVal(sol, x[i])
                # for e in G.edges:
                #     y_values[e] = bnc_model.getSolVal(sol, x[e])

                # cut = {(i, j): int(x_values[i] != x_values[j]) for (i, j) in G.edges}
                # nx.set_edge_attributes(G, cut, name='cut')
                # nx.set_edge_attributes(G, y_values, name='y')
                nx.set_node_attributes(G, x_values, name='x')

                # store elementary stats needed for training
                info[problem] = {'optimal_value': bnc_model.getObjVal(),
                                 'is_optimal': is_optimal,
                                 'lp_iterations_limit': dataset_config['lp_iterations_limit'],
                                 'baselines': baseline_stats}

                # store extensive stats needed for evaluation
                if save_all_stats:
                    info['problem']['baselines']['bnc'] = {bnc_seed: bnc_sepa.stats}

            with open(filepath, 'wb') as f:
                pickle.dump((G, info), f)
                print('saved instance to ', filepath)

            # if dataset_config['baseline_solver'] == 'gurobi':
            #     from utils.gurobi_models import maxcut_mccormic_model as gurobi_model
            #     bnc_model, x, y = gurobi_model(G)
            #     bnc_model.optimize()
            #     x_values = {}
            #     y_values = {}
            #
            #     for i in G.nodes:
            #         x_values[i] = x[i].X
            #     for e in G.edges:
            #         y_values[e] = y[e].X
            #
            #     cut = {(i, j): int(x_values[i] != x_values[j]) for (i, j) in G.edges}
            #     nx.set_edge_attributes(G, cut, name='cut')
            #     nx.set_edge_attributes(G, y_values, name='y')
            #     nx.set_node_attributes(G, x_values, name='x')
            #     info = {'optimal_value': bnc_model.getObjective().getValue()}
            #     with open(filepath, 'wb') as f:
            #         pickle.dump((G, info), f)
            if not quiet:
                print('saved graph to ', filepath)


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/maxcut',
                        help='path to generate/read data')
    parser.add_argument('--configfile', type=str, default='configs/maxcut_data_config.yaml',
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

    # read data config
    with open(args.configfile) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # product the dataset configs and worker ids.
    configs = []
    # for dataset_name, dataset_config in config['datasets'].items():
    #     for k, v in vars(args).items():
    #         dataset_config[k] = v
    for workerid in range(args.nworkers):
        cfg = deepcopy(vars(args))
        cfg['datasets'] = config['datasets']
        cfg['workerid'] = workerid
        configs.append(cfg)

    # first generate all graphs in the main thread and ensure no isomorphism
    generate_graphs(configs)

    if args.mp == 'mp':
        from multiprocessing import Pool
        with Pool() as p:
            res = p.map_async(solve_graphs, configs)
            res.wait()
            print(f'multiprocessing finished {"successfully" if res.successful() else "with errors"}')

    elif args.mp == 'ray':
        from ray.tune import track
        track.init(experiment_dir=args.datadir)
        tune_configs = tune.grid_search(configs)
        analysis = tune.run(solve_graphs,
                            config=tune_configs,
                            resources_per_trial={'cpu': 1, 'gpu': 0},
                            local_dir=args.datadir,
                            trial_name_creator=None,
                            max_failures=1  # TODO learn how to recover from checkpoints
                            )
    else:
        # process sequentially without threading
        for cfg in configs:
            solve_graphs(cfg)
    print('finished')

