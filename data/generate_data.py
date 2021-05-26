"""
Graph type: Barabasi-Albert
MaxCut formulation: McCormic
Solver: SCIP + cycles + default cut selection
Store optimal_dualbound as baseline
Store average of: lp_iterations, optimal_dualbound, initial_dualbound, initial_gap
Each graph is stored with its baseline in graph_<worker_id>_<idx>.pkl
"""
from utils.scip_models import mvc_model, maxcut_mccormic_model, CSBaselineSepa, CSResetSepa, set_aggresive_separation
import pickle
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from utils.functions import get_normalized_areas
import ray
import zmq
import pyarrow as pa


def random_ba_graphs(ngraphs, nmin, nmax, m, weights):
    graphs = []
    for _ in tqdm(range(ngraphs), desc='generating graphs'):
        n = np.random.randint(nmin, nmax)
        G = nx.barabasi_albert_graph(n, m)
        if weights == 'ones':
            w = 1.0
            c = 1.0
        elif weights == 'uniform01':
            w = {e: np.random.uniform() for e in G.edges}
            c = {v: np.random.uniform() for v in G.nodes}
        elif weights == 'normal':
            w = {e: np.random.normal() for e in G.edges}
            c = {v: np.random.normal() for v in G.nodes}
        nx.set_edge_attributes(G, w, name='weight')
        nx.set_node_attributes(G, c, name='c')
        graphs.append(G)
    return graphs


def generate_graphs(config):
    """
    Generate graphs and ensures no isomorphism
    """
    problem = config['problem']
    datadir = config['datadir']
    datasets = {}
    for dataset_name, dataset_config in config['datasets'].items():
        ngraphs = dataset_config['ngraphs']
        nmin, nmax = dataset_config["graph_size"]['min'], dataset_config["graph_size"]['max']
        m = dataset_config["barabasi_albert_m"]
        weights = dataset_config["weights"]
        seed = config['graph_generator_seed']
        dataset_dir = os.path.join(datadir, problem,
                                   dataset_config['dataset_name'],
                                   f"barabasi-albert-nmin{nmin}-nmax{nmax}-m{m}-weights-{weights}-seed{seed}")
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)
        graphs = random_ba_graphs(ngraphs, nmin, nmax, m, weights)
        datasets[dataset_name] = {}
        for graph_idx, G in tqdm(enumerate(graphs), desc=f'saving graphs...'):
            filepath = os.path.join(dataset_dir, f"graph_{graph_idx}.pkl")
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    g, info = pickle.load(f)
                datasets[dataset_name][filepath] = (g, info)

            else:
                with open(filepath, 'wb') as f:
                    pickle.dump((G, None), f)
                datasets[dataset_name][filepath] = (g, None)
    return datasets


def solve_graphs(config, workerid, worker2main_port, main2worker_port):
    """
    Worker thread. Solves graphs assigned to worker according to the specifications in config.
    """
    context = zmq.context()
    recv_socket = context.socket(zmq.PULL)
    recv_socket.connect(main2worker_port)
    send_socket = context.socket(zmq.PUSH)
    send_socket.connect(f'tcp://127.0.0.1:{worker2main_port}')
    problem = config['problem']
    quiet = config.get('quiet', False)
    while True:
        msg = recv_socket.recv()
        dataset_name, filepath = pa.deserialize(msg)
        dataset_config = config['datasets'][dataset_name]
        save_all_stats = dataset_config.get('save_all_stats', False)
        # todo continue from here

        with open(filepath, 'rb') as f:
            G, info = pickle.load(f)
            assert info is None

        # if dataset_config['baseline_solver'] == 'scip':
        # todo set problem type in config, and define graph sizes appropriately.
        # solve with B&C and the default cut selection
        if problem == 'MAXCUT':
            bnc_model, x, _ = maxcut_mccormic_model(G, allow_restarts=True, use_heuristics=True, use_random_branching=True)
        elif problem == 'MVC':
            bnc_model, x = mvc_model(G, allow_restarts=True, use_heuristics=True, use_random_branching=True)
            # set arbitrary random seed only for reproducibility and debug - doesn't matter for results
        bnc_seed = 52
        bnc_model.setBoolParam('randomization/permutevars', True)
        bnc_model.setIntParam('randomization/permutationseed', bnc_seed)
        bnc_model.setIntParam('randomization/randomseedshift', bnc_seed)
        bnc_model.setRealParam('limits/time', dataset_config['time_limit_sec'])
        bnc_model.hideOutput(quiet=quiet)
        bnc_sepa = CSBaselineSepa()
        bnc_model.includeSepa(bnc_sepa, '#CS_baseline', 'collect stats', priority=-10000000, freq=1)
        bnc_model.optimize()
        bnc_sepa.update_stats()

        # solve according to all baselines
        # solve also for the default baseline for having training AUC.
        # solve for three baselines 15-random, 15-most-violated, and default.
        # solve for all scip seeds provided, and store each seed stats separately
        baselines = ['default', '15_random', '15_most_violated']
        baseline_stats = {k: {} for k in baselines}
        for bsl in baselines:
            if bsl != 'default' and not save_all_stats:
                continue
            # solve graphs for all seeds.
            # for training graphs solve only once for seed=223
            for scip_seed in dataset_config.get('scip_seed', [223]):
                if problem == 'MAXCUT':
                    bsl_model, _, _ = maxcut_mccormic_model(G, use_heuristics=config['use_heuristics'])
                elif problem == 'MVC':
                    bsl_model, _ = mvc_model(G, use_heuristics=config['use_heuristics'])
                if config['aggressive_separation']:
                    set_aggresive_separation(bsl_model)
                bsl_model.setRealParam('limits/time', dataset_config['time_limit_sec'])
                bsl_model.setLongintParam('limits/nodes', 1)  # solve only at the root node
                bsl_model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever
                # set up randomization
                bsl_model.setBoolParam('randomization/permutevars', True)
                bsl_model.setIntParam('randomization/permutationseed', scip_seed)
                bsl_model.setIntParam('randomization/randomseedshift', scip_seed)
                bsl_model.hideOutput(quiet=quiet)
                hparams = {
                    'lp_iterations_limit': dataset_config['lp_iterations_limit'],
                    'policy': bsl,
                    'reset_maxcuts': config['reset_maxcuts'],
                    'reset_maxcutsroot': config['reset_maxcutsroot']
                }
                bsl_sepa = CSBaselineSepa(hparams=hparams)
                bsl_model.includeSepa(bsl_sepa, f'#CS_{bsl}', f'enforce cut selection: {bsl}', priority=-1000000, freq=1)
                reset_sepa = CSResetSepa(hparams=hparams)
                bsl_model.includeSepa(reset_sepa, '#CS_reset', 'reset maxcutsroot', priority=9999999, freq=1)
                bsl_model.optimize()
                bsl_sepa.update_stats()
                assert bsl_sepa.stats['lp_iterations'][-1] <= dataset_config['lp_iterations_limit']
                # compute some stats
                db, gap, lp_iter = bsl_sepa.stats['dualbound'], bsl_sepa.stats['gap'], bsl_sepa.stats['lp_iterations']
                db_auc = sum(get_normalized_areas(t=lp_iter, ft=db, t_support=dataset_config['lp_iterations_limit'], reference=bnc_model.getObjVal()))
                gap_auc = sum(get_normalized_areas(t=lp_iter, ft=gap, t_support=dataset_config['lp_iterations_limit'], reference=0))
                stats = {'db_auc': db_auc,
                         'gap_auc': gap_auc,
                         'lp_iterations': bsl_sepa.stats['lp_iterations'],
                         'dualbound': bsl_sepa.stats['dualbound'],
                         'gap': bsl_sepa.stats['gap']}
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
        # we are interested in the node variables only (both for maxcut and mvc)
        x_values = {}
        sol = bnc_model.getBestSol()
        for i in G.nodes:
            x_values[i] = bnc_model.getSolVal(sol, x[i])
        nx.set_node_attributes(G, x_values, name='x')

        # store elementary stats needed for training
        info = {'optimal_value': bnc_model.getObjVal(),
                'is_optimal': is_optimal,
                'lp_iterations_limit': dataset_config['lp_iterations_limit'],
                'baselines': baseline_stats}

        # store extensive stats needed for evaluation
        if save_all_stats:
            info['baselines']['bnc'] = {bnc_seed: bnc_sepa.stats}

        with open(filepath, 'wb') as f:
            pickle.dump((G, info), f)
            print('saved instance to ', filepath)

        # send msg to main
        msg = pa.serialize(workerid, (dataset_name, filepath)).to_buffer()
        send_socket.send(msg)


@ray.remote
def run_worker(config, workerid, worker2main_port, main2worker_port):
    solve_graphs(config, workerid, worker2main_port, main2worker_port)


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='.',
                        help='path to generate/read data')
    parser.add_argument('--data_configfile', type=str, default='maxcut_data_config.yaml',
                        help='path to config file')
    parser.add_argument('--experiment_configfile', type=str, default='../experiments/cut_selection_dqn/configs/exp5.yaml',
                        help='path to config file')
    parser.add_argument('--workerid', type=int, default=0,
                        help='worker id')
    parser.add_argument('--nworkers', type=int, default=1,
                        help='total number of workers')
    parser.add_argument('--graph_generator_seed', type=int, default=223,
                        help='random seed for graph generators')
    parser.add_argument('--mp', type=str, default='none',
                        help='use ray [ray] or multiprocessing [mp] with nworkers')
    parser.add_argument('--quiet', action='store_true',
                        help='hide scip solving messages')
    args = parser.parse_args()

    # read configs
    with open(args.data_configfile) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.experiment_configfile) as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)

    # product the dataset configs and worker ids.
    # configs = []
    # for dataset_name, dataset_config in config['datasets'].items():
    #     for k, v in vars(args).items():
    #         dataset_config[k] = v
    # for workerid in range(args.nworkers):
    #     cfg = deepcopy(vars(args))
    #     cfg.update(experiment_config)
    #     cfg.update(data_config)
    #     cfg['workerid'] = workerid
    #     configs.append(cfg)

    config = vars(args)
    config.update(experiment_config)
    config.update(data_config)

    # generate non-isomorphic graphs
    seed = config["graph_generator_seed"]
    np.random.seed(seed)
    datasets = generate_graphs(config)

    # find unfinished baselines, starting from the hardest ones
    unfinished = [(k, fp) for k, fps in reversed(list(datasets.items())) for fp, (_, info) in fps.items() if info is None]
    if len(unfinished) > 0:
        # complete the work
        # initialize zmq sockets
        context = zmq.Context()
        main2worker_sockets = {worker_id: context.socket(zmq.PUSH) for worker_id in range(args.nworkers)}
        workers2main_socket = context.socket(zmq.PULL)
        main2worker_ports = {worker_id: skt.bind_to_random_port('tcp://127.0.0.1', min_port=10000, max_port=60000) for worker_id, skt in main2worker_sockets.items()}
        workers2main_port = workers2main_socket.bind_to_random_port('tcp://127.0.0.1', min_port=10000, max_port=60000)

        # initialize workers
        ray.init()
        worker_handles = [run_worker.remote(config, worker_id, workers2main_port, main2worker_ports[worker_id]) for worker_id in range(args.nworkers)]
        pending_tasks = set()
        while unfinished:
            # assign graphs to ready workers
            msg = workers2main_socket.recv()
            worker_id, task_id = pa.deserialize(msg)
            if task_id:
                pending_tasks.remove(task_id)
            next_task = unfinished.pop(0)
            msg = pa.serialize(next_task).to_buffer()
            main2worker_sockets[worker_id].send(msg)
            pending_tasks.add(next_task)
        while pending_tasks:
            msg = workers2main_socket.recv()
            worker_id, task_id = pa.deserialize(msg)
            if task_id:
                pending_tasks.remove(task_id)

    # post processing - collect all graphs and save to a single file
    datasets = data_config['datasets']
    data = {k: {'instances': []} for k in datasets.keys()}
    for dataset_name, dataset in datasets.items():
        dataset['datadir'] = os.path.join(
            args.datadir, data_config['problem'], dataset['dataset_name'],
            f"barabasi-albert-nmin{dataset['graph_size']['min']}-nmax{dataset['graph_size']['max']}-m{dataset['barabasi_albert_m']}-weights-{dataset['weights']}-seed{dataset['seed']}")

        # read all graphs with their baselines from disk
        for filename in tqdm(os.listdir(dataset['datadir']), desc=f'Loading {dataset_name}'):
            with open(os.path.join(dataset['datadir'], filename), 'rb') as f:
                G, info = pickle.load(f)
                if info['is_optimal'] or 'train' not in dataset_name:
                    data[dataset_name]['instances'].append((G, info))
                if not info['is_optimal']:
                    print(filename, ' is not solved to optimality')
                # assert info['is_optimal'] or 'train' in dataset_name, 'validation/test instance not solved to optimality'
        data[dataset_name]['num_instances'] = len(data[dataset_name]['instances'])

    print('--------------------------')
    print('total number of instances:')
    print('--------------------------')
    for k, v in data.items():
        print(k, ':\t', v['num_instances'])
    # save the data
    with open(os.path.join(args.datadir, data_config['problem'], 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    print('saved all data to ', os.path.join(args.datadir, data_config['problem'], 'data.pkl'))
    print('finished')

