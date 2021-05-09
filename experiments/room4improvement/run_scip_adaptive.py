from utils.scip_models import mvc_model, CSBaselineSepa, set_aggresive_separation, CSResetSepa, maxcut_mccormic_model
from pathlib import Path
import numpy as np
import pyarrow as pa
from utils.functions import get_normalized_areas
from tqdm import tqdm
import pickle
from argparse import ArgumentParser
import ray
import zmq
from itertools import product
import os
parser = ArgumentParser()
parser.add_argument('--nnodes', type=int, default=1, help='number of machines')
parser.add_argument('--ncpus_per_node', type=int, default=6, help='ncpus available on each node')
parser.add_argument('--nodeid', type=int, default=0, help='node id for running on compute canada')
parser.add_argument('--rootdir', type=str, default='results', help='rootdir to store results')
parser.add_argument('--run_local', action='store_true', help='run on the local machine')
parser.add_argument('--run_node', action='store_true', help='run on the local machine')
args = parser.parse_args()
np.random.seed(777)
SEEDS = [46, 72, 101]
SCIP_ADAPTIVE_PARAMS_FILE = f'{args.rootdir}/scip_adaptive_params.pkl'


@ray.remote
def run_worker(data, configs, port, workerid):
    print(f'[worker {workerid}] connecting to {port}')
    context = zmq.Context()
    send_socket = context.socket(zmq.PUSH)
    send_socket.connect(f'tcp://127.0.0.1:{port}')
    baseline = 'adaptive'

    print(f'[worker {workerid}] loading adapted params from: {SCIP_ADAPTIVE_PARAMS_FILE}')
    with open(SCIP_ADAPTIVE_PARAMS_FILE, 'rb') as f:
        scip_adaptive_params = pickle.load(f)
    round_idx = len(scip_adaptive_params['mvc'][60][SEEDS[0]])
    logs = []
    best_db_aucs = {p: {gs: {seed: 0 for seed in SEEDS} for gs in gss.keys()} for p, gss in data.items()}
    best_configs = {p: {gs: {seed: None for seed in SEEDS} for gs in gss.keys()} for p, gss in data.items()}

    for config in tqdm(configs, desc=f'worker {workerid}'):
        cfg = {k: v for (k, v) in config}
        problem = cfg['problem']
        instances = data[problem]
        cfg_db_aucs = {problem: {gs: {} for gs in instances.keys()}}
        for graph_size, (g, info) in instances.items():
            for seed in SEEDS:
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
                               'policy': 'adaptive',
                               'reset_maxcuts': 100,
                               'reset_maxcutsroot': 100,
                               }
                # set the adapted params for the previous rounds and the current cfg for the next round.
                adapted_params = scip_adaptive_params[problem][graph_size][seed]
                adaptive_cfg = {k: {} for k in ['objparalfac', 'dircutoffdistfac', 'efficacyfac', 'intsupportfac', 'maxcutsroot', 'minorthoroot']}
                # for k in ['objparalfac', 'dircutoffdistfac', 'efficacyfac', 'intsupportfac', 'maxcutsroot', 'minorthoroot']:
                #     cfg[k] = {}  #{idx: v for idx, v in enumerate(vals + [cfg[k]])}
                for round, adapted_cfg in enumerate(adapted_params):
                    adapted_cfg = {prm: val for prm, val in adapted_cfg}
                    for k in adaptive_cfg.keys():
                        adaptive_cfg[k][round] = adapted_cfg[k]
                # set the current round params:
                for k in adaptive_cfg.keys():
                    adaptive_cfg[k][round_idx] = cfg[k]

                sepa_params.update(adaptive_cfg)
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
                db_auc = sum(get_normalized_areas(t=stats['lp_iterations'], ft=stats['dualbound'],
                             t_support=lp_iterations_limit, reference=info['optval']))

                if db_auc > best_db_aucs[problem][graph_size][seed]:
                    best_configs[problem][graph_size][seed] = config
                    best_db_aucs[problem][graph_size][seed] = db_auc
                cfg_db_aucs[problem][graph_size][seed] = db_auc

        logs.append((config, cfg_db_aucs))
        if len(logs) >= 5:
            # send logs to main process for checkpointing
            msg = (workerid, logs, best_configs, best_db_aucs)
            packet = pa.serialize(msg).to_buffer()
            send_socket.send(packet)
            logs = []

    if len(logs) > 0:
        # send remaining logs to main process for checkpointing
        msg = (workerid, logs, best_configs, best_db_aucs)
        packet = pa.serialize(msg).to_buffer()
        send_socket.send(packet)

    print(f'[worker {workerid}] finished')


def get_data_and_configs():
    print(f'loading data from: {args.rootdir}/data.pkl')
    with open(f'{args.rootdir}/data.pkl', 'rb') as f:
        data = pickle.load(f)

    search_space = {
        'objparalfac': [0.1, 0.5, 1],
        'dircutoffdistfac': [0.1, 0.5, 1],
        'efficacyfac': [0.1, 0.5, 1],
        'intsupportfac': [0.1, 0.5, 1],
        'maxcutsroot': [5, 15, 2000],
        'minorthoroot': [0.5, 0.9, 1],
        'problem': ['mvc', 'maxcut']
    }
    # seeds = [46, 72, 101]
    kv_list = []
    for k, vals in search_space.items():
        kv_list.append([(k, v) for v in vals])

    configs = list(product(*kv_list))
    return data, configs


def run_node(args):
    # socket for receiving results from workers
    context = zmq.Context()
    recv_socket = context.socket(zmq.PULL)
    port = recv_socket.bind_to_random_port('tcp://127.0.0.1', min_port=10000, max_port=60000)
    print(f'[node {args.nodeid} connected to port {port}')
    # load adapted params:
    with open(SCIP_ADAPTIVE_PARAMS_FILE, 'rb') as f:
        scip_adaptive_params = pickle.load(f)
    round_idx = len(scip_adaptive_params['mvc'][60][SEEDS[0]])

    # get missing configs:
    data, all_configs = get_data_and_configs()
    main_results_file = os.path.join(args.rootdir, f'scip_adaptive_round{round_idx}_results.pkl')
    with open(main_results_file, 'rb') as f:
        main_results = pickle.load(f)
    missing_configs = list(set(all_configs) - set(main_results['configs'].keys()))
    # assign configs to current machine
    node_configs = []
    for idx in range(args.nodeid, len(missing_configs), args.nnodes):
        node_configs.append(missing_configs[idx])
    # assign configs to workers
    nworkers = args.ncpus_per_node-1
    ray.init()
    # ray.get([run_worker.remote(cfg) for cfg in configs])
    worker_handles = []
    for workerid in range(nworkers):
        worker_configs = [node_configs[idx] for idx in range(workerid, len(node_configs), nworkers)]
        # if args.run_local:
        #     run_worker(data, worker_configs, port, workerid)
        # else:
        worker_handles.append(run_worker.remote(data, worker_configs, port, workerid))

    node_results_dir = os.path.join(args.rootdir, f'node{args.nodeid}_results')
    if not os.path.exists(node_results_dir):
        os.makedirs(node_results_dir)
    node_results_file = os.path.join(node_results_dir, f'scip_adaptive_node_results_round{round_idx}.pkl')
    node_results = {'best_db_aucs': {p: {gs: {seed: 0 for seed in SEEDS} for gs in gss.keys()} for p, gss in data.items()},
                    'best_configs': {p: {gs: {seed: None for seed in SEEDS} for gs in gss.keys()} for p, gss in data.items()},
                    'configs': {}}
    # wait for logs
    last_save = 0
    pbar = tqdm(total=len(node_configs), desc='receiving logs')
    while len(node_results['configs']) < len(node_configs):
        msg = recv_socket.recv()
        workerid, logs, worker_best_cfgs, worker_best_db_aucs = pa.deserialize(msg)
        for cfg, cfg_db_aucs in logs:
            node_results['configs'][cfg] = cfg_db_aucs
            for problem, graph_sizes in worker_best_db_aucs.items():
                for graph_size, seeds in graph_sizes.items():
                    for seed, db_auc in seeds.items():
                        if db_auc > node_results['best_db_aucs'][problem][graph_size][seed]:
                            node_results['best_db_aucs'][problem][graph_size][seed] = worker_best_db_aucs[problem][graph_size][seed]
                            node_results['best_configs'][problem][graph_size][seed] = worker_best_cfgs[problem][graph_size][seed]
        pbar.update(len(logs))
        # save to node results file every 5 configs
        if len(node_results['configs']) - last_save > 5:
            last_save = len(node_results['configs'])
            with open(node_results_file, 'wb') as f:
                pickle.dump(node_results, f)
    # save results and exit
    with open(node_results_file, 'wb') as f:
        pickle.dump(node_results, f)
    print(f'finished {len(node_results["configs"])}/{len(node_configs)} configs')
    print(f'saved node results to {node_results_file}')


def submit_job(jobname, nnodes, nodeid, time_limit_hours, time_limit_minutes):
    # CREATE SBATCH FILE
    job_file = os.path.join(args.rootdir, jobname + '.sh')
    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f'#SBATCH --time={time_limit_hours}:{time_limit_minutes}:00\n')
        fh.writelines('#SBATCH --account=def-alodi\n')
        fh.writelines(f'#SBATCH --output={args.rootdir}/{jobname}.out\n')
        fh.writelines('#SBATCH --mem=0\n')
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines(f'#SBATCH --job-name={jobname}\n')
        fh.writelines('#SBATCH --ntasks-per-node=1\n')
        fh.writelines(f'#SBATCH --cpus-per-task={args.ncpus_per_node}\n')
        fh.writelines('module load NiaEnv/2018a\n')
        fh.writelines('module load python\n')
        fh.writelines('source $HOME/server_bashrc\n')
        fh.writelines('source $HOME/venv/bin/activate\n')
        fh.writelines(f'python run_scip_adaptive.py --rootdir {args.rootdir} --nnodes {nnodes} --ncpus_per_node {args.ncpus_per_node} --nodeid {nodeid} --run_node\n')

    os.system("sbatch {}".format(job_file))


def main(args):
    data, all_configs = get_data_and_configs()
    # load/init prev rounds params
    if not os.path.exists(SCIP_ADAPTIVE_PARAMS_FILE):
        scip_adaptive_params = {p: {gs: {seed: [] for seed in SEEDS} for gs in insts.keys()} for p, insts in data.items()}
        with open(SCIP_ADAPTIVE_PARAMS_FILE, 'wb') as f:
            pickle.dump(scip_adaptive_params, f)
    else:
        with open(SCIP_ADAPTIVE_PARAMS_FILE, 'rb') as f:
            scip_adaptive_params = pickle.load(f)
    round_idx = len(scip_adaptive_params['mvc'][60][SEEDS[0]])

    # update main_results
    main_results_file = os.path.join(args.rootdir, f'scip_adaptive_round{round_idx}_results.pkl')
    if os.path.exists(main_results_file):
        with open(main_results_file, 'rb') as f:
            main_results = pickle.load(f)
    else:
        main_results = {'best_db_aucs': {p: {gs: {seed: 0 for seed in SEEDS} for gs in insts.keys()} for p, insts in data.items()},
                        'best_configs': {p: {gs: {seed: None for seed in SEEDS} for gs in insts.keys()} for p, insts in data.items()},
                        'configs': {}}
    for path in tqdm(Path(args.rootdir).rglob(f'scip_adaptive_node_results_round{round_idx}.pkl'), desc='Loading node files'):
        with open(path, 'rb') as f:
            node_results = pickle.load(f)
            main_results['configs'].update(node_results['configs'])
            for problem, instances in node_results['best_db_aucs'].items():
                for graph_size, db_aucs in instances.items():
                    for seed, db_auc in db_aucs.items():
                        if db_auc > main_results['best_db_aucs'][problem][graph_size][seed]:
                            main_results['best_db_aucs'][problem][graph_size][seed] = node_results['best_db_aucs'][problem][graph_size][seed]
                            main_results['best_configs'][problem][graph_size][seed] = node_results['best_configs'][problem][graph_size][seed]

    # save updated results to main results file
    with open(main_results_file, 'wb') as f:
        pickle.dump(main_results, f)
        print(f'saved main results to {main_results_file}')

    # check for missing results:
    missing_configs = set(all_configs) - set(main_results['configs'].keys())

    print(f'{len(missing_configs)} configs left to execute')
    if len(missing_configs) > 0:
        # submit jobs or run local
        if args.run_local:
            run_node(args)
        else:
            # submit up to nnodes jobs
            nnodes = int(min(args.nnodes, np.ceil(len(missing_configs) / (args.ncpus_per_node-1))))
            time_limit_minutes = max(int(np.ceil(len(missing_configs) * 40 / nnodes / (args.ncpus_per_node - 1))), 16)
            time_limit_hours = int(np.floor(time_limit_minutes / 60))
            time_limit_minutes = time_limit_minutes % 60
            assert 24 > time_limit_hours >= 0
            assert 60 > time_limit_minutes > 0

            for nodeid in range(nnodes):
                submit_job(f'scip_adapt{nodeid}', nnodes, nodeid, time_limit_hours, time_limit_minutes)
    else:
        # append best params for round_idx to scip_adaptive_params
        for problem, graph_sizes in main_results['best_configs'].items():
            for graph_size, seeds in graph_sizes.items():
                for seed, cfg in seeds.items():
                    scip_adaptive_params[problem][graph_size][seed].append(cfg)

        # save the new scip adaptive params
        with open(SCIP_ADAPTIVE_PARAMS_FILE, 'wb') as f:
            pickle.dump(scip_adaptive_params, f)
        print(f'saved scip_adaptive_params for rounds 0-{round_idx} to {SCIP_ADAPTIVE_PARAMS_FILE}')
        print(f'for adapting scip to lp round {round_idx+1} run the script again')


if __name__ == '__main__':
    if args.run_node:
        run_node(args)
    else:
        main(args)
    print('finished')
