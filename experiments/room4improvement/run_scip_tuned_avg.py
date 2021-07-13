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
import yaml
parser = ArgumentParser()
parser.add_argument('--nnodes', type=int, default=1, help='number of machines')
parser.add_argument('--ncpus_per_node', type=int, default=6, help='ncpus available on each node')
parser.add_argument('--nodeid', type=int, default=0, help='node id for running on compute canada')
parser.add_argument('--rootdir', type=str, default='results/large_action_space', help='rootdir to store results')
parser.add_argument('--configfile', type=str, default='configs/large_action_space.yaml', help='path to config yaml file')
parser.add_argument('--cluster', type=str, default='niagara', help='niagara or anything else')
parser.add_argument('--datadir', type=str, default='../../data', help='path to all data')
parser.add_argument('--run_local', action='store_true', help='run on the local machine')
parser.add_argument('--run_node', action='store_true', help='run on the local machine')
args = parser.parse_args()
np.random.seed(777)
SEEDS = [52, 176, 223]  # [46, 72, 101]
ROOTDIR = args.rootdir
with open(args.configfile) as f:
    action_space = yaml.load(f, Loader=yaml.FullLoader)


@ray.remote
def run_worker(data, training_data, configs, port, workerid):
    print(f'[worker {workerid}] connecting to {port}')
    context = zmq.Context()
    send_socket = context.socket(zmq.PUSH)
    send_socket.connect(f'tcp://127.0.0.1:{port}')
    baseline = 'scip_tuned'
    logs = []
    best_configs = {p: {gs: None for gs in gss.keys()} for p, gss in data.items()}
    best_db_aucs = {p: {gs: 0 for gs in gss.keys()} for p, gss in data.items()}
    for config in tqdm(configs, desc=f'worker {workerid}'):
        cfg = {k: v for (k, v) in config}
        problem = cfg['problem']
        cfg_db_aucs = {problem: {gs: [] for gs in data[problem].keys()}}
        for graph_size, maxcut_lp_iter_limit in zip(data[problem].keys(), [5000, 7000, 10000]):
            for k, d in training_data[problem].items():
                if 'valid' in k and int(k.split('_')[-2]) <= graph_size <= int(k.split('_')[-1]):
                    instances = d['instances']  # tune on all validation instances
                    break
            for g, info in instances:
                for seed in SEEDS:
                    if problem == 'mvc':
                        model, _ = mvc_model(g)
                        lp_iterations_limit = 1500
                    elif problem == 'maxcut':
                        model, _, _ = maxcut_mccormic_model(g)
                        lp_iterations_limit = maxcut_lp_iter_limit
                    else:
                        raise ValueError
                    set_aggresive_separation(model)
                    sepa_params = {'lp_iterations_limit': lp_iterations_limit,
                                   'policy': 'tuned',
                                   'reset_maxcuts': 100,
                                   'reset_maxcutsroot': 100,
                                   }
                    sepa_params.update(cfg)

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
                                                      t_support=lp_iterations_limit, reference=info['optimal_value']))
                    cfg_db_aucs[problem][graph_size].append(db_auc)
            cfg_db_aucs[problem][graph_size] = db_auc_avg = np.mean(cfg_db_aucs[problem][graph_size])
            if db_auc > best_db_aucs[problem][graph_size]:
                best_configs[problem][graph_size] = config
                best_db_aucs[problem][graph_size] = db_auc_avg

        logs.append((config, cfg_db_aucs))
        if len(logs) >= 1:
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


def get_data_and_configs(training=True):
    print(f'loading data from: {args.rootdir}/data.pkl')
    with open(f'{args.rootdir}/data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f'loading training data from: {args.datadir}')
    if training:
        training_data = {}
        with open(f'{args.datadir}/MVC/data.pkl', 'rb') as f:
            training_data['mvc'] = pickle.load(f)
        with open(f'{args.datadir}/MAXCUT/data.pkl', 'rb') as f:
            training_data['maxcut'] = pickle.load(f)
        for d in training_data.values():
            for k in list(d.keys()):
                if 'valid' not in k:
                    d.pop(k)

    else:
        training_data = None
    search_space = {**action_space, 'problem': ['mvc', 'maxcut']}

    kv_list = []
    for k, vals in search_space.items():
        kv_list.append([(k, v) for v in vals])
    configs = list(product(*kv_list))
    return data, training_data, configs


def run_node(args):
    # socket for receiving results from workers
    context = zmq.Context()
    recv_socket = context.socket(zmq.PULL)
    port = recv_socket.bind_to_random_port('tcp://127.0.0.1', min_port=10000, max_port=60000)
    print(f'[node {args.nodeid} connected to port {port}')
    # get missing configs:
    data, training_data, all_configs = get_data_and_configs()

    # # assign configs to current machine
    # node_configs = []
    # for idx in range(args.nodeid, len(missing_configs), args.nnodes):
    #     node_configs.append(missing_configs[idx])
    with open(f'{ROOTDIR}/scip_tuned_avg_node{args.nodeid}_configs.pkl', 'rb') as f:
        node_configs = pickle.load(f)
        print(f'[node {args.nodeid}] loaded configs from: {ROOTDIR}/scip_tuned_avg_node{args.nodeid}_configs.pkl')

    # assign configs to workers
    nworkers = args.ncpus_per_node-1
    ray.init()
    # ray.get([run_worker.remote(cfg) for cfg in configs])
    worker_handles = []
    for workerid in range(nworkers):
        worker_configs = [node_configs[idx] for idx in range(workerid, len(node_configs), nworkers)]
        worker_handles.append(run_worker.remote(data, training_data, worker_configs, port, workerid))

    node_results_dir = os.path.join(args.rootdir, f'scip_tuned_avg_node{args.nodeid}_results')
    if not os.path.exists(node_results_dir):
        os.makedirs(node_results_dir)
    node_results_file = os.path.join(node_results_dir, 'scip_tuned_avg_node_results.pkl')
    node_results = {'best_db_aucs': {p: {gs: 0 for gs in gss.keys()} for p, gss in data.items()},
                    'best_configs': {p: {gs: None for gs in gss.keys()} for p, gss in data.items()},
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
                for graph_size, db_auc in graph_sizes.items():
                    if db_auc > node_results['best_db_aucs'][problem][graph_size]:
                        node_results['best_db_aucs'][problem][graph_size] = worker_best_db_aucs[problem][graph_size]
                        node_results['best_configs'][problem][graph_size] = worker_best_cfgs[problem][graph_size]
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
        fh.writelines(f'#SBATCH --job-name={jobname}\n')
        fh.writelines(f'#SBATCH --cpus-per-task={args.ncpus_per_node}\n')
        if args.cluster == 'niagara':
            fh.writelines('#SBATCH --mem=0\n')
            fh.writelines('#SBATCH --nodes=1\n')
            fh.writelines('#SBATCH --ntasks-per-node=1\n')
            fh.writelines('module load NiaEnv/2018a\n')
            fh.writelines('module load python\n')
            fh.writelines('source $HOME/server_bashrc\n')
            fh.writelines('source $HOME/venv/bin/activate\n')

        else:
            fh.writelines(f'#SBATCH --mem={int(4 * args.ncpus_per_node)}G\n')
        fh.writelines(f'srun python run_scip_tuned_avg.py --configfile {args.configfile} --rootdir {args.rootdir} --datadir {args.datadir} --nnodes {nnodes} --ncpus_per_node {args.ncpus_per_node} --nodeid {nodeid} --run_node\n')

    os.system("sbatch {}".format(job_file))


def main(args):
    data, _, all_configs = get_data_and_configs(training=False)
    # update main_results
    main_results_file = os.path.join(args.rootdir, 'scip_tuned_avg_main_results.pkl')
    if os.path.exists(main_results_file):
        with open(main_results_file, 'rb') as f:
            main_results = pickle.load(f)
    else:
        main_results = {'best_db_aucs': {p: {gs: 0 for gs in gss.keys()} for p, gss in data.items()},
                        'best_configs': {p: {gs: None for gs in gss.keys()} for p, gss in data.items()},
                        'configs': {}}
    for path in tqdm(Path(args.rootdir).rglob('scip_tuned_avg_node_results.pkl'), desc='Loading node files'):
        with open(path, 'rb') as f:
            node_results = pickle.load(f)
            main_results['configs'].update(node_results['configs'])
            for problem, graph_sizes in node_results['best_db_aucs'].items():
                for graph_size, db_auc in graph_sizes.items():
                    if db_auc > main_results['best_db_aucs'][problem][graph_size]:
                        main_results['best_db_aucs'][problem][graph_size] = node_results['best_db_aucs'][problem][graph_size]
                        main_results['best_configs'][problem][graph_size] = node_results['best_configs'][problem][graph_size]

    # save updated results to main results file
    with open(main_results_file, 'wb') as f:
        pickle.dump(main_results, f)
        print(f'saved main results to {main_results_file}')

    # check for missing results:
    missing_configs = list(set(all_configs) - set(main_results['configs'].keys()))

    # filter configs which were already executed
    # resultsfile = f'{args.rootdir}/scip_tuned_results.pkl'
    # if os.path.exists(resultsfile):
    #     with open(resultsfile, 'rb') as f:
    #         results = pickle.load(f)
    #     already_executed = set(results['configs'].keys())
    #     print(f'loaded results for {len(already_executed)} configs')
    #     configs = list(set(configs) - already_executed)
    # else:
    #     results = {'best_db_auc': 0,
    #                'best_config': None,
    #                'configs': {}}
    print(f'{len(missing_configs)} configs left to execute')
    if len(missing_configs) > 0:
        # submit jobs or run local
        if args.run_local:
            run_node(args)
        else:
            # submit up to nnodes jobs
            nnodes = int(min(args.nnodes, np.ceil(len(missing_configs) / (args.ncpus_per_node-1))))
            time_limit_minutes = max(int(np.ceil(len(missing_configs) * 120 / nnodes / (args.ncpus_per_node - 1))), 40)
            time_limit_hours = int(np.floor(time_limit_minutes / 60))
            time_limit_minutes = time_limit_minutes % 60
            assert 24 > time_limit_hours >= 0
            assert 60 > time_limit_minutes >= 0

            # save node configs to pkls
            all_node_configs = []
            for nodeid in range(nnodes):
                node_configs = []
                for idx in range(nodeid, len(missing_configs), nnodes):
                    node_configs.append(missing_configs[idx])
                with open(f'{ROOTDIR}/scip_tuned_avg_node{nodeid}_configs.pkl', 'wb') as f:
                    pickle.dump(node_configs, f)
                all_node_configs += node_configs
            assert set(all_node_configs) == set(missing_configs)

            for nodeid in range(nnodes):
                submit_job(f'scip_tuned_avg{nodeid}', nnodes, nodeid, time_limit_hours, time_limit_minutes)
    else:
        # save scip tuned best config to
        scip_tuned_avg_best_configs_file = os.path.join(args.rootdir, 'scip_tuned_avg_best_config.pkl')
        scip_tuned_avg_best_configs = {p: {gs: cfg for gs, cfg in gss.items()} for p, gss in main_results['best_configs'].items()}
        with open(scip_tuned_avg_best_configs_file, 'wb') as f:
            pickle.dump(scip_tuned_avg_best_configs, f)
        print(f'saved scip_tuned_avg_best_configs to {scip_tuned_avg_best_configs_file}')


if __name__ == '__main__':
    if args.run_node:
        run_node(args)
    else:
        main(args)
    print('finished')
