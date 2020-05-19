""" Cut root
Graph type: Barabasi-Albert
MaxCut formulation: McCormic
Baseline: SCIP with defaults

Each graph is solved using different scip_seed,
and SCIP statistics are collected.

All results are written to experiment_results.pkl file
and should be post-processed using experiments/analyze_experiment.py

utils/analyze_experiment.py can generate tensorboard hparams,
and a csv file summarizing the statistics in a table (useful for latex).

In this experiment cutting planes are added only at the root node,
and the dualbound, lp_iterations and other statistics are collected.
The metric optimized is the dualbound integral w.r.t the number of lp iterations at each round.
"""
from ray import tune
from utils.scip_models import maxcut_mccormic_model, get_separator_cuts_applied
from separators.mccormick_cycle_separator import MccormickCycleSeparator
import pickle
import os
from tqdm import tqdm
from pathlib import Path

def experiment(config):
    log_dir = tune.track.trial_dir()
    # load config if experiment launched from complete_experiment.py
    if 'complete_experiment' in config.keys():
        config = config['complete_experiment']

    # set the current sweep trial parameters
    sweep_config = config['sweep_config']
    for k, v in sweep_config['constants'].items():
        config[k] = v

    # recover from checkpoints for adaptive policy experiment only.
    # for a case the long experiment terminated unexpectedly and launched again,
    # search in the iteration logdir if the current experiment has been already executed.
    if config['policy'] == 'adaptive':
        iterdir = os.path.dirname(os.path.dirname(log_dir))
        print('loading checkpoint from ', iterdir)
        with open(os.path.join(iterdir, 'checkpoint.pkl'), 'rb') as f:
            checkpoint = pickle.load(f)
        for res in checkpoint:
            cfg = res['config']
            match = True
            for k, v in config.items():
                if k != 'sweep_config' and v != cfg[k]:
                    match = False
                    # print('mismatch:', k, v, cfg[k])
                    break
            if match:
                print('experiment results found!')
                return

    if config['max_per_round'] == -1 and 'criterion' in sweep_config['sweep'].keys() and config['criterion'] != sweep_config['sweep']['criterion']['values'][0]:
        print('!!!!!!!!!!!!!!!!!!!!! SKIPPING EXPERIMENT !!!!!!!!!!!!!!!!!!!!!!')
        return

    # read graph
    graph_idx = config['graph_idx']
    filepath = os.path.join(config['data_abspath'], "graph_idx_{}.pkl".format(graph_idx))
    with open(filepath, 'rb') as f:
        G = pickle.load(f)

    scip_seed = config['scip_seed']
    model, x, y = maxcut_mccormic_model(G, use_general_cuts=False)

    sepa = MccormickCycleSeparator(G=G, x=x, y=y, name='MLCycles', hparams=config)

    model.includeSepa(sepa, 'MLCycles',
                      "Generate cycle inequalities for the MaxCut McCormic formulation",
                      priority=1000000, freq=1)
    #set scip params:
    model.setRealParam('separating/objparalfac', config['objparalfac'])
    model.setRealParam('separating/dircutoffdistfac', config['dircutoffdistfac'])
    model.setRealParam('separating/efficacyfac', config['efficacyfac'])
    model.setRealParam('separating/intsupportfac', config['intsupportfac'])
    model.setIntParam('separating/maxrounds', config['maxrounds'])
    model.setIntParam('separating/maxroundsroot', config['maxroundsroot'])
    model.setIntParam('separating/maxcuts', config['maxcuts'])
    model.setIntParam('separating/maxcutsroot', config['maxcutsroot'])

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', scip_seed)
    model.setIntParam('randomization/randomseedshift', scip_seed)

    # set time limit
    model.setRealParam('limits/time', config['time_limit_sec'])

    # set termination condition - exit after root node finishes
    model.setLongintParam('limits/nodes', 1)
    model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever.
    # run optimizer
    model.optimize()

    # collect statistics TODO collect stats every round in sepa
    sepa.finish_experiment()
    stats = sepa.stats
    if stats['total_ncuts_applied'][-1] < config['cuts_budget']:
        print('************* DID NOT EXPLOIT ALL CUTS BUDGET *************')

    # save stats to pkl
    experiment_results_filepath = os.path.join(log_dir, 'experiment_results.pkl')
    experiment_results = {}
    experiment_results['stats'] = stats
    experiment_results['config'] = config
    experiment_results['experiment'] = 'cut_root_fixed_maxcutsroot'
    with open(experiment_results_filepath, 'wb') as f:
        pickle.dump(experiment_results, f)
        print('Saved experiment results to: ' + experiment_results_filepath)


if __name__ == '__main__':
    # run the final adaptive policy found on Niagara
    import argparse
    from ray.tune import track
    import yaml
    from experiments.cut_root.data_generator import generate_data

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='results/adaptive_policy',
                        help='path to results root')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='path to generate/read data')
    parser.add_argument('--starting_policies_abspath', type=str, default='results/adaptive_policy/starting_policies.pkl',
                        help='path to load starting policies')

    args = parser.parse_args()

    track.init(experiment_dir=args.log_dir)
    log_dir = tune.track.trial_dir()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(os.path.dirname(os.path.dirname(log_dir)), 'checkpoint.pkl'), 'wb') as f:
        pickle.dump([], f)
    with open('adaptive_policy_config.yaml') as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
    config = sweep_config['constants']
    for k, v in sweep_config['sweep'].items():
        if k == 'graph_idx':
            config[k] = 0
        else:
            config[k] = v['values'][0]
    data_abspath = generate_data(sweep_config, 'data', solve_maxcut=True, time_limit=600)
    config['sweep_config'] = sweep_config
    config['data_abspath'] = data_abspath
    config['starting_policies_abspath'] = os.path.abspath(args.starting_policies_abspath)
    experiment(config)
