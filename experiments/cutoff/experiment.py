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
from utils.samplers import SepaSampler
import pickle
import os


def experiment(config):
    logdir = tune.track.trial_dir()
    # load config if experiment launched from complete_experiment.py
    if 'complete_experiment' in config.keys():
        config = config['complete_experiment']

    # set the current sweep trial parameters
    sweep_config = config['sweep_config']
    for k, v in sweep_config['constants'].items():
        config[k] = v

    # read graph
    graph_idx = config['graph_idx']
    datadir = config['datadir'][config['graph_size']][config['barabasi_albert_m']][config['weights']][config['dataset_generation_seed']]
    filepath = os.path.join(datadir, f"graph_idx_{graph_idx}.pkl")

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
    # model.setLongintParam('limits/nodes', 1)
    # model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever.
    # run optimizer
    model.optimize()
    with open(os.path.join(logdir, 'experiment_results.pkl'), 'wb') as f:
        pickle.dump(
            (sepa.stats, sepa.debug_cutoff_stats, sepa.debug_invalid_cut_stats), f
        )
    print('expeiment finished')

    return 0

if __name__ == '__main__':
    # run the final adaptive policy found on Niagara
    import argparse
    from ray.tune import track
    import yaml
    from utils.maxcut import generate_data

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='results/test',
                        help='path to results root')
    parser.add_argument('--datadir', type=str, default='data',
                        help='path to generate/read data')

    args = parser.parse_args()

    track.init(experiment_dir=args.logdir)
    logdir = tune.track.trial_dir()
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open('experiment_config.yaml') as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
    config = sweep_config['constants']
    for k, v in sweep_config['sweep'].items():
        if k == 'graph_idx':
            config[k] = 0
        else:
            config[k] = v['values'][0]
    datadir = generate_data(sweep_config, 'data', solve_maxcut=True, time_limit=600, mp=None)
    config['sweep_config'] = sweep_config
    config['datadir'] = datadir
    experiment(config)
