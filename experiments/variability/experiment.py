""" Variability
Graph type: Barabasi-Albert
MaxCut formulation: McCormic
Baseline: SCIP defaults

Each graph is solved using different scip_seed,
and SCIP statistics are collected.

All results are written to experiment_results.pkl file
and should be post-processed using experiments/analyze_experiment.py

utils/analyze_experiment.py can generate tensorboard hparams,
and a csv file summarizing the statistics in a table (useful for latex).

"""
from ray import tune
from utils.scip_models import maxcut_mccormic_model, get_separator_cuts_applied
from separators.mccormic_cycle_separator import MccormicCycleSeparator
import pickle
import os


def experiment(config):
    # load config if experiment launched from complete_experiment.py
    if 'complete_experiment' in config.keys():
        config = config['complete_experiment']

    # set the current sweep trial parameters
    sweep_config = config['sweep_config']
    for k, v in sweep_config['constants'].items():
        config[k] = v

    if not config['use_cycle_cuts']:
        # run this configuration only once with all hparams get their first choice
        for k, v in sweep_config['sweep'].items():
            if k != 'use_cycle_cuts' and k != 'graph_idx' and k != 'scip_seed' and config[k] != v['values'][0]:
                print('!!!!!!!!!!!!!!!!!!!!! SKIPPING DUPLICATED not use_cycle_cuts !!!!!!!!!!!!!!!!!!!!!!1')
                return

    # generate graph
    graph_idx = config['graph_idx']
    filepath = os.path.join(config['data_abspath'], "graph_idx_{}.pkl".format(graph_idx))
    with open(filepath, 'rb') as f:
        G = pickle.load(f)

    scip_seed = config['scip_seed']
    model, x, y = maxcut_mccormic_model(G)
    sepa = MccormicCycleSeparator(G=G, x=x, y=y, hparams=config)

    if config['use_cycle_cuts']:
        model.includeSepa(sepa, 'McCycles',
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
    model.optimize()

    cycles_sepa_time = sepa.time_spent / model.getSolvingTime() if config['use_cycle_cuts'] else 0
    if config['use_cycle_cuts']:
        cycle_cuts, cycle_cuts_applied = get_separator_cuts_applied(model, 'McCycles')
    else:
        cycle_cuts, cycle_cuts_applied = 0, 0

    # Statistics
    stats = {}
    stats['cycle_cuts'] = cycle_cuts
    stats['cycle_cuts_applied'] = cycle_cuts_applied
    stats['total_cuts_applied'] = model.getNCutsApplied()
    stats['cycles_sepa_time'] = cycles_sepa_time
    stats['solving_time'] = model.getSolvingTime()
    stats['processed_nodes'] = model.getNNodes()
    stats['gap'] = model.getGap()
    stats['LP_rounds'] = model.getNLPs()

    # set log-dir for tensorboard logging of the specific trial
    log_dir = tune.track.trial_dir()

    # save stats to pkl
    experiment_results_filepath = os.path.join(log_dir, 'experiment_results.pkl')
    experiment_results = {}
    experiment_results['stats'] = stats
    experiment_results['config'] = config
    experiment_results['experiment'] = 'variability'
    with open(experiment_results_filepath, 'wb') as f:
        pickle.dump(experiment_results, f)
        print('Saved experiment results to: ' + experiment_results_filepath)

