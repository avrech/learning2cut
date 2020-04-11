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

    if config['max_per_root'] == 0:
        # run this configuration only once with all hparams get their first choice (the default)
        for k, v in sweep_config['sweep'].items():
            if k != 'graph_idx' and k != 'scip_seed' and config[k] != v['values'][0]:
                print('!!!!!!!!!!!!!!!!!!!!! SKIPPING EXPERIMENT !!!!!!!!!!!!!!!!!!!!!!1')
                return

    if config['max_per_round'] == 1 and config['criterion'] != sweep_config['sweep']['criterion']['values'][0]:
        print('!!!!!!!!!!!!!!!!!!!!! SKIPPING EXPERIMENT !!!!!!!!!!!!!!!!!!!!!!1')
        return

    # read graph
    graph_idx = config['graph_idx']
    filepath = os.path.join(config['data_abspath'], "graph_idx_{}.pkl".format(graph_idx))
    with open(filepath, 'rb') as f:
        G = pickle.load(f)

    scip_seed = config['scip_seed']
    model, x, y = maxcut_mccormic_model(G, use_cuts=False)
    sepa = MccormicCycleSeparator(G=G, x=x, y=y, name='MLCycles', hparams=config)

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
    # set log-dir for tensorboard logging of the specific trial
    log_dir = tune.track.trial_dir()

    # save stats to pkl
    experiment_results_filepath = os.path.join(log_dir, 'experiment_results.pkl')
    experiment_results = {}
    experiment_results['stats'] = stats
    experiment_results['config'] = config
    experiment_results['experiment'] = 'cut_root_fixed_maxcutsroot'
    with open(experiment_results_filepath, 'wb') as f:
        pickle.dump(experiment_results, f)
        print('Saved experiment results to: ' + experiment_results_filepath)

