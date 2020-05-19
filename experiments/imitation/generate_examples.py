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


def generate_examples_from_graph(config):
    # load config if experiment launched from complete_experiment.py
    if 'complete_experiment' in config.keys():
        config = config['complete_experiment']

    # set the current sweep trial parameters
    sweep_config = config['sweep_config']
    for k, v in sweep_config['constants'].items():
        config[k] = v

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
    sampler = SepaSampler(G=G, x=x, y=y, name='g{}-samples'.format(graph_idx), hparams=config)
    # sampler = Sampler(G=G, x=x, y=y, name='g{}-samples'.format(graph_idx), hparams=config)
    model.includeSepa(sampler, 'g{}-samples1'.format(graph_idx),
                      "Store and save scip cut selection algorithm decisions",
                      priority=1, freq=1)
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
    # save the episode state-action pairs to a file
    sampler.save_data()
    print('expeiment finished')


def submit_job(jobname, taskid, time_limit_minutes):
    # CREATE SBATCH FILE
    job_file = os.path.join(args.log_dir, jobname + '.sh')
    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines('#SBATCH --time=00:{}:00\n'.format(time_limit_minutes))
        fh.writelines('#SBATCH --account=def-alodi\n')
        fh.writelines('#SBATCH --output={}/{}.out\n'.format(args.log_dir,jobname))
        fh.writelines('#SBATCH --mem=0\n')
        fh.writelines('#SBATCH --mail-user=avrech@campus.technion.ac.il\n')
        fh.writelines('#SBATCH --mail-type=END\n')
        fh.writelines('#SBATCH --mail-type=FAIL\n')
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines('#SBATCH --job-name={}\n'.format(jobname))
        fh.writelines('#SBATCH --ntasks-per-node=1\n')
        fh.writelines('#SBATCH --cpus-per-task={}\n'.format(args.cpus_per_task))
        fh.writelines('module load python\n')
        fh.writelines('source $HOME/server_bashrc\n')
        fh.writelines('source $HOME/venv/bin/activate\n')
        fh.writelines('python adaptive_policy_runner.py --experiment {} --log_dir {} --config_file {} --data_dir {} --taskid {} {} --product_keys {}\n'.format(
            args.experiment,
            args.log_dir,
            args.config_file,
            args.data_dir,
            taskid,
            '--auto' if args.auto else '',
            ' '.join(args.product_keys)
        ))

    os.system("sbatch {}".format(job_file))

if __name__ == '__main__':
    import argparse
    import yaml
    from experiments.imitation.data_generator import generate_data

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data',
                        help='path to generate/read data')
    parser.add_argument('--graphidx', type=str, default='data',
                        help='path to generate/read data')
    parser.add_argument('--configfile', type=str, default='experiment_config.yaml',
                        help='path to generate/read data')
    parser.add_argument('--ntasks-per-node', type=int, default=40,
                        help='Graham - 32, Niagara - 40')
    parser.add_argument('--graphs', nargs='+', default=[0],
                        help='list of hparam keys on which to product')

    args = parser.parse_args()

    with open('experiment_config.yaml') as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
    config = sweep_config['constants']
    for k, v in sweep_config['sweep'].items():
        if k == 'graph_idx':
            config[k] = args.graphidx
        else:
            config[k] = v['values'][0]
    data_abspath = generate_data(sweep_config, 'data', solve_maxcut=True, time_limit=600)
    config['sweep_config'] = sweep_config
    config['data_abspath'] = data_abspath
    generate_examples_from_graph(config)
