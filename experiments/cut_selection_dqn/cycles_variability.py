"""
File: cycles_variability
Description:
* Solves MAXCUT across multiple graphs and seeds using SCIP cut selection
* Records cycles added at each separation round
* Analyzes variability, i.e
    - percentage of cycles that are common to all seeds
    - percentage of cycles that were commonly applied across all seeds
    etc.
    TODO: Aleks, complete analysis description
"""
import os
import pickle
import numpy as np
from tqdm import tqdm
from utils.maxcut import maxcut_mccormic_model
from utils.scip_models import MccormickCycleSeparator


def collect_data(hparams):
    """
    Solves each of the validation set graphs using 10 different seeds.
    Uses SCIP default cut selection.
    Record at each separation round:
        - cycles added
        - cycles applied
        - dualbound, lp iterations etc.
    Stores all data in the following structure:
    {dataset_name:
        [
            {seed:
                {cycles: list of <n_rounds> sets, each set contains <n_cycles> cycle_dictionaries,
                         containing: {F: a list of odd number of cut-edges,
                                      C_minus_F: a list of the rest of the non-cut edges in the cycle,
                                      applied: bool - whether this cut was applied or not},
                 dualbound: nd array [n_rounds + 1],
                 lp_iterations: nd array [n_rounds + 1],
                 etc.
                }
            for seed in seeds}
        for graph in datasets[dataset_name]]
    for dataset_name in datasets}

    where the datasets are validset_20_30 and validset_50_60
    """
    # load datasets
    datasets = hparams['datasets']
    for key in list(datasets.keys()):
        if key not in hparams['focus_on_datasets']:
            datasets.pop(key)

    for dataset_name, dataset in datasets.items():
        datasets[dataset_name]['datadir'] = os.path.join(
            hparams['datadir'], dataset['dataset_name'],
            f"barabasi-albert-nmin{dataset['graph_size']['min']}-nmax{dataset['graph_size']['max']}-m{dataset['barabasi_albert_m']}-weights-{dataset['weights']}-seed{dataset['seed']}")

        # read all graphs with their baselines from disk
        dataset['instances'] = []
        for filename in tqdm(os.listdir(datasets[dataset_name]['datadir']), desc=f'Loading {dataset_name}'):
            with open(os.path.join(datasets[dataset_name]['datadir'], filename), 'rb') as f:
                G, baseline = pickle.load(f)
                dataset['instances'].append((G, baseline))
        dataset['num_instances'] = len(dataset['instances'])

    # solve MAXCUT
    seeds = np.arange(10)
    all_stats = {dataset_name: [] for dataset_name in datasets.keys()}
    for dataset_name, dataset in datasets.items():
        print('#########################')
        print('solving ', dataset_name)
        print('#########################')
        for idx, (G, baseline) in enumerate(dataset['instances']):
            stats = {}
            for seed in tqdm(seeds, desc=f'graph {idx}'):
                model, x, y = maxcut_mccormic_model(G, use_general_cuts=False)
                cycle_sepa = MccormickCycleSeparator(G=G, x=x, y=y, name='MLCycles', hparams=hparams)
                model.includeSepa(cycle_sepa, 'MLCycles',
                                  "Generate cycle inequalities for the MaxCut McCormick formulation",
                                  priority=1000000, freq=1)
                model.setLongintParam('limits/nodes', 1)  # solve only at the root node
                model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever
                # fix randomization seed
                model.setBoolParam('randomization/permutevars', True)
                model.setIntParam('randomization/permutationseed', seed)
                model.setIntParam('randomization/randomseedshift', seed)
                if hparams.get('hide_scip_output', False):
                    model.hideOutput()
                model.optimize()
                cycle_sepa.finish_experiment()
                stats[seed] = {**{'recorded_cycles': cycle_sepa.recorded_cycles}, **cycle_sepa.stats}
            all_stats[dataset_name].append(stats)

    # save all stats
    filename = os.path.join(hparams['logdir'], f'cycle_stats{"_chordless_only" if hparams["chordless_only"] else ""}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(all_stats, f)
    print('saved all stats to ', filename)


if __name__ == '__main__':
    from experiments.cut_selection_dqn.default_parser import parser, get_hparams

    # parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='results',
                        help='path to save results')
    parser.add_argument('--datadir', type=str, default='data/maxcut',
                        help='path to generate/read data')
    parser.add_argument('--data_config', type=str, default='configs/maxcut_data_config.yaml',
                        help='general experiment settings')
    parser.add_argument('--configfile', type=str, default='configs/experiment_config.yaml',
                        help='general experiment settings')
    parser.add_argument('--focus_on_datasets', type=str, nargs='+', default=['validset_20_30', 'validset_50_60'],
                        help='which datasets to solve')

    args = parser.parse_args()
    hparams = get_hparams(args)
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    collect_data(hparams)
    print('finished cycles_variability')
