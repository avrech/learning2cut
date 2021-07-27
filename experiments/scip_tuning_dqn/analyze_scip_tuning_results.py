import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.functions import get_normalized_areas, truncate
from tqdm import tqdm
from argparse import ArgumentParser
import yaml
import wandb


parser = ArgumentParser()
parser.add_argument('--rootdir', type=str, default='results/MAXCUT/reported_runs/', help='path to all run dirs')
parser.add_argument('--mdp_run_id', type=str, default='3n3uxetn', help='run_id of MDP run')
# parser.add_argument('--mdp_test_results_file', type=str, default='results/MAXCUT/reported_runs/3n3uxetn/test_results.pkl', help='directory to save results')
parser.add_argument('--ccmab_run_id', type=str, default='130m0n5o', help='run_id of CCMAB run')
parser.add_argument('--test_args', type=str, default='', help='special test args given to the test run in the same format (exactly! will be used to find the results path)')
# parser.add_argument('--ccmab_test_results_file', type=str, default='results/MAXCUT/reported_runs/130m0n5o/test_results.pkl', help='directory to save results')
# parser.add_argument('--baseline_test_results_file', type=str, default='results/MAXCUT/reported_runs/baseline/test_results.pkl', help='directory to save results')
# parser.add_argument('--baseline_test_results_file', type=str, default='results/MAXCUT/reported_runs/baseline/test_results.pkl', help='directory to save results')
args = parser.parse_args()
ROOTDIR = args.rootdir
PROBLEM = 'MAXCUT'
SEEDS = [52, 176, 223]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

if not os.path.isdir(ROOTDIR):
    os.makedirs(ROOTDIR)


print('###################################################################')
print('####################  Analyzing RL Results  #######################')
print('###################################################################')
ccmab_test_results_file = os.path.join(args.rootdir, args.ccmab_run_id, f'test{args.test_args}', 'test_results.pkl')
mdp_test_results_file = os.path.join(args.rootdir, args.mdp_run_id, f'test{args.test_args}', 'test_results.pkl')
baseline_test_results_file = os.path.join(args.rootdir, 'baseline', f'test{args.test_args}', 'test_results.pkl')
with open(ccmab_test_results_file, 'rb') as f:
    ccmab_results = pickle.load(f)

with open(mdp_test_results_file, 'rb') as f:
    mdp_results = pickle.load(f)

with open(baseline_test_results_file, 'rb') as f:
    baseline_results = pickle.load(f)

summary_root_only = []
summary_bnc = {'solved': [], 'soltime': [], 'gap': [], 'nnodes': []}
dataset_names = [dsname for dsname in list(mdp_results.values())[0]['root_only'].keys() if 'validset' in dsname] + \
                [dsname for dsname in list(mdp_results.values())[0]['root_only'].keys() if 'testset' in dsname]
columns = ['Method', 'Validated On'] + [f"{'Val' if 'valid' in dsname else 'Test'} {'-'.join(dsname.split('_')[1:])}" for dsname in dataset_names]

# Gap vs. Time scatter plots
for method, model_statss in zip(['CCMAB', 'MDP', 'Average Tuning'], [ccmab_results, mdp_results, baseline_results]):
    for model, stats in model_statss.items():
        
        if 'default' not in model:
            line_root_only = [method, f"Val {'-'.join(model.split('_')[2:-1])}"]
            bnc_lines = {k: [method, f"Val {'-'.join(model.split('_')[2:-1])}"] for k in summary_bnc.keys()}

        else:
            line_root_only = [method, 'default']
            bnc_lines = {k: [method, 'default'] for k in summary_bnc.keys()}

        for dsname in dataset_names:
            root_only_stats = stats['root_only'][dsname]
            db_aucs = [100 * (list(res[seed].values())[0]['db_auc_improvement']-1) for res in root_only_stats.values() for seed in SEEDS]
            db_auc_std = np.mean([np.std([100*(list(res[seed].values())[0]['db_auc_improvement']-1)  for seed in SEEDS]) for res in root_only_stats.values()])
            line_root_only.append("{:.1f} {} {:.1f}%".format(np.mean(db_aucs), u"\u00B1", db_auc_std))

            bnc_stats = stats['branch_and_cut'][dsname]
            solved = []
            soltimes, soltime_stds = [], []
            gaps, gap_stds = [], []
            nnodes, nnodes_stds = [] ,[]
            for inst_stats in bnc_stats.values():
                inst_soltimes = []
                inst_gaps = []
                inst_nodes = []
                for seed_stats in inst_stats.values():
                    for run_stats in seed_stats.values():
                        solved.append(run_stats['gap'][-1] == 0)
                        soltimes.append(run_stats['solving_time'][-1])
                        inst_soltimes.append(run_stats['solving_time'][-1])
                        gaps.append(100*run_stats['gap'][-1])
                        inst_gaps.append(100*run_stats['gap'][-1])
                        nnodes.append(run_stats['processed_nodes'][-1])
                        inst_nodes.append(run_stats['processed_nodes'][-1])
                soltime_stds.append(np.std(inst_soltimes))
                gap_stds.append(np.std(inst_gaps))
                nnodes_stds.append(np.std(inst_nodes))
            # take mean over all samples
            bnc_lines['solved'].append(f"{sum(solved)}/{len(solved)}")
            bnc_lines['soltime'].append("{:.1f} {} {:.1f}".format(np.mean(soltimes), u"\u00B1", np.mean(soltime_stds)))
            bnc_lines['gap'].append("{:.1f} {} {:.1f}%".format(np.mean(gaps), u"\u00B1", np.mean(gap_stds)))
            bnc_lines['nnodes'].append("{} {} {}".format(int(np.mean(nnodes)), u"\u00B1", int(np.mean(nnodes_stds))))

            # line.append("{:.1f} {} {:.1f}%".format(np.mean(db_aucs), u"\u00B1", np.std(db_aucs)))
        summary_root_only.append(line_root_only)
        for k, line in bnc_lines.items():
            summary_bnc[k].append(line)


savedir = f'{ROOTDIR}/{PROBLEM}_test{args.test_args}'
if not os.path.exists(savedir):
    os.makedirs(savedir)

# summary[baseline].append('{:.1f}{}{:.1f}% ({:.3f})'.format(db_auc_imps.mean(), u"\u00B1", db_auc_imps.std(), db_aucs.mean()))
for smr, title in zip([summary_root_only] + list(summary_bnc.values()), ['rootonly_dbaucimp', 'bnc_nsolved', 'bnc_soltimes', 'bnc_gaps', 'bnc_nnodes']):
    df = pd.DataFrame(smr, columns=columns)
    print(df)
    csvfile = f'{savedir}/{PROBLEM}_rl_{title}_results.csv'
    df.to_csv(csvfile)
    print(f'saved {PROBLEM} RL {title} results to: {csvfile}')

# save to csv the first 20 LP rounds parameters found by MDP model. and append the value found by CCMAB
# compare to overfitted exhaustive tuning/adaptive tuning.
columns = ['objparalfac', 'dircutoffdistfac', 'efficacyfac', 'intsupportfac', 'maxcutsroot', 'minorthoroot']
parameters_found = {}
for lp_round in range(20):
    parameters_found[lp_round] = [list(mdp_results['best_validset_40_50_params']['root_only']['validset_40_50'][0][223].values())[0]['selected_separating_parameters'][lp_round][k] for k in columns]
# append parameters found for CCMAB
q_values = ccmab_results['best_validset_40_50_params']['root_only']['validset_40_50'][0][223][0]['q_values']
q_keys = ccmab_results['best_validset_40_50_params']['root_only']['validset_40_50'][0][223][0]['q_keys']
ccmab_selection = {k: idx for k, idx in zip (q_keys, np.argmax(q_values, axis=1))}
action_set = {k: [0.1, 0.5, 1] for k in ['objparalfac', 'dircutoffdistfac', 'efficacyfac', 'intsupportfac']}
action_set['maxcutsroot'] = [5, 15, 2000]
action_set['minorthoroot'] = [0.1, 0.5, 0.9]
parameters_found['CCMAB'] = [action_set[k][ccmab_selection[k]] for k in columns]
parameters_found['Average Tuning'] = [1, 0.1, 1, 0.1, 2000, 0.5]
parameters_found['Exhaustive Tuning'] = [1, 1, 0.5, 0.1, 15, 0.5]
df = pd.DataFrame.from_dict(parameters_found, orient='index', columns=columns)
print(df)
csvfile = f'{savedir}/{PROBLEM}_rl_val40-50_inst0_seed223_trajectory.csv'
df.to_csv(csvfile)
print(f'saved {PROBLEM} RL example trajectory to: {csvfile}')

print(f'analyzed {PROBLEM} test results')
