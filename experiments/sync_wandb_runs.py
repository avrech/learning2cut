import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--job_ids', type=str, nargs='+', default=[], help='list of job ids to search in their "*<job_id>.out file')
args = parser.parse_args()

out_files = [f for f in os.listdir() if f.endswith('.out')]
if args.job_ids:
    out_files = [f for f in out_files if any([j in f for j in args.job_ids])]

for out_file in out_files:
    with open(out_file, 'r') as f:
        for line in f.readlines():
            if line.startswith('wandb: Run data is saved locally in'):
                run_dir = line.split(' ')[-1]
                print('syncing ', out_file)
                os.system(f'wandb sync {run_dir}')
                break

print('finished')
