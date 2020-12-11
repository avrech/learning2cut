import os
import pickle
import signal
from experiments.dqn.default_parser import parser

args = parser.parse_args()
run_dir = os.path.join(args.rootdir, args.run_id)
for file in os.listdir(run_dir):
    if file.endswith('pid.txt'):
        with open(f'{run_dir}/{file}', 'r') as f:
            pid = int(f.readline().strip())
        try:
            os.kill(pid, 0)
        except OSError:
            print(f'pid {pid} does not exist')
        else:
            print(f'killing pid {pid}')
            os.kill(pid, signal.SIGKILL)


