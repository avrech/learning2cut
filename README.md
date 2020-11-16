# learning2cut  
Reinforcement Learning for Cut Selection  

## Setup  
0. Clone this repo, create a `virtualenv` and install requirements:  
> git clone https://github.com/avrech/learning2cut.git  
> virtualenv --python=python3 venv  
> source venv/bin/activate
> pip install -r learning2cut/requirements.txt  

1. Append `export SCIPOPTDIR=/home/my-scip` to your `~/.bashrc`.  
2. Clone and install my `scipoptsuite-6.0.2` version (includes some extra features needed for the RL environment):  
> git clone https://github.com/avrech/scipoptsuite-6.0.2-avrech.git  
> cd scipoptsuite-6.0.2-avrech  
> cmake -Bbuild -H. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR  
> cmake --build build  
> cd build  
> make install  

3. Install my branch on PySCIPOpt  
> git clone https://github.com/ds4dm/PySCIPOpt.git  
> cd PySCIPOpt  
> git checkout ml-cutting-planes  
> pip install --debug_option='--debug' .  

## Reproducing Datasets  
### Maxcut  
> cd learning2cut/experiments/dqn  
> python generate_dataset.py --configfile configs/data_config.yaml --datadir /path/to/data/dir --mp ray --nworkers <num_cpu_cores>   

Or on `graham` (recommended):  

> . launch_graham_generate_dataset.sh  

This script does the following:  
- Randomizes `barabasi-albert` graphs according to `experiments/dqn/configs/data_config.yaml`
- Validates that there is no isomorphism between any pair of graphs
- For each graph, solves MAXCUT to optimality using `scip` and saves stats for training
