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
Inside `learning2cut/experiments/dqn` run:  
> python generate_dataset.py --configfile configs/data_config.yaml --datadir /path/to/data/dir --mp ray --nworkers <num_cpu_cores>   

Or on `graham` (recommended):  

> . launch_graham_generate_dataset.sh  

`generate_dataset.py` does the following:  
- Randomizes `barabasi-albert` graphs according to `experiments/dqn/configs/data_config.yaml`
- Validates that there is no isomorphism between any pair of graphs
- For each graph, solves MAXCUT to optimality using `scip` and saves stats for training

## Experiments
### Cycles Variability
Inside `learning2cut/experiments/dqn` run:  
> python cycles_variability.py --logdir results/cycles_variability [--simple_cycle_only --chordless_only --enable_chordality_check] --record_cycles  
`cycles_variability.py` will solve each graph in `validset_20_30` and `validset_50_60` 10 times with seeds ranging from 0 to 9. In each separation round it will save the cycles generated along with other related stats.  
The script will pickle a dictionary of the following structure:  
```
{dataset: [{seed: stats for seed in range(10)} for graph in dataset] for dataset in [`validset_20_30`, `validset_50_60`]}  
```  
The `recorded_cycles` are stored in `stats` alongside the `dualbound`, `lp_iterations` etc. A cycle is stored as dictionary with items:
- 'edges': a list of the edges in cycle  
- 'F': a list of odd number of cut edges  
- 'C_minus_F': a list of the rest of the edges  
- 'is_simple': True if the cycle is simple cycle else False  
- 'is_chordless': True if the cycle has no chords else False  
- 'applied': True if the cycle was selected to the LP else False  
