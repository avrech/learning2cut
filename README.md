# learning2cut  
Reinforcement Learning for Cut Selection  

## Installation  
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

4. Follow the instructions [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install `torch_geometric`.  

5. Install the rest of requirements  
> cd learning2cut  
> pip install -r requirements.txt  

6. Sign up to [Weights & Biases](https://www.wandb.com/), and follow the [instructions](https://docs.wandb.com/quickstart) to connect your device to your `wandb` account. 

## Reproducing Datasets  
### Maxcut  
Inside `learning2cut/experiments/dqn` run:  
> python generate_dataset.py --configfile configs/data_config.yaml --datadir /path/to/data/dir --mp ray --nworkers <num_cpu_cores>   

Or on `graham` (recommended):  

> . launch_graham_generate_dataset.sh  

`generate_dataset.py` does the following:  
- Randomizes `barabasi-albert` graphs according to `experiments/dqn/configs/data_config.yaml`
- Validates that there is no isomorphism between any pair of graphs
- For each graph, solves MAXCUT to optimality using SCIP saves stats for training

## Running Experiments
### Single run 
There are two run files, `run_single_thread_dqn.py` for single thread training, and `run_apex_dqn.py` for distributed training.
The distributed version is useful also for debugging and development, as each actor can run independently of the others. 
`run_apex_dqn.py` allows debugging and updating the code of a specific actor while the entire system keep running. 
Run
> python run_apex_dqn.py --rootdir /path/to/save/results --configfile /path/to/config/file --use-gpu  

Example config files can be found at `learning2cut/experiments/dqn/configs`. Those files conveniently pack parameters for training. 
All parameters are controlled also from command line, where the command line args override the config file setting. 
Each run is assigned a random 8-characters `run_id` which can be used for resuming and for viewing results on `wandb` dashboard. 

### Resuming
For resuming a run, add `--resume --run_id <run_id>` to the command line arguments. 

### Restarting Actors
Actors can be restarted (for updating code) without restarting the entire system. Useful cases:
* updating the tester code with additional tests/logs without shutting down the replay server.  
* fixing bugs and restarting an actor after crashing.  
Restart options are:
* Restarting the entire system: add `--restart` to the resuming command line. This will restart all crashed actors. 
* Restarting specific actors: add `--restart --restart-actors <list of actors>`. The list of actors can include any combination of `apex`, `replay_server`, `learner`, `tester` and `worker_<worker_id>` (`worker_id` running from 1 to `num_workers`).   
* Forcing restart when the target actors are still running: add `--force-restart` to the arguments above.  
Example:

### Debugging Remote Actors
In order to debug a remote actors, run:
> python run_apex_dqn.py --resume --run_id <run_id> --restart [--restart-actors <list of actors>] --debug-actor <actor_name>  

This will restart the debugged actor main loop in the debugger, so one can step into the actor code, while the rest of remote actors keep running.  


## Experiments
### Cycles Variability
Inside `learning2cut/experiments/dqn` run:  
> python cycles_variability.py --logdir results/cycles_variability [--simple_cycle_only --chordless_only --enable_chordality_check] --record_cycles  

`cycles_variability.py` will solve each graph in `validset_20_30` and `validset_50_60` 10 times with seeds ranging from 0 to 9. In each separation round it will save the cycles generated along with other related stats.  
The script will pickle a dictionary of the following structure:  
```
{dataset: [{seed: stats for seed in range(10)} for graph in dataset] for dataset in [`validset_20_30`, `validset_50_60`]}  
```  
The `recorded_cycles` are stored in `stats` alongside the `dualbound`, `lp_iterations` etc. A cycle is stored as a dictionary with items:
- `edges`: a list of the edges in cycle  
- `F`: a list of odd number of cut edges  
- `C_minus_F`: a list of the rest of the edges  
- `is_simple`: True if the cycle is simple cycle else False  
- `is_chordless`: True if the cycle has no chords else False  
- `applied`: True if the cycle was selected to the LP else False  

### Experiment 1
Inside `learning2cut/experiments/dqn` run:  
> python run_apex_dqn.py --rootdir results/exp1 --configfile configs/exp1-overfitVal25-demoLossOnly-fixedTrainingScipSeed.yaml --use-gpu  

In experiment 1 we fix a maxcut instance and SCIP random seed, and train the model to imitate SCIP, using only demonstrations. This sanity check shows that the model is capable of learning high quality sequential cut selection. 

