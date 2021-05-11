# learning2cut  
Reinforcement Learning for Cut Selection  

## TODO
- [ ] Deploy on Graham

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

## Running on Compute Canada
### Graham
* `pyarrow` cannot be installed directly, and must be loaded using `module load arrow`.  
* `torch_geometric` is compiled for specific `torch` and `cuda` versions. For available `torch` versions contact CC support. 
* The following setup was tested successfully on Graham:
> $ module load StdEnv/2018.3 gcc/7.3.0 python/3.7.4 arrow  
$ virtualenv env && source env/bin/activate  
(env) ~ $ pip install torch==1.4.0 torch_geometric torchvision torch-scatter torch-sparse torch-cluster torch-spline-conv -U --force  
(env) ~ $ python -c "import pyarrow; torch_geometric; import torch_cluster; import torch_cluster.graclus_cpu"  
(env) ~ $  
### Niagara
* This cluster was used only for experiment "room for improvement". All nodes support cpus only. No need to install cuda related packages.

## Experiment 1 - Room for Improvement
This experiment requires massive computation power. Performed on 20 nodes of 80 cpus each. 
### Generate Data
Inside `learning2cut/experiments/room4improvement` run:  
> python run_experiment.py  

The script will generate `data.pkl` file for the whole experiment. This can take a long time since we solve hard maxcut instances to optimality.  
After finishing, log in to Niagara and copy `data.pkl` to `$SCRATCH/room4improvement`.   

### Find `scip_tuned` baseline
On Niagara, inside `learning2cut/experiments/room4improvement` run:   
> python run_scip_tuned.py --rootdir $SCRATCH/room4improvement --nnodes 20 --ncpus_per_node 80  

Jobs for finding `scip_tuned` policy will be submitted. After all jobs have finished, run the same command line again to finalize stuff. 
In a case something went wrong in the first run, the script should be invoked again until it finishes the work.      

### Find `scip_adaptive` baseline
On Niagara, inside `learning2cut/experiments/room4improvement` run:  
> python run_scip_adaptive.py --rootdir $SCRATCH/room4improvement --nnodes 20 --ncpus_per_node 80  

Running this command line `K` times will generate adaptive policy for `K` lp rounds. 
The adaptive policies per problem, graph size and seed are stored as a list of key-vals. 

### Run all baselines ###
To compare all baselines in terms of solving time, 
run again `run_experiment` pointing to the rootdir where
scip tuned and adaptive results are stored. 
The script will test all baselines on the local machine one by one
without multiprocessing. 
Results will be saved to a csv and png files. 




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

After `generate_dataset.py` finished successfully, generate additional baselines:
> python generate_simple_baselines.py --nworkers 10 --mp ray  

This script will solve the validation and test instances using two simple cut selection policies:
- `10_random` - select 10 random cuts every separation round
- `10_most_violated` - select the 10 most violated cuts every round

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

### Experiment 2
Inside `learning2cut/experiments/dqn` run:  
> python run_apex_dqn.py --use-gpu --rootdir results/exp2 --configfile configs/exp2-overfitVal25-demoLossOnly.yaml




|Done |Exp | Train Set | Behaviour | Loss | SCIP Seed  | Goal | Results |
|---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| &#9745; |1 | Fixed graph| Demo | Demo | Fixed | Perfect overfitting | [here](https://app.wandb.ai/avrech/learning2cut/runs/2v0lez39)|  
| &#9745; |2 | Fixed graph| Demo | Demo | Random | Generalization across seeds | [here](https://app.wandb.ai/avrech/learning2cut/runs/3i8f068p)|  
| &#9745; |3 | Random | Demo | Demo | Random | Generalization across graphs | [here](https://app.wandb.ai/avrech/learning2cut/runs/dyvqmmp9)|  
| &#9744; |4 | Random | Demo | Demo+DQN | Random | See convergence to "interesting" policy | [here](https://app.wandb.ai/avrech/learning2cut/runs/1jmcareo)|
| &#9744; |5 | Random | Demo+DQN| Demo+DQN | Random | Improving over SCIP | [here](https://wandb.ai/avrech/learning2cut/runs/1jmcareo?workspace=user-avrech)|

