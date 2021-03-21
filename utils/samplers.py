from pyscipopt import  Sepa, Conshdlr, SCIP_RESULT, SCIP_STAGE
from time import time
import networkx as nx
import numpy as np
from utils.scip_models import maxcut_mccormic_model, MccormickCycleSeparator
from utils.misc import get_separator_cuts_applied
from utils.data import get_gnn_data
import os
import torch
import pickle

class SepaSampler(Sepa):
    def __init__(self, G, x, y, name='Sampler',
                 hparams={}
                 ):
        """
        Sample scip.Model state every time self.sepaexeclp is invoked.
        Store the generated data object in
        """
        self.G = G
        self.x = x
        self.y = y
        self.name = name
        self.hparams = hparams

        # data list
        self.data_list = []
        self.nsamples = 0
        self.datapath = hparams.get('data_abspath', 'data')
        self.savedir = hparams.get('relative_savedir', 'examples')
        self.savedir = os.path.join(self.datapath, self.savedir)
        self.data_filepath = os.path.join(self.savedir, self.name + '_scip_state.pkl')
        self.stats_filepath = os.path.join(self.savedir, self.name + '_stats.pkl')

        # saving mode: 'episode' | 'state'
        # 'episode': save all the state-action pairs in a single file,
        # as a Batch object.
        # 'state': save each state-action pair in a separate file
        # as a Data object.
        self.saving_mode = hparams.get('saving_mode', 'episode')
        self.reward_func = hparams.get('reward_func', 'db_integral_credit')
        self.db_scale = hparams.get('db_scale', 1.0)
        self.lpiter_scale = hparams.get('lpiter_scale', 1.0)
        self.prev_action = None
        self.prev_state = None
        self.data_list = []
        self.time_spent = 0
        self.finished_episode = False
        # stats
        self.sample_format = hparams.get('sample_format', "sars")
        self.stats = {
            'ncuts': [],
            'ncuts_applied': [],
            'solving_time': [],
            'processed_nodes': [],
            'gap': [],
            'lp_rounds': [],
            'lp_iterations': [],
            'dualbound': []
        }

    def sepaexeclp(self):
        self.sample()
        return {"result": SCIP_RESULT.DIDNOTRUN}

    def update_stats(self):
        # collect statistics at the beginning of each round, starting from the second round.
        # the statistics are collected before taking any action, and refer to the last round.
        # NOTE: the last update must be done after the solver terminates optimization,
        # outside of this module, by calling McCormicCycleSeparator.update_stats() one more time.
        self.stats['ncuts'].append(self.model.getNCuts())
        self.stats['ncuts_applied'].append(self.model.getNCutsApplied())
        self.stats['solving_time'].append(self.model.getSolvingTime())
        self.stats['processed_nodes'].append(self.model.getNNodes())
        self.stats['gap'].append(self.model.getGap())
        self.stats['lp_rounds'].append(self.model.getNLPs())
        self.stats['lp_iterations'].append(self.model.getNLPIterations())
        self.stats['dualbound'].append(self.model.getDualbound())

    def get_reward(self):
        """
        compute action-wise reward according to self.reward_func
        :return: np.ndarray of size len(self.last_action['activity'])
        """
        # compute reward
        db_improvement = np.abs(self.stats['dualbound'][-1] - self.stats['dualbound'][-2]) * self.db_scale
        lp_iterations = (self.stats['lp_iterations'][-1] - self.stats['lp_iterations'][-2]) * self.lpiter_scale
        activity = self.prev_action['activity']
        if self.reward_func == 'db_improvement':
            return np.full_like(activity, fill_value=db_improvement)

        elif self.reward_func == 'db_integral':
            return np.full_like(activity, fill_value=- db_improvement * lp_iterations)

        elif self.reward_func == 'db_improvement_credit':
            return db_improvement * (1 + activity)

        elif self.reward_func == 'db_integral_credit':
            return db_improvement * lp_iterations * (activity - 1)

        elif self.reward_func == 'db_lpiter_fscore':
            # compute the harmonic average of p=db_improvement and q=1/lp_iterations
            # fscore = p*q/(p+q)
            fscore = db_improvement / (db_improvement * lp_iterations + 1)
            # this fscore will be high iff its both elements will be high,
            # i.e great dual bound improvement in a few lp iterations
            return np.full_like(activity, fill_value=fscore)

        elif self.reward_func == 'db_lpiter_fscore_credit':
            # compute the fscore as above,
            # and assign the credit to the active constraints only.
            fscore = db_improvement / (db_improvement * lp_iterations + 1)
            return fscore * (1 + activity)

    def sample(self):
        t0 = time()
        self.update_stats()
        cur_state = self.model.getState(state_format='tensor', get_available_cuts=True, query=self.prev_action)
        # compute the reward as the dual bound integral vs. LP iterations

        if self.prev_action is not None:
            action = self.prev_action['applied']
            reward = self.get_reward()
            if self.sample_format == 'sa':
                data = (self.prev_state, action)
            elif self.sample_format == 'sars':
                # TODO verify
                data = (self.prev_state, action, reward, cur_state)
            self.data_list.append(data)

        # termination condition. TODO: should never happen here
        if self.model.getGap() == 0:
            self.finished_episode = True

        self.prev_action = cur_state['cut_names']
        self.prev_state = cur_state

        t_left = time() - t0
        self.time_spent += t_left

    def close(self):
        """ query the last action, build the last state-action pair of the episode,
        and save the episode to file """
        if not self.finished_episode and self.prev_action is not None:
            self.finished_episode = True
            self.update_stats()
            self.model.isInLPRows(self.prev_action)  # TODO this function doesn't really work.
            data = (self.prev_state.copy(), self.prev_action.copy())
            self.data_list.append(data)
        self.save_data()

    def save_data(self):
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        with open(self.data_filepath, 'wb') as f:
            pickle.dump(self.data_list, f)
        print('Saved data to: ', self.data_filepath)

    def save_stats(self):
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        with open(self.stats_filepath, 'wb') as f:
            pickle.dump(self.stats, f)
        print('Saved stats to: ', self.stats_filepath)


def testSepaSampler():
    import sys
    if '--mixed-debug' in sys.argv:
        import ptvsd

        port = 3000
        # ptvsd.enable_attach(secret='my_secret', address =('127.0.0.1', port))
        ptvsd.enable_attach(address=('127.0.0.1', port))
        ptvsd.wait_for_attach()
    n = 20
    m = 10
    seed = 223
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    nx.set_edge_attributes(G, {e: np.random.normal() for e in G.edges}, name='weight')
    model, x, y = maxcut_mccormic_model(G, use_general_cuts=False)
    # model.setRealParam('limits/time', 1000 * 1)
    """ Define a controller and appropriate callback to add user's cuts """
    hparams = {'max_per_root': 2000,
               'max_per_round': 20,
               'criterion': 'random',
               'forcecut': False,
               'cuts_budget': 2000,
               'policy': 'default'
               }

    cycle_sepa = MccormickCycleSeparator(G=G, x=x, y=y, hparams=hparams)
    model.includeSepa(cycle_sepa, "MLCycles", "Generate cycle inequalities for MaxCut using McCormic variables exchange",
                      priority=1000000,
                      freq=1)
    sampler = SepaSampler(G=G, x=x, y=y, name='samplertest')
    model.includeSepa(sampler, sampler.name,
                      "Reinforcement learning separator",
                      priority=100000,
                      freq=1)
    model.setIntParam('separating/maxcuts', 20)
    model.setIntParam('separating/maxcutsroot', 100)
    model.setIntParam('separating/maxstallroundsroot', -1)
    model.setIntParam('separating/maxroundsroot', 2100)
    model.setRealParam('limits/time', 300)
    # model.setLongintParam('limits/nodes', 1)
    model.optimize()
    cycle_sepa.finish_experiment()
    stats = cycle_sepa.stats
    print("Solved using user's cutting-planes callback. Objective {}".format(model.getObjVal()))
    cycle_cuts_applied = -1
    # TODO: avrech - find a more elegant way to retrive cycle_cuts_applied
    cuts, cuts_applied = get_separator_cuts_applied(model, 'MLCycles')
    # model.printStatistics()
    print('cycles added: ', cuts, ', cycles applied: ', cuts_applied)
    # print(ci_cut.stats)
    print('total cuts applied: ', model.getNCutsApplied())
    print('separation time frac: ', stats['cycles_sepa_time'][-1] / stats['solving_time'][-1])
    print('cuts applied vs time', stats['total_ncuts_applied'])
    print('finish')
    sampler.save_data()
    from torch_geometric.data import DataLoader
    data_list = torch.load(sampler.data_filepath)
    from experiments.imitation.cutting_planes_dataset import CuttingPlanesDataset
    dataset = CuttingPlanesDataset(sampler.savedir, savefile=False)
    loader = DataLoader(dataset, batch_size=2, follow_batch=['x_s', 'x_t'])
    batch = next(iter(loader))
    print('finished')

if __name__ == '__main__':
    testSepaSampler()
