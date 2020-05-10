from pyscipopt import  Sepa, Conshdlr, SCIP_RESULT, SCIP_STAGE
from time import time
import networkx as nx
import numpy as np
from utils.scip_models import maxcut_mccormic_model, get_separator_cuts_applied
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
        self.filepath = os.path.join(self.savedir, self.name + '_scip_state.pkl')

        # saving mode: 'episode' | 'state'
        # 'episode': save all the state-action pairs in a single file,
        # as a Batch object.
        # 'state': save each state-action pair in a separate file
        # as a Data object.
        self.saving_mode = hparams.get('saving_mode', 'episode')
        self.last_action = None
        self.last_state = None
        self.data_list = []
        self.time_spent = 0
        self.finished_episode = False


    def sepaexeclp(self):
        self.sample()
        return {"result": SCIP_RESULT.DIDNOTRUN}

    def sample(self):
        t0 = time()
        cur_state = self.model.getState(state_format='tensor', return_cut_names=True, query_rows=self.last_action)
        if self.last_action is not None:
            data = (self.last_state, self.last_action)
            self.data_list.append(data)

        # termination condition. TODO: should never happen here
        if self.model.getGap() == 0:
            self.finished_episode = True

        self.last_action = cur_state['cut_names']
        self.last_state = cur_state

        t_left = time() - t0
        self.time_spent += t_left

    def close(self):
        """ query the last action, build the last state-action pair of the episode,
        and save the episode to file """
        if not self.finished_episode and self.last_action is not None:
            self.finished_episode = True
            self.model.isInLPRows(self.last_action)  # TODO this function doesn't really work.
            data = (self.last_state.copy(), self.last_action.copy())
            self.data_list.append(data)
        self.save_data()

    def save_data(self):
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        with open(self.filepath, 'wb') as f:
            pickle.dump(self.data_list, f)
        print('Saved data to: ', self.filepath)


def testSepaSampler():
    import sys
    from separators.mccormic_cycle_separator import MccormicCycleSeparator
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
    model, x, y = maxcut_mccormic_model(G, use_cuts=False)
    # model.setRealParam('limits/time', 1000 * 1)
    """ Define a controller and appropriate callback to add user's cuts """
    hparams = {'max_per_root': 2000,
               'max_per_round': 20,
               'criterion': 'random',
               'forcecut': False,
               'cuts_budget': 2000,
               'policy': 'default'
               }

    cycle_sepa = MccormicCycleSeparator(G=G, x=x, y=y, hparams=hparams)
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
    data_list = torch.load(sampler.filepath)
    from experiments.imitation.cutting_planes_dataset import CuttingPlanesDataset
    dataset = CuttingPlanesDataset(sampler.savedir, savefile=False)
    loader = DataLoader(dataset, batch_size=2, follow_batch=['x_s', 'x_t'])
    batch = next(iter(loader))
    print('finished')


# class ConshdlrSampler(Conshdlr):
#     def __init__(self, G, x, y, name='Sampler',
#                  hparams={}
#                  ):
#         """
#         Sample scip.Model state every time self.sepaexeclp is invoked.
#         Store the generated data object in
#         """
#         self.G = G
#         self.x = x
#         self.y = y
#         self.name = name
#         self.hparams = hparams
#
#         # data list
#         self.data_list = []
#         self.nsamples = 0
#         self.datapath = hparams.get('data_abspath', 'data')
#         self.savedir = hparams.get('relative_savedir', 'examples')
#         self.savedir = os.path.join(self.datapath, self.savedir)
#         self.filepath = os.path.join(self.savedir, self.name + '.pkl')
#         # saving mode: 'episode' | 'state'
#         # 'episode': save all the state-action pairs in a single file,
#         # as a Batch object.
#         # 'state': save each state-action pair in a separate file
#         # as a Data object.
#         self.saving_mode = hparams.get('saving_mode', 'episode')
#         self.last_action = None
#         self.last_state = None
#         self.data_list = []
#         self.time_spent = 0
#         self.finished_episode = False
#
#     def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
#         # comes here only after integrality conshdlr is satisfied, so
#         # it is the end of the episode.
#         # store the last state-action and terminate
#         if self.model.getStage() == SCIP_STAGE.SOLVED and not self.finished_episode:
#             # return infeasible in order to get back to solving stage,
#             # so we will be able to getState (inside consenfolp)
#             return {"result": SCIP_RESULT.INFEASIBLE}
#         return {"result": SCIP_RESULT.FEASIBLE}
#
#     def consenfolp(self, constraints, nusefulconss, solinfeasible):
#         # store last state-action pair
#         self.sample()
#         self.finished_episode = True
#         return {"result": SCIP_RESULT.FEASIBLE}
#
#         if solinfeasible:
#             return {"result": SCIP_RESULT.INFEASIBLE}
#         else:
#             # termination condition
#             self.finished_episode = True
#             return {"result": SCIP_RESULT.FEASIBLE}
#
#     def conssepalp(self, constraints, nusefulconss):
#         self.sample()
#         return {"result": SCIP_RESULT.DIDNOTFIND}
#
#     def conslock(self, constraint, locktype, nlockspos, nlocksneg):
#         pass
#
#     def sample(self):
#         t0 = time()
#         cur_state = self.model.getState(state_format='tensor', return_cut_names=True, query_rows=self.last_action)
#         if self.last_action is not None:
#             data = (self.last_state.copy(), self.last_action.copy())
#             self.data_list.append(data)
#
#         self.last_action = cur_state['cut_names']
#         self.last_state = cur_state
#
#         t_left = time() - t0
#         self.time_spent += t_left
#
#     def close(self):
#         """ query the last action, build the last state-action pair of the episode,
#         and save the episode to file """
#         if not self.finished_episode and self.last_action is not None:
#             self.model.queryRows(self.last_action)
#             data = get_bipartite_graph(self.last_state, scip_action=self.last_action)
#             self.data_list.append(data)
#         self.save_data()
#
#     def save_data(self):
#         if not os.path.exists(self.savedir):
#             os.makedirs(self.savedir)
#         with open(self.filepath, 'wb') as f:
#             pickle.dump(self.data_list, self.filepath)
#         print('Saved data to: ', self.filepath)
#
#
#
# def testConshdlrSampler():
#     import sys
#     from separators.mccormic_cycle_separator import MccormicCycleSeparator
#     if '--mixed-debug' in sys.argv:
#         import ptvsd
#
#         port = 3000
#         # ptvsd.enable_attach(secret='my_secret', address =('127.0.0.1', port))
#         ptvsd.enable_attach(address=('127.0.0.1', port))
#         ptvsd.wait_for_attach()
#     n = 20
#     m = 10
#     seed = 223
#     G = nx.barabasi_albert_graph(n, m, seed=seed)
#     nx.set_edge_attributes(G, {e: np.random.normal() for e in G.edges}, name='weight')
#     model, x, y = maxcut_mccormic_model(G, use_cuts=False)
#     # model.setRealParam('limits/time', 1000 * 1)
#     """ Define a controller and appropriate callback to add user's cuts """
#     hparams = {'max_per_root': 2000,
#                'max_per_round': 20,
#                'criterion': 'random',
#                'forcecut': False,
#                'cuts_budget': 2000,
#                'policy': 'default'
#                }
#
#     cycle_sepa = MccormicCycleSeparator(G=G, x=x, y=y, hparams=hparams)
#     model.includeSepa(cycle_sepa, "MLCycles", "Generate cycle inequalities for MaxCut using McCormic variables exchange",
#                       priority=1000000,
#                       freq=1)
#     sampler = ConshdlrSampler(G=G, x=x, y=y, name='samplertest')
#     # TODO: include conshdlr properly
#     # model.includeConshdlr(sampler, sampler.name,
#     #                       "Reinforcement learning separator",
#     #                       sepapriority=100000,
#     #                       freq=1)
#     model.setIntParam('separating/maxcuts', 20)
#     model.setIntParam('separating/maxcutsroot', 100)
#     model.setIntParam('separating/maxstallroundsroot', -1)
#     model.setIntParam('separating/maxroundsroot', 2100)
#     model.setRealParam('limits/time', 300)
#     # model.setLongintParam('limits/nodes', 1)
#     model.optimize()
#     cycle_sepa.finish_experiment()
#     stats = cycle_sepa.stats
#     print("Solved using user's cutting-planes callback. Objective {}".format(model.getObjVal()))
#     cycle_cuts_applied = -1
#     # TODO: avrech - find a more elegant way to retrive cycle_cuts_applied
#     cuts, cuts_applied = get_separator_cuts_applied(model, 'MLCycles')
#     # model.printStatistics()
#     print('cycles added: ', cuts, ', cycles applied: ', cuts_applied)
#     # print(ci_cut.stats)
#     print('total cuts applied: ', model.getNCutsApplied())
#     print('separation time frac: ', stats['cycles_sepa_time'][-1] / stats['solving_time'][-1])
#     print('cuts applied vs time', stats['total_ncuts_applied'])
#     print('finish')
#     sampler.save_data()
#     print('saved data to: ', sampler.filepath)
#
#     data_list = torch.load(sampler.filepath)
#     from torch_geometric.data import DataLoader
#     loader = DataLoader(data_list, batchsize=2, follow_batch=['x_s', 'x_t'])
#     batch = next(iter(loader))
#     print('finished')

if __name__ == '__main__':
    testSepaSampler()
    # testConshdlrSampler()