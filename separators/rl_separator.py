from pyscipopt import Sepa
from time import time
import networkx as nx
from pyscipopt import SCIP_RESULT
import numpy as np
from utils.scip_models import maxcut_mccormic_model, get_separator_cuts_applied
from utils.functions import dijkstra
import operator
import pickle
from torch_geometric.data import Data


class RLSeparator(Sepa):
    def __init__(self, G, x, y, name='RLSeparator',
                 hparams={}
                 ):
        """
        Add violated cycle inequalities to the separation storage.
        """
        self.G = G
        self.x = x
        self.y = y
        self.name = name
        self.hparams = hparams
        self.local = hparams.get('local', True)
        self.removable = hparams.get('removable', True)
        self.forcecut = hparams.get('forcecut', False)

        self.max_per_node = hparams.get('max_per_node', 5)
        self.max_per_round = hparams.get('max_per_round', -1)  # -1 means unlimited
        self.max_per_root = hparams.get('max_per_root', 100)
        self.criterion = hparams.get('criterion', 'most_violated_cycle')
        self.cuts_budget = hparams.get('cuts_budget', 2000)

        self.chordless_only = hparams.get('chordless_only', False)

        self._dijkstra_edge_list = None

        # policy
        self.policy = hparams.get('policy', 'baseline')

        # adaptive policy
        self.starting_policies = []
        self.policy_update_freq = hparams.get('policy_update_freq',
                                              -1)  # number of LP rounds between each params update.
        if self.policy == 'adaptive':
            with open(hparams['starting_policies_abspath'], 'rb') as f:
                self.starting_policies = pickle.load(f)
            # append the given hparams to the starting policies, so when they will finish,
            # the separator will use the given hparams for an additional iteration.
            # after that, the defaults are used.
            self.starting_policies.append(hparams)

        # statistics
        self.ncuts = 0
        self.ncuts_probing = 0

        # accumulate probing stats overhead for subtracting from the problem solving stats
        self._cuts_probing = 0
        self._cuts_applied_probing = 0
        self._lp_iterations_probing = 0
        self._lp_rounds_probing = 0
        self.time_spent = 0
        self.stats = {
            'cycle_ncuts': [],
            'cycle_ncuts_applied': [],
            'total_ncuts_applied': [],
            'cycles_sepa_time': [],
            'solving_time': [],
            'processed_nodes': [],
            'gap': [],
            'lp_rounds': [],
            'lp_iterations': [],
            'dualbound': []
        }
        self._n_lp_rounds = 0
        self._sepa_cnt = 0
        self._separation_efficiency = 0
        self._ncuts_at_cur_node = 0
        self._cur_node = 0  # invalid. root node index is 1
        self.finished = False

    def sepaexeclp(self):
        state_dict = self.model.getState(state_format='dict')
        state_tensor = self.model.getState(state_format='tensor')
        gnn_state = 0

        if self.model.getNCutsApplied() - self._cuts_applied_probing >= self.cuts_budget:
            # terminate
            self.finish_experiment()
            return {"result": SCIP_RESULT.DIDNOTRUN}

        self.update_stats()

        if self.policy == 'adaptive' and self._n_lp_rounds % self.policy_update_freq == 0:
            config = self.starting_policies.pop(0) if len(self.starting_policies) > 0 else {}
            self.update_cut_selection_policy(config=config)

        t0 = time()
        result = self.separate()
        t_left = time() - t0
        self.time_spent += t_left
        self._n_lp_rounds += 1
        return result

    def finish_experiment(self):
        if not self.finished:
            # record stats the last time
            self.update_stats()
            self.finished = True

    def update_stats(self):
        # collect statistics at the beginning of each round, starting from the second round.
        # the statistics are collected before taking any action, and refer to the last round.
        # NOTE: the last update must be done after the solver terminates optimization,
        # outside of this module, by calling McCormicCycleSeparator.update_stats() one more time.
        # if self._round_cnt > 0:
        cycle_cuts, cycle_cuts_applied = get_separator_cuts_applied(self.model, 'MLCycles')
        self.stats['cycle_ncuts'].append(cycle_cuts - self._cuts_probing)
        self.stats['cycle_ncuts_applied'].append(cycle_cuts_applied - self._cuts_applied_probing)
        self.stats['total_ncuts_applied'].append(self.model.getNCutsApplied())
        self.stats['cycles_sepa_time'].append(self.time_spent)
        self.stats['solving_time'].append(self.model.getSolvingTime())
        self.stats['processed_nodes'].append(self.model.getNNodes())
        self.stats['gap'].append(self.model.getGap())
        self.stats['lp_rounds'].append(self.model.getNLPs() - self._lp_rounds_probing)
        self.stats['lp_iterations'].append(self.model.getNLPIterations() - self._lp_iterations_probing)
        self.stats['dualbound'].append(self.model.getDualbound())

    def separate(self):
        # if exceeded limit of cuts per node ,then exit and branch or whatever else.
        cur_node = self.model.getCurrentNode().getNumber()
        if cur_node != self._cur_node:
            self._cur_node = cur_node
            self._ncuts_at_cur_node = 0
        max_cycles = self.max_per_root if cur_node == 1 else self.max_per_node
        if self._ncuts_at_cur_node >= max_cycles:
            print('Reached max_per_node cycles. DIDNOTRUN occured! node {}'.format(cur_node))
            return {"result": SCIP_RESULT.DIDNOTRUN}

        x = np.zeros(self.G.number_of_nodes())
        for i in range(self.G.number_of_nodes()):
            x[i] = self.model.getSolVal(None, self.x[i])
        # search for fractional nodes
        feasible = all(np.logical_or(x == 0, x == 1))
        # if solution is feasible, then return immediately DIDNOTRUN
        if feasible:  # or model.getDualbound() == 6.0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # otherwise, enforce the violated inequalities:
        self.update_dijkstra_edge_list()
        violated_cycles = self.find_violated_cycles(x)
        cut_found = False
        for cycle in violated_cycles:
            if self._ncuts_at_cur_node < max_cycles:
                result = self.add_cut(cycle)
                if result['result'] == SCIP_RESULT.CUTOFF:
                    print('CUTOFF')
                    return result
                self.ncuts += 1
                self._ncuts_at_cur_node += 1
                cut_found = True

        if cut_found:
            self._sepa_cnt += 1
            return {"result": SCIP_RESULT.SEPARATED}
        else:
            print("Cut not found!")
            return {"result": SCIP_RESULT.DIDNOTFIND}

    def update_dijkstra_edge_list(self):
        """
        Construct the weighted edge list representing the undirected graph of Sontag,
        in which the dijkstra algorithm needs to find violated cycle inequalities.
        :return: None
        """
        edge_list = []
        for i, j in self.G.edges:
            xi = self.model.getSolVal(None, self.x[i])
            xj = self.model.getSolVal(None, self.x[j])
            wij = self.model.getSolVal(None, self.y[(i, j)])
            e_in_cut = xi + xj - 2*wij
            edge_list += [((i, 1), (j, 1), e_in_cut),
                          ((i, 2), (j, 2), e_in_cut),
                          ((i, 1), (j, 2), 1 - e_in_cut),
                          ((i, 2), (j, 1), 1 - e_in_cut),
                          ((j, 1), (i, 1), e_in_cut),
                          ((j, 2), (i, 2), e_in_cut),
                          ((j, 1), (i, 2), 1 - e_in_cut),
                          ((j, 2), (i, 1), 1 - e_in_cut)]

            self._dijkstra_edge_list = edge_list

    def find_violated_cycles(self, x):
        # sort the variables according to most infeasibility:
        distance_from_half = np.abs(x - 0.5)
        most_infeasible_nodes = np.argsort(distance_from_half)
        violated_cycles = []
        costs = []
        already_added = set()
        max_cycles = self.max_per_root if self.model.getCurrentNode().getNumber() == 1 else self.max_per_node
        num_cycles_to_add = self.max_per_round if self.max_per_round > 0 else self.G.number_of_nodes()
        num_cycles_to_add = np.min([num_cycles_to_add,
                                    max_cycles - self._ncuts_at_cur_node,
                                    self.cuts_budget - self.model.getNCutsApplied()])

        for runs, i in enumerate(most_infeasible_nodes):
            cost, path = dijkstra(self._dijkstra_edge_list, (i, 1), (i, 2))
            if cost < 1 and (not self.chordless_only or self.is_chordless(path)):
                cycle_edges, F, C_minus_F = [], [], []
                for idx, (i, i_side) in enumerate(path[:-1]):
                    j, j_side = path[idx + 1]
                    e = (i, j) if i < j else (j, i)
                    cycle_edges.append(e)
                    if i_side != j_side:
                        F.append(e)
                    else:
                        C_minus_F.append(e)
                # to avoid double adding the same cycle, sort F and C_minus_F, and store in a set.
                F.sort(key=operator.itemgetter(0, 1))
                C_minus_F.sort(key=operator.itemgetter(0, 1))
                if (tuple(F), tuple(C_minus_F)) not in already_added:
                    violated_cycles.append((cycle_edges, F, C_minus_F))
                    costs.append(cost)
                    already_added.add((tuple(F), tuple(C_minus_F)))

        # define how many cycles to add


        if self.criterion == 'most_violated_cycle':
            # sort the violated cycles, most violated first (lower cost):
            most_violated_cycles = np.argsort(costs)
            return np.array(violated_cycles)[most_violated_cycles[:num_cycles_to_add]]
        elif self.criterion == 'most_infeasible_var':
            return np.array(violated_cycles)[:num_cycles_to_add]
        elif self.criterion == 'random':
            # choose random cycles
            random_cycles = np.array(violated_cycles)
            np.random.shuffle(random_cycles)
            return random_cycles[:num_cycles_to_add]
        elif self.criterion == 'strong':
            """ test all the violated cycles in probing mode, and sort them by their 
            dualbound improvement """
            assert not self.model.inRepropagation()
            assert not self.model.inProbing()
            new_dualbound = []
            cuts_probing_start, cuts_applied_probing_start = get_separator_cuts_applied(self.model, self.name)
            lp_iterations_probing_start = self.model.getNLPIterations()
            lp_rounds_probing_start = self.model.getNLPs()
            for cycle in violated_cycles:
                self.model.startProbing()
                assert not self.model.isObjChangedProbing()
                # self.model.fixVarProbing(self.cont, 2.0)
                # self.model.constructLP()
                # todo - check if separation storage flush is needed
                # add cycle to the separation storage
                self.add_cut(cycle, probing=True)
                self.model.applyCutsProbing()
                self.ncuts_probing += 1
                # solve the LP
                lperror, _ = self.model.solveProbingLP()
                if not lperror:
                    dualbound = -self.model.getLPObjVal()  # for some reason it returns as negative.
                    new_dualbound.append(dualbound)
                else:
                    print('LPERROR OCCURED IN STRONG_CUTTING')
                    new_dualbound.append(1000000)
                self.model.endProbing()
            # calculate how many cuts applied in probing to subtract from stats
            ncuts_probing_end, ncuts_applied_probing_end = get_separator_cuts_applied(self.model, self.name)
            self._cuts_probing += ncuts_probing_end - cuts_probing_start
            self._cuts_applied_probing += ncuts_applied_probing_end - cuts_applied_probing_start
            self._lp_iterations_probing += self.model.getNLPIterations() - lp_iterations_probing_start
            self._lp_rounds_probing += self.model.getNLPs() - lp_rounds_probing_start
            assert len(new_dualbound) == len(violated_cycles)
            # sort the violated cycles, most effective first (lower dualbound):
            strongest = np.argsort(new_dualbound)
            return np.array(violated_cycles)[strongest[:num_cycles_to_add]]


    def add_cut(self, violated_cycle, probing=False):
        result = SCIP_RESULT.DIDNOTRUN
        model = self.model

        if not model.isLPSolBasic():
            return {"result": result}

        result = SCIP_RESULT.DIDNOTFIND
        # add cut
        #TODO: here it might make sense just to have a function `addCut` just like `addCons`. Or maybe better `createCut`
        # so that then one can ask stuff about it, like its efficacy, etc. This function would receive all coefficients
        # and basically do what we do here: cacheRowExtension etc up to releaseRow

        # add the cycle variables to the new row
        cycle_edges, F, C_minus_F = violated_cycle

        cutrhs = len(F) - 1
        name = "probingcycle%d" % self.ncuts_probing if probing else "cycle%d" % self.ncuts
        cut = model.createEmptyRowSepa(self, name, lhs=None, rhs=cutrhs,
                                       local=self.local,
                                       removable=self.removable)
        model.cacheRowExtensions(cut)
        x = self.x
        w = self.y

        for e in F:
            i, j = e
            model.addVarToRow(cut, x[i], 1)
            model.addVarToRow(cut, x[j], 1)
            model.addVarToRow(cut, w[e], -2)

        for e in C_minus_F:
            i, j = e
            model.addVarToRow(cut, x[i], -1)
            model.addVarToRow(cut, x[j], -1)
            model.addVarToRow(cut, w[e], 2)

        if cut.getNNonz() == 0:
            assert model.isFeasNegative(cutrhs)
            # print("Gomory cut is infeasible: 0 <= ", cutrhs)
            return {"result": SCIP_RESULT.CUTOFF}

        # Only take efficacious cuts, except for cuts with one non-zero coefficient (= bound changes)
        # the latter cuts will be handeled internally in sepastore.
        if cut.getNNonz() == 1 or model.isCutEfficacious(cut):
            # flush all changes before adding the cut
            model.flushRowExtensions(cut)

            infeasible = model.addCut(cut, forcecut=self.forcecut)

            if infeasible:
                result = SCIP_RESULT.CUTOFF
            else:
                result = SCIP_RESULT.SEPARATED
        model.releaseRow(cut)
        return {"result": result}


    def update_cut_selection_policy(self, config={}):
        """
        Set self.model params to config. If config is empty, set SCIP defauls.
        :param config: a dictionary containing the following key-value pairs.
        """
        # set scip params:
        if self.hparams.get('debug', False):
            oldparams = {p: self.model.getParam('separating/'+p) for p in ['objparalfac', 'dircutoffdistfac', 'efficacyfac', 'intsupportfac', 'maxcutsroot']}
        self.model.setRealParam('separating/objparalfac', config.get('objparalfac',0.1))
        self.model.setRealParam('separating/dircutoffdistfac', config.get('dircutoffdistfac',0.5))
        self.model.setRealParam('separating/efficacyfac', config.get('efficacyfac', 1))
        self.model.setRealParam('separating/intsupportfac', config.get('intsupportfac', 0.1))
        self.model.setIntParam('separating/maxcutsroot', config.get('maxcutsroot', 2000))
        if self.hparams.get('debug', False):
            newparams = {p: self.model.getParam('separating/'+p) for p in ['objparalfac', 'dircutoffdistfac', 'efficacyfac', 'intsupportfac', 'maxcutsroot']}
            print('changed scip params from:')
            print(oldparams)
            print('to:')
            print(newparams)


if __name__ == "__main__":
    import sys
    from separators.mccormic_cycle_separator import MccormicCycleSeparator
    if '--mixed-debug' in sys.argv:
        import ptvsd

        port = 3000
        # ptvsd.enable_attach(secret='my_secret', address =('127.0.0.1', port))
        ptvsd.enable_attach(address=('127.0.0.1', port))
        ptvsd.wait_for_attach()
    n = 50
    m = 10
    seed = 223
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    nx.set_edge_attributes(G, {e: np.random.normal() for e in G.edges}, name='weight')
    model, x, y = maxcut_mccormic_model(G, use_cuts=False)
    # model.setRealParam('limits/time', 1000 * 1)
    """ Define a controller and appropriate callback to add user's cuts """
    hparams = {'max_per_root': 200000,
               'max_per_round': -1,
               'criterion': 'random',
               'forcecut': False,
               'cuts_budget': 2000,
               'policy': 'default'
               }

    cycle_sepa = MccormicCycleSeparator(G=G, x=x, y=y, hparams=hparams)
    model.includeSepa(cycle_sepa, "MLCycles", "Generate cycle inequalities for MaxCut using McCormic variables exchange",
                      priority=1000000,
                      freq=1)
    rlsepa = RLSeparator(G=G, x=x, y=y)
    model.includeSepa(rlsepa, rlsepa.name,
                      "Reinforcement learning separator",
                      priority=100000,
                      freq=1)
    model.setIntParam('separating/maxcuts', 20)
    model.setIntParam('separating/maxcutsroot', 100)
    model.setIntParam('separating/maxstallroundsroot', -1)
    model.setIntParam('separating/maxroundsroot', 2100)
    model.setRealParam('limits/time', 300)
    model.setLongintParam('limits/nodes', 1)
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
