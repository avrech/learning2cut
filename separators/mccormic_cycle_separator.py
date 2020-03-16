from pyscipopt import Sepa
from time import time
import networkx as nx
from pyscipopt import SCIP_RESULT
import numpy as np
from utils.scip_models import maxcut_mccormic_model
from utils.functions import dijkstra
import operator


class MccormicCycleSeparator(Sepa):
    def __init__(self, G, x, y,
                 hparams={}
                 ):
        """
        Add violated cycle inequalities to the separation storage if any.
        """
        self.hparams = hparams
        self.local = hparams.get('local', True)
        self.removable = hparams.get('removable', True)
        self.forcecut = hparams.get('forcecut', False)

        self.max_per_node = hparams.get('max_per_node', 5)
        self.max_per_round = hparams.get('max_per_round', 0.1)
        self.max_per_root = hparams.get('max_per_root', 100)
        self.criterion = hparams.get('criterion', 'most_infeasible_var')
        self.max_per_round_relative_to = hparams.get('max_per_round_relative_to', 'num_vars')

        self.chordless_only = hparams.get('chordless_only', False)
        self._num_added_cycles_at_cur_node = 0
        self._cur_node = 0  # invalid. root node index is 1

        self.G = G
        self._dijkstra_edge_list = None
        self.x = x
        self.y = y
        self.num_added_cycles = 0
        self._sepa_cnt = 0
        self.dijkstra_efficiency = 0
        self.time_spent = 0
        self.ncuts = 0

    def sepaexeclp(self):
        t0 = time()
        result = self.separate()
        timing = time() - t0
        self.time_spent += timing
        return result

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
        if self.max_per_round_relative_to == 'num_vars':
            num_cycles_to_add = int(np.ceil(self.max_per_round * self.G.number_of_nodes()))
        elif self.max_per_round_relative_to == 'num_fractions':
            num_fractions = self.G.number_of_nodes() - sum(np.logical_or(x == 0, x == 1))
            num_cycles_to_add = int(np.ceil(self.max_per_round * num_fractions))
        elif self.max_per_round_relative_to == 'num_violations':
            num_cycles_to_add = -1
        num_cycles_to_add = np.min([num_cycles_to_add, self.max_per_node-self._num_added_cycles_at_cur_node])

        if self.criterion == 'most_infeasible_var':
            early_exit = True
        elif self.criterion == 'most_violated_cycle':
            early_exit = False

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
            if early_exit and len(violated_cycles) == num_cycles_to_add:
                # record the fraction of useful dijkstra runs
                self.dijkstra_efficiency = num_cycles_to_add / (runs + 1)
                return violated_cycles

        # define how many cycles to add
        num_violations = len(violated_cycles)
        if self.max_per_round_relative_to == 'num_violations':
            num_cycles_to_add = int(np.ceil(self.max_per_round * num_violations))

        # record the fraction of useful dijkstra runs
        self.dijkstra_efficiency = num_cycles_to_add / len(most_infeasible_nodes)

        # sort the violated cycles, most violated first (lower cost):
        most_violated_cycles = np.argsort(costs)
        return np.array(violated_cycles)[most_violated_cycles[:num_cycles_to_add]]

    def add_cut(self, violated_cycle):
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
        cut = model.createEmptyRowSepa(self, "cycle%d" % self.num_added_cycles, lhs=None, rhs=cutrhs,
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

    def is_chordless(self, path):
        """
        Check if any non-adjacent pair of nodes in the cycle are connected directly in the graph.
        :param path: a list of edges ((from, _), (to, _)), forming a simple cycle.
        :return: True if chordless, otherwise False.
        """
        cycle_nodes = [e[0] for e in path[:-1]]
        cycle = nx.subgraph(self.G, cycle_nodes)
        return nx.is_chordal(cycle)

    def separate(self):
        x = np.zeros(self.G.number_of_nodes())
        for i in range(self.G.number_of_nodes()):
            x[i] = self.model.getSolVal(None, self.x[i])
        # search for fractional nodes
        feasible = all(np.logical_or(x == 0, x == 1))

        # if solution is feasible, then return immediately DIDNOTRUN
        if feasible:  # or model.getDualbound() == 6.0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # if exceeded limit of cuts per node ,then exit and branch or whatever else.
        cur_node = self.model.getCurrentNode().getNumber()
        if cur_node != self._cur_node:
            self._cur_node = cur_node
            self._num_added_cycles_at_cur_node = 0
        max_cycles = self.max_per_root if cur_node == 1 else self.max_per_node
        if self._num_added_cycles_at_cur_node >= max_cycles:
            print('Reached max_per_node cycles. DIDNOTRUN occured! node {}'.format(cur_node))
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # otherwise, enforce the violated inequalities:
        self.update_dijkstra_edge_list()
        violated_cycles = self.find_violated_cycles(x)
        cut_found = False
        for cycle in violated_cycles:
            if self._num_added_cycles_at_cur_node < max_cycles:
                result = self.add_cut(cycle)
                if result['result'] == SCIP_RESULT.CUTOFF:
                    print('CUTOFF')
                    return result
                self.num_added_cycles += 1
                self._num_added_cycles_at_cur_node += 1
                cut_found = True

        if cut_found:
            self._sepa_cnt += 1
            return {"result": SCIP_RESULT.SEPARATED}
        else:
            print("Cut not found!")
            return {"result": SCIP_RESULT.DIDNOTFIND}


if __name__ == "__main__":
    n = 20
    m = 10
    seed = 223
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    nx.set_edge_attributes(G, {e: np.random.normal() for e in G.edges}, name='weight')
    model, x, y = maxcut_mccormic_model(G, use_cuts=False)
    model.setRealParam('limits/time', 1000 * 1)
    """ Define a controller and appropriate callback to add user's cuts """
    hparams = {'max_per_node': 200, 'max_per_round': 1, 'method': 'sepa'}
    ci_cut = MccormicCycleSeparator(G=G, x=x, y=y, hparams=hparams)
    model.includeSepa(ci_cut, "MccormicCycles", "Generate cycle inequalities for MaxCut using McCormic variables exchange",
                      priority=1000000,
                      freq=1)
    model.optimize()
    print("Solved using user's cutting-planes callback. Objective {}".format(model.getObjVal()))
    cycle_cuts_applied = -1
    # TODO: avrech - find a more elegant way to retrive cycle_cuts_applied
    try:
        tmpfile = 'tmp_stats.txt'
        model.writeStatistics(filename=tmpfile)
        with open(tmpfile, 'r') as f:
            for line in f.readlines():
                if 'Mccormic' in line:
                    cycle_cuts_applied = line.split()[-2]
                    print('line cached:')
                    print(line)
                    print(cycle_cuts_applied)
    except:
        print('Failed to retrieve cycle_cuts_applied')

    print('finish')
