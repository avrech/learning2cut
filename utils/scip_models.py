import operator
import pickle
from time import time
import torch
import networkx as nx
import numpy as np
from pyscipopt import quicksum
import pyscipopt as scip
from collections import OrderedDict

from pyscipopt import Sepa, SCIP_RESULT, SCIP_STAGE

from utils.functions import dijkstra_best_shortest_path
from utils.misc import get_separator_cuts_applied


def set_aggresive_separation(model):
    """ set SCIP separating parameters so that separators are called every round,
     and increase the limit of cuts per round """
    # model.setIntParam("separating/strongcg/freq", -1)
    # model.setIntParam("separating/gomory/freq", -1)
    # model.setIntParam("separating/aggregation/freq", -1)
    # model.setIntParam("separating/mcf/freq", -1)
    # model.setIntParam("separating/closecuts/freq", -1)

    # model.setIntParam("separating/zerohalf/freq", -1)
    # for sepa in ['clique', 'closecuts', 'flowcover', 'cmir', 'gomory', 'strongcg', 'zerohalf', 'mcf', 'aggregation']:
    # the following cuts are disabled by default: cgmip, closecuts, convexproj, eccuts, gauge, oddcycle, rapidlearning,
    # todo what is maxbounddist? (maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <disjunctive> (0.0: only on current best node, 1.0: on all nodes)
    model.setIntParam("separating/clique/maxsepacuts", 100)  # 10
    model.setIntParam("separating/clique/freq", 1)  # 0
    model.setIntParam("separating/flowcover/freq", 1)  # 10
    model.setIntParam("separating/cmir/freq", 1)  # 10
    model.setIntParam("separating/aggregation/freq", 1)  # 10
    model.setIntParam("separating/gomory/freq", 1)  # 10
    model.setIntParam("separating/gomory/maxroundsroot", -1)  # 10
    model.setIntParam("separating/strongcg/freq", -1)  # 10  # todo this separator makes problems with null rows
    model.setIntParam("separating/strongcg/maxroundsroot", -1)  # 20
    model.setIntParam("separating/strongcg/maxrounds", -1)  # 5
    model.setIntParam("separating/zerohalf/freq", 1)  # 10
    model.setIntParam("separating/zerohalf/maxroundsroot", -1)  # 20
    model.setIntParam("separating/zerohalf/maxrounds", -1)  # 5

    # todo - what is it?
    # minimal integrality violation of a basis variable in order to try Gomory cut
    # [type: real, advanced: FALSE, range: [0.0001,0.5], default: 0.01]
    # separating / gomory / away = 0.01


def mvc_model(G, model_name='MVC Model',
              use_presolve=True, use_heuristics=False, use_general_cuts=True, use_propagation=True,
              use_random_branching=True, use_cut_pool=True, allow_restarts=False, add_trivial_sol=True, time_limit=1800):
    r"""
        Returns Minimum Vertex Cover model defined by G(V,E)

        MVC:

        :math:`min_{x} \ \sum_i x_i`
        subject to:
            .. math::
                x_i \in \{0,1\}

                x_i + x_j \geq 1 \ \ \ \forall (i,j) in E

        :param G (nx.Graph): Undirected graph with node attributes 'c'
        :param model_name (str): SCIP model name
        :param use_presolve (bool): Enable presolving
        :param use_heuristics (bool): Enable heuristics
        :param use_cuts (bool): Enable default cuts
        :param use_propagation (bool): Enable propagation
        :return: pyscipopt.Model
    """
    V = G.nodes
    E = G.edges
    c = nx.get_node_attributes(G, 'c')
    if len(c) == 0:
        c = {i: 1 for i in V}

    # create SCIP model:
    model = scip.Model(model_name)
    # x = OrderedDict([(i, model.addVar(name=f'{i}', obj=c[i], vtype='B')) for i in V])
    # x = {i: model.addVar(name=f'{i}', obj=c[i], vtype='B') for i in V}
    x = {}
    for i in V:
        x[i] = model.addVar(name=f'{i}', obj=c[i], vtype='B')

    """ McCormic Inequalities """
    for ij in E:
        i, j = ij
        model.addCons(1 <= (x[i] + x[j]), name=f'cover{ij}')

    model.setMinimize()
    model.setBoolParam("misc/allowdualreds", 0)

    # turn off propagation
    if not use_propagation:
        model.setIntParam("propagating/maxrounds", 0)
        model.setIntParam("propagating/maxroundsroot", 0)
    # turn off some cuts
    model.setIntParam("separating/strongcg/freq", -1)
    model.setIntParam("separating/aggregation/freq", -1)
    model.setIntParam("separating/mcf/freq", -1)
    model.setIntParam("separating/closecuts/freq", -1)
    if not use_general_cuts:
        model.setIntParam("separating/gomory/freq", -1)
        model.setIntParam("separating/clique/freq", -1)
        model.setIntParam("separating/zerohalf/freq", -1)
    if not use_presolve:
        model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        model.disablePropagation()
    if not use_heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
    if use_random_branching:
        model.setIntParam('branching/random/priority', 999999)
    if not use_cut_pool:
        model.setIntParam('separating/poolfreq', -1)
        # model.setBoolParam('separating/cgmip/usecutpool', False)
    if not allow_restarts:
        model.setIntParam('presolving/maxrestarts', 0)
    if add_trivial_sol:
        s = model.createSol()
        for x_i in x.values():
            s[x_i] = 1
        assert model.addSol(s, free=True)
    model.setRealParam('limits/time', time_limit)
    return model, x


def maxcut_mccormic_model(G, model_name='MAXCUT McCormic Model',
                          use_presolve=True, use_heuristics=False, use_general_cuts=True, use_propagation=True,
                          use_random_branching=True, use_cycles=True, hparams={}, allow_restarts=False,
                          add_trivial_sol=True, time_limit=1800):
    r"""
    Returns MAXCUT model of G assuming edge attributes named 'weight', denoted by `w`.

    MAXCUT:

    :math:`max_{x,y} \ \sum_ij w_ij (x_i + x_j - 2y_ij)`
    subject to:
        .. math::
            x_i \in \{0,1\}

            y_ij \geq 0

            y_ij \leq x_i

            y_ij \leq x_j

            x_i + x_j - y_ij \leq 1

    :param G (nx.Graph): Undirected graph with edges attribute 'weight'
    :param model_name (str): SCIP model name
    :param use_presolve (bool): Enable presolving
    :param use_heuristics (bool): Enable heuristics
    :param use_cuts (bool): Enable default cuts
    :param use_propagation (bool): Enable propagation
    :return: pyscipopt.Model
    """
    V = G.nodes
    E = G.edges
    w = nx.get_edge_attributes(G, 'weight')
    if len(w) == 0:
        w = {e: 1 for e in E}

    # compute coefficients:
    x_coef = {}
    y_coef = {}
    for i in V:
        x_coef[i] = sum([w[e] for e in E if i in e])
    for e in E:
        y_coef[e] = -2 * w[e]

    # create SCIP model:
    model = scip.Model(model_name)
    x = OrderedDict([(i, model.addVar(name='{}'.format(i), obj=x_coef[i], vtype='B')) for i in V])
    y = OrderedDict([(ij, model.addVar(name='{}'.format(ij), obj=y_coef[ij], vtype='C', lb=0, ub=1)) for ij in E])

    """ McCormic Inequalities """
    for ij in E:
        i, j = ij
        model.addCons(0 <= (quicksum([x[i], x[j], -y[ij]]) <= 1), name='{}'.format(ij))
        model.addCons(0 <= (x[i] - y[ij] <= 1), name='{}'.format((ij, i)))
        model.addCons(0 <= (x[j] - y[ij] <= 1), name='{}'.format((ij, j)))

    model.setMaximize()
    model.setBoolParam("misc/allowdualreds", 0)

    # turn off propagation
    if not use_propagation:
        model.setIntParam("propagating/maxrounds", 0)
        model.setIntParam("propagating/maxroundsroot", 0)
    # turn off some cuts
    if not use_general_cuts:
        model.setIntParam("separating/strongcg/freq", -1)
        model.setIntParam("separating/gomory/freq", -1)
        model.setIntParam("separating/aggregation/freq", -1)
        model.setIntParam("separating/mcf/freq", -1)
        model.setIntParam("separating/closecuts/freq", -1)
        model.setIntParam("separating/clique/freq", -1)
        model.setIntParam("separating/zerohalf/freq", -1)
    if not use_presolve:
        model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        model.disablePropagation()
    if not use_heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
    if use_random_branching:
        model.setIntParam('branching/random/priority', 999999)
    if use_cycles:
        # include cycle inequalities separator with high priority
        cycle_sepa = MccormickCycleSeparator(G=G, x=x, y=y, name='MLCycles', hparams=hparams)
        model.includeSepa(cycle_sepa, 'MLCycles',
                          "Generate cycle inequalities for the MaxCut McCormick formulation",
                          priority=1000000, freq=1)
    else:
        cycle_sepa = None
    if not allow_restarts:
        model.setIntParam('presolving/maxrestarts', 0)
    if add_trivial_sol:
        s = model.createSol()
        for x_i in x.values():
            s[x_i] = 1
        for y_ij in y.values():
            s[y_ij] = 1
        assert model.addSol(s, free=True)
    model.setRealParam('limits/time', time_limit)
    # unify x and y to a single dictionary
    x_dict = {**x, **y}
    return model, x_dict, cycle_sepa


class MccormickCycleSeparator(Sepa):
    def __init__(self, G, x, y, name='MLCycles',
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

        self.max_per_node = hparams.get('max_per_node', 100000000)
        self.max_per_round = hparams.get('max_per_round', -1)  # -1 means unlimited
        self.max_per_root = hparams.get('max_per_root', 100000000)
        self.criterion = hparams.get('criterion', 'most_violated_cycle')
        self.cuts_budget = hparams.get('cuts_budget', 100000000)
        self.max_cuts_node = hparams.get('max_cuts_node', 100000000)
        self.max_cuts_root = hparams.get('max_cuts_root', 100000000)
        self.max_cuts_applied_node = hparams.get('max_cuts_applied_node', 100000000)
        self.max_cuts_applied_root = hparams.get('max_cuts_applied_root', 100000000)

        # cycle separation routine
        self.enable_chordality_check = hparams.get('enable_chordality_check', False)
        self.chordless_only = hparams.get('chordless_only', False)
        self.simple_cycle_only = hparams.get('simple_cycle_only', True)
        self._dijkstra_edge_list = None
        self.added_cuts = set()

        # policy
        self.policy = 'notusedanymore'  # hparams.get('policy', 'baseline')

        # adaptive policy
        self.starting_policies = []
        self.policy_update_freq = hparams.get('policy_update_freq', -1)  # number of LP rounds between each params update.
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

        # general statistics
        self.time_spent = 0
        self.record = hparams.get('record', False)
        self.stats = {
            'ncuts': [],
            'ncuts_applied': [],
            'cycles_sepa_time': [],
            'solving_time': [],
            'processed_nodes': [],
            'gap': [],
            'lp_rounds': [],
            'lp_iterations': [],
            'dualbound': [],
            'nchordless': [],
            'nsimple': [],
            'ncycles': [],
            'nchordless_applied': [],
        }
        self.current_round_cycles = {}
        self.current_round_cycles_done = False
        self.nseparounds = 0
        self._sepa_cnt = 0
        self._separation_efficiency = 0
        self.ncuts_at_cur_node = 0
        self.ncuts_applied_at_cur_node = 0
        self.ncuts_applied_at_entering_cur_node = 0
        self._cur_node = 0  # invalid. root node index is 1
        self.finished = False
        self.nchordless = 0
        self.nsimple = 0
        self.ncycles = 0
        self.recorded_cycles = []
        self.record_cycles = hparams.get('record_cycles', False)

        # debug cutoff events
        self.debug_cutoff = hparams.get('debug_cutoff', False)
        if self.debug_cutoff:
            # restore the optimal solution from G:
            self.x_opt = nx.get_node_attributes(G, 'x')
            self.y_opt = nx.get_edge_attributes(G, 'y')
            cut_opt = nx.get_edge_attributes(G, 'cut')
            w = nx.get_edge_attributes(G, 'weight')
            self.opt_obj = sum([w[e] for e, is_cut in cut_opt.items() if is_cut])
            self.debug_invalid_cut_stats = {}
            self.debug_cutoff_stats = {}
            self.cutoff_occured = False

    def sepaexeclp(self):
        if self.hparams.get('debug_events', False):
            print('DEBUG MSG: cycles separator called')

        # reset solver maxcuts and maxcutsroot
        # because dqn_agent can set it to a low number
        if self.hparams.get('reset_maxcuts_every_round', False):
            self.model.setIntParam('separating/maxcuts', 100000)
            self.model.setIntParam('separating/maxcutsroot', 100000)
        if self.debug_cutoff and not self.cutoff_occured:
            self.catch_cutoff()

        if self.model.getNCutsApplied() - self._cuts_applied_probing >= self.cuts_budget:
            # terminate
            self.finish_experiment()
            return {"result": SCIP_RESULT.DIDNOTRUN}

        lp_iterations_limit = self.hparams.get('lp_iterations_limit', -1)
        if lp_iterations_limit > 0 and self.model.getNLPIterations() >= lp_iterations_limit:
            # terminate
            if self.hparams.get('verbose', 0) == 2:
                print('mccormick_cycle_separator: LP_ITERATIONS_LIMIT reached. terminating!')
            self.finish_experiment()
            return {"result": SCIP_RESULT.DIDNOTRUN}

        self.update_stats()  # must be called before self.separate() is called

        if self.policy == 'adaptive' and self.nseparounds % self.policy_update_freq == 0:
            config = self.starting_policies.pop(0) if len(self.starting_policies) > 0 else {}
            self.update_cut_selection_policy(config=config)

        t0 = time()
        result = self.separate()
        t_left = time() - t0
        self.time_spent += t_left
        self.nseparounds += 1
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
        # cycle_cuts, cycle_cuts_applied = get_separator_cuts_applied(self.model, self.name)
        self.stats['ncuts'].append(self.model.getNCuts() - self._cuts_probing)
        self.stats['ncuts_applied'].append(self.model.getNCutsApplied() - self._cuts_applied_probing)
        self.stats['cycles_sepa_time'].append(self.time_spent)
        self.stats['solving_time'].append(self.model.getSolvingTime())
        self.stats['processed_nodes'].append(self.model.getNNodes())
        self.stats['lp_rounds'].append(self.model.getNLPs() - self._lp_rounds_probing)
        self.stats['gap'].append(self.model.getGap())
        self.stats['lp_iterations'].append(self.model.getNLPIterations() - self._lp_iterations_probing)
        self.stats['dualbound'].append(self.model.getDualbound())

        # enforce the lp_iterations_limit:
        lp_iterations_limit = self.hparams.get('lp_iterations_limit', -1)
        if lp_iterations_limit > 0 and self.stats['lp_iterations'][-1] > lp_iterations_limit:
            # interpolate the dualbound and gap at the limit
            assert self.stats['lp_iterations'][-2] < lp_iterations_limit
            t = self.stats['lp_iterations'][-2:]
            for k in ['dualbound', 'gap']:
                ft = self.stats[k][-2:]
                # compute ft slope in the last interval [t[-2], t[-1]]
                slope = (ft[-1] - ft[-2]) / (t[-1] - t[-2])
                # compute the linear interpolation of ft at the limit
                interpolated_ft = ft[-2] + slope * (lp_iterations_limit - t[-2])
                self.stats[k][-1] = interpolated_ft
            # finally truncate the lp_iterations to the limit
            self.stats['lp_iterations'][-1] = lp_iterations_limit

        self.stats['nsimple'].append(self.nsimple)
        self.stats['ncycles'].append(self.ncycles)

        if self.enable_chordality_check:
            self.stats['nchordless'].append(self.nchordless)
            if self.model.getStage() == SCIP_STAGE.SOLVING:
                self.model.queryRows(self.current_round_cycles)
                self.stats['nchordless_applied'].append(sum([cycle['applied'] for cycle in self.current_round_cycles.values() if cycle['is_chordless']]))
            else:
                self.stats['nchordless_applied'].append(0)  # pessimistic estimate

        # record cycles
        if self.record_cycles and len(self.current_round_cycles) > 0 and not self.current_round_cycles_done:
            cut_names = self.model.getSelectedCutsNames()
            for cut_name in cut_names:
                self.current_round_cycles[cut_name]['applied'] = True
            self.recorded_cycles.append(self.current_round_cycles)
            self.current_round_cycles_done = True

    def separate(self):
        self.current_round_cycles = {}
        self.current_round_cycles_done = False
        self.added_cuts = set()

        # if exceeded limit of cuts per node ,then exit and branch or whatever else.
        cur_node = self.model.getCurrentNode().getNumber()
        self.ncuts_applied_at_cur_node = self.model.getNCutsApplied() - self.ncuts_applied_at_entering_cur_node
        if cur_node != self._cur_node:
            self._cur_node = cur_node
            self.ncuts_at_cur_node = 0
            self.ncuts_applied_at_cur_node = 0
            self.ncuts_applied_at_entering_cur_node = self.model.getNCutsApplied()

        max_cycles = self.max_per_root if cur_node == 1 else self.max_per_node
        max_cuts = self.max_cuts_root if cur_node == 1 else self.max_cuts_node
        max_cuts_applied = self.max_cuts_applied_root if cur_node == 1 else self.max_cuts_applied_node

        if self.ncuts_at_cur_node >= max_cuts:
            print('Reached max number of cuts added at node. DIDNOTRUN occured! node {}'.format(cur_node))
            return {"result": SCIP_RESULT.DIDNOTRUN}

        if self.ncuts_applied_at_cur_node >= max_cuts_applied:
            print('Reached max number of cuts applied at node. DIDNOTRUN occured! node {}'.format(cur_node))
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
            if self.ncuts_at_cur_node < max_cycles:
                result, cut_added = self.add_cut(cycle)
                if result['result'] == SCIP_RESULT.CUTOFF:
                    print('CUTOFF')
                    return result
                elif cut_added:
                    self.ncuts += 1
                    self.ncuts_at_cur_node += 1
                    cut_found = True

        # # record cycles
        # if self.record_cycles:
        #     # cycles = [{'edges': cycle[0],
        #     #            'C_minus_F': cycle[1],
        #     #            'F': cycle[2],
        #     #            'is_chordless': cycle[3],
        #     #            'is_simple': cycle[4]} for cycle in violated_cycles]
        #     # self.recorded_cycles.append(cycles)
        #     # prob SCIP which cycles will be selected:
        #     self.model.startProbing()
        #     self.model.applyCutsProbing()
        #     cut_names = self.model.getSelectedCutsNames()
        #     self.model.endProbing()
        #     for cut_name in cut_names:
        #         self.current_round_cycles[cut_name]['applied'] = True
        #     self.recorded_cycles.append(self.current_round_cycles)

        if cut_found:
            self._sepa_cnt += 1
            return {"result": SCIP_RESULT.SEPARATED}
        else:
            if self.hparams.get('debug', False):
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
        # reset counters for the current separation round
        self.nchordless = 0
        self.nsimple = 0
        self.ncycles = 0

        # sort the variables according to most infeasibility:
        distance_from_half = np.abs(x - 0.5)
        most_infeasible_nodes = np.argsort(distance_from_half)
        violated_cycles = []
        costs = []
        already_added = set()
        max_cycles = self.max_per_root if self.model.getCurrentNode().getNumber() == 1 else self.max_per_node
        num_cycles_to_add = self.max_per_round if self.max_per_round > 0 else self.G.number_of_nodes()
        num_cycles_to_add = np.min([num_cycles_to_add,
                                    max_cycles - self.ncuts_at_cur_node,
                                    self.cuts_budget - self.model.getNCutsApplied()])

        for runs, i in enumerate(most_infeasible_nodes):
            cost, closed_walk = dijkstra_best_shortest_path(self._dijkstra_edge_list, (i, 1), (i, 2))
            if self.enable_chordality_check:
                is_chordless = self.is_chordless(closed_walk)
            else:
                is_chordless = -1
            is_simple = self.is_simple_cycle(closed_walk)
            if cost < 1 \
                    and (not (self.chordless_only and self.enable_chordality_check) or is_chordless) \
                    and (not self.simple_cycle_only or is_simple):

                self.nchordless += is_chordless
                self.nsimple += is_simple
                self.ncycles += 1

                cycle_edges, F, C_minus_F = [], [], []
                for idx, (i, i_side) in enumerate(closed_walk[:-1]):
                    j, j_side = closed_walk[idx + 1]
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
                    violated_cycles.append((cycle_edges, F, C_minus_F, is_chordless, is_simple))
                    costs.append(cost)
                    already_added.add((tuple(F), tuple(C_minus_F)))

        # # record cycles
        # if self.record_cycles:
        #     cycles = [{'edges': cycle[0],
        #                'C_minus_F': cycle[1],
        #                'F': cycle[2],
        #                'is_chordless': cycle[3],
        #                'is_simple': cycle[4]} for cycle in violated_cycles]
        #     self.recorded_cycles.append(cycles)

        # define how many cycles to add
        if self.criterion == 'most_violated_cycle':
            # sort the violated cycles, most violated first (lower cost):
            most_violated_cycles = np.argsort(costs)
            return np.array(violated_cycles, dtype=object)[most_violated_cycles[:num_cycles_to_add]]
        elif self.criterion == 'most_infeasible_var':
            return np.array(violated_cycles, dtype=object)[:num_cycles_to_add]
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
                _, cut_added = self.add_cut(cycle, probing=True)
                if cut_added:
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
                else:
                    print('CUT SKIPPED IN STRONG CUTTING')
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
        elif self.criterion == 'none':
            return np.array(violated_cycles)
        else:
            raise ValueError

    def add_cut(self, violated_cycle, probing=False):
        scip_result = SCIP_RESULT.DIDNOTRUN
        cut_added = False
        model = self.model

        if not model.isLPSolBasic():
            return {"result": scip_result}, cut_added

        scip_result = SCIP_RESULT.DIDNOTFIND
        # add cut
        #TODO: here it might make sense just to have a function `addCut` just like `addCons`. Or maybe better `createCut`
        # so that then one can ask stuff about it, like its efficacy, etc. This function would receive all coefficients
        # and basically do what we do here: cacheRowExtension etc up to releaseRow

        # add the cycle variables to the new row
        cycle_edges, F, C_minus_F, is_chordless, is_simple = violated_cycle

        x = self.x
        y = self.y

        # compute variable coefficients
        x_coef = {i: 0 for e in cycle_edges for i in e}
        y_coef = {e: 0 for e in cycle_edges}

        for e in F:
            i, j = e
            x_coef[i] += 1
            x_coef[j] += 1
            y_coef[e] -= 2

        for e in C_minus_F:
            i, j = e
            x_coef[i] -= 1
            x_coef[j] -= 1
            y_coef[e] += 2

        cutrhs = len(F) - 1
        cutlhs = -len(C_minus_F)

        # filter "empty" cycles: 0 <= 0
        all_zeros = all(np.array(list(x_coef.values()) + list(y_coef.values())) == 0)
        if all_zeros:
            assert cutlhs <= 0 <= cutrhs
            # skip this cut
            return {'result': scip_result}, cut_added

        # debug
        # check if inequality is valid with respect to the optimal solution found
        if self.debug_cutoff:
            assert self.is_valid_inequality(x_coef, y_coef, cutrhs)

        # filter duplicated cuts
        # after coefficient aggregation, to different cycles can collapse into the same inequality.
        x_items = list(x_coef.items())
        x_items.sort(key=operator.itemgetter(0)) # sort variables
        y_keys = list(y_coef.keys())
        y_keys.sort(key=operator.itemgetter(0, 1)) # sort edges
        y_items = [(ij, y_coef[ij]) for ij in y_keys]
        cut_id = (tuple(x_items), tuple(y_items), cutrhs, cutlhs)
        if cut_id in self.added_cuts:
            # skip this cycle
            return {'result': scip_result}, cut_added

        # else - add to the added_cuts set, and add the cut to scip separation storage
        self.added_cuts.add(cut_id)

        # create a Row object, and add the variables and the coefficients
        name = "pc%d" % self.ncuts_probing if probing else "c%d" % self.ncuts
        cut = model.createEmptyRowSepa(self, name, rhs=cutrhs, lhs=cutlhs,
                                       local=self.local,
                                       removable=self.removable)
        model.cacheRowExtensions(cut)
        self.current_round_cycles[name] = {'edges': cycle_edges,
                                           'C_minus_F': C_minus_F,
                                           'F': F,
                                           'is_chordless': is_chordless,
                                           'is_simple': is_simple,
                                           'applied': False,
                                           }

        for i, c in x_coef.items():
            if c != 0:
                model.addVarToRow(cut, x[i], c)
        for e, c in y_coef.items():
            if c != 0:
                model.addVarToRow(cut, y[e], c)

        # todo: model.isFeasNegative(0) returns False, why?
        #       a cycle 0 <= 0 is valid, but here it fails.
        #       we filter such cases beforehand.
        if cut.getNNonz() == 0:
            # debug
            if not model.isFeasNegative(cutrhs):
                print(f'cutrhs: {cutrhs}')
                print(f'x_coef: {x_coef}')
                print(f'y_coef: {y_coef}')
                print(violated_cycle)
            assert model.isFeasNegative(cutrhs)
            # print("Gomory cut is infeasible: 0 <= ", cutrhs)
            return {"result": SCIP_RESULT.CUTOFF}, cut_added

        # Only take efficacious cuts, except for cuts with one non-zero coefficient (= bound changes)
        # the latter cuts will be handeled internally in sepastore.
        if cut.getNNonz() == 1 or model.isCutEfficacious(cut):
            # flush all changes before adding the cut
            model.flushRowExtensions(cut)

            infeasible = model.addCut(cut, forcecut=self.forcecut)

            if infeasible:
                scip_result = SCIP_RESULT.CUTOFF
            else:
                scip_result = SCIP_RESULT.SEPARATED
            cut_added = True

        model.releaseRow(cut)
        return {"result": scip_result}, cut_added

    def is_valid_inequality(self, x_coef, y_coef, cutrhs):
        """
        Check if the single cut is cutting off the optimal solution found without cycle inequalities
        :param x_coef:
        :param y_coef:
        :param cutrhs:
        :return:
        """
        cutlhs = 0
        for i, c_i in x_coef.items():
            cutlhs += c_i * self.x_opt[i]
        for ij, c_ij in y_coef.items():
            cutlhs += c_ij * self.y_opt[ij]
        cutoff = cutlhs > cutrhs

        if cutoff:
            print('found invalid inequality')
            self.debug_invalid_cut_stats['violation'] = cutlhs - cutrhs
            self.debug_invalid_cut_stats['dualbound'] = self.model.getDualbound()
            self.debug_invalid_cut_stats['primalbound'] = self.model.getPrimalbound()
            self.debug_invalid_cut_stats['lp_round'] = self.model.getNLPs()
            self.debug_invalid_cut_stats['ncuts'] = self.model.getNCuts() - self._cuts_probing
            self.debug_invalid_cut_stats['ncuts_applied'] = self.model.getNCutsApplied() - self._cuts_applied_probing
            self.debug_invalid_cut_stats['solving_time'] = self.model.getSolvingTime()
            self.debug_invalid_cut_stats['processed_nodes'] = self.model.getNNodes()
            self.debug_invalid_cut_stats['gap'] = self.model.getGap()
            self.debug_invalid_cut_stats['lp_rounds'] = self.model.getNLPs() - self._lp_rounds_probing
            self.debug_invalid_cut_stats['lp_iterations'] = self.model.getNLPIterations() - self._lp_iterations_probing
            self.debug_invalid_cut_stats['ncuts_applied_before_cutoff'] = self.ncuts_applied_at_cur_node
        return not cutoff

    def catch_cutoff(self):
        cur_dualbound = self.model.getDualbound()
        if cur_dualbound < self.opt_obj:
            print('Catched cutoff event')
            self.cutoff_occured = True
            self.debug_cutoff_stats['violation'] = cur_dualbound - self.opt_obj
            self.debug_cutoff_stats['dualbound'] = self.model.getDualbound()
            self.debug_cutoff_stats['primalbound'] = self.model.getPrimalbound()
            self.debug_cutoff_stats['lp_round'] = self.model.getNLPs()
            self.debug_cutoff_stats['ncuts'] = self.model.getNCuts() - self._cuts_probing
            self.debug_cutoff_stats['ncuts_applied'] = self.model.getNCutsApplied() - self._cuts_applied_probing
            self.debug_cutoff_stats['solving_time'] = self.model.getSolvingTime()
            self.debug_cutoff_stats['processed_nodes'] = self.model.getNNodes()
            self.debug_cutoff_stats['gap'] = self.model.getGap()
            self.debug_cutoff_stats['lp_rounds'] = self.model.getNLPs() - self._lp_rounds_probing
            self.debug_cutoff_stats['lp_iterations'] = self.model.getNLPIterations() - self._lp_iterations_probing
            self.debug_cutoff_stats['ncuts_applied_when_cutoff'] = self.ncuts_applied_at_cur_node

    def is_chordless(self, closed_walk):
        """
        Check if any non-adjacent pair of nodes in the cycle are connected directly in the graph.
        :param closed_walk: a list of edges ((from, _), (to, _)), forming a simple cycle.
        :return: True if chordless, otherwise False.
        """
        subset = [i for i, _ in closed_walk[:-1]]
        subgraph = nx.subgraph(self.G, subset)
        return nx.is_chordal(subgraph)

    def is_simple_cycle(self, closed_walk):
        path = [i for i, _ in closed_walk[:-1]]
        # subgraph = nx.subgraph(self.G, path)
        # return nx.is_simple_path()
        # return True iff each node in the path appears exactly once.
        # equivalent to iff each node in closed_walk is visited exactly once.
        return len(set(path)) == len(path)

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


class CSBaselineSepa(Sepa):
    def __init__(self, hparams={}):
        """
        Applies criterion to cut selection, supporting:
        default (do nothing), k_random, k_most_violated, all_cuts.
        Sample scip.Model state every time self.sepaexeclp is invoked.
        """
        super(CSBaselineSepa, self).__init__()
        self.name = 'Baseline Separator'
        self.hparams = hparams
        self.policy = hparams.get('policy', 'default')
        self.add_k = 0
        if self.policy not in ['default', 'all_cuts', 'tuned', 'adaptive']:
            self.add_k = int(self.policy.split('_')[0])
            assert self.policy.endswith('random') or self.policy.endswith('most_violated')
        self.lp_round_idx = 0
        # set default params for using after the adapted params
        self.default_separating_params = hparams.get('default_separating_params', {'objparalfac': 0.1,
                                                                                   'dircutoffdistfac': 0.5,
                                                                                   'efficacyfac': 1.0,
                                                                                   'intsupportfac': 0.1,
                                                                                   'maxcutsroot': 2000,
                                                                                   'minorthoroot': 0.9})
        # instance specific data needed to be reset every episode
        # todo unifiy x and y to x only (common for all combinatorial problems)
        self.G = None
        self.x = None
        self.stats = {
            'ncuts': [],
            'ncuts_applied': [],
            'solving_time': [],
            'processed_nodes': [],
            'gap': [],
            'lp_rounds': [],
            'lp_iterations': [],
            'dualbound': [],
        }
        if hparams.get('cut_stats', False):
            self.stats.update({
                'selected_minortho_avg': [],
                'selected_minortho_std': [],
                'selected_efficacy_avg': [],
                'selected_efficacy_std': [],
                'selected_dircutoffdist_avg': [],
                'selected_dircutoffdist_std': [],
                'selected_objparal_avg': [],
                'selected_objparal_std': [],
                'selected_intsupport_avg': [],
                'selected_intsupport_std': [],
                'discarded_minortho_avg': [],
                'discarded_minortho_std': [],
                'discarded_efficacy_avg': [],
                'discarded_efficacy_std': [],
                'discarded_dircutoffdist_avg': [],
                'discarded_dircutoffdist_std': [],
                'discarded_objparal_avg': [],
                'discarded_objparal_std': [],
                'discarded_intsupport_avg': [],
                'discarded_intsupport_std': [],
                'scip_selected_cuts': [],
                'policy_selected_cuts': [],
            })
            self.states_and_cuts = []
            self.prev_cuts = None
            self.prev_state = None
        self.stats_updated = False
        self.node_limit_reached = False
        self.terminal_state = False
        self.lp_iterations_limit = hparams.get('lp_iterations_limit', -1)
        self.terminal_state = False
        self.node_limit_reached = False
        self.print_prefix = '[Baseline Sepa]'
        self.prev_ncuts = 0

    # done
    def sepaexeclp(self):
        # finish with the previous step:
        self.update_stats()
        if self.hparams.get('cut_stats', False):
            self.update_cut_stats()
        # if for some reason we terminated the episode (lp iterations limit reached / empty action etc.
        # we dont want to run any further dqn steps, and therefore we return immediately.
        if self.terminal_state:
            # discard all the cuts in the separation storage and return
            self.model.clearCuts()
            # self.model.setIntParam('separating/maxcuts', 0)
            # self.model.setIntParam('separating/maxcutsroot', 0)
            self.model.interruptSolve()
            result = {"result": SCIP_RESULT.DIDNOTFIND}

        elif self.lp_iterations_limit == -1 or self.model.getNLPIterations() < self.lp_iterations_limit:
            result = self.separate_baseline()

        else:
            # stop optimization (implicitly), and don't add any more cuts
            if self.hparams.get('verbose', 0) == 2:
                self.print('LP_ITERATIONS_LIMIT reached. DIDNOTRUN!')
            self.terminal_state = 'LP_ITERATIONS_LIMIT_REACHED'
            # clear cuts and terminate
            self.model.clearCuts()
            # self.model.setIntParam('separating/maxcuts', 0)
            # self.model.setIntParam('separating/maxcutsroot', 0)
            self.model.interruptSolve()
            result = {"result": SCIP_RESULT.DIDNOTFIND}

        return result

    # done
    def separate_baseline(self):
        """
        Baselines:
        default - do nothing
        10_random - force 10 random cuts and discard the rest
        10_most_violated - force the 10 most efficacious cuts and discard the rest
        """
        # get the current state, a dictionary of available cuts (keyed by their names,
        # and query statistics related to the previous action (cut activeness etc.)
        cur_state, available_cuts = self.model.getState(state_format='tensor', get_available_cuts=True)
        # self.print(f'debug - model.getNCcuts={self.model.getNCuts()}, ncuts={available_cuts["ncuts"]}, NLP{self.model.getNLPs()}')
        # if there are available cuts, select action and continue to the next state
        # print(list(available_cuts['cuts'].keys()))
        assert self.model.getNCuts() == len(available_cuts['cuts']), "cuts duplicated with same name"
        # print('n pool cuts ', self.model.getNPoolCuts())
        if available_cuts['ncuts'] > 0:
            if self.hparams.get('cut_stats', False):
                self.stats['scip_selected_cuts'].append(self.prob_scip_cut_selection())

            # prob what scip cut selection algorithm would do in this state
            if self.policy == 'default':
                # use SCIP's cut selection (don't do anything)
                result = {"result": SCIP_RESULT.DIDNOTRUN}

            elif self.policy == 'all_cuts':
                selected = np.ones((available_cuts['ncuts'],))
                self.model.forceCuts(selected)
                # set SCIP maxcutsroot and maxcuts to the number of selected cuts,
                # in order to prevent it from adding more or less cuts
                self.model.setIntParam('separating/maxcuts', int(sum(selected)))
                self.model.setIntParam('separating/maxcutsroot', int(sum(selected)))
                # continue to the next state
                result = {"result": SCIP_RESULT.SEPARATED}

            elif self.policy.endswith('random'):
                # apply the action
                random_idxes = torch.randperm(available_cuts['ncuts'])[:self.add_k].numpy()
                selected = np.zeros((available_cuts['ncuts'],))
                selected[random_idxes] = 1
                # force SCIP to take the selected cuts and discard the others
                self.model.forceCuts(selected)
                # set SCIP maxcutsroot and maxcuts to the number of selected cuts,
                # in order to prevent it from adding more or less cuts
                self.model.setIntParam('separating/maxcuts', int(sum(selected)))
                self.model.setIntParam('separating/maxcutsroot', int(sum(selected)))
                # continue to the next state
                result = {"result": SCIP_RESULT.SEPARATED}

            elif self.policy.endswith('most_violated'):
                # available_cuts['selected_by_scip'] = np.array(
                #     [cut_name in cut_names_selected_by_scip for cut_name in available_cuts['cuts'].keys()])
                info = self.model.getState(state_format='dict')
                efficacy = info['cut']['efficacy']
                most_efficacious = list(reversed(np.argsort(efficacy)))[:self.add_k]
                selected = np.zeros((available_cuts['ncuts'],))
                selected[most_efficacious] = 1
                # force SCIP to take the selected cuts and discard the others
                self.model.forceCuts(selected)
                # set SCIP maxcutsroot and maxcuts to the number of selected cuts,
                # in order to prevent it from adding more or less cuts
                self.model.setIntParam('separating/maxcuts', int(sum(selected)))
                self.model.setIntParam('separating/maxcutsroot', int(sum(selected)))
                result = {"result": SCIP_RESULT.SEPARATED}

            elif self.policy == 'tuned':
                # reset separating parameters
                self.model.setRealParam('separating/objparalfac', self.hparams['objparalfac'])
                self.model.setRealParam('separating/dircutoffdistfac', self.hparams['dircutoffdistfac'])
                self.model.setRealParam('separating/efficacyfac', self.hparams['efficacyfac'])
                self.model.setRealParam('separating/intsupportfac', self.hparams['intsupportfac'])
                self.model.setIntParam('separating/maxcutsroot', self.hparams['maxcutsroot'])
                self.model.setRealParam('separating/minorthoroot', self.hparams['minorthoroot'])
                result = {"result": SCIP_RESULT.DIDNOTRUN}

            elif self.policy == 'adaptive':
                # reset separating params according to lp_round.
                # set defaults if no adapted params exist
                self.model.setRealParam('separating/objparalfac', self.hparams['objparalfac'].get(self.lp_round_idx, self.default_separating_params['objparalfac']))
                self.model.setRealParam('separating/dircutoffdistfac', self.hparams['dircutoffdistfac'].get(self.lp_round_idx, self.default_separating_params['dircutoffdistfac']))
                self.model.setRealParam('separating/efficacyfac', self.hparams['efficacyfac'].get(self.lp_round_idx, self.default_separating_params['efficacyfac']))
                self.model.setRealParam('separating/intsupportfac', self.hparams['intsupportfac'].get(self.lp_round_idx, self.default_separating_params['intsupportfac']))
                self.model.setIntParam('separating/maxcutsroot', self.hparams['maxcutsroot'].get(self.lp_round_idx, self.default_separating_params['maxcutsroot']))
                self.model.setRealParam('separating/minorthoroot', self.hparams['minorthoroot'].get(self.lp_round_idx, self.default_separating_params['minorthoroot']))
                self.lp_round_idx += 1
                result = {"result": SCIP_RESULT.DIDNOTRUN}

            self.stats_updated = False  # mark false to record relevant stats after this action will make effect
            self.prev_ncuts = available_cuts['ncuts']

        elif available_cuts['ncuts'] == 0:
            result = {"result": SCIP_RESULT.DIDNOTFIND}

        return result

    def assert_behavior(self):
        if self.policy != 'default':
            assert self.model.getParam('separating/maxcuts') == self.hparams.get('reset_maxcuts', 100)
            assert self.model.getParam('separating/maxcutsroot') == self.hparams.get('reset_maxcutsroot', 2000)

    def prob_scip_cut_selection(self):
        available_cuts = self.model.getCuts()
        lp_iter = self.model.getNLPIterations()
        self.model.startProbing()
        for cut in available_cuts:
            self.model.addCut(cut)
        self.model.applyCutsProbing()
        cut_names = self.model.getSelectedCutsNames()
        self.model.endProbing()
        if self.model.getNLPIterations() != lp_iter:
            # todo - investigate why with scip_seed = 562696653 probing increments lp_iter by one.
            #  it seems not to make any damage, however.
            print('Warning! SCIP probing mode changed num lp iterations.')
        # assert self.model.getNLPIterations() == lp_iter
        return cut_names

    # done
    def update_stats(self):
        """ Collect statistics related to the action taken at the previous round.
        This function is assumed to be called in the consequent separation round
        after the action was taken.
        A corner case is when choosing "EMPTY_ACTION" (shouldn't happen if we force selecting at least one cut)
        then the function is called immediately, and we need to add 1 to the number of lp_rounds.
        """
        if not self.stats_updated:  # or self.prev_action is None:   <- todo: this was a bug. missed the initial stats
            # todo verify recording initial state stats before taking any action
            self.stats['ncuts'].append(self.prev_ncuts)
            self.stats['ncuts_applied'].append(self.model.getNCutsApplied())
            self.stats['solving_time'].append(self.model.getSolvingTime())
            self.stats['processed_nodes'].append(self.model.getNNodes())
            self.stats['gap'].append(self.model.getGap())
            self.stats['lp_iterations'].append(self.model.getNLPIterations())
            self.stats['dualbound'].append(self.model.getDualbound())
            # todo - we always store the stats referring to the previous lp round action, so we need to subtract 1 from the
            #  the current LP round counter
            if self.terminal_state and self.terminal_state == 'EMPTY_ACTION':
                self.stats['lp_rounds'].append(self.model.getNLPs() + 1)  # todo - check if needed to add 1 when EMPTY_ACTION
            else:
                self.stats['lp_rounds'].append(self.model.getNLPs())
            self.truncate_to_lp_iterations_limit()
            self.stats_updated = True
            if self.hparams.get('cut_stats', False):
                self.stats['policy_selected_cuts'].append(self.model.getSelectedCutsNames())

    def update_cut_stats(self):
        state, cuts = self.model.getState(state_format='dict', get_available_cuts=True, query=self.prev_cuts)

        # compute stats related to the selected/discarded cuts at the previous round
        if self.prev_cuts is not None and self.prev_cuts['ncuts'] > 0:
            selected = self.prev_cuts['applied'].astype(np.bool)
            discarded = np.logical_not(selected)
            cut_efficacy = self.prev_state['cut']['efficacy']
            cut_dircutoffdist = self.prev_state['cut']['dircutoffdist']
            cut_objparal = self.prev_state['cut']['objparal']
            cut_intsupport = self.prev_state['cut']['intsupport']
            cuts_orthogonality = self.prev_state['cuts_orthogonality']
            cuts_orthogonality += np.eye(len(selected))
            # compute orthogonality w.r.t the selected group
            cut_minortho_wrt_selected = np.array([cuts_orthogonality[idx, selected].min() for idx in range(len(selected))])
            self.stats['selected_minortho_avg'].append(cut_minortho_wrt_selected[selected].mean())
            self.stats['selected_minortho_std'].append(cut_minortho_wrt_selected[selected].std())
            self.stats['selected_efficacy_avg'].append(cut_efficacy[selected].mean())
            self.stats['selected_efficacy_std'].append(cut_efficacy[selected].std())
            self.stats['selected_dircutoffdist_avg'].append(cut_dircutoffdist[selected].mean())
            self.stats['selected_dircutoffdist_std'].append(cut_dircutoffdist[selected].std())
            self.stats['selected_objparal_avg'].append(cut_objparal[selected].mean())
            self.stats['selected_objparal_std'].append(cut_objparal[selected].std())
            self.stats['selected_intsupport_avg'].append(cut_intsupport[selected].mean())
            self.stats['selected_intsupport_std'].append(cut_intsupport[selected].std())
            self.stats['discarded_minortho_avg'].append(cut_minortho_wrt_selected[discarded].mean() if any(discarded) else None)
            self.stats['discarded_minortho_std'].append(cut_minortho_wrt_selected[discarded].std() if any(discarded) else None)
            self.stats['discarded_efficacy_avg'].append(cut_efficacy[discarded].mean() if any(discarded) else None)
            self.stats['discarded_efficacy_std'].append(cut_efficacy[discarded].std() if any(discarded) else None)
            self.stats['discarded_dircutoffdist_avg'].append(cut_dircutoffdist[discarded].mean() if any(discarded) else None)
            self.stats['discarded_dircutoffdist_std'].append(cut_dircutoffdist[discarded].std() if any(discarded) else None)
            self.stats['discarded_objparal_avg'].append(cut_objparal[discarded].mean() if any(discarded) else None)
            self.stats['discarded_objparal_std'].append(cut_objparal[discarded].std() if any(discarded) else None)
            self.stats['discarded_intsupport_avg'].append(cut_intsupport[discarded].mean() if any(discarded) else None)
            self.stats['discarded_intsupport_std'].append(cut_intsupport[discarded].std() if any(discarded) else None)

        # ignore zero ncuts cases.
        if cuts['ncuts'] > 0:
            # append to history only if LP round was executed
            self.states_and_cuts.append((state, cuts))
            # assert len(self.states_and_cuts) == self.model.getNLPs()
        self.prev_cuts = cuts
        self.prev_state = state

    def truncate_to_lp_iterations_limit(self):
        # enforce the lp_iterations_limit on the last two records
        lp_iterations_limit = self.lp_iterations_limit
        if lp_iterations_limit > 0 and self.stats['lp_iterations'][-1] > lp_iterations_limit:
            # interpolate the dualbound and gap at the limit
            assert self.stats['lp_iterations'][-2] < lp_iterations_limit
            t = self.stats['lp_iterations'][-2:]
            for k in ['dualbound', 'gap']:
                ft = self.stats[k][-2:]
                # compute ft slope in the last interval [t[-2], t[-1]]
                slope = (ft[-1] - ft[-2]) / (t[-1] - t[-2])
                # compute the linear interpolation of ft at the limit
                interpolated_ft = ft[-2] + slope * (lp_iterations_limit - t[-2])
                self.stats[k][-1] = interpolated_ft
            # finally truncate the lp_iterations to the limit
            self.stats['lp_iterations'][-1] = lp_iterations_limit

    def print(self, expr):
        print(self.print_prefix, expr)


class CSResetSepa(Sepa):
    def __init__(self, hparams={'reset_maxcuts': 100, 'reset_maxcutsroot': 2000}):
        """
        Sample scip.Model state every time self.sepaexeclp is invoked.
        Store the generated data object in
        """
        super(CSResetSepa, self).__init__()
        self.name = 'Reset Separator'
        self.maxcuts = hparams['reset_maxcuts']
        self.maxcutsroot = hparams['reset_maxcutsroot']

    def sepaexeclp(self):
        # reset maxncuts and maxncutsroot for the next cut selection round
        self.model.setIntParam('separating/maxcuts', self.maxcuts)
        self.model.setIntParam('separating/maxcutsroot', self.maxcutsroot)
        return {"result": SCIP_RESULT.DIDNOTRUN}

    def print(self, expr):
        print(self.print_prefix, expr)


if __name__ == "__main__":
    # todo this step should be called from another file. e.g from scip_models
    port = 3000
    n = 50
    m = 10
    seed = 223

    import sys
    if '--mixed-debug' in sys.argv:
        import ptvsd

        # ptvsd.enable_attach(secret='my_secret', address =('127.0.0.1', port))
        ptvsd.enable_attach(address=('127.0.0.1', port))
        ptvsd.wait_for_attach()

    G = nx.barabasi_albert_graph(n, m, seed=seed)
    hparams = {'max_per_root': 200000,
               'max_per_round': -1,
               'criterion': 'random',
               'forcecut': False,
               'cuts_budget': 2000,
               'policy': 'adaptive',
               'policy_update_freq': 10,
               'starting_policies': [{'objparalfac': 0.1,
                                      'dircutoffdistfac': 0.5,
                                      'efficacyfac': 1,
                                      'intsupportfac': 0.1,
                                      'maxcutsroot': 1},
                                     {'objparalfac': 0.1,
                                      'dircutoffdistfac': 0.5,
                                      'efficacyfac': 1,
                                      'intsupportfac': 0.1,
                                      'maxcutsroot': 5},

                                     ]
               }
    nx.set_edge_attributes(G, {e: np.random.normal() for e in G.edges}, name='weight')
    model, x, ci_cut = maxcut_mccormic_model(G, use_general_cuts=False)
    # model.setRealParam('limits/time', 1000 * 1)
    """ Define a controller and appropriate callback to add user's cuts """
    # model.setRealParam('separating/objparalfac', 0.1)
    # model.setRealParam('separating/dircutoffdistfac', 0.5)
    # model.setRealParam('separating/efficacyfac', 1)
    # model.setRealParam('separating/intsupportfac', 0.1)
    # model.setIntParam('separating/maxrounds', -1)
    # model.setIntParam('separating/maxroundsroot', 10)
    # model.setIntParam('separating/maxcuts', 20)
    # model.setIntParam('separating/maxcutsroot', 100)
    model.setIntParam('separating/maxstallroundsroot', -1)
    # model.setIntParam('separating/maxroundsroot', 2100)
    model.setRealParam('limits/time', 300)
    model.setLongintParam('limits/nodes', 1)
    model.optimize()
    ci_cut.finish_experiment()
    stats = ci_cut.stats
    print("Solved using user's cutting-planes callback. Objective {}".format(model.getObjVal()))
    # TODO: avrech - find a more elegant way to retrive cycle_cuts_applied
    cuts, cuts_applied = get_separator_cuts_applied(model, 'MLCycles')
    # model.printStatistics()
    print('cycles added: ', cuts, ', cycles applied: ', cuts_applied)
    # print(ci_cut.stats)
    print('total cuts applied: ', model.getNCutsApplied())
    print('separation time frac: ', stats['cycles_sepa_time'][-1] / stats['solving_time'][-1])
    print('cuts applied vs time', stats['total_ncuts_applied'])
    print('finish')

