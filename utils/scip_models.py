import networkx as nx
from pyscipopt import quicksum
import pyscipopt as scip
from collections import OrderedDict


def maxcut_mccormic_model(G, model_name='MAXCUT McCormic Model',
                          use_presolve=True, use_heuristics=True, use_general_cuts=True, use_propagation=True):
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

    return model, x, y


def get_separator_cuts_applied(model, separator_name):
    # TODO: avrech - find a more elegant way to retrive cycle_cuts_applied
    cycle_cuts_added, cycle_cuts_applied = -2, -2
    try:
        tmpfile = 'tmp_stats.txt'
        model.writeStatistics(filename=tmpfile)
        with open(tmpfile, 'r') as f:
            for line in f.readlines():
                if separator_name in line:
                    cycle_cuts_applied = int(line.split()[-2])
                    cycle_cuts_added = int(line.split()[-3])
    except:
        cycle_cuts_added, cycle_cuts_applied = -1, -1
    return cycle_cuts_added, cycle_cuts_applied