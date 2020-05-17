import networkx as nx
from collections import OrderedDict
import os
try:
    import gurobipy as grb
    from gurobipy import GRB
except:
    import sys
    print('Adding Gurobi to sys.path')
    sys.path.insert(0, '/home/avrech/gurobi900/linux64/lib/')
    import gurobipy as grb
    from gurobipy import GRB


def maxcut_mccormic_model(G):
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
    :return: gurobipy.Model
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

    # create GUROBI model:
    model = grb.Model()
    model.setAttr(GRB.Attr.ModelSense, GRB.MAXIMIZE)
    x = model.addVars(list(x_coef.keys()), name='x', obj=x_coef, vtype=GRB.BINARY)
    y = model.addVars(list(y_coef.keys()), name='y', obj=y_coef, vtype=GRB.BINARY)
    # x = OrderedDict([(i, primal_model.addVar(name='{}'.format(i), obj=x_coef[i], vtype='B')) for i in V])
    # y = OrderedDict([(ij, primal_model.addVar(name='{}'.format(ij), obj=y_coef[ij], vtype='C')) for ij in E])

    """ McCormic Inequalities """
    for ij in E:
        i, j = ij
        model.addConstr(x[i] + x[j] -y[ij] <= 1, name='{}'.format(ij))
        model.addConstr(y[ij] <= x[i], name='{}'.format((ij, i)))
        model.addConstr(y[ij] <= x[j], name='{}'.format((ij, j)))

    return model, x, y

