from heapq import heappop, heappush
from collections import defaultdict
import torch_geometric as tg
import torch
import numpy as np

def dijkstra(edges, s, t):
    """
    Find the shortest path from node s to t in graph G.
    :param edges: a list of tuples (i, j, w), where (i,j) is an undirected edge, and w is its weight
    :param s: source node
    :param t: target node
    :return: cost, path if any, otherwise Inf
    """
    g = defaultdict(list)
    for l, r, c in edges:
        g[l].append((c, r))

    q, seen, mins = [(0, s, [])], set(), {s: 0}
    while q:
        cost, v1, path = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            # path.append(v1)
            path = [v1] + path
            if v1 == t:
                return cost, path

            for c, v2 in g.get(v1, ()):
                if v2 in seen:
                    continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf"), []


def get_bipartite_graph(scip_state, scip_action=None):
    """
    Creates a torch_geometric.data.Data object from SCIP state
    produced by scip.Model.getState(format='tensor')
    :param scip_state: scip.getState(format='tensor')
    :param scip_action: numpy array of size ? containing 0-1.
    :return: torch_geometric.data.Data
    """
    C = torch.from_numpy(scip_state['C'])
    V = torch.from_numpy(scip_state['V'])
    A = torch.from_numpy(scip_state['A'])
    cut_parallelism = scip_state['cut_parallelism']
    nzrcoef = scip_state['nzrcoef']['vals']
    nzrrows = scip_state['nzrcoef']['rowidxs']
    nzrcols = scip_state['nzrcoef']['colidxs']
    ncons, cons_feats_dim = C.shape
    nvars, vars_feats_dim = V.shape
    ncuts, cuts_feats_dim = A.shape
    cuts_nzrcoef = scip_state['cut_nzrcoef']['vals']
    cuts_nzrrows = scip_state['cut_nzrcoef']['rowidxs']
    cuts_nzrcols = scip_state['cut_nzrcoef']['colidxs']
    stats = torch.tensor([v for v in scip_state['stats'].values()], dtype=torch.float32)
    if scip_action is not None:
        assert len(scip_action) == ncuts

    # Combine the constraint, variable and cut nodes into a single graph:
    # Hold a combined features set, x, composed of C, V and A.
    # Pad with zeros if not(cons_feats_dim == vars_feats_dim == cuts_feats_dim),
    # and store the appropriate feats_dim for each node.
    # The edge attributes will be the nzrcoef of the constraint/cut.
    # Edges are directed, to be able to distinguish between C and A to V.
    # In this way the data object is standard torch geometric object, so we can use
    # all the torch_geometric utilities.

    # Constraint nodes are mapped to the nodes 0:(ncons-1),
    # variable nodes are mapped to the nodes ncons:ncons+nvars-1, and
    # cut nodes are mapped to the nodes ncons+nvars:ncons+nvars+ncuts-1

    # edge_index:
    # shift nzrcols by ncons (because the indices of the vars are now shifted by ncons)
    # and build the directed edge_index of the graph representation
    lp_edge_index = np.vstack([nzrrows, nzrcols+ncons])
    cuts_edge_index = np.vstack([cuts_nzrrows+ncons+nvars, cuts_nzrcols+ncons])
    edge_index = torch.from_numpy(np.hstack([lp_edge_index, cuts_edge_index]))
    edge_attributes = torch.from_numpy(np.concatenate([nzrcoef, cuts_nzrcoef]))

    # Build the features tensor x:
    # if the variable and constraint features dimensionality is not equal,
    # pad with zeros to the maximal length
    max_dim = max([cons_feats_dim, vars_feats_dim, cuts_feats_dim])
    if cons_feats_dim < max_dim:
        C = torch.constant_pad_nd(C, [0, max_dim - cons_feats_dim], value=0)
    if vars_feats_dim < max_dim:
        V = torch.constant_pad_nd(V, [0, max_dim - vars_feats_dim], value=0)
    if cuts_feats_dim < max_dim:
        A = torch.constant_pad_nd(A, [0, max_dim - cuts_feats_dim], value=0)

    x = torch.cat([C, V, A], dim=0)

    # for imitation learning, store the target in y
    if scip_action is not None:
        y = torch.zeros(size=(x.shape[0],))
        y[-ncuts:] = torch.tensor(list(scip_action.values()))
    else:
        y = None
    # create the data object
    data = tg.data.Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attributes,
                        y=y,
                        cons_feats_dim=cons_feats_dim,
                        vars_feats_dim=vars_feats_dim,
                        ncons=ncons,
                        nvars=nvars,
                        stats=stats)
    return data
