from heapq import heappop, heappush
from collections import defaultdict
import networkx as nx
import numpy as np


def dijkstra(edges, s, t):
    """
    Find the shortest path from node s to t in graph G.
    :param edges: a list of tuples (i, j, w), where (i,j) is an undirected edge, and w is its weight
    :param s: source node
    :param t: target node
    :return: cost, path if any, otherwise Inf
    """
    # adjacency dictionary,
    # for each node l, store a list of its neighbors (r) and their distances (c) from l
    g = defaultdict(list)
    for l, r, c in edges:
        g[l].append((c, r))

    q = [(0, s, [])]  # priority queue, prioritizing according to distances from s
    visited = set()      # visited nodes
    mindist = {s: 0}     # min distances from s

    while q:
        cost, v1, path = heappop(q)
        if v1 not in visited:
            visited.add(v1)
            # path.append(v1)
            path = [v1] + path
            if v1 == t:
                return cost, path

            for c, v2 in g.get(v1, ()):
                if v2 in visited:
                    continue
                prev = mindist.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mindist[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf"), []


def dijkstra_best_shortest_path(edges, s, t):
    """
    Find the shortest path from node s to t in graph G.
    :param edges: a list of tuples (i, j, w), where (i,j) is an undirected edge, and w is its weight
    :param s: source node
    :param t: target node
    :return: cost, path if any, otherwise Inf
    """
    # adjacency dictionary,
    # for each node l, store a list of its neighbors (r) and their distances (c) from l
    g = defaultdict(list)
    for l, r, c in edges:
        g[l].append((c, r))

    # priority queue:
    # structure: (cost, pathlen, node, path)
    # where: cost - the sum of edge weights of the path from s
    #        pathlen - path length (number of edges from s)
    #        node - node identifier (could be a tuple)
    #        path - a list of the nodes on the path from s to node
    # the heap "should" sort the elements in q according to the tuple elements.
    # so first will come the node with the smaller cost,
    # an if there are two nodes with the same cost, we will prefer the closest one
    # in terms of path length.
    q = [(0, 0, s, [])]
    visited = set()      # visited nodes
    mincosts = {s: 0}     # min distances from s
    pathlens = {s: 0}
    while q:
        v1_cost, v1_pathlen, v1, path = heappop(q)
        if v1 not in visited:
            visited.add(v1)
            # path.append(v1)
            path = [v1] + path
            if v1 == t:
                return v1_cost, path

            # relax the costs of v1 neighbors
            for c, v2 in g.get(v1, ()):
                if v2 in visited:
                    continue
                v2_cur_cost = mincosts.get(v2, None)
                v2_new_cost = v1_cost + c
                v2_cur_pathlen = pathlens.get(v2, None)
                v2_new_pathlen = v1_pathlen + 1
                # if the path to v2 via v1 is cheaper,
                # or even if it is equal cost, but shorter in terms of pathlen,
                # then update v2
                if v2_cur_cost is None or v2_new_cost < v2_cur_cost or (v2_new_cost == v2_cur_cost and v2_new_pathlen < v2_cur_pathlen):
                    mincosts[v2] = v2_new_cost
                    pathlens[v2] = v2_new_pathlen
                    heappush(q, (v2_new_cost, v2_new_pathlen, v2, path))

    return float("inf"), []


def verify_maxcut_sol(model, x, G):
    edge_weights = nx.get_edge_attributes(G, 'weight')
    sol = {i: model.getVal(x[i]) for i in x.keys()}
    for v in sol.values():
        assert v == 0 or v == 1
    cut = 0
    for i, j in G.edges:
        if sol[i] != sol[j]:
            cut += edge_weights[(i, j)]
    return cut


def get_normalized_areas(t, ft, t_support=None, reference=0):
    """
    Compute the area under f(t) vs. t on t_support
    :param t: lp_iterations
    :param ft: dualbound (or gap)
    :param t_support:
    :param reference: optimal dualbound (or 0 for gap integral)
    :return: array of length = len(t) -1 , containing the area under the normalized curve
    for each interval in t,
    using 1st order interpolation to approximate ft between each adjacent points in t.
    """

    # if t[-1] < t_support, extend t to t_support
    # and extend ft with a constant value ft[-1]
    extended = False
    t_support = t[-1] if t_support is None else t_support

    if t[-1] < t_support:
        ft = ft + [ft[-1]]
        t = t + [t_support]
        extended = True
    ft = np.array(ft)
    t = np.array(t)

    # normalize ft to [0,1] according to the reference value,
    # such that it will start from 1 and end at zero (if optimal).
    # if ft is increasing function, we flip it to be decreasing.
    ft = np.abs(ft - reference) / np.abs(ft[0])

    # normalize t to [0,1]
    t = t / t_support

    # compute the area under the curve using first order interpolation
    ft_diff = ft[1:] - ft[:-1]
    t_diff = t[1:] - t[:-1]
    ft_areas = t_diff * (ft[:-1] + ft_diff / 2)
    if extended:
        # add the extension area to the last transition area
        ft_areas[-2] += ft_areas[-1]
        # truncate the extension, and leave n-areas for the n-transition done
        ft_areas = ft_areas[:-1]

    return ft_areas
