from heapq import heappop, heappush
from collections import defaultdict


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

