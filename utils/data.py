import torch_geometric as tg
import torch
import numpy as np
from torch_geometric.data.data import Data
from torch_geometric.utils import dense_to_sparse

class PairData(Data):
    """
    Hold two graphs in one data object as described in
    https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    This will be used for storing the bipartite graph of each instance,
    besides the complete graph of each instance cuts.
    The bipartite graph edge_attr_s will be the constraints/cuts coefficients,
    and the edge_attr_t of the cuts complete graph will be the orthogonality
    between each pair of cuts.
    """
    def __init__(self,
                 x_s,             # bipartite graph features       - LP state
                 edge_index_s,    # bipartite graph edge_index     - LP nonzero indices
                 edge_attr_s,     # bipartite graph edge weights   - LP nonzero coefficients
                 x_t,             # cuts clique graph features     - empty tensor of size [ncuts, 1] reserved for the cuts embedding
                 edge_index_t,    # cuts clique graph edge index   - fully connected
                 edge_attr_t,     # cuts clique graph edge weights - orthogonality between each two cuts
                 y_t,             # cuts clique graph node labels  - the action whether to apply the cut or not.
                 # meta-data needed for processing the bipartite graph
                 cons_feats_dim,
                 vars_feats_dim,
                 cuts_feats_dim,
                 ncons,
                 nvars,
                 ncuts,
                 ncons_edges,
                 ncuts_edges,
                 cons_edges,
                 vars_nodes,
                 cuts_nodes,
                 stats
                 ):
        super(PairData, self).__init__()
        self.x_s = x_s
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.x_t = x_t
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.y_t = y_t
        self.cons_feats_dim = cons_feats_dim
        self.vars_feats_dim = vars_feats_dim
        self.cuts_feats_dim = cuts_feats_dim
        self.ncons = ncons
        self.nvars = nvars
        self.ncuts = ncuts
        self.ncons_edges = ncons_edges
        self.ncuts_edges = ncuts_edges
        self.cons_edges = cons_edges
        self.vars_nodes = vars_nodes
        self.cuts_nodes = cuts_nodes
        self.stats = stats

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super(PairData, self).__inc__(key, value)


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
    cuts_orthogonality = scip_state['cuts_orthogonality']
    stats = torch.tensor([v for v in scip_state['stats'].values()], dtype=torch.float32).view(1, -1)
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
    edge_index_s = torch.from_numpy(np.hstack([lp_edge_index, cuts_edge_index]))
    edge_attr_s = torch.from_numpy(np.concatenate([nzrcoef, cuts_nzrcoef]))
    ncons_edges = len(nzrcoef)
    ncuts_edges = len(cuts_nzrcoef)
    cons_edges = torch.ones_like(edge_attr_s, dtype=torch.int32)
    cons_edges[ncons_edges:] = 0
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

    x_s = torch.cat([C, V, A], dim=0)
    vars_nodes = torch.zeros(x_s.shape[0], dtype=torch.int32)
    vars_nodes[ncons:ncons+nvars] = 1
    cuts_nodes = torch.zeros(x_s.shape[0], dtype=torch.int32)
    cuts_nodes[ncons+nvars:] = 1

    # build the clique graph the candidate cuts:
    x_t = torch.empty((ncuts, 1)) # only to reserve the keyword. will be overidden later by the cuts embeddings
    edge_index_t, edge_attr_t = dense_to_sparse(torch.from_numpy(cuts_orthogonality))

    # for imitation learning, store scip_action as cut labels in y,
    # for reinforcement learning store zeros
    if scip_action is not None:
        y_t = torch.tensor(list(scip_action.values()), dtype=torch.int32)
    else:
        y_t = torch.zeros(size=(ncuts, ))

    # create a pair-data object consist of both the bipartite graph
    # and the cuts clique graph
    # NOTE: in order to process PairData in mini batches,
    # follow the example in https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    # and do:
    # data_list = [PairData_1, PairData_2, ... PairData_n]
    # loader = DataLoader(data_list, batch_size=2, follow_batch=['x_s', 'x_t'])
    pair_data = PairData(x_s=x_s,
                         edge_index_s=edge_index_s,
                         edge_attr_s=edge_attr_s,
                         x_t=x_t,
                         edge_index_t=edge_index_t,
                         edge_attr_t=edge_attr_t,
                         y_t=y_t,
                         cons_feats_dim=cons_feats_dim,
                         vars_feats_dim=vars_feats_dim,
                         cuts_feats_dim=cuts_feats_dim,
                         ncons=ncons,
                         nvars=nvars,
                         ncuts=ncuts,
                         ncons_edges=ncons_edges,
                         ncuts_edges=ncuts_edges,
                         cons_edges=cons_edges,
                         vars_nodes=vars_nodes,
                         cuts_nodes=cuts_nodes,
                         stats=stats)

    return pair_data

def get_data_memory(data, units='M'):
    """
    Computes the memory consumption of torch_geometric.data.Data object in MBytes
    Counts for all torch.Tensor elements in data: x, y, edge_index, edge_attr, stats, etc.
    :param data:
    :return:
    """
    membytes = 0
    for k in data.keys:
        v = data[k]
        if type(v) is torch.Tensor:
            membytes += v.element_size() * v.nelement()
    mem = {'B': membytes,
           'K': membytes / 2**10,
           'M': membytes / 2**20,
           'G': membytes / 2**30,
           }.get(units)
    return mem
