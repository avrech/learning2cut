import torch_geometric as tg
import torch
import numpy as np
from torch_geometric.data.data import Data
from torch_geometric.utils import dense_to_sparse


class PairTripartiteAndCliqueData(Data):
    """
    Hold two graphs in one data object as described in
    https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    The first graph is a TripartiteGraph, in which the left nodes are
    the constraint nodes,
    in the middle are the variable nodes, and on the right there are the cut nodes.
    The edge attributes will be the LP coefficients. This tripartite graph
    represents the LP system together with the candidate cuts.
    The second graph in the "pair-data" is a clique graph consists of the cut nodes,
    with the intra-cuts orthogonality as edge attributes. This graph will use
    after computing cuts embedding to factorize the cuts joint classification.

    In this way we can do mini-batch SGD easily with torch_geometric DataLoader.
    """
    def __init__(self,
                 x_c=None,  # tripartite graph features       - LP state
                 x_v=None,
                 x_a=None,
                 edge_index_c2v=None,  # bipartite graph edge_index     - LP nonzero indices
                 edge_attr_c2v=None,  # bipartite graph edge weights   - LP nonzero coefficients
                 edge_index_a2v=None,
                 edge_attr_a2v=None,
                 edge_index_a2a=None,  # cuts clique graph edge index   - fully connected
                 edge_attr_a2a=None,  # cuts clique graph edge weights - orthogonality between each two cuts
                 y=None,  # cuts clique graph node labels  - the action whether to apply the cut or not.
                 # meta-data needed for processing the bipartite graph
                 n_c_nodes=None,
                 n_v_nodes=None,
                 n_a_nodes=None,
                 n_c2v_edges=None,
                 n_a2v_edges=None,
                 n_a2a_edges=None,
                 stats=None
                 ):
        super(PairTripartiteAndCliqueData, self).__init__()
        self.x_c = x_c
        self.x_v = x_v
        self.x_a = x_a
        self.edge_index_c2v = edge_index_c2v
        self.edge_attr_c2v = edge_attr_c2v
        self.edge_index_a2v = edge_index_a2v
        self.edge_attr_a2v = edge_attr_a2v
        self.edge_index_a2a = edge_index_a2a
        self.edge_attr_a2a = edge_attr_a2a
        self.y = y
        self.n_c_nodes = n_c_nodes
        self.n_v_nodes = n_v_nodes
        self.n_a_nodes = n_a_nodes
        self.n_c2v_edges = n_c2v_edges
        self.n_a2v_edges = n_a2v_edges
        self.n_a2a_edges = n_a2a_edges
        self.stats = stats

    def __inc__(self, key, value):
        if key == 'edge_index_c2v':
            return torch.tensor([[self.x_c.size(0)], [self.x_v.size(0)]])
        if key == 'edge_index_a2v':
            return torch.tensor([[self.x_a.size(0)], [self.x_v.size(0)]])
        if key == 'edge_index_a2a':
            return self.x_a.size(0)
        else:
            return super(PairTripartiteAndCliqueData, self).__inc__(key, value)


def get_gnn_data(scip_state, scip_action=None):
    """
    Creates a torch_geometric.data.Data object from SCIP state
    produced by scip.Model.getState(format='tensor')
    :param scip_state: scip.getState(format='tensor')
    :param scip_action: numpy array of size ? containing 0-1.
    :return: torch_geometric.data.Data
    """
    x_c = torch.from_numpy(scip_state['C'])
    x_v = torch.from_numpy(scip_state['V'])
    x_a = torch.from_numpy(scip_state['A'])
    nzrcoef = scip_state['nzrcoef']['vals']
    nzrrows = scip_state['nzrcoef']['rowidxs']
    nzrcols = scip_state['nzrcoef']['colidxs']
    n_c_nodes, c_feats_dim = x_c.shape
    n_v_nodes, v_feats_dim = x_v.shape
    n_a_nodes, c_feats_dim = x_a.shape
    cuts_nzrcoef = scip_state['cut_nzrcoef']['vals']
    cuts_nzrrows = scip_state['cut_nzrcoef']['rowidxs']
    cuts_nzrcols = scip_state['cut_nzrcoef']['colidxs']
    cuts_orthogonality = scip_state['cuts_orthogonality']
    stats = torch.tensor([v for v in scip_state['stats'].values()], dtype=torch.float32).view(1, -1)
    if scip_action is not None:
        assert len(scip_action) == n_a_nodes

    # Combine the constraint, variable and cut nodes into a single graph:
    # The edge attributes will be the nzrcoef of the constraint/cut.
    # Edges are directed, to be able to distinguish between C and A to V.
    # In this way the data object is a proper torch_geometric Data object,
    # so we can use all torch_geometric utilities.

    # edge_index:
    edge_index_c2v = torch.from_numpy(np.vstack([nzrrows, nzrcols])).long()
    edge_index_a2v = torch.from_numpy(np.vstack([cuts_nzrrows, cuts_nzrcols])).long()
    edge_attr_c2v = torch.from_numpy(nzrcoef).unsqueeze(dim=1)
    edge_attr_a2v = torch.from_numpy(cuts_nzrcoef).unsqueeze(dim=1)
    n_c2v_edges = len(nzrcoef)
    n_a2v_edges = len(cuts_nzrcoef)

    # build the clique graph the candidate cuts:
    # x_t = torch.empty((ncuts, 1)) # only to reserve the keyword. will be overidden later by the cuts embeddings
    edge_index_a2a, edge_attr_a2a = dense_to_sparse(torch.from_numpy(cuts_orthogonality))
    edge_attr_a2a.unsqueeze_(dim=1)
    n_a2a_edges = edge_index_a2a.shape[1]

    # for imitation learning, store scip_action as cut labels in y,
    # for reinforcement learning store zeros
    if scip_action is not None:
        y = torch.tensor(list(scip_action.values()), dtype=torch.float32)
    else:
        y = torch.zeros(size=(n_a_nodes, ), dtype=torch.float32)

    # create the pair-tripartite-and-clique-data object consist of both the LP tripartite graph
    # and the cuts clique graph
    # NOTE: in order to process PairTripartiteAndCliqueData in mini batches,
    # follow the example in https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    # and do:
    # data_list = [data_1, data_2, ... data_n]
    # loader = DataLoader(data_list, batch_size=<whatever>, follow_batch=['x_c', 'x_v', 'x_a'])

    data = PairTripartiteAndCliqueData(
        x_c=x_c,
        x_v=x_v,
        x_a=x_a,
        edge_index_c2v=edge_index_c2v,
        edge_attr_c2v=edge_attr_c2v,
        edge_index_a2v=edge_index_a2v,
        edge_attr_a2v=edge_attr_a2v,
        edge_index_a2a=edge_index_a2a,
        edge_attr_a2a=edge_attr_a2a,
        y=y,
        n_c_nodes=n_c_nodes,
        n_v_nodes=n_v_nodes,
        n_a_nodes=n_a_nodes,
        n_c2v_edges=n_c2v_edges,
        n_a2v_edges=n_a2v_edges,
        n_a2a_edges=n_a2a_edges,
        stats=stats,
    )
    return data


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
