import torch
import numpy as np
from torch_geometric.data.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.utils import dense_to_sparse, add_remaining_self_loops, sort_edge_index
from collections import namedtuple
TransitionNumpyTuple = namedtuple(
    'TransitionTuple',
    (
        'x_c',
        'x_v',
        'x_a',
        'edge_index_c2v',
        'edge_attr_c2v',
        'edge_index_a2v',
        'edge_attr_a2v',
        'edge_index_a2a',
        'edge_attr_a2a',
        # 'edge_index_dec',
        # 'edge_attr_dec',
        'stats',
        'a',
        'r',
        'ns_x_c',
        'ns_x_v',
        'ns_x_a',
        'ns_edge_index_c2v',
        'ns_edge_attr_c2v',
        'ns_edge_index_a2v',
        'ns_edge_attr_a2v',
        'ns_edge_index_a2a',
        'ns_edge_attr_a2a',
        'ns_stats',
        'ns_terminal',
    )
)


class Transition(Data):
    """
    Hold (s,a,r,s') in one data object to make batching easier
    """
    def __init__(self,
                 # state
                 x_c=None,  # tripartite graph features       - LP state
                 x_v=None,
                 x_a=None,
                 edge_index_c2v=None,  # bipartite graph edge_index     - LP nonzero indices
                 edge_attr_c2v=None,  # bipartite graph edge weights   - LP nonzero coefficients
                 edge_index_a2v=None,
                 edge_attr_a2v=None,
                 edge_index_a2a=None,  # cuts clique graph edge index   - fully connected
                 edge_attr_a2a=None,  # cuts clique graph edge weights - orthogonality between each two cuts
                 # edge_index_dec=None, # transformer decoder context
                 # edge_attr_dec=None, # transformer decoder context
                 stats=None,
                 # action
                 a=None,  # cuts clique graph node labels  - the action whether to apply the cut or not.
                 # reward
                 r=None,
                 # next_state
                 ns_x_c=None,  # tripartite graph features       - LP state
                 ns_x_v=None,
                 ns_x_a=None,
                 ns_edge_index_c2v=None,  # bipartite graph edge_index     - LP nonzero indices
                 ns_edge_attr_c2v=None,  # bipartite graph edge weights   - LP nonzero coefficients
                 ns_edge_index_a2v=None,
                 ns_edge_attr_a2v=None,
                 ns_edge_index_a2a=None,  # cuts clique graph edge index   - fully connected
                 ns_edge_attr_a2a=None,  # cuts clique graph edge weights - orthogonality between each two cuts
                 ns_stats=None,
                 ns_terminal=None
                 ):
        super(Transition, self).__init__()
        self.x_c = x_c
        self.x_v = x_v
        self.x_a = x_a
        self.edge_index_c2v = edge_index_c2v
        self.edge_attr_c2v = edge_attr_c2v
        self.edge_index_a2v = edge_index_a2v
        self.edge_attr_a2v = edge_attr_a2v
        self.edge_index_a2a = edge_index_a2a
        self.edge_attr_a2a = edge_attr_a2a
        # self.edge_index_dec = edge_index_dec
        # self.edge_attr_dec = edge_attr_dec
        self.stats = stats
        self.a = a
        self.r = r
        self.ns_x_c = ns_x_c
        self.ns_x_v = ns_x_v
        self.ns_x_a = ns_x_a
        self.ns_edge_index_c2v = ns_edge_index_c2v
        self.ns_edge_attr_c2v = ns_edge_attr_c2v
        self.ns_edge_index_a2v = ns_edge_index_a2v
        self.ns_edge_attr_a2v = ns_edge_attr_a2v
        self.ns_edge_index_a2a = ns_edge_index_a2a
        self.ns_edge_attr_a2a = ns_edge_attr_a2a
        self.ns_stats = ns_stats
        self.ns_terminal = ns_terminal

    def __inc__(self, key, value):
        if key == 'edge_index_c2v':
            return torch.tensor([[self.x_c.size(0)], [self.x_v.size(0)]])
        if key == 'edge_index_a2v':
            return torch.tensor([[self.x_a.size(0)], [self.x_v.size(0)]])
        if key == 'edge_index_a2a':
            return self.x_a.size(0)
        # if key == 'edge_index_dec':
        #     return self.x_a.size(0)
        if key == 'ns_edge_index_c2v':
            return torch.tensor([[self.ns_x_c.size(0)], [self.ns_x_v.size(0)]])
        if key == 'ns_edge_index_a2v':
            return torch.tensor([[self.ns_x_a.size(0)], [self.ns_x_v.size(0)]])
        if key == 'ns_edge_index_a2a':
            return self.ns_x_a.size(0)
        else:
            return super(Transition, self).__inc__(key, value)

    def to_numpy_tuple(self):
        return tuple([self[k].numpy() for k in self.keys])

    @staticmethod
    def from_numpy_tuple(transition_numpy_tuple):
        return Transition(*[torch.from_numpy(np_array) for np_array in transition_numpy_tuple])

    @staticmethod
    def create(scip_state,
               action=None,
               transformer_decoder_context=None,
               reward=None,
               scip_next_state=None,
               tqnet_version='v3'):
        """
        Creates a torch_geometric.data.Data object from SCIP state
        produced by scip.Model.getState(state_format='tensor')
        :param scip_state: scip.getState(state_format='tensor')
        :param transformer_decoder_context: required only if using transformer
        :param action: np.ndarray
        :param reward: np.ndarray
        :param scip_next_state: scip.getState(state_format='tensor')
        :return: Transition
        """
        x_c = torch.from_numpy(scip_state['C'])
        x_v = torch.from_numpy(scip_state['V'])
        x_a = torch.from_numpy(scip_state['A'])
        nzrcoef = scip_state['nzrcoef']['vals']
        nzrrows = scip_state['nzrcoef']['rowidxs']
        nzrcols = scip_state['nzrcoef']['colidxs']
        cuts_nzrcoef = scip_state['cut_nzrcoef']['vals']
        cuts_nzrrows = scip_state['cut_nzrcoef']['rowidxs']
        cuts_nzrcols = scip_state['cut_nzrcoef']['colidxs']
        cuts_orthogonality = scip_state['cuts_orthogonality']
        stats = torch.tensor([v for v in scip_state['stats'].values()], dtype=torch.float32).view(1, -1)

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

        # build the clique graph of the candidate cuts:
        # if using transformer, take the decoder context and edge_index from the input, else generate empty one
        if transformer_decoder_context is not None:
            edge_index_a2a, edge_attr_a2a = transformer_decoder_context
        else:
            # create basic edge_attr_a2a
            if x_a.shape[0] > 1:
                # we add 1 to cuts_orthogonality to ensure that all edges are created
                # including the self edges
                edge_index_a2a, edge_attr_a2a = dense_to_sparse(torch.from_numpy(cuts_orthogonality + 1))
                edge_attr_a2a -= 1  # subtract 1 to set back the correct orthogonality values
                edge_attr_a2a.unsqueeze_(dim=1)
            elif x_a.shape[0] == 1:
                edge_index_a2a = torch.tensor([[0], [0]], dtype=torch.long)  # single self loop
                edge_attr_a2a = torch.zeros(size=(1, 1), dtype=torch.float32)  # self orthogonality
            else:
                raise ValueError

            # attach initial transformer context if needed
            if tqnet_version == 'v1':
                # edge_attr_a2a = [o_ij, processed_i, selected_i]
                edge_attr_a2a = torch.cat([edge_attr_a2a, torch.zeros((edge_attr_a2a.shape[0], 2))], dim=-1)
            elif tqnet_version == 'v2':
                # edge_attr_a2a = [o_ij, selected_i]
                edge_attr_a2a = torch.cat([edge_attr_a2a, torch.zeros((edge_attr_a2a.shape[0], 1))], dim=-1)
            elif tqnet_version == 'v3':
                # edge_attr_a2a = [o_ij, o_iS, selected_i]
                # o_iS, the orthogonality to the selected group is initialized to 1 (the orthogonality to "nothing" is 1 by convenetion)
                edge_attr_a2a = torch.cat([edge_attr_a2a, torch.ones_like(edge_attr_a2a), torch.zeros_like(edge_attr_a2a)], dim=-1)
            elif tqnet_version == 'none':
                # don't do anything. transformer is not in use.
                pass
            else:
                raise ValueError

        if action is not None:
            a = torch.from_numpy(action).long()
            assert a.shape[0] == x_a.shape[0]  # n_a_nodes
        else:
            # don't care
            a = torch.zeros(size=(x_a.shape[0], ), dtype=torch.long)
        if reward is not None:
            r = torch.from_numpy(reward).float()
            assert r.shape[0] == x_a.shape[0]  # n_a_nodes
        else:
            # don't care
            r = torch.zeros(size=(x_a.shape[0], ), dtype=torch.float32)

        # process the next state:
        if scip_next_state is not None:
            # non terminal state
            ns_x_c = torch.from_numpy(scip_next_state['C'])
            ns_x_v = torch.from_numpy(scip_next_state['V'])
            ns_x_a = torch.from_numpy(scip_next_state['A'])
            ns_nzrcoef = scip_next_state['nzrcoef']['vals']
            ns_nzrrows = scip_next_state['nzrcoef']['rowidxs']
            ns_nzrcols = scip_next_state['nzrcoef']['colidxs']
            ns_cuts_nzrcoef = scip_next_state['cut_nzrcoef']['vals']
            ns_cuts_nzrrows = scip_next_state['cut_nzrcoef']['rowidxs']
            ns_cuts_nzrcols = scip_next_state['cut_nzrcoef']['colidxs']
            ns_cuts_orthogonality = scip_next_state['cuts_orthogonality']
            ns_stats = torch.tensor([v for v in scip_next_state['stats'].values()], dtype=torch.float32).view(1, -1)

            # edge_index:
            ns_edge_index_c2v = torch.from_numpy(np.vstack([ns_nzrrows, ns_nzrcols])).long()
            ns_edge_index_a2v = torch.from_numpy(np.vstack([ns_cuts_nzrrows, ns_cuts_nzrcols])).long()
            ns_edge_attr_c2v = torch.from_numpy(ns_nzrcoef).unsqueeze(dim=1)
            ns_edge_attr_a2v = torch.from_numpy(ns_cuts_nzrcoef).unsqueeze(dim=1)

            # build the clique graph of the candidate cuts,
            if ns_x_a.shape[0] > 1:
                ns_edge_index_a2a, ns_edge_attr_a2a = dense_to_sparse(torch.from_numpy(ns_cuts_orthogonality + 1))
                ns_edge_attr_a2a -= 1
                # ns_edge_index_a2a, ns_edge_attr_a2a = add_remaining_self_loops(ns_edge_index_a2a, edge_weight=ns_edge_attr_a2a,
                #                                                          fill_value=0)
                ns_edge_attr_a2a.unsqueeze_(dim=1)
            elif ns_x_a.shape[0] == 1:
                ns_edge_index_a2a = torch.tensor([[0], [0]], dtype=torch.long)  # single self loop
                ns_edge_attr_a2a = torch.zeros(size=(1, 1), dtype=torch.float32)  # self orthogonality
            else:
                raise ValueError
            # attach initial transformer context if needed
            if tqnet_version == 'v1':
                # edge_attr_a2a = [o_ij, processed_i, selected_i]
                ns_edge_attr_a2a = torch.cat([ns_edge_attr_a2a, torch.zeros((ns_edge_attr_a2a.shape[0], 2))], dim=-1)
            elif tqnet_version == 'v2':
                # edge_attr_a2a = [o_ij, selected_i]
                ns_edge_attr_a2a = torch.cat([ns_edge_attr_a2a, torch.zeros_like(ns_edge_attr_a2a)], dim=-1)
            elif tqnet_version == 'v3':
                # edge_attr_a2a = [o_ij, o_iS, selected_i]
                # o_iS, the orthogonality to the selected group is initialized to 1 (the orthogonality to "nothing" is 1 by convenetion)
                ns_edge_attr_a2a = torch.cat([ns_edge_attr_a2a, torch.ones_like(ns_edge_attr_a2a), torch.zeros_like(ns_edge_attr_a2a)], dim=-1)
            elif tqnet_version == 'none':
                # don't do anything. transformer is not in use.
                pass
            else:
                raise ValueError

            ns_terminal = torch.tensor([0], dtype=torch.bool)
        else:
            # build a redundant graph with empty features, only
            # to allow batching with all other transitions.
            ns_x_c = torch.zeros(size=(1, x_c.shape[1]))  # single node with zero features
            ns_x_v = torch.zeros(size=(1, x_v.shape[1]))  # single node with zero features
            ns_x_a = torch.zeros(size=(1, x_a.shape[1]))  # single node with zero features
            ns_edge_index_c2v = torch.tensor([[0], [0]], dtype=torch.long)  # self loops
            ns_edge_attr_c2v = torch.zeros(size=(1, 1), dtype=torch.float32)  # null attributes
            ns_edge_index_a2v = torch.tensor([[0], [0]], dtype=torch.long)  # self loops
            ns_edge_attr_a2v = torch.zeros(size=(1, 1), dtype=torch.float32)  # null attributes
            ns_edge_index_a2a = torch.tensor([[0], [0]], dtype=torch.long)  # self loops
            edge_attr_a2a_dim = {'v1': 3, 'v2': 2, 'v3': 3, 'none': 1}.get(tqnet_version)
            ns_edge_attr_a2a = torch.zeros(size=(1, edge_attr_a2a_dim), dtype=torch.float32)  # null attributes
            ns_stats = torch.zeros_like(stats)
            ns_terminal = torch.tensor([1], dtype=torch.bool)

        # create the pair-tripartite-and-clique-data object consist of both the LP tripartite graph
        # and the cuts clique graph
        # NOTE: in order to process Transition in mini batches,
        # follow the example in https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        # and do:
        # data_list = [data_1, data_2, ... data_n]
        # loader = DataLoader(data_list, batch_size=<whatever>, follow_batch=['x_c', 'x_v', 'x_a', 'ns_x_c', 'ns_x_v', 'ns_x_a'])

        data = Transition(
            x_c=x_c,
            x_v=x_v,
            x_a=x_a,
            edge_index_c2v=edge_index_c2v,
            edge_attr_c2v=edge_attr_c2v,
            edge_index_a2v=edge_index_a2v,
            edge_attr_a2v=edge_attr_a2v,
            edge_index_a2a=edge_index_a2a,
            edge_attr_a2a=edge_attr_a2a,
            # edge_index_dec=edge_index_dec,
            # edge_attr_dec=edge_attr_dec,
            stats=stats,
            a=a,
            r=r,
            ns_x_c=ns_x_c,
            ns_x_v=ns_x_v,
            ns_x_a=ns_x_a,
            ns_edge_index_c2v=ns_edge_index_c2v,
            ns_edge_attr_c2v=ns_edge_attr_c2v,
            ns_edge_index_a2v=ns_edge_index_a2v,
            ns_edge_attr_a2v=ns_edge_attr_a2v,
            ns_edge_index_a2a=ns_edge_index_a2a,
            ns_edge_attr_a2a=ns_edge_attr_a2a,
            ns_stats=ns_stats,
            ns_terminal=ns_terminal
        )
        return data

    def as_batch(self):
        return Batch.from_data_list([self], follow_batch=['x_c', 'x_v', 'x_a', 'ns_x_c', 'ns_x_v', 'ns_x_a'])

    @staticmethod
    def create_batch(transition_list):
        return Batch.from_data_list(transition_list, follow_batch=['x_c', 'x_v', 'x_a', 'ns_x_c', 'ns_x_v', 'ns_x_a'])

    @staticmethod
    def get_initial_decoder_context(scip_state, tqnet_version='v3'):
        cuts_orthogonality = scip_state['cuts_orthogonality']
        ncuts = scip_state['A'].shape[0]
        # create basic edge_attr_a2a
        if ncuts > 1:
            # we add 1 to cuts_orthogonality to ensure that all edges are created
            # including the self edges
            edge_index_a2a, edge_attr_a2a = dense_to_sparse(torch.from_numpy(cuts_orthogonality + 1))
            edge_attr_a2a -= 1  # subtract 1 to set back the correct orthogonality values
            edge_attr_a2a.unsqueeze_(dim=1)
        elif ncuts == 1:
            edge_index_a2a = torch.tensor([[0], [0]], dtype=torch.long)  # single self loop
            edge_attr_a2a = torch.zeros(size=(1, 1), dtype=torch.float32)  # self orthogonality
        else:
            raise ValueError

        # attach initial transformer context if needed
        if tqnet_version == 'v1':
            # edge_attr_a2a = [o_ij, processed_i, selected_i]
            edge_attr_a2a = torch.cat([edge_attr_a2a, torch.zeros((edge_attr_a2a.shape[0], 2))], dim=-1)
        elif tqnet_version == 'v2':
            # edge_attr_a2a = [o_ij, selected_i]
            edge_attr_a2a = torch.cat([edge_attr_a2a, torch.zeros((edge_attr_a2a.shape[0], 1))], dim=-1)
        elif tqnet_version == 'v3':
            # edge_attr_a2a = [o_ij, o_iS, selected_i]
            # o_iS, the orthogonality to the selected group is initialized to 1 (the orthogonality to "nothing" is 1 by convenetion)
            edge_attr_a2a = torch.cat([edge_attr_a2a, torch.ones_like(edge_attr_a2a), torch.zeros_like(edge_attr_a2a)], dim=-1)
        else:
            raise ValueError

        return edge_index_a2a, edge_attr_a2a


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
                 r=None,
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
        self.r = r
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


def get_gnn_data(scip_state, action=None, reward=None, scip_next_state=None):
    """
    Creates a torch_geometric.data.Data object from SCIP state
    produced by scip.Model.getState(state_format='tensor')
    :param scip_state: scip.getState(state_format='tensor')
    :param action: np.ndarray
    :param reward: np.ndarray
    :param scip_next_state: scip.getState(state_format='tensor')
    :return: PairTripartiteAndCliqueData
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
    if action is not None:
        y = torch.from_numpy(action, dtype=torch.float32)
        assert len(y) == n_a_nodes
    else:
        y = None
    if reward is not None:
        r = torch.from_numpy(reward, dtype=torch.float32)
        assert len(r) == n_a_nodes
    else:
        r = None

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
        r=r,
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
