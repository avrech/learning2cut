import torch
from torch_geometric.nn import GCNConv, GATConv, GraphConv, MetaLayer
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_add
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing



class CutsEmbedding(torch.nn.Module):
    r"""Left to right and right to left graph neural network convolution,
        inspired from the `"Exact Combinatorial Optimizationwith Graph Convolutional Neural Networks"
        <https://arxiv.org/pdf/1906.01629.pdf>`_ paper
        The right nodes are the variables,
        and the left nodes consist of `ncons` constraint nodes,
        and `ncuts` candidate cuts nodes.
        The left to right update rule is:

        .. math::
            \mathbf{v}^{\prime}_j = \mathbf{f}_v ([\mathbf{v}_j ,
            \square_{i \in \mathcal{Nc}(j)} \mathbf{g}_v (\mathbf{v}_j, \mathbf{c}_i, \mathbf{e}_{ij}),
            \square_{i \in \mathcal{Na}(j)} \mathbf{h}_v (\mathbf{v}_j, \mathbf{c}_i, \mathbf{e}_{ij})])

        where

        :math:`\square_{i \in \mathcal{Nc}(j)}` is some permutation invariant aggregation function, e.g. add or mean

        :math:`\mathcal{Nc}(j)` and :math:`\mathcal{Na}(j)` are the neighboring nodes of
          :math:`\mathbf{v}_j` among the constraints and the cuts nodes respectively.

        :math:`\mathbf{f}_v, \mathbf{g}_v` and :math:`\mathbf{h}_v` are 2-layer MLP operators with Relu activation.

        In the same manner, the right to left convolution for updating
        the cuts (cons) features is

        .. math::
            \mathbf{c}^{\prime}_i = \mathbf{f}_{cuts} ([\mathbf{c}_i ,
            \square_{j \in \mathcal{N}(i)} \mathbf{g}_{cuts} (\mathbf{c}_i, \mathbf{v}_j, \mathbf{e}_{ij})])

        where

        :math:`\square_{j \in \mathcal{N}(i)}` is some permutation invariant aggregation function, e.g. add or mean

        :math:`\mathcal{N}(i)` are the neighboring variable nodes of a cut (constraint) :math:`\mathbf{c}_i`.

        :math:`\mathbf{f}_{cuts}` and :math:`\mathbf{g}_{cuts}` are 2-layer MLP operators with Relu activation

        Since cuts and constraints do not have edges between them,
        and since cuts and constraints have different features,
        they are updated using two separated right-to-left operators.

        Updated constraint features are computed only if requested (cuts_only=False),
        because they are interesting only when cascading such CutEmbedding modules.
        In the case cuts_only=False, the returned tensor is
        corresponding to the input features tensor x, but only with the
        updated features of the constraints, variables and cuts.

        Args:
            in_channels (int): Size of each input sample.
            out_channels (int): Size of each output sample.
            aggr (string, optional): The aggregation scheme to use
                (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
                (default: :obj:`"add"`)
            bias (bool, optional): If set to :obj:`False`, the layer will not learn
                an additive bias. (default: :obj:`True`)
            **kwargs (optional): Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.
        """

    def __init__(self, vars_feats_dim, cons_feats_dim, cuts_feats_dim, edge_attr_dim,
                 out_channels=32, aggr='mean', cuts_only=True):
        super(CutsEmbedding, self).__init__()
        self.vars_feats_dim = vars_feats_dim
        self.cons_feats_dim = cons_feats_dim
        self.cuts_feats_dim = cuts_feats_dim
        self.edge_attr_dim = edge_attr_dim
        self.out_channels = out_channels
        self.aggr = aggr
        self.cuts_only = cuts_only
        if aggr == 'add':
            self.aggr_func = scatter_add
        elif aggr == 'mean':
            self.aggr_func = scatter_mean

        ### LEFT TO RIGHT LAYERS ###
        # vars embedding
        self.g_v_in_channels = vars_feats_dim + cons_feats_dim + edge_attr_dim
        self.h_v_in_channels = vars_feats_dim + cuts_feats_dim + edge_attr_dim
        self.f_v_in_channels = vars_feats_dim + out_channels * 2
        self.g_v = Seq(Lin(self.g_v_in_channels, out_channels), ReLU(), Lin(out_channels, out_channels))
        self.h_v = Seq(Lin(self.h_v_in_channels, out_channels), ReLU(), Lin(out_channels, out_channels))
        self.f_v = Seq(Lin(self.f_v_in_channels, out_channels), ReLU(), Lin(out_channels, out_channels))

        ### RIGHT TO LEFT LAYERS ###
        # cuts embedding
        self.g_cuts_in_channels = cuts_feats_dim + out_channels + edge_attr_dim
        self.f_cuts_in_channels = cuts_feats_dim + out_channels
        self.g_cuts = Seq(Lin(self.g_cuts_in_channels, out_channels), ReLU(), Lin(out_channels, out_channels))
        self.f_cuts = Seq(Lin(self.f_cuts_in_channels, out_channels), ReLU(), Lin(out_channels, out_channels))
        # cons embedding:
        if not cuts_only:
            self.g_cons_in_channels = cons_feats_dim + out_channels + edge_attr_dim
            self.f_cons_in_channels = cons_feats_dim + out_channels
            self.g_cons = Seq(Lin(self.g_cons_in_channels, out_channels), ReLU(), Lin(out_channels, out_channels))
            self.f_cons = Seq(Lin(self.f_cons_in_channels, out_channels), ReLU(), Lin(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr, vars_nodes, cuts_nodes, cons_edges, cuts_edges):
        """
        Compute the left-to-right convolution of a bipartite graph.
        Assuming a Data or Batch object, d, produced by
        utils.functions.get_bipartite_graph,
        the module inputs should be as follows:
        :param x:           = d.x
        :param edge_index:  = d.edge_index
        :param edge_attr:   = d.edge_attr
        :param vars_nodes:  = d.vars_nodes
        :param cuts_nodes:  = d.cuts_nodes
        :param cons_edges:  = d.cons_edges
        :param cuts_edges:  = 1 - cons_edges
        :return: torch.Tensor([nvars, out_channels]) if self.cuts_only=True
                 torch.Tensor([x.shape[0], out_channels]) otherwise
        """
        ### LEFT TO RIGHT CONVOLUTION ###
        cons_edge_index = edge_index[cons_edges]  # constraints connectivity
        cuts_edge_index = edge_index[cuts_edges]  # candidate cuts connectivity
        cons_row, cons_col = cons_edge_index      # row and col correspond to constraint and variable nodes repectively
        cuts_row, cuts_col = cuts_edge_index
        nvars = vars_nodes.sum()
        ncuts = cuts_nodes.sum()
        ncons = x.shape[0] - nvars - ncuts
        cons_nodes = torch.ones_like(vars_nodes, dtype=torch.int32) - vars_nodes - cuts_nodes

        # cons to vars messages:
        g_v_input = torch.cat([x[cons_col][:, :self.vars_feats_dim], # v_j
                               x[cons_row][:, :self.cons_feats_dim], # c_i
                               edge_attr[cons_edges]],               # e_ij
                              dim=1)
        g_v_out = self.g_v(g_v_input)

        # cuts to vars messages:
        h_v_input = torch.cat([x[cuts_col][:, :self.vars_feats_dim], # v_j
                               x[cuts_row][:, :self.cuts_feats_dim], # c_i
                               edge_attr[cuts_edges]],               # e_ij
                              dim=1)
        h_v_out = self.h_v(h_v_input)

        # aggregate messages to a tensor of size [nvars, out_channels]:
        aggr_g_v_out = self.aggr_func(g_v_out, cons_col, dim=0, dim_size=nvars) # TODO verify that dim_size-None is correct
        aggr_h_v_out = self.aggr_func(h_v_out, cuts_col, dim=0, dim_size=nvars) # TODO verify that dim_size-None is correct

        # update vars features with f:
        f_v_input = torch.cat(x[vars_nodes], aggr_g_v_out, aggr_h_v_out, dim=1)
        f_v_out = self.f_v(f_v_input)

        # return a tensor of size [total_nvars, out_channels]
        # this tensor should be propagated to the next layer as the updated variable nodes features
        # return f_v_out


        ### RIGHT TO LEFT CONVOLUTION ###
        # construct the edge_index for the updated vars features tensor:
        # now, the updated vars features are indexed by range(nvars).
        # we need to map the 2nd row of edge_index 1:1 to range(nvars).
        # so only take the previous vars indices, and replace them by range(nvars)
        # in the ascending order mapping.
        vars_index_mapping = torch.zeros_like(vars_nodes)
        vars_index_mapping[vars_nodes] = torch.arange(nvars)

        # vars to cuts messages, using the updated vars features:
        g_cuts_input = torch.cat([x[cuts_row][:, :self.cuts_feats_dim],  # c_i
                             f_v_out[vars_index_mapping[cuts_col]], # v'_j
                             edge_attr[cuts_edges]],                # e_ij
                            dim=1)
        g_cuts_out = self.g_cuts(g_cuts_input)

        # aggregate messages to a tensor of size [ncuts, out_channels]:
        aggr_g_cuts_out = self.aggr_func(g_cuts_out, cuts_row, dim=0, dim_size=ncuts)  # TODO verify that dim_size-None is correct

        # update cuts features with f_cuts:
        f_cuts_input = torch.cat(x[cuts_nodes], aggr_g_cuts_out, dim=1)
        f_cuts_out = self.f_cuts(f_cuts_input)

        if not self.cuts_only:
            # do the same for the constraint nodes:
            # vars to cons messages, using the updated vars features:
            g_cons_input = torch.cat([x[cons_row][:, :self.cons_feats_dim],  # c_i
                                      f_v_out[vars_index_mapping[cons_col]],  # v'_j
                                      edge_attr[cons_edges]],  # e_ij
                                     dim=1)
            g_cons_out = self.g_cons(g_cons_input)

            # aggregate messages to a tensor of size [ncons, out_channels]:
            aggr_g_cons_out = self.aggr_func(g_cons_out, cons_row, dim=0,
                                             dim_size=ncons)  # TODO verify that dim_size-None is correct

            # update cons features with f_cons:
            f_cons_input = torch.cat(x[cons_nodes], aggr_g_cons_out, dim=1)
            f_cons_out = self.f_cons(f_cons_input)

            # construct the output tensor corresponding to x
            # with the updated features.
            out = torch.empty((x.shape[0], self.out_channels), dtype=torch.float32)
            out[vars_nodes] = f_v_out
            out[cons_nodes] = f_cons_out
            out[cuts_nodes] = f_cuts_out
            return out

        # return a tensor of size [ncuts, out_channels]
        return f_cuts_out


class FGConv(MessagePassing):
    r"""Factor Graph Convolution

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{f} \left([\mathbf{x}_i, \square_{j \in \mathcal{N}(i)}
        \mathbf{g}(\mathbf{z}_{ij})]\right)

    Where :math:`\mathbf{f}` and :math:`\mathbf{g}` denote some NN operator, e.g. mlp.

    and :math:`\mathbf{z}_{ij} = [ \mathbf{x}_i, \mathbf{x}_j, \mathbf{e}_{i,j} ]`

    The aggregation function :math:`\square` can be either 'add' or 'mean'

    Args:
        channels (int): Size of each input sample.
        edge_attr_dim (int): Edge feature dimensionality.
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, channels, edge_attr_dim, aggr='add', **kwargs):
        super(FGConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = channels
        self.out_channels = channels
        self.edge_attr_dim = edge_attr_dim

        self.f = Lin(2 * channels, channels)
        self.g = Lin(2 * channels + edge_attr_dim, channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.f.reset_parameters()
        self.g.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """"""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)


    def message(self, x_i, x_j, edge_attr):
        z_ij = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.g(z_ij)

    def update(self, aggr_out, x):
        x = torch.cat([x, aggr_out], dim=-1)
        return self.f(x)

    def __repr__(self):
        return '{}({}, {}, edge_attr_dim={})'.format(self.__class__.__name__,
                                           self.in_channels, self.out_channels,
                                           self.edge_attr_dim)


class CutsSelector(torch.nn.Module):
    def __init__(self, channels, edge_attr_dim, factorization='fg'):
        super(CutsSelector, self).__init__()
        self.channels = channels
        self.edge_attr_dim = edge_attr_dim
        self.factorization = factorization
        # TODO: support more factorizations, e.g. GCNConv, GATConv, etc.
        # In addition, support sequential selection
        self.f = {'fg': FGConv(channels, edge_attr_dim, aggr='add')}[factorization]
        self.classifier = Lin(channels, 1)  # binary decision, wheter to apply the cut or not.

    def forward(self, cuts_embedding, cuts_edge_index, cuts_edge_attr):
        if self.factorization = 'fg':
            # apply factor graph convolution, and the classify
            x = self.f(cuts_embedding, cuts_edge_index, cuts_edge_attr)
            prob = self.classifier(x)
            # classify cuts as 1 ("good") if prob > 0.5, else 0 ("bad")
            y = prob > 0.5
            return y, prob
        else:
            raise NotImplementedError()


class CutSelectionAgent(torch.nn.Module):
    def __init__(self, hparams={}):
        self.hparams = hparams
        assert hparams['cuts_embedding/cuts_only'], "not implemented"

        # cuts embedding
        self.cuts_embedding = CutsEmbedding(
            vars_feats_dim=hparams['state/vars_feats_dim'],         # mandatory - derived from state features
            cons_feats_dim=hparams['state/cons_feats_dim'],         # mandatory - derived from state features
            cuts_feats_dim=hparams['state/cuts_feats_dim'],         # mandatory - derived from state features
            edge_attr_dim=hparams['state/edge_attr_dim'],           # mandatory - derived from state features
            out_channels=hparams.get('channels', 32),               # default
            aggr=hparams.get('cuts_embedding/aggr', 'mean'),        # default
            cuts_only=hparams.get('cuts_embedding/cuts_only', True) # default
        )

        # cut selector
        self.cuts_selector = CutsSelector(
            channels=hparams.get('channels', 32),  # default
            edge_attr_dim=1,  # this is the inter cuts parallelism coefficient
            factorization=hparams.get('cuts_selector/factorization', 'fg') # default
        )

    def forward(self, state):
        """
        :return: torch.Tensor([nvars, out_channels]) if self.cuts_only=True
                 torch.Tensor([x.shape[0], out_channels]) otherwise
        """
        cuts_embedding = self.cuts_embedding(x=state.x,
                                             edge_index=state.edge_index,
                                             edge_attr=state.edge_attr,
                                             vars_nodes=state.vars_nodes,
                                             cuts_nodes=state.cuts_nodes,
                                             cons_edges=state.cons_edges,
                                             cuts_edges=1 - state.cons_edges)

        # build the complete graphs edge_index for each instance candidate cuts

        # partition of the cuts nodes according to their instances
        b = state.batch[state.cuts_nodes]


# class MetaLayer(torch.nn.Module):
#     r"""A meta layer for building any kind of graph network, inspired by the
#     `"Relational Inductive Biases, Deep Learning, and Graph Networks"
#     <https://arxiv.org/abs/1806.01261>`_ paper.
#
#     A graph network takes a graph as input and returns an updated graph as
#     output (with same connectivity).
#     The input graph has node features :obj:`x`, edge features :obj:`edge_attr`
#     as well as global-level features :obj:`u`.
#     The output graph has the same structure, but updated features.
#
#     Edge features, node features as well as global features are updated by
#     calling the modules :obj:`edge_model`, :obj:`node_model` and
#     :obj:`global_model`, respectively.
#
#     To allow for batch-wise graph processing, all callable functions take an
#     additional argument :obj:`batch`, which determines the assignment of
#     edges or nodes to their specific graphs.
#
#     Args:
#         edge_model (Module, optional): A callable which updates a graph's edge
#             features based on its source and target node features, its current
#             edge features and its global features. (default: :obj:`None`)
#         node_model (Module, optional): A callable which updates a graph's node
#             features based on its current node features, its graph
#             connectivity, its edge features and its global features.
#             (default: :obj:`None`)
#         global_model (Module, optional): A callable which updates a graph's
#             global features based on its node features, its graph connectivity,
#             its edge features and its current global features.
#
#     .. code-block:: python
#
#         from torch.nn import Sequential as Seq, Linear as Lin, ReLU
#         from torch_scatter import scatter_mean
#         from torch_geometric.nn import MetaLayer
#
#         class EdgeModel(torch.nn.Module):
#             def __init__(self):
#                 super(EdgeModel, self).__init__()
#                 self.edge_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
#
#             def forward(self, src, dest, edge_attr, u, batch):
#                 # source, target: [E, F_x], where E is the number of edges.
#                 # edge_attr: [E, F_e]
#                 # u: [B, F_u], where B is the number of graphs.
#                 # batch: [E] with max entry B - 1.
#                 out = torch.cat([src, dest, edge_attr, u[batch]], 1)
#                 return self.edge_mlp(out)
#
#         class NodeModel(torch.nn.Module):
#             def __init__(self):
#                 super(NodeModel, self).__init__()
#                 self.node_mlp_1 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
#                 self.node_mlp_2 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
#
#             def forward(self, x, edge_index, edge_attr, u, batch):
#                 # x: [N, F_x], where N is the number of nodes.
#                 # edge_index: [2, E] with max entry N - 1.
#                 # edge_attr: [E, F_e]
#                 # u: [B, F_u]
#                 # batch: [N] with max entry B - 1.
#                 row, col = edge_index
#                 out = torch.cat([x[row], edge_attr], dim=1)
#                 out = self.node_mlp_1(out)
#                 out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
#                 out = torch.cat([x, out, u[batch]], dim=1)
#                 return self.node_mlp_2(out)
#
#         class GlobalModel(torch.nn.Module):
#             def __init__(self):
#                 super(GlobalModel, self).__init__()
#                 self.global_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
#
#             def forward(self, x, edge_index, edge_attr, u, batch):
#                 # x: [N, F_x], where N is the number of nodes.
#                 # edge_index: [2, E] with max entry N - 1.
#                 # edge_attr: [E, F_e]
#                 # u: [B, F_u]
#                 # batch: [N] with max entry B - 1.
#                 out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
#                 return self.global_mlp(out)
#
#         op = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())
#         x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)
#     """
#
#     def __init__(self, edge_model=None, node_model=None, global_model=None):
#         super(MetaLayer, self).__init__()
#         self.edge_model = edge_model
#         self.node_model = node_model
#         self.global_model = global_model
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for item in [self.node_model, self.edge_model, self.global_model]:
#             if hasattr(item, 'reset_parameters'):
#                 item.reset_parameters()
#
#     def forward(self, x, edge_index, edge_attr=None, u=None, batch=None):
#         """"""
#         row, col = edge_index
#
#         if self.edge_model is not None:
#             edge_attr = self.edge_model(x[row], x[col], edge_attr, u,
#                                         batch if batch is None else batch[row])
#
#         if self.node_model is not None:
#             x = self.node_model(x, edge_index, edge_attr, u, batch)
#
#         if self.global_model is not None:
#             u = self.global_model(x, edge_index, edge_attr, u, batch)
#
#         return x, edge_attr, u
#
#     def __repr__(self):
#         return ('{}(\n'
#                 '    edge_model={},\n'
#                 '    node_model={},\n'
#                 '    global_model={}\n'
#                 ')').format(self.__class__.__name__, self.edge_model,
#                             self.node_model, self.global_model)
#
#
# class Right2LeftConv(torch.nn.Module):
#     r"""Right to left graph neural network operator
#         inspired from the `"Exact Combinatorial Optimizationwith Graph Convolutional Neural Networks"
#         <https://arxiv.org/pdf/1906.01629.pdf>`_ paper
#         The right nodes are the `nvars` variables,
#         and the left nodes consist of `ncons` constraint nodes,
#         and `ncuts` candidate cuts nodes.
#         The update rule is:
#
#         .. math::
#             \mathbf{c}^{\prime}_i = \mathbf{f} ([\mathbf{c}_i ,
#             \square_{j \in \mathcal{N}(i)} \mathbf{g} (\mathbf{c}_i, \mathbf{v}_j, \mathbf{e}_{ij})])
#
#         where
#
#         :math:`\square_{i \in \mathcal{N}(j)}` is some permutation invariant aggregation function, e.g. add or mean
#
#         :math:`\mathcal{N}(i)` are the neighboring variable nodes of :math:`\mathbf{c}_i`.
#
#         :math:`\mathbf{f}` and :math:`\mathbf{g}` are 2-layer MLP operators with Relu activation
#
#         This module can be used both for updating the constraint nodes and the cut nodes.
#         Since cuts and constraints do not have edges between them,
#         and since cuts and constraints have different features, they can be updated using two
#         separated instances of this class.
#         We prefer using the key-word cuts upon cons, because in this work cons are not the
#         object of interest. However, it applies also for constraints.
#
#         Args:
#             in_channels (int): Size of each input sample.
#             out_channels (int): Size of each output sample.
#             aggr (string, optional): The aggregation scheme to use
#                 (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
#                 (default: :obj:`"add"`)
#             bias (bool, optional): If set to :obj:`False`, the layer will not learn
#                 an additive bias. (default: :obj:`True`)
#             **kwargs (optional): Additional arguments of
#                 :class:`torch_geometric.nn.conv.MessagePassing`.
#         """
#
#     def __init__(self, vars_feats_dim, cuts_feats_dim, edge_attr_dim,
#                  out_channels, aggr='mean'):
#         super(Right2LeftConv, self).__init__()
#         self.vars_feats_dim = vars_feats_dim
#         self.cuts_feats_dim = cuts_feats_dim
#         self.edge_attr_dim = edge_attr_dim
#         self.out_channels = out_channels
#         self.aggr = aggr
#         if aggr == 'add':
#             self.aggr_func = scatter_add
#         elif aggr == 'mean':
#             self.aggr_func = scatter_mean
#
#         self.g_in_channels = cuts_feats_dim + vars_feats_dim + edge_attr_dim
#         self.f_in_channels = cuts_feats_dim + out_channels
#         self.g = Seq(Lin(self.g_in_channels, out_channels), ReLU(), Lin(out_channels, out_channels))
#         self.f = Seq(Lin(self.f_in_channels, out_channels), ReLU(), Lin(out_channels, out_channels))
#
#     def forward(self, x, edge_index, edge_attr, cuts_nodes, cuts_edges):
#         """
#         Compute the left-to-right convolution of a bipartite graph.
#         Assuming a Data or Batch object, d, produced by
#         utils.functions.get_bipartite_graph,
#         the module inputs should be as follows:
#         :param x:           = d.x
#         :param edge_index:  = d.edge_index
#         :param edge_attr:   = d.edge_attr
#         :param cuts_nodes:  = d.cuts_nodes
#         :param cuts_edges:  = 1 - d.cons_edges
#         :return: torch.Tensor of size [ncuts, out_channels] containing the
#         updated cuts features.
#         """
#         cuts_edge_index = edge_index[cuts_edges]  # candidate cuts connectivity
#         cuts_row, cuts_col = cuts_edge_index      # row and col correspond to cuts and variable nodes repectively
#
#         # vars to cuts messages:
#         g_input = torch.cat([x[cuts_row][:, :self.cuts_feats_dim],
#                              x[cuts_col][:, :self.vars_feats_dim],
#                              edge_attr[cuts_edges]], dim=1)
#         g_out = self.g(g_input)
#
#         # aggregate messages to a tensor of size [nvars, out_channels]:
#         aggr_g_out = self.aggr_func(g_out, cuts_row, dim=0, dim_size=cuts_nodes.sum()) # TODO verify that dim_size-None is correct
#
#         # update cuts features with f:
#         f_input = torch.cat(x[cuts_nodes], aggr_g_out, dim=1)
#         f_out = self.f(f_input)
#
#         # return a tensor of size [total ncuts, out_channels]
#         # this tensor should be propagated to the next layer as the updated cut nodes features
#         return f_out