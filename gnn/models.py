import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_add
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

    def forward(self, x_s, edge_index_s, edge_attr_s, vars_nodes, cuts_nodes, cons_edges, cuts_edges):
        """
        Compute the left-to-right convolution of a bipartite graph.
        Assuming a PairData or Batch object, d, produced by
        utils.data.get_bipartite_graph,
        the module inputs should be as follows:
        :param x_s:           = d.x_s
        :param edge_index_s:  = d.edge_index_s
        :param edge_attr_s:   = d.edge_attr_s
        :param vars_nodes:  = d.vars_nodes
        :param cuts_nodes:  = d.cuts_nodes
        :param cons_edges:  = d.cons_edges
        :param cuts_edges:  = 1 - cons_edges
        :return: torch.Tensor([d.ncuts.sum(), out_channels]) if self.cuts_only=True
                 torch.Tensor([x_s.shape[0], out_channels]) otherwise
        """
        ### LEFT TO RIGHT CONVOLUTION ###
        cons_edge_index = edge_index_s[cons_edges]  # constraints connectivity
        cuts_edge_index = edge_index_s[cuts_edges]  # candidate cuts connectivity
        cons_row, cons_col = cons_edge_index      # row and col correspond to constraint and variable nodes repectively
        cuts_row, cuts_col = cuts_edge_index
        nvars = vars_nodes.sum()
        ncuts = cuts_nodes.sum()
        ncons = x_s.shape[0] - nvars - ncuts
        cons_nodes = torch.ones_like(vars_nodes, dtype=torch.int32) - vars_nodes - cuts_nodes

        # cons to vars messages:
        g_v_input = torch.cat([x_s[cons_col][:, :self.vars_feats_dim],  # v_j
                               x_s[cons_row][:, :self.cons_feats_dim],  # c_i
                               edge_attr_s[cons_edges]],  # e_ij
                              dim=1)
        g_v_out = self.g_v(g_v_input)

        # cuts to vars messages:
        h_v_input = torch.cat([x_s[cuts_col][:, :self.vars_feats_dim],  # v_j
                               x_s[cuts_row][:, :self.cuts_feats_dim],  # c_i
                               edge_attr_s[cuts_edges]],  # e_ij
                              dim=1)
        h_v_out = self.h_v(h_v_input)

        # aggregate messages to a tensor of size [nvars, out_channels]:
        aggr_g_v_out = self.aggr_func(g_v_out, cons_col, dim=0, dim_size=nvars) # TODO verify that dim_size-None is correct
        aggr_h_v_out = self.aggr_func(h_v_out, cuts_col, dim=0, dim_size=nvars) # TODO verify that dim_size-None is correct

        # update vars features with f:
        f_v_input = torch.cat(x_s[vars_nodes], aggr_g_v_out, aggr_h_v_out, dim=1)
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
        g_cuts_input = torch.cat([x_s[cuts_row][:, :self.cuts_feats_dim],  # c_i
                                  f_v_out[vars_index_mapping[cuts_col]],  # v'_j
                                  edge_attr_s[cuts_edges]],  # e_ij
                                 dim=1)
        g_cuts_out = self.g_cuts(g_cuts_input)

        # aggregate messages to a tensor of size [ncuts, out_channels]:
        aggr_g_cuts_out = self.aggr_func(g_cuts_out, cuts_row, dim=0, dim_size=ncuts)  # TODO verify that dim_size-None is correct

        # update cuts features with f_cuts:
        f_cuts_input = torch.cat(x_s[cuts_nodes], aggr_g_cuts_out, dim=1)
        f_cuts_out = self.f_cuts(f_cuts_input)

        if not self.cuts_only:
            # do the same for the constraint nodes:
            # vars to cons messages, using the updated vars features:
            g_cons_input = torch.cat([x_s[cons_row][:, :self.cons_feats_dim],  # c_i
                                      f_v_out[vars_index_mapping[cons_col]],  # v'_j
                                      edge_attr_s[cons_edges]],  # e_ij
                                     dim=1)
            g_cons_out = self.g_cons(g_cons_input)

            # aggregate messages to a tensor of size [ncons, out_channels]:
            aggr_g_cons_out = self.aggr_func(g_cons_out, cons_row, dim=0,
                                             dim_size=ncons)  # TODO verify that dim_size-None is correct

            # update cons features with f_cons:
            f_cons_input = torch.cat(x_s[cons_nodes], aggr_g_cons_out, dim=1)
            f_cons_out = self.f_cons(f_cons_input)

            # construct the output tensor corresponding to x
            # with the updated features.
            out = torch.empty((x_s.shape[0], self.out_channels), dtype=torch.float32)
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

    def forward(self, x_t, edge_index_t, edge_attr_t):
        """
        Assuming a PairData (or Batch) object, d, produced by utils.data.get_bipartite_graph,
        this module work on the cuts clique graph of d.
        The module applies some factorization function on the complete graph,
        and then a classifier to select cuts.
        The module inputs are as follows
        :param x_t: d.x_t (the updated cuts features from CutsEmbeddind)
        :param edge_index_t: d.edge_index_t (batch-wise complete graph connectivity)
        :param edge_attr_t: d.edge_attr_t (pairwise orthogonalities)
        :return:
        """
        if self.factorization == 'fg':
            # apply factor graph convolution, and the classify
            x_t = self.f(x_t, edge_index_t, edge_attr_t)
            probs = self.classifier(x_t)
            # classify cuts as 1 ("good") if prob > 0.5, else 0 ("bad")
            y_t = probs > 0.5
            return y_t, probs
        else:
            raise NotImplementedError()


class CutSelectionModel(torch.nn.Module):
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
            edge_attr_dim=1,  # this is the intra cuts orthogonalities
            factorization=hparams.get('cuts_selector/factorization', 'fg') # default
        )

    def forward(self, state):
        """
        :return: torch.Tensor([nvars, out_channels]) if self.cuts_only=True
                 torch.Tensor([x.shape[0], out_channels]) otherwise
        """
        cuts_embedding = self.cuts_embedding(x_s=state.x_s,
                                             edge_index_s=state.edge_index_s,
                                             edge_attr_s=state.edge_attr_s,
                                             vars_nodes=state.vars_nodes,
                                             cuts_nodes=state.cuts_nodes,
                                             cons_edges=state.cons_edges,
                                             cuts_edges=1 - state.cons_edges)
        state.x_t = cuts_embedding
        y_t, probs = self.cuts_selector(x_t=state.x_t,
                                        edge_index_t=state.edge_index_t,
                                        edge_attr_t=state.edge_attr_t)
        return y_t, probs

