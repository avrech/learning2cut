import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.conv import MessagePassing, GCNConv
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops
from torch_geometric.utils.repeat import repeat


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
            emb_dim (int): Size of each output sample.
            aggr (string, optional): The aggregation scheme to use
                (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
                (default: :obj:`"add"`)
            bias (bool, optional): If set to :obj:`False`, the layer will not learn
                an additive bias. (default: :obj:`True`)
            **kwargs (optional): Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.
        """

    def __init__(self, x_v_channels, x_c_channels, x_a_channels, edge_attr_dim,
                 emb_dim=32, aggr='mean', cuts_only=True):
        super(CutsEmbedding, self).__init__()
        self.x_v_channels = x_v_channels
        self.x_c_channels = x_c_channels
        self.x_a_channels = x_a_channels
        self.edge_attr_dim = edge_attr_dim
        self.emb_dim = emb_dim
        self.aggr = aggr
        self.cuts_only = cuts_only
        if aggr == 'add':
            self.aggr_func = scatter_add
        elif aggr == 'mean':
            self.aggr_func = scatter_mean

        ### LEFT TO RIGHT LAYERS ###
        # vars embedding
        self.g_v_in_channels = x_v_channels + x_c_channels + edge_attr_dim
        self.h_v_in_channels = x_v_channels + x_a_channels + edge_attr_dim
        self.f_v_in_channels = x_v_channels + emb_dim * 2
        self.g_v = Seq(Lin(self.g_v_in_channels, emb_dim), ReLU(), Lin(emb_dim, emb_dim))
        self.h_v = Seq(Lin(self.h_v_in_channels, emb_dim), ReLU(), Lin(emb_dim, emb_dim))
        self.f_v = Seq(Lin(self.f_v_in_channels, emb_dim), ReLU(), Lin(emb_dim, emb_dim))

        ### RIGHT TO LEFT LAYERS ###
        # cuts embedding
        self.g_a_in_channels = x_a_channels + emb_dim + edge_attr_dim
        self.f_a_in_channels = x_a_channels + emb_dim
        self.g_a = Seq(Lin(self.g_a_in_channels, emb_dim), ReLU(), Lin(emb_dim, emb_dim))
        self.f_a = Seq(Lin(self.f_a_in_channels, emb_dim), ReLU(), Lin(emb_dim, emb_dim))
        # cons embedding:
        if not cuts_only:
            self.g_c_in_channels = x_c_channels + emb_dim + edge_attr_dim
            self.f_c_in_channels = x_c_channels + emb_dim
            self.g_c = Seq(Lin(self.g_c_in_channels, emb_dim), ReLU(), Lin(emb_dim, emb_dim))
            self.f_c = Seq(Lin(self.f_c_in_channels, emb_dim), ReLU(), Lin(emb_dim, emb_dim))

    def forward(self, x_c, x_v, x_a, edge_index_c2v, edge_index_a2v, edge_attr_c2v, edge_attr_a2v):
        """
        Compute the left-to-right convolution of a bipartite graph.
        Assuming a PairTripartiteAndCliqueData or Batch object, d, produced by
        utils.data.get_gnn_data,
        the module inputs should be as follows:
        :param x_c:             = d.x_c
        :param x_v:             = d.x_v
        :param x_a:             = d.x_a
        :param edge_index_c2v:  = d.edge_index_c2v
        :param edge_index_a2v:  = d.edge_index_a2v
        :param edge_attr_c2v:   = d.edge_attr_c2v
        :param edge_attr_a2v:   = d.edge_attr_a2v
        :return: torch.Tensor([d.ncuts.sum(), out_channels]) if self.cuts_only=True
                 torch.Tensor([x_s.shape[0], out_channels]) otherwise
        """
        ### LEFT TO RIGHT CONVOLUTION ###
        c2v_s, c2v_t = edge_index_c2v
        a2v_s, a2v_t = edge_index_a2v
        n_v_nodes = x_v.shape[0]
        n_a_nodes = x_a.shape[0]
        n_c_nodes = x_c.shape[0]

        # cons to vars messages:
        g_v_input = torch.cat([x_v[c2v_t],      # v_j
                               x_c[c2v_s],      # c_i
                               edge_attr_c2v],  # e_ij
                              dim=1)
        g_v_out = self.g_v(g_v_input)

        # cuts to vars messages:
        h_v_input = torch.cat([x_v[a2v_t],      # v_j
                               x_a[a2v_s],      # c_i
                               edge_attr_a2v],  # e_ij
                              dim=1)
        h_v_out = self.h_v(h_v_input)

        # aggregate messages to a tensor of size [nvars, out_channels]:
        aggr_g_v_out = self.aggr_func(g_v_out, c2v_t, dim=0, dim_size=n_v_nodes)  # TODO verify that dim_size-None is correct
        aggr_h_v_out = self.aggr_func(h_v_out, a2v_t, dim=0, dim_size=n_v_nodes)  # TODO verify that dim_size-None is correct

        # update vars features with f:
        f_v_input = torch.cat([x_v, aggr_g_v_out, aggr_h_v_out], dim=1)
        f_v_out = self.f_v(f_v_input)

        # return a tensor of size [total_nvars, out_channels]
        # this tensor should be propagated to the next layer as the updated variable nodes features
        # return f_v_out


        ### RIGHT TO LEFT CONVOLUTION ###
        # vars to cuts messages, using the updated vars features:
        g_a_input = torch.cat([x_a[a2v_s],      # a_i
                               f_v_out[a2v_t],  # v'_j
                               edge_attr_a2v],  # e_ij
                              dim=1)
        g_a_out = self.g_a(g_a_input)

        # aggregate messages to a tensor of size [ncuts, out_channels]:
        aggr_g_a_out = self.aggr_func(g_a_out, a2v_s, dim=0, dim_size=n_a_nodes)  # TODO verify that dim_size-None is correct

        # update cuts features with f_cuts:
        f_a_input = torch.cat([x_a, aggr_g_a_out], dim=-1)
        f_a_out = self.f_a(f_a_input)

        if not self.cuts_only:
            # do the same for the constraint nodes:
            # vars to cons messages, using the updated vars features:
            g_c_input = torch.cat([x_c[c2v_s],      # c_i
                                   f_v_out[c2v_t],  # v'_j
                                   edge_attr_c2v],  # e_ij
                                  dim=1)
            g_c_out = self.g_c(g_c_input)

            # aggregate messages to a tensor of size [ncons, out_channels]:
            aggr_g_c_out = self.aggr_func(g_c_out, c2v_s, dim=0, dim_size=n_c_nodes)  # TODO verify that dim_size-None is correct

            # update cons features with f_cons:
            f_c_input = torch.cat([x_c, aggr_g_c_out], dim=-1)
            f_c_out = self.f_c(f_c_input)

            # return the updated features of the constraint, variable and cut nodes
            return f_c_out, f_v_out, f_a_out

        # if embedding only cuts:
        return f_a_out


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

    def forward(self, x, edge_index, edge_attr, batch=None):
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
    def __init__(self, channels, edge_attr_dim, hparams={}):
        super(CutsSelector, self).__init__()
        self.channels = channels
        self.edge_attr_dim = edge_attr_dim
        self.factorization_arch = hparams.get('factorization_arch', 'FGConv')
        self.factorization_aggr = hparams.get('factorization_aggr', 'mean')
        # TODO: support more factorizations, e.g. GCNConv, GATConv, etc.
        # In addition, support sequential selection
        self.f = {
            'FGConv': FGConv(channels, edge_attr_dim, aggr=self.factorization_aggr),
            'GraphUNet': GraphUNet(channels, channels, channels, depth=3)
        }.get(self.factorization_arch, 'FGConv')
        self.classifier = Seq(Lin(channels, 1))  # binary decision, wheter to apply the cut or not.

    def forward(self, x_a, edge_index_a2a, edge_attr_a2a, batch=None):
        """
        Assuming a PairTripartiteAndClique (or Batch) object, d,
        produced by utils.data.get_gnn_data,
        this module works on the cuts clique graph of d.
        The module applies some factorization function on the clique graph,
        and then applies a classifier to select cuts.
        The module inputs are as follows
        :param x_a: d.x_a (the updated cut features from CutsEmbedding)
        :param edge_index_a2a: d.edge_index_a2a (instance-wise cuts clique graph connectivity)
        :param edge_attr_a2a: d.edge_attr_a2a (intra-cuts orthogonality)
        :return:
        """
        # apply factorization module
        x_a = self.f(x_a, edge_index_a2a, edge_attr_a2a, batch)

        # classify
        probs = self.classifier(x_a).sigmoid()

        # classify cuts as 1 ("good") if probs > 0.5, else 0 ("bad")
        y = probs > 0.5
        return y, probs


class Qhead(torch.nn.Module):
    def __init__(self, channels, edge_attr_dim, hparams={}):
        super(Qhead, self).__init__()
        self.channels = channels
        self.edge_attr_dim = edge_attr_dim
        self.factorization_arch = hparams.get('factorization_arch', 'FGConv')
        self.factorization_aggr = hparams.get('factorization_aggr', 'mean')
        # TODO: support more factorizations, e.g. GCNConv, GATConv, etc.
        # In addition, support sequential selection
        self.f = {
            'FGConv': FGConv(channels, edge_attr_dim, aggr=self.factorization_aggr),
            'GraphUNet': GraphUNet(channels, channels, channels, depth=3)
        }.get(self.factorization_arch, 'FGConv')
        self.q = Lin(channels, 2)  # Q-values for adding a cut or not

    def forward(self, x_a, edge_index_a2a, edge_attr_a2a, batch=None):
        """
        Assuming a PairTripartiteAndClique (or Batch) object, d,
        produced by utils.data.get_gnn_data,
        this module works on the cuts clique graph of d.
        The module applies some factorization function on the clique graph,
        and then applies a classifier to select cuts.
        The module inputs are as follows
        :param x_a: d.x_a (the updated cut features from CutsEmbedding)
        :param edge_index_a2a: d.edge_index_a2a (instance-wise cuts clique graph connectivity)
        :param edge_attr_a2a: d.edge_attr_a2a (intra-cuts orthogonality)
        :return:
        """
        # apply factorization module
        x_a = self.f(x_a, edge_index_a2a, edge_attr_a2a, batch)

        # approximate the q value of each action
        q_a = self.q(x_a)
        return q_a


class Qnet(torch.nn.Module):
    def __init__(self, hparams={}):
        super(Qnet, self).__init__()
        self.hparams = hparams
        assert hparams['cuts_embedding_layers'] == 1, "Not implemented"

        # cuts embedding
        self.cuts_embedding = CutsEmbedding(
            x_v_channels=hparams['state_x_v_channels'],             # mandatory - derived from state features
            x_c_channels=hparams['state_x_c_channels'],             # mandatory - derived from state features
            x_a_channels=hparams['state_x_a_channels'],             # mandatory - derived from state features
            edge_attr_dim=hparams['state_edge_attr_dim'],           # mandatory - derived from state features
            emb_dim=hparams.get('emb_dim', 32),                     # default
            aggr=hparams.get('cuts_embedding_aggr', 'mean')         # default
        )

        # cut selector
        self.q_head = Qhead(
            channels=hparams.get('emb_dim', 32),                    # default
            edge_attr_dim=hparams['state_edge_attr_dim'],           # this is the intra cuts orthogonalities
            hparams=hparams
        )

    def forward(self,
                x_c,
                x_v,
                x_a,
                edge_index_c2v,
                edge_index_a2v,
                edge_attr_c2v,
                edge_attr_a2v,
                edge_index_a2a,
                edge_attr_a2a,
                x_a_batch
                ):
        """
        :return: torch.Tensor([nvars, out_channels]) if self.cuts_only=True
                 torch.Tensor([x.shape[0], out_channels]) otherwise
        """
        cuts_embedding = self.cuts_embedding(x_c=x_c,
                                             x_v=x_v,
                                             x_a=x_a,
                                             edge_index_c2v=edge_index_c2v,
                                             edge_index_a2v=edge_index_a2v,
                                             edge_attr_c2v=edge_attr_c2v,
                                             edge_attr_a2v=edge_attr_a2v)

        q_a = self.q_head(x_a=cuts_embedding,
                          edge_index_a2a=edge_index_a2a,
                          edge_attr_a2a=edge_attr_a2a,
                          batch=x_a_batch)
        return q_a


class CutSelectionModel(torch.nn.Module):
    def __init__(self, hparams={}):
        super(CutSelectionModel, self).__init__()
        self.hparams = hparams
        assert hparams['cuts_embedding_layers'] == 1, "Not implemented"

        # cuts embedding
        self.cuts_embedding = CutsEmbedding(
            x_v_channels=hparams['state_x_v_channels'],             # mandatory - derived from state features
            x_c_channels=hparams['state_x_c_channels'],             # mandatory - derived from state features
            x_a_channels=hparams['state_x_a_channels'],             # mandatory - derived from state features
            edge_attr_dim=hparams['state_edge_attr_dim'],           # mandatory - derived from state features
            emb_dim=hparams.get('emb_dim', 32),                     # default
            aggr=hparams.get('cuts_embedding_aggr', 'mean')         # default
        )

        # cut selector
        self.cuts_selector = CutsSelector(
            channels=hparams.get('emb_dim', 32),                    # default
            edge_attr_dim=hparams['state_edge_attr_dim'],           # this is the intra cuts orthogonalities
            hparams=hparams
        )

    def forward(self, state):
        """
        :return: torch.Tensor([nvars, out_channels]) if self.cuts_only=True
                 torch.Tensor([x.shape[0], out_channels]) otherwise
        """
        cuts_embedding = self.cuts_embedding(x_c=state.x_c,
                                             x_v=state.x_v,
                                             x_a=state.x_a,
                                             edge_index_c2v=state.edge_index_c2v,
                                             edge_index_a2v=state.edge_index_a2v,
                                             edge_attr_c2v=state.edge_attr_c2v,
                                             edge_attr_a2v=state.edge_attr_a2v)

        y, probs = self.cuts_selector(x_a=cuts_embedding,
                                      edge_index_a2a=state.edge_index_a2a,
                                      edge_attr_a2a=state.edge_attr_a2a,
                                      batch=state.x_a_batch)
        return y, probs





class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.
    This version allows also edge attributes.
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """

    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(GraphUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))
        else:
            edge_weight.squeeze_()

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)
