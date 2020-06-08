import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops
from torch_geometric.utils.repeat import repeat
from collections import OrderedDict, namedtuple

from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
TransformerDecoderContext = namedtuple('TransformerDecoderContext', ('edge_index', 'edge_attr'))


# Cut convolution with attention and pairwise edge attributes
class CATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper, extended with edge attributes :math:`e_{ij}`

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j \, \Vert e_{ij}]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k \, \Vert e_{ij}]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        edge_attr_dim (int): edge attributes dimensionality.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, edge_attr_dim, edge_attr_emb=4, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(CATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_attr_dim = edge_attr_dim
        self.edge_attr_emb = edge_attr_emb
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # projection of x_a to attention heads
        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        # projection of edge_attr to attention heads
        self.edge_attr_weight = Parameter(torch.Tensor(edge_attr_dim, heads * edge_attr_emb))  # todo
        # extend the attention projection vector according to the additional edge_attr dimensions
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_attr_emb))  # todo

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, inputs):
        """
        Compute multi head attention, where the compatibility coefficient \alpha
        takes into account the edge attributes.
        For cut encoder, edge attributes e_ij can be the orthogonality of cut i and cut j.
        For cut decoder, e_ij can be the decoder information about cut i, while processing cut j.
        In the latter case, e_ij will be two bits: (processed, selected).
        :param x: torch.Tensor [|V|, d_v]
        :param edge_index: torch.Tensor [2, |E|]
        :param edge_index: torch.Tensor [|E|, d_e]
        """
        x, edge_index, edge_attr = inputs

        # masked out this original lines, we don't need them, and I don't know why they exist.
        # in addition, I don't know what is the meaning of the input size
        # originally there was input keyword size=None, and then:
        # if size is None and torch.is_tensor(x):
        #     edge_index, _ = remove_self_loops(edge_index)
        #     edge_index, _ = add_self_loops(edge_index,
        #                                    num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))
        edge_attr = torch.matmul(edge_attr, self.edge_attr_weight)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr), edge_index, edge_attr  # todo verify

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):  # todo verify
        # Compute attention coefficients.
        # split x_j and edge_attr projections to attention heads
        x_j = x_j.view(-1, self.heads, self.out_channels)
        edge_attr = edge_attr.view(-1, self.heads, self.edge_attr_emb)
        if x_i is None:  # why should it happen?
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            # split x_i projections to attention heads
            x_i = x_i.view(-1, self.heads, self.out_channels)
            # todo - split edge_attr projections to the attention heads
            # concatenate x_i to each one of its neighbors
            # including the associated edge attributes,
            # then multiply and sum to generate \alpha_ij for each attention head
            alpha = (torch.cat([x_i, x_j, edge_attr], dim=-1) * self.att).sum(dim=-1)  # todo

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)  # todo - what is size role?

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


# LP tripartite graph convolution
class LPConv(torch.nn.Module):
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
        super(LPConv, self).__init__()
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

    def forward(self, inputs):
        """
        Compute the left-to-right convolution of a bipartite graph.
        Assuming a PairTripartiteAndCliqueData or Batch object, d, produced by
        utils.data.get_gnn_data,
        the module inputs should be as follows:
        :param inputs: a tuple consists of
                       x_c            : d.x_c
                       x_v            : d.x_v
                       x_a            : d.x_a
                       edge_index_c2v : d.edge_index_c2v
                       edge_index_a2v : d.edge_index_a2v
                       edge_attr_c2v  : d.edge_attr_c2v
                       edge_attr_a2v  : d.edge_attr_a2v
        :return: if self.cuts_only==True: torch.Tensor([d.ncuts.sum(), out_channels])
                 else: tuple like inputs, where x_c, x_v and x_a have emb_dim features, and the rest are the same.
        """
        x_c, x_v, x_a, edge_index_c2v, edge_index_a2v, edge_attr_c2v, edge_attr_a2v = inputs
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
            return f_c_out, f_v_out, f_a_out, edge_index_c2v, edge_index_a2v, edge_attr_c2v, edge_attr_a2v

        # if embedding only cuts:
        return f_a_out


# classic convolution for cuts
class CutConv(MessagePassing):
    r"""Inter cuts convolution

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
    def __init__(self, channels, edge_attr_dim, aggr='mean', **kwargs):
        super(CutConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = channels
        self.out_channels = channels
        self.edge_attr_dim = edge_attr_dim

        self.f = Lin(2 * channels, channels)
        self.g = Lin(2 * channels + edge_attr_dim, channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.f.reset_parameters()
        self.g.reset_parameters()

    def forward(self, inputs):
        """"""
        x, edge_index, edge_attr = inputs
        return self.propagate(edge_index, x=x, edge_attr=edge_attr), edge_index, edge_attr

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


# transformer Q network
class TQnet(torch.nn.Module):
    def __init__(self, hparams={}):
        super(TQnet, self).__init__()
        self.hparams = hparams
        self.device = torch.device(f"cuda:{hparams['gpu_id']}" if torch.cuda.is_available() and hparams.get('gpu_id', None) is not None else "cpu")
        ###########
        # Encoder #
        ###########
        # stack lp conv layers todo consider skip connections
        self.lp_conv = Seq(OrderedDict([(f'lp_conv_{i}', LPConv(x_v_channels=hparams.get('state_x_v_channels', 13) if i==0 else hparams.get('emb_dim', 32),   # mandatory - derived from state features
                                                                x_c_channels=hparams.get('state_x_c_channels', 14) if i==0 else hparams.get('emb_dim', 32),   # mandatory - derived from state features
                                                                x_a_channels=hparams.get('state_x_a_channels', 16) if i==0 else hparams.get('emb_dim', 32),   # mandatory - derived from state features
                                                                edge_attr_dim=hparams.get('state_edge_attr_dim', 1),  # mandatory - derived from state features
                                                                emb_dim=hparams.get('emb_dim', 32),                   # default
                                                                aggr=hparams.get('lp_conv_aggr', 'mean'),             # default
                                                                cuts_only=(i == hparams.get('encoder_lp_conv_layers', 1) - 1)))
                                        for i in range(hparams.get('encoder_lp_conv_layers', 1))]))

        # stack cut conv layers todo consider skip connections
        self.cut_conv = {
            'CutConv': Seq(OrderedDict([(f'cut_conv_{i}', CutConv(channels=hparams.get('emb_dim', 32),
                                                                  edge_attr_dim=1,
                                                                  aggr=hparams.get('cut_conv_aggr', 'mean')))
                                        for i in range(hparams.get('encoder_cut_conv_layers', 1))])),
            'CATConv': Seq(OrderedDict([(f'cat_conv_{i}', CATConv(in_channels=hparams.get('emb_dim', 32),
                                                                  out_channels=hparams.get('emb_dim', 32) // hparams.get('attention_heads', 4),
                                                                  edge_attr_dim=1,
                                                                  edge_attr_emb=1,
                                                                  heads=hparams.get('attention_heads', 4)))
                                        for i in range(hparams.get('encoder_cut_conv_layers', 1))])),
        }.get(hparams.get('cut_conv', 'CATConv'))

        ###########
        # Decoder #
        ###########
        self.decoder_conv = {
            'CutConv': Seq(OrderedDict([(f'cut_conv_{i}', CutConv(channels=hparams.get('emb_dim', 32),
                                                                  edge_attr_dim=2,
                                                                  aggr=hparams.get('cut_conv_aggr', 'mean')))
                                        for i in range(hparams.get('decoder_cut_conv_layers', 1))])),
            'CATConv': Seq(OrderedDict([(f'cat_conv_{i}', CATConv(in_channels=hparams.get('emb_dim', 32),
                                                                  out_channels=hparams.get('emb_dim', 32) // hparams.get('attention_heads', 4),
                                                                  edge_attr_dim=2,
                                                                  edge_attr_emb=4,
                                                                  heads=hparams.get('attention_heads', 4)))
                                        for i in range(hparams.get('decoder_cut_conv_layers', 1))])),
        }.get(hparams.get('cut_conv', 'CATConv'))
        self.decoder_edge_attr_list = None
        self.decoder_edge_index_list = None
        self.decoder_context = None

        ##########
        # Q head #
        ##########
        self.q = Lin(hparams.get('emb_dim', 32), 2)  # Q-values for adding a cut or not

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
                edge_attr_dec=None,
                edge_index_dec=None
                ):
        """
        :return: torch.Tensor([nvars, out_channels]) if self.cuts_only=True
                 torch.Tensor([x.shape[0], out_channels]) otherwise
        """
        # encoding
        # run lp conv and generate cut embedding
        lp_conv_inputs = x_c, x_v, x_a, edge_index_c2v, edge_index_a2v, edge_attr_c2v, edge_attr_a2v
        x_a = self.lp_conv(lp_conv_inputs)
        # run cut conv and generate cut encoding
        cut_conv_inputs = x_a, edge_index_a2a, edge_attr_a2a
        cut_encoding, _, _ = self.cut_conv(cut_conv_inputs)

        # decoding - inference
        if edge_attr_dec is None:
            ncuts = cut_encoding.shape[0]

            # rand permutation over available cuts
            inference_order = torch.randperm(ncuts)
            edge_index_dec = torch.cat([torch.arange(ncuts).view(1, -1),
                                        torch.empty((1, ncuts), dtype=torch.long)], dim=0).to(self.device)

            # initialize the decoder with all cuts marked as not (processed, selected)
            self.decoder_edge_index_list = []
            self.decoder_edge_attr_list = []
            edge_attr_dec = torch.zeros((ncuts, 2), dtype=torch.float32).to(self.device)

            # create a tensor of all q values to return to user
            q_vals = torch.empty_like(edge_attr_dec)

            # iterate over all cuts in random order, and process one cut each time
            for cut_index in inference_order:
                # set all edges to point from all cuts to the currently processed one (focus the attention mechanism)
                edge_index_dec[1, :] = cut_index

                # store the context (edge_index_dec and edge_attr_dec) of the current iteration
                self.decoder_edge_attr_list.append(edge_attr_dec.detach().cpu().clone())
                self.decoder_edge_index_list.append(edge_index_dec.detach().cpu().clone())

                # decode
                decoder_inputs = (cut_encoding, edge_index_dec, edge_attr_dec)
                cut_decoding, _, _ = self.decoder_conv(decoder_inputs)
                # take the decoder output only at the cut_index and estimate q values
                q = self.q(cut_decoding[cut_index, :])
                edge_attr_dec[cut_index, 0] = 1           # mark the current cut as processed
                edge_attr_dec[cut_index, 1] = q.argmax()  # mark the cut as selected or not, greedily according to q
                # store q in the output q_vals tensor
                q_vals[cut_index, :] = q

            # finally, stack the decoder edge_attr and edge_index tensors,
            # and make a transformer context in order to generate later a Transition for training,
            # allowing by that fast parallel backprop
            edge_attr_dec = torch.cat(self.decoder_edge_attr_list, dim=0)
            edge_index_dec = torch.cat(self.decoder_edge_index_list, dim=1)
            self.decoder_context = TransformerDecoderContext(edge_index_dec, edge_attr_dec)
            return q_vals

        else:
            # we are in training.
            # produce all q values in parallel
            decoder_inputs = (cut_encoding, edge_index_dec, edge_attr_dec)
            cut_decoding, _, _ = self.decoder_conv(decoder_inputs)
            # take the decoder output only at the cut_index and estimate q values
            return self.q(cut_decoding)


# standard Q network
class Qnet(torch.nn.Module):
    def __init__(self, hparams={}):
        super(Qnet, self).__init__()
        self.hparams = hparams

        ###########
        # Encoder #
        ###########
        # stack lp conv layers todo consider skip connections
        self.lp_conv = Seq(OrderedDict([(f'lp_conv_{i}', LPConv(x_v_channels=hparams.get('state_x_v_channels', 13) if i == 0 else hparams.get('emb_dim', 32),
                                                                x_c_channels=hparams.get('state_x_c_channels', 14) if i == 0 else hparams.get('emb_dim', 32),
                                                                x_a_channels=hparams.get('state_x_a_channels', 16) if i == 0 else hparams.get('emb_dim', 32),
                                                                edge_attr_dim=hparams.get('state_edge_attr_dim', 1),  # mandatory - derived from state features
                                                                emb_dim=hparams.get('emb_dim', 32),  # default
                                                                aggr=hparams.get('lp_conv_aggr', 'mean'),  # default
                                                                cuts_only=(i == hparams.get('encoder_lp_conv_layers', 1) - 1)))
                                        for i in range(hparams.get('encoder_lp_conv_layers', 1))]))

        # stack cut conv layers todo consider skip connections
        self.cut_conv = {
            'CutConv': Seq(OrderedDict([(f'cut_conv_{i}', CutConv(channels=hparams.get('emb_dim', 32),
                                                                  edge_attr_dim=1,
                                                                  aggr=hparams.get('cut_conv_aggr', 'mean')))
                                        for i in range(hparams.get('encoder_cut_conv_layers', 1))])),
            'CATConv': Seq(OrderedDict([(f'cat_conv_{i}', CATConv(in_channels=hparams.get('emb_dim', 32),
                                                                  out_channels=hparams.get('emb_dim', 32) // hparams.get('attention_heads', 4),
                                                                  edge_attr_dim=1,
                                                                  edge_attr_emb=1,
                                                                  heads=hparams.get('attention_heads', 4)))
                                        for i in range(hparams.get('encoder_cut_conv_layers', 1))])),
        }.get(hparams.get('cut_conv', 'CATConv'))

        ###########
        # Decoder #
        ###########
        # todo add some standard sequential model, e.g. LSTM

        ##########
        # Q head #
        ##########
        self.q = Lin(hparams.get('emb_dim', 32), 2)  # Q-values for adding a cut or not

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
                **kwargs
                ):
        """
        :return: torch.Tensor([nvars, out_channels]) if self.cuts_only=True
                 torch.Tensor([x.shape[0], out_channels]) otherwise
        """
        # encoding
        # run lp conv and generate cut embedding
        lp_conv_inputs = x_c, x_v, x_a, edge_index_c2v, edge_index_a2v, edge_attr_c2v, edge_attr_a2v
        x_a = self.lp_conv(lp_conv_inputs)
        # run cut conv and generate cut encoding
        cut_conv_inputs = x_a, edge_index_a2a, edge_attr_a2a
        cut_encoding, _, _ = self.cut_conv(cut_conv_inputs)

        # decoding
        # todo - add here the sequential decoder stuff.

        # compute q values
        return self.q(cut_encoding)


# imitation learning models - not relevant
class CutsSelector(torch.nn.Module):
    def __init__(self, channels, edge_attr_dim, hparams={}):
        super(CutsSelector, self).__init__()
        self.channels = channels
        self.edge_attr_dim = edge_attr_dim
        self.factorization_arch = hparams.get('factorization_arch', 'CutConv')
        self.factorization_aggr = hparams.get('factorization_aggr', 'mean')
        # TODO: support more factorizations, e.g. GCNConv, GATConv, etc.
        # In addition, support sequential selection
        self.f = {
            'CutConv': CutConv(channels, edge_attr_dim, aggr=self.factorization_aggr),
            'GraphUNet': GraphUNet(channels, channels, channels, depth=3)
        }.get(self.factorization_arch, 'CutConv')
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


class CutSelectionModel(torch.nn.Module):
    def __init__(self, hparams={}):
        super(CutSelectionModel, self).__init__()
        self.hparams = hparams
        assert hparams.get('cuts_embedding_layers', 1) == 1, "Not implemented"

        # cuts embedding
        self.cuts_embedding = LPConv(
            x_v_channels=hparams.get('state_x_v_channels', 13),     # mandatory - derived from state features
            x_c_channels=hparams.get('state_x_c_channels', 14),     # mandatory - derived from state features
            x_a_channels=hparams.get('state_x_a_channels', 16),     # mandatory - derived from state features
            edge_attr_dim=hparams.get('state_edge_attr_dim', 1),    # mandatory - derived from state features
            emb_dim=hparams.get('emb_dim', 32),                     # default
            aggr=hparams.get('cuts_embedding_aggr', 'mean')         # default
        )

        # cut selector
        self.cuts_selector = CutsSelector(
            channels=hparams.get('emb_dim', 32),                    # default
            edge_attr_dim=hparams.get('state_edge_attr_dim', 1),           # this is the intra cuts orthogonalities
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
