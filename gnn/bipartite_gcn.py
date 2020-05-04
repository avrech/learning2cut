import torch
from torch_geometric.nn import GCNConv, GATConv


class BiGraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, attention_heads=4, bidirectional=True):
        super(BiGraphConv, self).__init__()
        self.use_gatconv = attention_heads > 0
        self.bidirectional = bidirectional
        if bidirectional:
            out_channels //= 2 # to preserve out_channels dimensions after concatenating outputs
        if self.use_gatconv:
            out_channels //= attention_heads
            self.inbound_conv = GATConv(in_channels, out_channels, heads=attention_heads)
            self.outbound_conv = GATConv(in_channels, out_channels, heads=attention_heads)
        else:
            self.inbound_conv = GCNConv(in_channels, out_channels, improved=True)
            self.outbound_conv = GCNConv(in_channels, out_channels, improved=True)

    def reset_parameters(self):
        self.inbound_conv.reset_parameters()
        self.outbound_conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        inbound_edges = edge_index
        outbound_edges = torch.flip(edge_index, (0,))
        if self.use_gatconv:
            inbound_x = self.inbound_conv(x, inbound_edges)
            if self.bidirectional:
                outbound_x = self.outbound_conv(x, outbound_edges)
        else:
            inbound_x = self.inbound_conv(x, inbound_edges, edge_weight)
            if self.bidirectional:
                outbound_x = self.outbound_conv(x, outbound_edges, edge_weight)
        # concatenate the convolutions of the inbound and outbound edges
        if self.bidirectional:
            x_out = torch.cat([inbound_x, outbound_x], dim=1)
        else:
            x_out = inbound_x
        return x_out

