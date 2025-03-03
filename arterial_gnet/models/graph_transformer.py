import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax

class MultiHeadAttention(MessagePassing):
    def __init__(self, in_dim, head_dim, num_heads, use_bias=True, concat=True):
        super().__init__(aggr='add', node_dim=0) # aggr is add because we want to sum the attention scores
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.concat = concat
        self.scaling = self.head_dim ** -0.5

        self.Q = nn.Linear(in_dim, head_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, head_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, head_dim * num_heads, bias=use_bias)

    def forward(self, x, edge_index):
        q = self.Q(x).view(-1, self.num_heads, self.head_dim)
        k = self.K(x).view(-1, self.num_heads, self.head_dim)
        v = self.V(x).view(-1, self.num_heads, self.head_dim)

        return self.propagate(edge_index, q=q, k=k, v=v, size=None)

    def message(self, q_i, k_j, v_j, edge_index, size):
        attention = (q_i * k_j).sum(dim=-1) * self.scaling
        attention = softmax(attention, edge_index[1], num_nodes=size[1])
        return v_j * attention.unsqueeze(-1)

    def update(self, aggr_out):
        if self.concat:
            return aggr_out.view(-1, self.num_heads * self.head_dim)
        else:
            return aggr_out.mean(dim=1)

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, concat=True, use_bias=True, dropout=0.2, residual=True, layer_norm=False):
        super().__init__()

        self.dropout = dropout
        self.residual = residual

        self.attention = MultiHeadAttention(in_dim, out_dim // num_heads if concat else out_dim, num_heads, concat=concat, use_bias=use_bias)
        self.O = nn.Linear(out_dim, out_dim)
        self.norm_1 = nn.LayerNorm(out_dim) if layer_norm else nn.BatchNorm1d(out_dim)
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)
        self.norm_2 = nn.LayerNorm(out_dim) if layer_norm else nn.BatchNorm1d(out_dim)

    def forward(self, h, edge_index):
        h_in1 = h # save input for residual connection

        h = self.attention(h, edge_index)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h

        h = self.norm_1(h)

        h_in2 = h # save input for residual connection

        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h   

        h = self.norm_2(h)

        return h
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.num_heads, self.residual)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_out_layers = 2):
        super().__init__()
        list_linear_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(num_out_layers - 1)]
        list_linear_layers.append(nn.Linear(input_dim // 2 ** (num_out_layers - 1), output_dim, bias=True))
        self.linear_layers = nn.ModuleList(list_linear_layers)
        self.num_out_layers = num_out_layers
        
    def forward(self, x):
        for l in range(self.num_out_layers - 1):
            x = self.linear_layers[l](x)
            x = F.relu(x)
        return self.linear_layers[self.num_out_layers - 1](x)
    
def pna_aggregation(x, batch_index):
    agg_mean = global_mean_pool(x, batch_index)
    agg_max = global_max_pool(x, batch_index)
    agg_add = global_add_pool(x, batch_index)
    h = torch.cat([agg_mean, agg_max, agg_add], dim=1).to(x.device)

    return h

class GraphTransformerNet(nn.Module):
    def __init__(
            self,
            dense_node_in_dim=25,
            hidden_dim=32,
            out_dim=2,
            num_layers=1,
            num_out_layers=1,
            attn_heads=8,
            aggregation="mean",
            dropout=0.2,
            concat=True,
            use_bias=True,
            residual=True,
            layer_norm=False,
            use_pos_enc=True,
            pos_enc_dim=8
    ):
        super().__init__()
        self.use_pos_enc = use_pos_enc
        self.aggregation_type = aggregation
        assert pos_enc_dim > 0, "pos_enc_dim must be greater than 0"

        # laplacian positional encoding (tensor should be provided by the user)
        if use_pos_enc:
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        # input encoding
        self.embedding_h = nn.Linear(dense_node_in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphTransformerLayer(hidden_dim, hidden_dim, attn_heads, concat, use_bias, dropout, residual, layer_norm))

        if self.aggregation_type == "mean":
            self.aggregation = global_mean_pool
        elif self.aggregation_type == "max":
            self.aggregation = global_max_pool
        elif self.aggregation_type == "add":
            self.aggregation = global_add_pool
        elif self.aggregation_type == "pna":
            self.aggregation = pna_aggregation

        self.out_mlp = MLP(hidden_dim * 3 if self.aggregation_type == "pna" else hidden_dim, out_dim, num_out_layers)

    def forward(self, batch):
        x, edge_index, batch_index, pos_enc = batch.x, batch.edge_index, batch.batch, batch.pos_enc

        h = self.embedding_h(x)
        h = self.in_feat_dropout(h)

        # laplacian positional encoding
        if self.use_pos_enc:
            h_pos_enc = self.embedding_pos_enc(pos_enc)
            h = h + h_pos_enc

        for layer in self.layers:
            h = layer(h, edge_index)

        # aggregate node embeddings
        if isinstance(self.aggregation, list):
            h = torch.cat([agg(h, batch_index) for agg in self.aggregation], dim=1)
        else:
            h = self.aggregation(h, batch_index)

        out = self.out_mlp(h)

        out = F.softmax(out, dim=1)

        return out.unsqueeze(0)
