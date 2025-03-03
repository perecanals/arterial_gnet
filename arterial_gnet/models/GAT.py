import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool, global_max_pool, LayerNorm, BatchNorm

class GATv2Layer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=None, dropout_rate=0.2, attn_heads=1, concat=False, use_residual=False):
        super(GATv2Layer, self).__init__()
        self.concat = concat
        self.edge_dim = edge_dim
        self.attn_heads = attn_heads
        self.use_residual = use_residual

        self.conv = GATv2Conv(in_channels=in_channels, out_channels=out_channels, edge_dim=self.edge_dim, heads=self.attn_heads, concat=self.concat)
        self.bn1 = BatchNorm(out_channels * self.attn_heads if self.concat else out_channels)
        self.ff = nn.Linear(out_channels * self.attn_heads if self.concat else out_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout_rate)

        if self.use_residual:
            self.skip_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        identity = x
        
        if self.edge_dim is not None:
            x = self.conv(x, edge_index, edge_attr)
            attention_weights = None
        else:
            x, attention_weights = self.conv(x, edge_index, return_attention_weights=True)
        
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn1(x)
        
        x = self.ff(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn2(x)
        
        x = self.dropout(x)

        if self.use_residual:
            x = x + self.skip_proj(identity)

        return x, attention_weights

class GATv2LayerNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=None, dropout_rate=0.2, attn_heads=1, concat=False, use_residual=False):
        super(GATv2LayerNeXt, self).__init__()
        self.concat = concat
        self.edge_dim = edge_dim
        self.attn_heads = attn_heads
        self.use_residual = use_residual

        self.conv = GATv2Conv(in_channels=in_channels, out_channels=out_channels, edge_dim=self.edge_dim, heads=self.attn_heads, concat=self.concat)
        self.norm = LayerNorm(out_channels)
        self.ff1 = nn.Linear(out_channels * self.attn_heads if self.concat else out_channels, 2 * out_channels)
        self.act = nn.GELU()
        self.ff2 = nn.Linear(2 * out_channels, out_channels)

        if self.use_residual:
            self.skip_proj = nn.Linear(in_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr=None):
        x0 = x
        
        if self.edge_dim is not None:
            x = self.conv(x, edge_index, edge_attr)
            attention_weights = None
        else:
            x, attention_weights = self.conv(x, edge_index, return_attention_weights=True)
        
        x = self.norm(x)
        
        x = self.ff1(x)
        x = self.act(x)
        x = self.ff2(x)
        
        x = self.dropout(x)

        if self.use_residual:
            x0 = self.skip_proj(x0)
            x = x + x0

        return x, attention_weights

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

class GATv2Net(nn.Module):
    """
    GATv2Net.
    
    """
    def __init__(
            self,
            node_in_dim,
            edge_in_dim=None,
            hidden_dim=32,
            out_dim=1,
            num_layers=1,
            num_out_layers=1,
            attn_heads=8,
            aggregation="mean",
            dropout=0.2,
            concat=False,
            is_classification=True,
            use_residual=False,
            next_layer=True
    ):
        super(GATv2Net, self).__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_out_layers = num_out_layers
        self.attn_heads = attn_heads
        self.aggregation_type = aggregation 
        self.dropout = dropout
        self.concat = concat
        self.is_classification = is_classification
        self.attention_weights = None
        self.use_residual = use_residual
        self.attention_weights = None

        # self.embedding_h = nn.Linear(self.node_in_dim, self.hidden_dim)
        # self.in_feat_dropout = nn.Dropout(dropout)
        
        self.convs = torch.nn.ModuleList()
        for idx in range(self.num_layers):
            if next_layer:
                self.convs.append(GATv2LayerNeXt(self.hidden_dim if idx > 0 else self.node_in_dim, self.hidden_dim, edge_dim=self.edge_in_dim, dropout_rate=self.dropout, attn_heads=self.attn_heads, concat=self.concat, use_residual=self.use_residual))
            else:
                self.convs.append(GATv2Layer(self.hidden_dim if idx > 0 else self.node_in_dim, self.hidden_dim, edge_dim=self.edge_in_dim, dropout_rate=self.dropout, attn_heads=self.attn_heads, concat=self.concat, use_residual=self.use_residual))

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
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch

        # x = self.embedding_h(x)
        # x = self.in_feat_dropout(x)

        for conv in self.convs:
            x, self.attention_weights = conv(x, edge_index)

        x = self.aggregation(x, batch_index)

        out = self.out_mlp(x)

        out = F.softmax(out, dim=1)

        return out.unsqueeze(0)