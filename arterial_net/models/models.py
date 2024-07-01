import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool, global_max_pool, TopKPooling, BatchNorm

def get_model(args, dataset_description, device = "cpu"):
    """
    Define model for the dataset.

    Paramaters
    ----------
    args : argparse.Namespace
        Arguments. Contains:
            - base_model_name : string
                Name of the base model to use.
            - batch_size : int
                Batch size.
            - hidden_channels : int
                Number of hidden channels.
            - depth : int
                Depth of the graph U-Net.
    dataset_description : dict
        Dictionary containing dataset description. Contains:
            - num_edge_features : int
                Number of edge features.
            - num_edge_classes : int
                Number of edge classes.
    device : string, optional
        Device to use for training. The default is "cpu".

    Returns
    -------
    model : torch.nn.Module
        Model to train.
    model_name : string
        String with the model name, where data (train and test) will be 
        in os.path.join(root, "models", model_name).
    """
    model_name = "{}_bs-{}_te-{}_hc-{}_hcd-{}_op-{}_lr-{}_lrs-{}_ngl-{}_nsl-{}_ndl-{}_nol-{}_ah-{}_agg-{}_drop-{}_wl-{}_os-{}_rs-{}_trs-{}".format(args.base_model_name, args.batch_size, \
                                                                           args.total_epochs, args.hidden_channels, args.hidden_channels_dense, args.optimizer, args.learning_rate, args.lr_scheduler, \
                                                                           args.num_global_layers, args.num_segment_layers, args.num_dense_layers, args.num_out_layers, args.attn_heads, args.aggregation, \
                                                                                args.dropout, args.weighted_loss, args.oversampling, args.random_state, args.test_random_state)
    if args.is_classification:
        model_name += "_class"
    if args.tag is not None:
        model_name += "_tag-{}".format(args.tag)

    print("------------------------------------------------ Model information")
    print(f"Training model:                 {args.base_model_name}")
    print(f"Hidden channels:                {args.hidden_channels}")
    print(f"Hidden channels (dense layer):  {args.hidden_channels_dense}")
    print(f"Number of global layers:        {args.num_global_layers}")
    print(f"Number of segment layers:       {args.num_segment_layers}")
    print(f"Number of dense layers:         {args.num_dense_layers}")
    print(f"Number of output layers:        {args.num_out_layers}")
    print(f"Number of attention heads:      {args.attn_heads}")
    print(f"Aggregation method:             {args.aggregation}")
    print(f"Dropout:                        {args.dropout}")
    print(f"Oversampling:                   {args.oversampling}")

    # Initialize model with the corresponding parameters
    if args.base_model_name == "ArterialGNet":
        model = ArterialGNet(
            global_in_dim=dataset_description["num_global_features"], 
            segment_node_in_dim=dataset_description["num_segment_node_features"], 
            segment_edge_in_dim=dataset_description["num_segment_edge_features"],
            dense_node_in_dim=dataset_description["num_dense_node_features"],
            hidden_dim=args.hidden_channels, 
            hidden_dim_dense=args.hidden_channels_dense,
            out_dim=2 if args.is_classification else 1,
            num_global_layers=args.num_global_layers,
            num_segment_layers=args.num_segment_layers, 
            num_dense_layers=args.num_dense_layers,
            num_out_layers=args.num_out_layers,
            attn_heads=args.attn_heads,
            aggregation=args.aggregation,
            dropout=args.dropout,
            is_classification=args.is_classification
            ).to(device)
        
    return model, model_name

class GATv2Layer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=None, dropout_rate=0.2, attn_heads=1):
        super(GATv2Layer, self).__init__()
        self.conv = GATv2Conv(in_channels=in_channels, out_channels=out_channels, edge_dim=edge_dim, heads=attn_heads)
        # self.conv = GATv2Conv(in_channels=in_channels, out_channels=out_channels, edge_dim=edge_dim, heads=attn_heads, return_attention_weights=True)
        # What does this return when forwarded?
        self.bn = BatchNorm(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.edge_dim = edge_dim

    def forward(self, x, edge_index, edge_attr=None):
        if self.edge_dim is not None:
            x = self.conv(x, edge_index, edge_attr)
        else:
            x = self.conv(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class MLPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class ArterialGNet(nn.Module):
    """
    ArterialGNet.
    
    """
    def __init__(
            self,
            global_in_dim,
            segment_node_in_dim,
            segment_edge_in_dim,
            dense_node_in_dim,
            hidden_dim=8,
            hidden_dim_dense=8,
            out_dim=2,
            num_global_layers=1,
            num_segment_layers=1,
            num_dense_layers=1,
            num_out_layers=1,
            attn_heads=1,
            aggregation="mean",
            dropout=0.2,
            is_classification=True):
        super(ArterialGNet, self).__init__()
        self.global_in_dim = global_in_dim
        self.segment_node_in_dim = segment_node_in_dim
        self.segment_edge_in_dim = segment_edge_in_dim
        self.dense_node_in_dim = dense_node_in_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_dense = hidden_dim_dense
        self.out_dim = out_dim
        self.num_global_layers = num_global_layers
        self.num_segment_layers = num_segment_layers
        self.num_dense_layers = num_dense_layers
        self.num_out_layers = num_out_layers
        self.attn_heads = attn_heads
        self.aggregation_ = aggregation 
        self.dropout = dropout
        self.is_classification = is_classification

        if self.num_global_layers > 0:
            self.global_path = torch.nn.ModuleList()
            for idx in range(self.num_global_layers):
                self.global_path.append(MLPLayer(self.global_in_dim if idx == 0 else self.hidden_dim, self.hidden_dim, dropout_rate=self.dropout))
        else:
            self.global_path = None
        
        if self.num_segment_layers > 0:
            self.segment_path = torch.nn.ModuleList()
            for idx in range(self.num_segment_layers):
                self.segment_path.append(GATv2Layer(self.segment_node_in_dim if idx == 0 else self.hidden_dim, self.hidden_dim, edge_dim=self.segment_edge_in_dim, dropout_rate=self.dropout, attn_heads=self.attn_heads))
        else:
            self.segment_path = None
        
        if self.num_dense_layers > 0:
            self.dense_path = torch.nn.ModuleList()
            for idx in range(self.num_dense_layers):
                self.dense_path.append(GATv2Layer(self.dense_node_in_dim if idx == 0 else self.hidden_dim_dense, self.hidden_dim_dense, dropout_rate=self.dropout, attn_heads=self.attn_heads))
        else:
            self.dense_path = None
        
        self.output_path = torch.nn.ModuleList()
        if self.num_out_layers > 1:
            for idx in range(num_out_layers - 1):
                self.output_path.append(MLPLayer(self.hidden_dim * sum([self.num_global_layers > 0, self.num_segment_layers > 0]) + self.hidden_dim_dense * sum([self.num_dense_layers > 0]) if idx == 0 else hidden_dim, self.hidden_dim, dropout_rate=self.dropout))
            self.output_path.append(nn.Linear(self.hidden_dir, self.out_dim))
        else:
            self.output_path.append(nn.Linear(self.hidden_dim * sum([self.num_global_layers > 0, self.num_segment_layers > 0]) + self.hidden_dim_dense * sum([self.num_dense_layers > 0]), self.out_dim))
        if self.is_classification:
            self.output_path.append(nn.Softmax(dim=1))

        if self.aggregation_ == "mean":
            self.aggregation = global_mean_pool
        elif self.aggregation_ == "max":
            self.aggregation = global_max_pool
        elif self.aggregation_ == "add":
            self.aggregation = global_add_pool
        # elif self.aggregation_ == "topk":
        #     self.aggregation = TopKPooling(self.hidden_dim, ratio=0.5)

    def forward(self, data):
        global_data, segment_data, dense_data = data.global_data, data.segment_data, data.dense_data
        
        # Process global features
        if self.global_path is not None:
            global_features = global_data
            for layer in self.global_path:
                global_features = layer(global_features)
        else:
            global_features = None

        # Process segment_data
        segment_x = segment_data.x
        if self.segment_path is not None:
            for layer in self.segment_path:
                segment_x = layer(segment_x, segment_data.edge_index, segment_data.edge_attr)
            segment_x = self.aggregation(segment_x, segment_data.batch)
        else:
            segment_x = None

        # Process dense_graph
        dense_x = dense_data.x
        if self.dense_path is not None:
            for layer in self.dense_path:
                dense_x = layer(dense_x, dense_data.edge_index)
            dense_x = self.aggregation(dense_x, dense_data.batch)
        else:
            dense_x = None

        # Concatenate all features
        out = torch.cat([x for x in [global_features, segment_x, dense_x] if x is not None], dim=1)

        # Final output layers
        for layer in self.output_path:
            out = layer(out)

        return out