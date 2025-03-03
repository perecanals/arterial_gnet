from arterial_gnet.models.arterial_gnet import ArterialGNet
from arterial_gnet.models.graph_transformer import GraphTransformerNet
from arterial_gnet.models.transformer import TransformerNet
from arterial_gnet.models.GAT import GATv2Net
from arterial_gnet.models.HGPSL import HGPSLModel

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

    # Initialize model with the corresponding parameters
    if args.base_model_name == "ArterialGNet":
        model_name = "{}_bs-{}_te-{}_hc-{}_hcd-{}_op-{}_lr-{}_lrs-{}_warmup-{}_ngl-{}_nsl-{}_ndl-{}_nol-{}_ah-{}_agg-{}_drop-{}_r-{}_concat-{}_skip-{}_cl-{}_alpha-{}_wl-{}_os-{}_rs-{}_trs-{}".format(args.base_model_name, args.batch_size, \
                    args.total_epochs, args.hidden_channels, args.hidden_channels_dense, args.optimizer, args.learning_rate, args.lr_scheduler, args.warmup_steps, \
                        args.num_global_layers, args.num_segment_layers, args.num_dense_layers, args.num_out_layers, args.attn_heads, args.aggregation, \
                            args.dropout, args.radius, args.concat, args.use_skip, args.class_loss, args.alpha, args.weighted_loss, args.oversampling, args.random_state, args.test_random_state)
        if args.is_classification:
            model_name += "_class"
        if args.tag is not None:
            model_name += "_tag-{}".format(args.tag)

        print("------------------------------------------------ Model information")
        print(f"Training model:                   {args.base_model_name}")
        print(f"Hidden channels:                  {args.hidden_channels}")
        print(f"Hidden channels (dense layer):    {args.hidden_channels_dense}")
        print(f"Number of global layers:          {args.num_global_layers}")
        print(f"Number of segment layers:         {args.num_segment_layers}")
        print(f"Number of dense layers:           {args.num_dense_layers}")
        print(f"Number of output layers:          {args.num_out_layers}")
        print(f"Number of attention heads:        {args.attn_heads}")
        print(f"Aggregation method:               {args.aggregation}")
        print(f"Dropout:                          {args.dropout}")
        print(f"Radius [mm] (connectivity):       {args.radius}")
        print(f"Concatenate output of GAT layers: {args.concat}")

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
            concat=args.concat,
            is_classification=args.is_classification,
            use_skip=args.use_skip,
            combined_loss=True if args.class_loss == "combined" else False
            ).to(device)
    elif args.base_model_name == "GraphTransformer":
        model_name = "{}_bs-{}_te-{}_hc-{}_op-{}_lr-{}_lrs-{}_warmup-{}_nl-{}_nol-{}_ah-{}_agg-{}_drop-{}_r-{}_concat-{}_lpe-{}_ped-{}_os-{}_rs-{}_trs-{}".format(args.base_model_name, args.batch_size, \
                    args.total_epochs, args.hidden_channels, args.optimizer, args.learning_rate, args.lr_scheduler, args.warmup_steps, \
                        args.num_dense_layers, args.num_out_layers, args.attn_heads, args.aggregation, \
                            args.dropout, args.radius, args.concat, args.use_lap_pos_enc, args.pos_enc_dim, \
                                args.oversampling, args.random_state, args.test_random_state)
        if args.is_classification:
            model_name += "_class"
        if args.tag is not None:
            model_name += "_tag-{}".format(args.tag)

        print("------------------------------------------------ Model information")
        print(f"Training model:                    {args.base_model_name}")
        print(f"Hidden channels:                   {args.hidden_channels}")
        print(f"Number of transformer layers:      {args.num_dense_layers}")
        print(f"Number of output layers:           {args.num_out_layers}")
        print(f"Number of attention heads:         {args.attn_heads}")
        print(f"Aggregation method:                {args.aggregation}")
        print(f"Dropout:                           {args.dropout}")
        print(f"Radius [mm] (connectivity):        {args.radius}")
        print(f"Concatenate output of MHA layers:  {args.concat}")
        print(f"Use Laplacian positional encoding: {args.use_lap_pos_enc}")
        print(f"Positional encoding dimension:     {args.pos_enc_dim}")

        model = GraphTransformerNet(
            dense_node_in_dim=dataset_description["num_dense_node_features"],
            hidden_dim=args.hidden_channels,
            out_dim=2 if args.is_classification else 1,
            num_layers=args.num_dense_layers,
            num_out_layers=args.num_out_layers,
            attn_heads=args.attn_heads,
            aggregation=args.aggregation,
            dropout=args.dropout,
            concat=args.concat,
            use_pos_enc=args.use_lap_pos_enc,
            pos_enc_dim=args.pos_enc_dim
            ).to(device)
    
    elif args.base_model_name == "Transformer":
        model_name = "{}_bs-{}_te-{}_hc-{}_op-{}_lr-{}_lrs-{}_warmup-{}_nl-{}_nol-{}_ah-{}_agg-{}_drop-{}_os-{}_rs-{}_trs-{}".format(args.base_model_name, args.batch_size, \
                    args.total_epochs, args.hidden_channels, args.optimizer, args.learning_rate, args.lr_scheduler, args.warmup_steps, \
                        args.num_dense_layers, args.num_out_layers, args.attn_heads, args.aggregation, \
                            args.dropout, args.oversampling, args.random_state, args.test_random_state)
        if args.is_classification:
            model_name += "_class"
        if args.tag is not None:
            model_name += "_tag-{}".format(args.tag)

        print("------------------------------------------------ Model information")
        print(f"Training model:                    {args.base_model_name}")
        print(f"Hidden channels:                   {args.hidden_channels}")
        print(f"Number of transformer layers:      {args.num_dense_layers}")
        print(f"Number of output layers:           {args.num_out_layers}")
        print(f"Number of attention heads:         {args.attn_heads}")
        print(f"Aggregation method:                {args.aggregation}")
        print(f"Dropout:                           {args.dropout}")

        model = TransformerNet(
            in_dim=dataset_description["num_dense_node_features"],
            hidden_dim=args.hidden_channels,
            out_dim=2 if args.is_classification else 1,
            num_layers=args.num_dense_layers,
            num_out_layers=args.num_out_layers,
            attn_heads=args.attn_heads,
            aggregation=args.aggregation,
            dropout=args.dropout,
            pos_enc_dim=args.pos_enc_dim
            ).to(device)
    elif args.base_model_name == "GAT":
        model_name = "{}_bs-{}_te-{}_hc-{}_op-{}_lr-{}_lrs-{}_warmup-{}_nl-{}_nol-{}_ah-{}_agg-{}_drop-{}_r-{}_concat-{}_next-{}_skip-{}_os-{}_rs-{}_trs-{}".format(args.base_model_name, args.batch_size, \
                    args.total_epochs, args.hidden_channels, args.optimizer, args.learning_rate, args.lr_scheduler, args.warmup_steps, \
                        args.num_dense_layers, args.num_out_layers, args.attn_heads, args.aggregation, \
                            args.dropout, args.radius, args.concat, args.next_layer, args.use_skip, args.oversampling, args.random_state, args.test_random_state)
        if args.is_classification:
            model_name += "_class"
        if args.tag is not None:
            model_name += "_tag-{}".format(args.tag)

        model = GATv2Net(
            node_in_dim=dataset_description["num_dense_node_features"],
            edge_in_dim=None,
            hidden_dim=args.hidden_channels,
            out_dim=2 if args.is_classification else 1,
            num_layers=args.num_dense_layers,
            num_out_layers=args.num_out_layers,
            attn_heads=args.attn_heads,
            aggregation=args.aggregation,
            dropout=args.dropout,
            concat=args.concat,
            use_skip=args.use_skip,
            next_layer=args.next_layer
        ).to(device)

    elif args.base_model_name == "HGPSL":
        model_name = "{}_bs-{}_te-{}_hc-{}_op-{}_lr-{}_lrs-{}_warmup-{}_nl-{}_nol-{}_ah-{}_agg-{}_drop-{}_r-{}_pr-{}_sn-{}_sa-{}_sl-{}_lamb-{}_os-{}_rs-{}_trs-{}".format(
                    args.base_model_name, args.batch_size,
                        args.total_epochs, args.hidden_channels, args.optimizer, args.learning_rate, args.lr_scheduler, args.warmup_steps, \
                            args.num_dense_layers, args.num_out_layers, args.attn_heads, args.aggregation,
                                args.dropout, args.radius, args.pooling_ratio, args.sample_neighbor, 
                                    args.sparse_attention, args.structure_learning, args.lamb,
                                        args.oversampling, args.random_state, args.test_random_state)
        if args.is_classification:
            model_name += "_class"
        if args.tag is not None:
            model_name += "_tag-{}".format(args.tag)

        model = HGPSLModel(
            node_in_dim=dataset_description["num_dense_node_features"],
            hidden_dim=args.hidden_channels,
            num_classes=2 if args.is_classification else 1,
            pooling_ratio=args.pooling_ratio,
            dropout=args.dropout,
            sample_neighbor=args.sample_neighbor,
            sparse_attention=args.sparse_attention,
            structure_learning=args.structure_learning,
            lamb=args.lamb
        ).to(device)

    print(f"Number of parameters:             {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
        
    return model, model_name

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool, global_max_pool, BatchNorm

class GATv2Layer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=None, dropout_rate=0.2, attn_heads=1, concat=False):
        super(GATv2Layer, self).__init__()
        self.concat = concat
        self.edge_dim = edge_dim
        self.attn_heads = attn_heads

        self.conv = GATv2Conv(in_channels=in_channels, out_channels=out_channels, edge_dim=self.edge_dim, heads=self.attn_heads, concat=self.concat)
        self.bn1 = BatchNorm(out_channels * self.attn_heads if self.concat else out_channels)
        self.ff = nn.Linear(out_channels * self.attn_heads if self.concat else out_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr=None):
        if self.edge_dim is not None:
            x = self.conv(x, edge_index, edge_attr)
            attention_weights = None
        else:
            x, attention_weights = self.conv(x, edge_index, return_attention_weights=True)
        
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn1(x)
        
        # Apply ff layer in both cases
        x = self.ff(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn2(x)
        
        x = self.dropout(x)
        return x, attention_weights

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
            concat=False,
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
        self.concat = concat
        self.is_classification = is_classification
        self.attention_weights = None

        if self.num_global_layers > 0:
            self.global_path = torch.nn.ModuleList()
            for idx in range(self.num_global_layers):
                self.global_path.append(MLPLayer(self.global_in_dim if idx == 0 else self.hidden_dim, self.hidden_dim, dropout_rate=self.dropout))
        else:
            self.global_path = None
        
        if self.num_segment_layers > 0:
            self.segment_path = torch.nn.ModuleList()
            for idx in range(self.num_segment_layers):
                in_channels = self.segment_node_in_dim if idx == 0 else self.hidden_dim
                self.segment_path.append(GATv2Layer(in_channels, self.hidden_dim, edge_dim=self.segment_edge_in_dim, dropout_rate=self.dropout, attn_heads=self.attn_heads, concat=self.concat))
        else:
            self.segment_path = None
        
        if self.num_dense_layers > 0:
            self.dense_path = torch.nn.ModuleList()
            for idx in range(self.num_dense_layers):
                in_channels = self.dense_node_in_dim if idx == 0 else self.hidden_dim_dense
                self.dense_path.append(GATv2Layer(in_channels, self.hidden_dim_dense, dropout_rate=self.dropout, attn_heads=self.attn_heads, concat=self.concat))
        else:
            self.dense_path = None
        
        self.output_path = torch.nn.ModuleList()
        if self.num_out_layers > 1:
            for idx in range(num_out_layers - 1):
                self.output_path.append(MLPLayer(self.hidden_dim * sum([self.num_global_layers > 0, self.num_segment_layers > 0]) + self.hidden_dim_dense * sum([self.num_dense_layers > 0]) if idx == 0 else hidden_dim, self.hidden_dim, dropout_rate=self.dropout))
            self.output_path.append(nn.Linear(self.hidden_dim, self.out_dim))
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
                segment_x, _ = layer(segment_x, segment_data.edge_index, segment_data.edge_attr)
            segment_x = self.aggregation(segment_x, segment_data.batch)
        else:
            segment_x = None

        # Process dense_graph
        dense_x = dense_data.x
        if self.dense_path is not None:
            for idx, layer in enumerate(self.dense_path):
                if idx == 0:  # Only extract attention weights from the first layer
                    dense_x, attention_weights = layer(dense_x, dense_data.edge_index)
                    self.attention_weights = attention_weights
                else:
                    dense_x, _ = layer(dense_x, dense_data.edge_index)
            dense_x = self.aggregation(dense_x, dense_data.batch)
        else:
            dense_x = None

        # Concatenate all features
        out = torch.cat([x for x in [global_features, segment_x, dense_x] if x is not None], dim=1)

        # Final output layers
        for layer in self.output_path:
            out = layer(out)

        return out, self.attention_weights