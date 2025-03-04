import torch
from torch_geometric.transforms import Compose, ToDevice, RadiusGraph, RandomJitter, RandomFlip, RandomScale, RandomRotate

class DenseRadiusGraph(object):
    def __init__(self, r, loop=True):
        self.transform = RadiusGraph(r=r, loop=loop)

    def __call__(self, data):
        if hasattr(data, 'dense_data'):
            dense_data = data.dense_data
            original_edge_index = dense_data.edge_index
            new_dense_data = self.transform(dense_data)
            new_edge_index = new_dense_data.edge_index

            combined_edge_index = torch.cat([original_edge_index, new_edge_index], dim=1)

            # Transpose to treat each edge as a row
            transposed_edges = combined_edge_index.T
            # Use torch.unique to find unique rows (edges)
            unique_edges = torch.unique(transposed_edges, dim=0)
            # Transpose back to the original edge_index format
            combined_edge_index = unique_edges.T

            dense_data.edge_index = combined_edge_index
            data.dense_data = dense_data
        else:
            original_edge_index = data.edge_index
            new_data = self.transform(data)
            new_edge_index = new_data.edge_index

            combined_edge_index = torch.cat([original_edge_index, new_edge_index], dim=1)

            # Transpose to treat each edge as a row
            transposed_edges = combined_edge_index.T
            # Use torch.unique to find unique rows (edges)
            unique_edges = torch.unique(transposed_edges, dim=0)
            # Transpose back to the original edge_index format
            combined_edge_index = unique_edges.T

            data.edge_index = combined_edge_index

        return data

class ApplyPositionTransformToFeatures(object):
    def __init__(self):
        pass

    def __call__(self, data):
        self.update_features(data)
        return data

    def update_features(self, graph_data):
        if hasattr(graph_data, 'pos') and hasattr(graph_data, 'x'):
            # Calculate the scale factor
            original_norm = torch.norm(graph_data.x[:, :3], dim=1, keepdim=True)
            new_norm = torch.norm(graph_data.pos, dim=1, keepdim=True)
            scale_factor = original_norm / new_norm

            # Update the first 3 features of x to match the new positions
            graph_data.x[:, :3] = graph_data.pos * scale_factor

        return graph_data

def get_transforms(device, args):
    """
    Define transforms for the dataset.

    Parameters
    ----------
    device : string
        Device to use for training.
    
    Returns
    -------
    pre_transform : torch_geometric.transforms.Compose
        Pre-transforms to apply to the dataset.
    train_transform : torch_geometric.transforms.Compose
        Dynamic transforms to apply for data augmentation during training.
    test_transform : torch_geometric.transforms.Compose
        Dynamic transforms to apply during testing.
    """
    # Define transforms
    if args.radius > 0:
        pre_transform = Compose([
            DenseRadiusGraph(r=args.radius, loop=True),
        ])
        train_transform = Compose([
            # RandomJitter(2),
            # # RandomFlip(0.2),
            # RandomScale([0.9, 1.1]),
            # RandomRotate(15),
            # ApplyPositionTransformToFeatures(),
            ToDevice(device)
        ])
        test_transform = Compose([
            DenseRadiusGraph(r=args.radius, loop=True),
            ToDevice(device)
        ])
    else: 
        pre_transform = Compose([
        ])
        train_transform = Compose([
            ToDevice(device)
        ])
        test_transform = Compose([
            ToDevice(device)
        ])

    return pre_transform, train_transform, test_transform