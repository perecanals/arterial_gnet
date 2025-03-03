import pickle

import torch

from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

import scipy.sparse as sp

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
def z_score_normalization(features, mean, std):
    return (features - mean) / std

def min_max_normalization(features, min, max):
    return (features - min) / (max - min)

def mean_centering_normalization(features, mean):
    return features - mean

def normalize_vector(features):
    return features / torch.norm(features, dim=1, keepdim=True)

def laplacian_positional_encoding(edge_index, num_nodes, pos_enc_dim=8):
    """
    Graph positional encoding v/ Laplacian eigenvectors for PyTorch Geometric
    
    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity in COO format with shape [2, num_edges]
    num_nodes : int
        Number of nodes in the graph
    pos_enc_dim : int
        Desired dimension of positional encoding
    
    Returns
    -------
    torch.Tensor
        Laplacian positional encoding matrix with shape [num_nodes, pos_enc_dim]
    """
    # Get normalized Laplacian
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
    
    # Convert to scipy sparse matrix
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    
    # Eigenvectors with scipy
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', return_eigenvectors=True)
    EigVec = EigVec.real
    
    # Sort and keep top K eigenvectors
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], EigVec[:, idx]
    
    # Discard the first eigenvector/value
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()
    
    return lap_pos_enc

def sinusoidal_positional_encoding(num_nodes, pos_enc_dim):  
    """
    Sinusoidal positional encoding.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    pos_enc_dim : int
        Dimension of the positional encoding.

    Returns
    -------
    torch.Tensor
        Positional encoding matrix with shape [num_nodes, dim].
    """
    pos = torch.arange(0, num_nodes).unsqueeze(1)
    i = torch.arange(0, pos_enc_dim // 2).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * i) / pos_enc_dim)
    pos_encoding = torch.zeros(num_nodes, pos_enc_dim)
    pos_encoding[:, 0::2] = torch.sin(pos * angle_rates)
    pos_encoding[:, 1::2] = torch.cos(pos * angle_rates)
    return pos_encoding
    