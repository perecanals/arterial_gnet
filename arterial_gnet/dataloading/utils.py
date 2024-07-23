import pickle

import torch

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