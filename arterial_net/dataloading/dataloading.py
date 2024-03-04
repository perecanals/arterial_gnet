import torch
from torch_geometric.data import Data, Batch

import os

from sklearn.model_selection import train_test_split, StratifiedKFold

from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader # This does not allow you to override the collate_fn

from arterial_maps.dataloading.dataset import ArterialMapsDataset
from arterial_maps.dataloading.utils import load_pickle

def get_data_loaders(root, args, fold = None, pre_transform = None, train_transform = None, test_transform = None):
    """
    Define train, validation and test data loaders for the dataset.

    Parameters
    ----------
    root : string or path-like object
        Path to root folder.
    args : argparse.Namespace
        Arguments.
    fold : int
        Fold for cross validation. By default is None.
    pre_transform : torch_geometric.transforms.Compose, optional
        Pre-transforms to apply to the dataset. The default is None.
    train_transform : torch_geometric.transforms.Compose, optional
        Dynamic transforms to apply for data augmentation during training, optional
    test_transforms : torch_geometric.transforms.Compose, optional
        Dynamic transforms to apply during testing, optional

    Returns
    -------
    t
    """
    # Define datasets . First make division of train + val and test
    dataset_filenames = [f for f in os.listdir(os.path.join(root, "raw")) if f.endswith(".pickle")]
    y_class = [load_pickle(os.path.join(root, "raw", f))["classification"] for f in dataset_filenames]
    if not args.test_size == args.val_size == 0:
        train_val_filenames, test_filenames, y_train_val, y_test = train_test_split(dataset_filenames, y_class, test_size = args.test_size, random_state = args.random_state, stratify = y_class)
        # Now, depending on the folds being specified or not, perform cross validation (select the k-fold corresponding to "fold", with k being "args.folds") or not
        if fold is not None:
            kf = StratifiedKFold(n_splits = args.folds, shuffle = True, random_state = args.random_state)
            folds = list(kf.split(train_val_filenames, y_train_val))
            train_filenames = [train_val_filenames[i] for i in folds[fold][0]]
            val_filenames = [train_val_filenames[i] for i in folds[fold][1]]    
        else:
            train_filenames, val_filenames = train_test_split(train_val_filenames, test_size = args.val_size, random_state = args.random_state, stratify=y_train_val)
        # Now define dataset classes
        train_dataset = ArterialMapsDataset(root, raw_file_names_list = train_filenames, pre_transform = pre_transform, transform = train_transform)
        val_dataset = ArterialMapsDataset(root, raw_file_names_list = val_filenames, pre_transform = pre_transform, transform = test_transform)
        test_dataset = ArterialMapsDataset(root, raw_file_names_list = test_filenames, pre_transform = pre_transform, transform = test_transform)
    else:
        print("Training with the whole dataset (ignore validation and test results)\n")
        train_dataset = ArterialMapsDataset(root, pre_transform = pre_transform, transform = train_transform)
        val_dataset = ArterialMapsDataset(root, raw_file_names_list = dataset_filenames[:2], pre_transform = pre_transform, transform = test_transform)
        test_dataset = ArterialMapsDataset(root, raw_file_names_list = dataset_filenames[:2], pre_transform = pre_transform, transform = test_transform)

    print("------------------------------------------------ Dataset information")
    print("Total number of samples:        {}".format(len(dataset_filenames)))
    if fold is not None:    
        print(f"Fold:                           {fold}")
    print("Number of training samples:     {} ({:.2f}%)".format(len(train_dataset), 100 * len(train_dataset) / len(dataset_filenames)))
    print("Number of validation samples:   {} ({:.2f}%)".format(len(val_dataset), 100 * len(val_dataset) / len(dataset_filenames)))
    print("Number of testing samples:      {} ({:.2f}%)\n".format(len(test_dataset), 100 * len(test_dataset) / len(dataset_filenames)))

    # Define loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader

def custom_collate_fn(batch):
    # Separate the lists of global_data, segment_data, and dense_data
    y_list = [item.y for item in batch]
    y_class_list = [item.y_class for item in batch]
    global_data_list = [item.global_data for item in batch]
    segment_data_list = [item.segment_data for item in batch]
    dense_data_list = [item.dense_data for item in batch]
    ids = [item.id for item in batch]

    # Standard batching for global data and labels
    batched_y = torch.stack(y_list, dim=0)
    batched_y_class = torch.stack(y_class_list, dim=0)
    batched_global_data = torch.stack(global_data_list, dim=0)
    batched_ids = ids

    # Use PyG's Batch to batch segment_data and dense_data
    # Since these are PyG Data objects, they can be directly batched
    batched_segment_data = Batch.from_data_list(segment_data_list)
    batched_dense_data = Batch.from_data_list(dense_data_list)

    # Create a new Data object for the batch
    batch_data = Data()
    batch_data.y = batched_y
    batch_data.y_class = batched_y_class
    batch_data.global_data = batched_global_data
    batch_data.segment_data = batched_segment_data
    batch_data.dense_data = batched_dense_data
    batch_data.id = batched_ids 

    return batch_data

