import torch
from torch_geometric.data import Data, Batch

import os

from sklearn.model_selection import train_test_split, StratifiedKFold

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from arterial_gnet.dataloading.dataset import ArterialMapsDataset, DenseGraphDataset, SequenceGraphDataset
from arterial_gnet.dataloading.utils import load_pickle

def get_data_loaders(root, args, fold=None, pre_transform=None, train_transform=None, test_transform=None):
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
    train_loader : torch_geometric.loader.DataLoader
        Training data loader.
    val_loader : torch_geometric.loader.DataLoader
        Validation data loader.
    test_loader : torch_geometric.loader.DataLoader
        Testing data loader.

    """
    def compute_class_weights(y):
        """
        Compute class weights for a given list of classes.

        Parameters
        ----------
        y : list
            List of classes.

        Returns
        -------
        class_weights : list
            List of class weights.

        """
        class_counts = [y.count(i) for i in range(max(y) + 1)]
        class_weights = [max(class_counts) / count for count in class_counts]
        return class_weights
    # Define datasets . First make division of train + val and test
    dataset_filenames = [f for f in os.listdir(os.path.join(root, "raw")) if f.endswith(".pickle")]
    y_class = [load_pickle(os.path.join(root, "raw", f))["classification"] for f in dataset_filenames]
    # Pass y_class list to binary
    # y_class = [1 if item > 1.5 else 0 for item in y_class]
    if not args.test_size == args.val_size == 0:
        train_val_filenames, test_filenames, y_class_train_val, y_class_test = train_test_split(dataset_filenames, y_class, test_size = args.test_size, random_state = args.random_state, stratify = y_class)
        # Now, depending on the folds being specified or not, perform cross validation (select the k-fold corresponding to "fold", with k being "args.folds") or not
        if fold is not None:
            kf = StratifiedKFold(n_splits = args.folds, shuffle = True, random_state = args.random_state)
            folds = list(kf.split(train_val_filenames, y_class_train_val))
            train_filenames = [train_val_filenames[i] for i in folds[fold][0]]
            val_filenames = [train_val_filenames[i] for i in folds[fold][1]]  
            y_class_train = [y_class_train_val[i] for i in folds[fold][0]]  
            y_class_val = [y_class_train_val[i] for i in folds[fold][1]]
        else:
            train_filenames, val_filenames, y_class_train, y_class_val = train_test_split(train_val_filenames, y_class_train_val, test_size = args.val_size, random_state = args.random_state, stratify=y_class_train_val)
        # Now define dataset classes
        if args.base_model_name.startswith("ArterialGNet"):
            train_dataset = ArterialMapsDataset(root, raw_file_names_list = train_filenames, pre_transform = pre_transform, transform = train_transform, radius = args.radius)
            val_dataset = ArterialMapsDataset(root, raw_file_names_list = val_filenames, pre_transform = pre_transform, transform = test_transform, radius = args.radius)
            test_dataset = ArterialMapsDataset(root, raw_file_names_list = test_filenames, pre_transform = pre_transform, transform = test_transform, radius = args.radius)
        elif args.base_model_name.startswith("Transformer"):
            train_dataset = SequenceGraphDataset(root, raw_file_names_list = train_filenames, pre_transform = pre_transform, transform = train_transform, pos_enc_dim = args.pos_enc_dim, max_seq_len = args.max_seq_len)
            val_dataset = SequenceGraphDataset(root, raw_file_names_list = val_filenames, pre_transform = pre_transform, transform = test_transform, pos_enc_dim = args.pos_enc_dim, max_seq_len = args.max_seq_len)
            test_dataset = SequenceGraphDataset(root, raw_file_names_list = test_filenames, pre_transform = pre_transform, transform = test_transform, pos_enc_dim = args.pos_enc_dim, max_seq_len = args.max_seq_len)
        else:
            train_dataset = DenseGraphDataset(root, raw_file_names_list = train_filenames, pre_transform = pre_transform, transform = train_transform, radius = args.radius, use_lap_pos_enc = args.use_lap_pos_enc, pos_enc_dim = args.pos_enc_dim)
            val_dataset = DenseGraphDataset(root, raw_file_names_list = val_filenames, pre_transform = pre_transform, transform = test_transform, radius = args.radius, use_lap_pos_enc = args.use_lap_pos_enc, pos_enc_dim = args.pos_enc_dim)
            test_dataset = DenseGraphDataset(root, raw_file_names_list = test_filenames, pre_transform = pre_transform, transform = test_transform, radius = args.radius, use_lap_pos_enc = args.use_lap_pos_enc, pos_enc_dim = args.pos_enc_dim)
    else:
        print("Training with the whole dataset (ignore validation and test results)\n")
        if args.base_model_name.startswith("ArterialGNet"):
            train_dataset = ArterialMapsDataset(root, pre_transform = pre_transform, transform = train_transform, radius = args.radius)
            val_dataset = ArterialMapsDataset(root, raw_file_names_list = dataset_filenames[:2], pre_transform = pre_transform, transform = test_transform, radius = args.radius)
            test_dataset = ArterialMapsDataset(root, raw_file_names_list = dataset_filenames[:2], pre_transform = pre_transform, transform = test_transform, radius = args.radius)
        elif args.base_model_name.startswith("Transformer"):
            train_dataset = SequenceGraphDataset(root, raw_file_names_list = train_filenames, pre_transform = pre_transform, transform = train_transform, pos_enc_dim = args.pos_enc_dim, max_seq_len = args.max_seq_len)
            val_dataset = SequenceGraphDataset(root, raw_file_names_list = val_filenames, pre_transform = pre_transform, transform = test_transform, pos_enc_dim = args.pos_enc_dim, max_seq_len = args.max_seq_len)
            test_dataset = SequenceGraphDataset(root, raw_file_names_list = test_filenames, pre_transform = pre_transform, transform = test_transform, pos_enc_dim = args.pos_enc_dim, max_seq_len = args.max_seq_len)
        else:
            train_dataset = DenseGraphDataset(root, pre_transform = pre_transform, transform = train_transform, radius = args.radius, use_lap_pos_enc = args.use_lap_pos_enc, pos_enc_dim = args.pos_enc_dim)
            val_dataset = DenseGraphDataset(root, raw_file_names_list = dataset_filenames[:2], pre_transform = pre_transform, transform = test_transform, radius = args.radius, use_lap_pos_enc = args.use_lap_pos_enc, pos_enc_dim = args.pos_enc_dim)
            test_dataset = DenseGraphDataset(root, raw_file_names_list = dataset_filenames[:2], pre_transform = pre_transform, transform = test_transform, radius = args.radius, use_lap_pos_enc = args.use_lap_pos_enc, pos_enc_dim = args.pos_enc_dim)

    print("------------------------------------------------ Dataset information")
    print("Total number of samples:        {}".format(len(dataset_filenames)))
    if fold is not None:    
        print(f"Fold:                           {fold}")
    print("Number of training samples:     {} ({:.2f}%)".format(len(train_dataset), 100 * len(train_dataset) / len(dataset_filenames)))
    for class_ in range(max(y_class_train) + 1):
        print(f"\tNumber of samples of class {class_}: {y_class_train.count(class_)} ({100 * y_class_train.count(class_)/len(y_class_train):.2f}%)")
    print("Number of validation samples:   {} ({:.2f}%)".format(len(val_dataset), 100 * len(val_dataset) / len(dataset_filenames)))
    for class_ in range(max(y_class_val) + 1):
        print(f"\tNumber of samples of class {class_}: {y_class_val.count(class_)} ({100 * y_class_val.count(class_)/len(y_class_val):.2f}%)")
    print("Number of testing samples:      {} ({:.2f}%)\n".format(len(test_dataset), 100 * len(test_dataset) / len(dataset_filenames)))
    for class_ in range(max(y_class_test) + 1):
        print(f"\tNumber of samples of class {class_}: {y_class_test.count(class_)} ({100 * y_class_test.count(class_)/len(y_class_test):.2f}%)")

    # Apply oversampling if enabled
    if args.oversampling:
        # Calculate class weights for all training samples
        class_weights_train = compute_class_weights(y_class_train)
        sample_weights_train = [class_weights_train[y] for y in y_class_train]  # Assign each sample its class's weight
        # Divide by 2 the weights of the classes that are not the majority class
        sample_weights_train = [weight / 2 if y > 0.5 else weight for y, weight in zip(y_class_train, sample_weights_train)]
        sampler_train = WeightedRandomSampler(sample_weights_train, len(sample_weights_train))
        # Repeat for validation
        class_weights_val = compute_class_weights(y_class_val)
        sample_weights_val = [class_weights_val[y] for y in y_class_val]
        # Divide by 2 the weights of the classes that are not the majority class
        sample_weights_val = [weight / 2 if y > 0.5 else weight for y, weight in zip(y_class_val, sample_weights_val)]
        sampler_val = WeightedRandomSampler(sample_weights_val, len(sample_weights_val))

    # Define loaders
    if args.oversampling:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=choose_collate_fn(args.base_model_name), sampler=sampler_train)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=choose_collate_fn(args.base_model_name), sampler=sampler_val)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=choose_collate_fn(args.base_model_name))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=choose_collate_fn(args.base_model_name))

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=choose_collate_fn(args.base_model_name))

    return train_loader, val_loader, test_loader

def get_test_folds(root, args):
    """
    Define test data filename lists for the dataset. Extracts 5 different pairs of 
    train+val and test lists of filenames.

    Parameters
    ----------
    root : string or path-like object
        Path to root folder.
    args : argparse.Namespace
        Arguments.
    """
    dataset_filenames = [f for f in os.listdir(os.path.join(root, "raw")) if f.endswith(".pickle")]
    y_class = [load_pickle(os.path.join(root, "raw", f))["classification"] for f in dataset_filenames]

    kf = StratifiedKFold(n_splits = args.test_folds, shuffle = True, random_state = args.random_state)

    folds = list(kf.split(dataset_filenames, y_class))

    folds_filenames = []
    for (train_val_indices, test_indices) in folds:
        train_val_filenames = [dataset_filenames[idx] for idx in train_val_indices]
        test_filenames = [dataset_filenames[idx] for idx in test_indices]
        folds_filenames.append((train_val_filenames, test_filenames))

    return folds_filenames

def get_data_loaders_with_filenames(root, args, train_val_filenames, test_filenames, fold=None, pre_transform=None, train_transform=None, test_transform=None):
    """
    Define train, validation and test data loaders for the dataset.

    Parameters
    ----------
    root : string or path-like object
        Path to root folder.
    train_val_filenames : list
        List of filenames for training and validation.
    test_filenames : list
        List of filenames for testing.
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
    train_loader : torch_geometric.loader.DataLoader
        Training data loader.
    val_loader : torch_geometric.loader.DataLoader
        Validation data loader.
    test_loader : torch_geometric.loader.DataLoader
        Testing data loader.

    """
    def compute_class_weights(y):
        """
        Compute class weights for a given list of classes.

        Parameters
        ----------
        y : list
            List of classes.

        Returns
        -------
        class_weights : list
            List of class weights.

        """
        class_counts = [y.count(i) for i in range(max(y) + 1)]
        class_weights = [max(class_counts) / count for count in class_counts]
        return class_weights
    # Define datasets . First make division of train + val and test
    dataset_filenames = [f for f in os.listdir(os.path.join(root, "raw")) if f.endswith(".pickle")]
    y_class = [load_pickle(os.path.join(root, "raw", f))["classification"] for f in dataset_filenames]

    y_class_train_val = [y_class[dataset_filenames.index(f)] for f in train_val_filenames]
    y_class_test = [y_class[dataset_filenames.index(f)] for f in test_filenames]

    if fold is not None:
        kf = StratifiedKFold(n_splits = args.folds, shuffle = True, random_state = args.random_state)
        folds = list(kf.split(train_val_filenames, y_class_train_val))
        train_filenames = [train_val_filenames[i] for i in folds[fold][0]]
        val_filenames = [train_val_filenames[i] for i in folds[fold][1]]  
        y_class_train = [y_class_train_val[i] for i in folds[fold][0]]  
        y_class_val = [y_class_train_val[i] for i in folds[fold][1]]
    else:
        train_filenames, val_filenames, y_class_train, y_class_val = train_test_split(train_val_filenames, y_class_train_val, test_size = args.val_size, random_state = args.random_state, stratify=y_class_train_val)
    # Now define dataset classes
    if args.base_model_name.startswith("ArterialGNet"):
        train_dataset = ArterialMapsDataset(root, raw_file_names_list = train_filenames, pre_transform = pre_transform, transform = train_transform, radius = args.radius)
        val_dataset = ArterialMapsDataset(root, raw_file_names_list = val_filenames, pre_transform = pre_transform, transform = test_transform, radius = args.radius)
        test_dataset = ArterialMapsDataset(root, raw_file_names_list = test_filenames, pre_transform = pre_transform, transform = test_transform, radius = args.radius)
    elif args.base_model_name.startswith("Transformer"):
        train_dataset = SequenceGraphDataset(root, raw_file_names_list = train_filenames, pre_transform = pre_transform, transform = train_transform, pos_enc_dim = args.pos_enc_dim, max_seq_len = args.max_seq_len)
        val_dataset = SequenceGraphDataset(root, raw_file_names_list = val_filenames, pre_transform = pre_transform, transform = test_transform, pos_enc_dim = args.pos_enc_dim, max_seq_len = args.max_seq_len)
        test_dataset = SequenceGraphDataset(root, raw_file_names_list = test_filenames, pre_transform = pre_transform, transform = test_transform, pos_enc_dim = args.pos_enc_dim, max_seq_len = args.max_seq_len)
    else:   
        train_dataset = DenseGraphDataset(root, raw_file_names_list = train_filenames, pre_transform = pre_transform, transform = train_transform, radius = args.radius, use_lap_pos_enc = args.use_lap_pos_enc, pos_enc_dim = args.pos_enc_dim)
        val_dataset = DenseGraphDataset(root, raw_file_names_list = val_filenames, pre_transform = pre_transform, transform = test_transform, radius = args.radius, use_lap_pos_enc = args.use_lap_pos_enc, pos_enc_dim = args.pos_enc_dim)
        test_dataset = DenseGraphDataset(root, raw_file_names_list = test_filenames, pre_transform = pre_transform, transform = test_transform, radius = args.radius, use_lap_pos_enc = args.use_lap_pos_enc, pos_enc_dim = args.pos_enc_dim)

    print("------------------------------------------------ Dataset information")
    print("Total number of samples:        {}".format(len(dataset_filenames)))
    for class_ in range(max(y_class) + 1):
        print(f"\tNumber of samples of class {class_}: {y_class.count(class_)} ({100 * y_class.count(class_)/len(y_class):.2f}%)")
    if fold is not None:    
        print(f"Fold:                           {fold}")
    print("Number of training samples:     {} ({:.2f}%)".format(len(train_dataset), 100 * len(train_dataset) / len(dataset_filenames)))
    for class_ in range(max(y_class_train) + 1):
        print(f"\tNumber of samples of class {class_}: {y_class_train.count(class_)} ({100 * y_class_train.count(class_)/len(y_class_train):.2f}%)")
    print("Number of validation samples:   {} ({:.2f}%)".format(len(val_dataset), 100 * len(val_dataset) / len(dataset_filenames)))
    for class_ in range(max(y_class_val) + 1):
        print(f"\tNumber of samples of class {class_}: {y_class_val.count(class_)} ({100 * y_class_val.count(class_)/len(y_class_val):.2f}%)")
    print("Number of testing samples:      {} ({:.2f}%)".format(len(test_dataset), 100 * len(test_dataset) / len(dataset_filenames)))
    for class_ in range(max(y_class_test) + 1):
        print(f"\tNumber of samples of class {class_}: {y_class_test.count(class_)} ({100 * y_class_test.count(class_)/len(y_class_test):.2f}%)")

    print()

    # Apply oversampling if enabled
    if args.oversampling:
        # Calculate class weights for all training samples
        class_weights_train = compute_class_weights(y_class_train)
        sample_weights_train = [class_weights_train[y] for y in y_class_train]  # Assign each sample its class's weight
        # Divide by 2 the weights of the classes that are not the majority class
        sample_weights_train = [weight / 2 if y > 0.5 else weight for y, weight in zip(y_class_train, sample_weights_train)]
        sampler_train = WeightedRandomSampler(sample_weights_train, len(sample_weights_train))
        # Repeat for validation
        class_weights_val = compute_class_weights(y_class_val)
        sample_weights_val = [class_weights_val[y] for y in y_class_val]
        # Divide by 2 the weights of the classes that are not the majority class
        sample_weights_val = [weight / 2 if y > 0.5 else weight for y, weight in zip(y_class_val, sample_weights_val)]
        sampler_val = WeightedRandomSampler(sample_weights_val, len(sample_weights_val))

    # Define loaders
    if args.oversampling:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=choose_collate_fn(args.base_model_name), sampler=sampler_train)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=choose_collate_fn(args.base_model_name), sampler=sampler_val)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=choose_collate_fn(args.base_model_name))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=choose_collate_fn(args.base_model_name))

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=choose_collate_fn(args.base_model_name))

    return train_loader, val_loader, test_loader

def get_test_data_loader(root, base_model_name, pre_transform=None, test_transform=None, radius=0, use_lap_pos_enc=False, pos_enc_dim=8):
    if base_model_name.startswith("Transformer"):
        use_lap_pos_enc = False
    elif base_model_name.startswith("GraphTransformer"):
        use_lap_pos_enc = True
    test_filenames = [f for f in os.listdir(os.path.join(root, "raw")) if f.endswith(".pickle")]
    y_class = [load_pickle(os.path.join(root, "raw", f))["classification"] for f in test_filenames]
    if base_model_name.startswith("ArterialGNet"):
        test_dataset = ArterialMapsDataset(root, raw_file_names_list = test_filenames, pre_transform = pre_transform, transform = test_transform, radius = radius)
    else:
        test_dataset = DenseGraphDataset(root, raw_file_names_list = test_filenames, pre_transform = pre_transform, transform = test_transform, radius = radius, use_lap_pos_enc = use_lap_pos_enc, pos_enc_dim = pos_enc_dim)
    print("Number of testing samples:      {} ({:.2f}%)\n".format(len(test_dataset), 100 * len(test_dataset) / len(test_filenames)))
    for class_ in range(max(y_class) + 1):
        print(f"\tNumber of samples of class {class_}: {y_class.count(class_)} ({100 * y_class.count(class_)/len(y_class):.2f}%)")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=choose_collate_fn(base_model_name))

    return test_loader

def choose_collate_fn(base_model_name):
    if base_model_name.startswith("ArterialGNet"):
        return collate_ArterialGNet
    elif base_model_name.startswith("Transformer"):
        return custom_collate_sequence
    else:
        return collate_DenseGraphDataset
    
def collate_ArterialGNet(data_list):
    device = data_list[0].dense_data.x.device
    # Create a new Data object for the batch
    batch = Data()

    # Standard batching for global data and labels
    batch.y = torch.tensor([data.y for data in data_list], dtype=torch.float).to(device)
    batch.y_class = torch.tensor([data.y_class for data in data_list], dtype=torch.long).to(device)
    batch.id = [data.id for data in data_list]
    batch.global_data = torch.stack([item.global_data for item in data_list], dim=0).to(device)

    # Use PyG's Batch to batch segment_data and dense_data
    # Since these are PyG Data objects, they can be directly batched
    batch.segment_data = Batch.from_data_list([item.segment_data for item in data_list])
    batch.dense_data = Batch.from_data_list([item.dense_data for item in data_list])
    batch.batch = batch.dense_data.batch

    return batch

def collate_DenseGraphDataset(data_list):
    # Use PyG's Batch to efficiently handle graphs of different sizes
    batch = Batch.from_data_list(data_list)
    
    # More efficient handling of single-value tensors
    batch.y = torch.tensor([data.y for data in data_list], dtype=torch.float).to(batch.x.device)
    batch.y_class = torch.tensor([data.y_class for data in data_list], dtype=torch.long).to(batch.x.device)
    batch.id = [data.id for data in data_list]

    return batch


def custom_collate_sequence(data_list):
    batch = Batch.from_data_list(data_list)
    batch.x = batch.x.view(len(batch), -1, batch.x.shape[-1])
    batch.pos_enc = batch.pos_enc.view(len(batch), -1, batch.pos_enc.shape[-1]).to(batch.x.device)
    batch.mask = batch.mask.view(len(batch), -1, batch.mask.shape[-1]).to(batch.x.device)
    
    batch.y = torch.tensor([data.y for data in batch], dtype=torch.float).to(batch.x.device)
    batch.y_class = torch.tensor([data.y_class for data in batch], dtype=torch.long).to(batch.x.device)
    batch.seq_len = torch.tensor([data.seq_len for data in batch], dtype=torch.long).to(batch.x.device)
    batch.id = [data.id for data in batch]

    return batch

