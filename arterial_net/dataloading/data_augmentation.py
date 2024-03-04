from torch_geometric.transforms import Compose, ToDevice

def get_transforms(device):
    """
    Define transforms for the dataset.

    Paramaters
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
    pre_transform = Compose([
        ToDevice(device)
    ])
    train_transform = Compose([
        ToDevice(device)
    ])
    test_transform = Compose([
        ToDevice(device)
    ])

    return pre_transform, train_transform, test_transform