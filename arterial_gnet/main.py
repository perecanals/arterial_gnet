
def main(root, args):
    import os, json

    from arterial_gnet.dataloading.data_augmentation import get_transforms
    from arterial_gnet.dataloading.dataloading import get_data_loaders
    from arterial_gnet.models.models import get_model
    from arterial_gnet.train.train import run_training
    from arterial_gnet.train.losses import LinearWeightedMSELoss, ScaledExponentialWeightedMSELoss, LogarithmicWeightedMSELoss, NLLLoss
    from torch.nn.modules.loss import MSELoss
    from arterial_gnet.test.test import run_testing

    import torch

    print("------------------------------------------------")
    print("Running training and testing for ArterialGNet\n")

    # Read device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #################################### Dataset organization ############################################
    # Define pre-transforms (applied to the graph before batching, regardless of training or testing)
    pre_transform, train_transform, test_transform = get_transforms(device)
    for fold in (range(args.folds) if args.folds is not None else [None]):
        if args.folds is not None and args.skip_folds is not None and fold < args.skip_folds:
            continue
        # Get data loaders
        train_loader, val_loader, test_loader = get_data_loaders(root, args, fold, pre_transform, train_transform, test_transform)

        with open(os.path.join(root, "dataset.json")) as f:
            dataset_description = json.load(f)

        #################################### Model definition ################################################
        model, model_name = get_model(args, dataset_description, device)
        
        if args.train:
            if args.is_classification:
                # Define loss function
                if dataset_description["graph_class_frequencies"] is not None and not args.oversampling:
                    if args.class_loss == "ce":
                        # dataset_description["graph_class_frequencies"] = [sum(dataset_description["graph_class_frequencies"][:-1]), dataset_description["graph_class_frequencies"][-1]]
                        loss_function = torch.nn.CrossEntropyLoss(weight = 1 / torch.tensor(dataset_description["graph_class_frequencies"], dtype=torch.float).to(device))
                    elif args.class_loss == "nll":
                        loss_function = NLLLoss(class_frequencies=dataset_description["graph_class_frequencies"])
                else:
                    if args.class_loss == "ce":
                        loss_function = torch.nn.CrossEntropyLoss()
                    elif args.class_loss == "nll":
                        loss_function = NLLLoss()
            else:
                if args.weighted_loss is None or args.oversampling:
                    loss_function = MSELoss()
                elif args.weighted_loss == "lin":
                    loss_function = LinearWeightedMSELoss()
                elif args.weighted_loss == "exp":
                    loss_function = ScaledExponentialWeightedMSELoss()
                elif args.weighted_loss == "log":
                    loss_function = LogarithmicWeightedMSELoss()
            # Run training
            run_training(
                root,
                model,
                model_name,
                train_loader,
                val_loader,
                loss_function,
                args,
                fold = fold,
                device = device
            )

        if args.test:
            # Run testing
            # for model_test in ["best", "latest", "metric"]:
            for model_test in ["best"]:
                run_testing(
                    root,
                    model_name,
                    test_loader,
                    model = model_test,
                    device = device,
                    fold = fold,
                    is_classification = args.is_classification
                )

if __name__ == "__main__":
    import os, sys
    import argparse

    sys.path.append("/home/vhir/github/arterial_net/arterial_net")
    root = os.environ["arterial_maps_root"]

    # Create argument parser
    parser = argparse.ArgumentParser(description='Train and test the model for ArterialMaps.')
    # Add arguments
    parser.add_argument('-ts', '--test_size', type=float, default=0.2, 
        help='Dataset ratio for testing, with respect to the total size of the dataset. Default is 0.2.')
    parser.add_argument('-vs', '--val_size', type=float, default=0.2,
        help='Dataset ratio for validation, with respect to the size of the training dataset (after subtracting testing set)'
        'It will be overrun if args.folds is not None (size will be (1 - test_size) / args.folds). Default is 0.2.')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, 
        help='Batch size for training and validation. Default is 64.')
    parser.add_argument('-bnm', '--base_model_name', type=str, default="ArterialGNet", 
        help='Base model name. Default is ArterialGNet.')
    parser.add_argument('-hc', '--hidden_channels', type=int, default=8, 
        help='Number of hidden channels. Default is 8.')
    parser.add_argument('-hcd', '--hidden_channels_dense', type=int, default=None, 
        help='Number of hidden channels of the dense layers. Default is None, which defaults to the number --hidden_channels.')
    parser.add_argument('-ngl', '--num_global_layers', type=int, default=1,
        help='Number of global layers. Default is 1.')
    parser.add_argument('-nsl', '--num_segment_layers', type=int, default=1,
        help='Number of segment layers. Default is 1.')
    parser.add_argument('-ndl', '--num_dense_layers', type=int, default=1,
        help='Number of dense layers. Default is 1.')
    parser.add_argument('-te', '--total_epochs', type=int, default=500, 
        help='Total number of epochs. Default is 500.')
    parser.add_argument('-optim', '--optimizer', type=str, default="adam", choices=["adam", "sgd"],
        help='Optimizer. Default is adam.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, 
        help='Initial learning rate. Default is 0.001.')
    parser.add_argument('-lrs', '--lr_scheduler', type=str, default="poly", choices=['poly', 'plateau', 'None'],
        help='Learning rate scheduler. Default is True.')
    parser.add_argument('-agg', '--aggregation', type=str, default="mean", choices=["mean", "add", "max"],
        help='Aggregation method. Default is mean.')
    parser.add_argument('-wl', '--weighted_loss', type=str, default="exp", choices=['exp', "lin", "log", 'None'],
        help='Whether to use weighted loss and what type. Default is exp.')
    parser.add_argument('-drop', '--dropout', type=float, default=0.8,
        help='Dropout probability. Default is 0.8.')
    parser.add_argument('-rs', '--random_state', type=int, default=42,
        help='Random state for splitting the dataset. Default is 42.')
    parser.add_argument('-trs', '--test_random_state', type=int, default=42,
        help='Random state for splitting the dataset for k-fold in test. Default is 42.')
    parser.add_argument('-f', '--folds', type=int, default=None,
        help='Folds number. Default is None.')
    parser.add_argument('-tf', '--test_folds', type=int, default=5,
        help='Folds number for test. Default is 5.')
    parser.add_argument('-sf', '--skip_folds', type=int, default=None,
        help='Skip to fold. Default is None.')
    parser.add_argument('-train', '--train', type=str, default=True, choices=['True', 'False'],
        help='Whether to train the model. Default is True.')
    parser.add_argument('-test', '--test', type=str, default=True, choices=['True', 'False'],
        help='Whether to test the model. Default is True.')
    parser.add_argument("-class", "--is_classification", action="store_true",
        help="Whether the task is is_classification or regression. Default is False.")
    parser.add_argument('-cl', '--class_loss', type=str, default="ce", choices=["ce", "nll"],
        help='Loss function for classification. Default is ce.')
    parser.add_argument('-tag', '--tag', type=str, default=None,
        help='Additional tag to add to the model name for identification. Default is None.')
    parser.add_argument('-nw', '--num_workers', type=int, default=2,
        help='Number of workers for the DataLoader (ATM unused). Default is 2.')
    parser.add_argument('-os', '--oversampling', type=str, default=False, choices=['True', 'False'],
        help='Whether to use oversampling. Default is False.')
    
    # Parse arguments
    args = parser.parse_args()

    if args.lr_scheduler == "None":
        args.lr_scheduler = None
    if args.hidden_channels_dense is None:
        args.hidden_channels_dense = args.hidden_channels

    # Set random state
    import random
    import numpy as np
    import torch

    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import time

    start = time.time()
    
    main(root, args)

    end = time.time()

    print("Time elapsed: {:.2f} seconds".format(end - start))