import os

import numpy as np

from arterial_net.train.lr_schedulers import PolyLRScheduler
from arterial_net.train.utils import make_train_plot
from arterial_net.utils.metrics import compute_accuracy, compute_rmse

import torch

def run_training(root, model, model_name, train_loader, val_loader, loss_function, args, fold=None, device="cpu"):
    """
    Trainer function for a given model. Thought out for GraphUNet for node classification.

    Parameters
    ----------
    root : string or path-like object
        Path to root folder.
    model : torch.nn.Module
        Model to train.
    model_name : string 
        Name of the model.
    train_loader : torch_geometric.loader.DataLoader
        DataLoader for training.
    val_loader : torch_geometric.loader.DataLoader
        DataLoader for validation.
    loss_function : torch.nn.Module
        Loss function to use.
    args : argparse.Namespace
        Arguments. Contains:
            - total_epochs : int
                Total number of epochs to train.
            - optim : string
                Optimizer to use.
            - learning_rate : float
                Initial learning rate.
            - lr_scheduler : bool
                Whether to use a learning rate scheduler.
            - is_classification : bool
                Whether the task is classification or regression.
    fold : int, optional
        Fold for cross validation. The default is None.
    device : string, optional
        Device to use for training. The default is "cpu".

    """
    def train_step(model, batch):
        """
        Performs a training step for a batch of graphs.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        batch : torch_geometric.data.Batch
            Batch of graphs.

        Returns
        -------
        loss : torch.tensor
            Loss for the batch.
        metric : float
            Accuracy for the batch if is_classification is True otherwise RMSE.
        """
        # Set model in training mode
        model.train()
        # Set gradients to 0
        optimizer.zero_grad() 
        # Perform forward pass with batch
        out = model(batch.to(device)).to(device)
        out = torch.squeeze(out)
        # Raise error if nan in out
        if torch.isnan(out).any():
            raise ValueError("NaN in output.")
        # Compute loss
        if args.is_classification:
            loss = loss_function(out, batch.y_class)
            # Compute back propagation
            loss.backward() 
            # Update weights with optimizer
            optimizer.step()
            # Compute accuracy for training
            metric = compute_accuracy(out.argmax(dim=1), batch.y_class)
        else:
            loss = loss_function(out, batch.y)
            # Compute back propagation
            loss.backward() 
            # Update weights with optimizer
            optimizer.step()
            # Compute RMSE for training
            metric = compute_rmse(out, batch.y)
        return loss, metric, out

    def val_step(model, batch):
        """
        Performs a validation step for a batch of graphs.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        batch : torch_geometric.data.Batch
            Batch of graphs.

        Returns
        -------
        loss : torch.tensor
            Loss for the batch.
        metric : float
            Accuracy for the batch if is_classification is True otherwise RMSE.
        """
        # Set model in evaluation mode
        model.eval()
        # In validation we do not keep track of gradients
        with torch.no_grad():
            # Perform forward pass with batch
            out = model(batch.to(device)).to(device)
            out = torch.squeeze(out)
            # Compute validation loss
            if args.is_classification:
                loss = loss_function(out, batch.y_class)
                # Compute accuracy for validation
                metric = compute_accuracy(out.argmax(dim=1), batch.y_class)
            else:
                loss = loss_function(out, batch.y)
                # Compute RMSE for training
                metric = compute_rmse(out, batch.y)
        return loss, metric, out
    
    print("\n------------------------------------------------ Training parameters")
    print(f"Batch size:                     {train_loader.batch_size}")
    print(f"Total epochs:                   {args.total_epochs}")
    print(f"Optimizer:                      {args.optimizer}")
    print(f"Initial learning rate:          {args.learning_rate}")
    print(f"Learning rate scheduler:        {args.lr_scheduler}")
    print(f"Loss function:                  {loss_function}")
    print(f"Running on device:              {device}")
    print(f"Number of parameters:           {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    # Define model path
    model_path = os.path.join(root, "models", model_name)
    if fold is not None:
        model_path = os.path.join(model_path, f"fold_{fold}")
    os.makedirs(model_path, exist_ok=True)
    
    # Define optimizer and scheduler
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-03
            )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.learning_rate, 
            momentum=0.9, 
            weight_decay=1e-03)
    
    if args.lr_scheduler is not None:
        if args.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=50, 
                threshold=0.01, 
                threshold_mode='rel', 
                eps=1e-04,
                verbose=True
                )
        elif args.lr_scheduler == "poly":
            scheduler = PolyLRScheduler(optimizer, 
                                        initial_lr=args.learning_rate,
                                        max_steps=args.total_epochs,
                                        exponent=0.9)
        
    # Initializes lists for loss and accuracy evolution during training
    losses_train = []
    losses_val = []
    metric_train = []
    metric_val = []

    ewma_losses_train = []
    ewma_losses_val = []
    ewma_metric_train = []
    ewma_metric_val = []

    # if args.is_classification:
    #     roc_auc_train = []
    #     pr_auc_train = []
    #     roc_auc_val = []
    #     pr_auc_val = []
    #     ewma_roc_auc_train = []
    #     ewma_pr_auc_train = []
    #     ewma_roc_auc_val = []
    #     ewma_pr_auc_val = []

    # Starts training
    for epoch in range(0, args.total_epochs + 1):
        print("Epoch: {}/{}".format(epoch, args.total_epochs), end="\r")
        # Initializes in-epoch variables
        total_epoch_loss_train, total_epoch_loss_val = 0, 0
        metric_train_epoch_list, metric_val_epoch_list = [], []
        num_graphs_train, num_graphs_val = 0, 0
        # if args.is_classification:
        #     preds_train, preds_val = [], []
        #     labels_train, labels_val = [], []
        #     roc_auc_train_epoch_list, roc_auc_val_epoch_list = [], []
        #     pr_auc_train_epoch_list, pr_auc_val_epoch_list = [], []

        # Iterates over training DataLoader and performs a training step for each batch
        for batch in train_loader:
            # Performs training step
            loss_train, met_train, out_train = train_step(model, batch)
            # Adds loss to epoch loss
            total_epoch_loss_train += loss_train.detach()
            # Adds batch accuracy/rmse to list for epoch
            metric_train_epoch_list.append(met_train)
            # Update the number of graphs
            num_graphs_train += batch.segment_data.batch.max().item() + 1

            # preds_train = out_train[:, 1].cpu().detach().numpy()
            # labels_train = batch.y_class.cpu().detach().numpy()

        # Divides epoch accumulated training loss by number of graphs
        total_epoch_loss_train = total_epoch_loss_train.cpu() / num_graphs_train
        # Appends training loss to tracking
        losses_train.append(total_epoch_loss_train.cpu())
        # Computes mean training accuracy across batches
        metric_train_epoch = np.mean(metric_train_epoch_list)
        # Appends training accuracy to tracking
        metric_train.append(metric_train_epoch)
        # Computes exponentially weighted moving average for training loss and accuracy
        ewma_losses_train.append(0.9 * ewma_losses_train[-1] + 0.1 * total_epoch_loss_train if len(ewma_losses_train) > 0 else total_epoch_loss_train)
        ewma_metric_train.append(0.9 * ewma_metric_train[-1] + 0.1 * metric_train_epoch if len(ewma_metric_train) > 0 else metric_train_epoch)

        # Iterates over validation DataLoader and performs a training step for each batch
        for batch in val_loader:
            # Performs validation step
            loss_val, met_val, out = val_step(model, batch)
            # Adds loss to epoch loss
            total_epoch_loss_val += loss_val
            # Adds accuracy to list for epoch
            metric_val_epoch_list.append(met_val)
            # Update the number of graphs
            num_graphs_val += batch.segment_data.batch.max().item() + 1

        # Divides epoch accumulated validation loss by number of graphs
        total_epoch_loss_val = total_epoch_loss_val.cpu() / num_graphs_val
        # Appends validation loss to tracking
        losses_val.append(total_epoch_loss_val.cpu())
        # Computes mean validation accuracy across batches
        metric_val_epoch = np.mean(metric_val_epoch_list)
        # Appends validation accuracy to tracking
        metric_val.append(metric_val_epoch)
        # Computes exponentially weighted moving average for validation loss and accuracy
        ewma_losses_val.append(0.9 * ewma_losses_val[-1] + 0.1 * total_epoch_loss_val if len(ewma_losses_val) > 0 else total_epoch_loss_val)
        ewma_metric_val.append(0.9 * ewma_metric_val[-1] + 0.1 * metric_val_epoch if len(ewma_metric_val) > 0 else metric_val_epoch)

        # Saves best model in terms of validation loss
        if epoch > 0:
            if ewma_losses_val[-1] < np.amin(ewma_losses_val[:-1]):
                # print(f"New best model in epoch {epoch} (validation loss: {total_epoch_loss_val:.2f}). Saving model...")
                # print(f"\t{os.path.join(model_path, "model_best_loss.pth")}\n")
                torch.save(model, os.path.join(model_path, "model_best_loss.pth"))
            # Saves best model in terms of validation accuracy
            elif ewma_metric_val[-1] > np.max(ewma_metric_val[:-1]) and args.is_classification:
                # print(f"New best model in epoch {epoch} (validation accuracy: {metric_val_epoch:.2f}). Saving model...")
                # print(f"\t{os.path.join(model_path, "model_best_metric.pth")}\n")
                torch.save(model, os.path.join(model_path, "model_best_metric.pth"))
            elif ewma_metric_val[-1] < np.min(ewma_metric_val[:-1]) and not args.is_classification:
                # print(f"New best model in epoch {epoch} (validation rmse: {metric_val_epoch:.2f}). Saving model...")
                # print(f"\t{os.path.join(model_path, "model_best_metric.pth")}\n")
                torch.save(model, os.path.join(model_path, "model_best_metric.pth"))
        else:
            torch.save(model, os.path.join(model_path, "model_best_loss.pth"))
            torch.save(model, os.path.join(model_path, "model_best_metric.pth"))

        # Updates learning rate policy if scheduler is used
        if args.lr_scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(total_epoch_loss_val)
            elif args.lr_scheduler == "poly":
                scheduler.step(current_step=epoch)
        else:
            pass

        # Prints checkpoint every 100 epochs
        if epoch % 100 == 0:
            torch.save(model, os.path.join(model_path, f"model_{epoch}_epochs.pth"))
            torch.save(model, os.path.join(model_path, f"model_latest.pth"))
            if fold is not None:
                print(f"Epoch: {epoch:04d} (model {model_name}, fold {fold})")
            else:
                print(f"Epoch: {epoch:04d} (model {model_name})")
            print(f"Training loss (EWMA):       {ewma_losses_train[-1]:.4f}")
            if args.is_classification:
                print(f"Training accuracy (EWMA):   {ewma_metric_train[-1]:.4f}")
            else:
                print(f"Training RMSE (EWMA):       {ewma_metric_train[-1]:.4f}")
            print(f"Validation loss (EWMA):     {ewma_losses_val[-1]:.4f}")
            if args.is_classification:
                print(f"Validation accuracy (EWMA): {ewma_metric_val[-1]:.4f}")
            else:
                print(f"Validation RMSE (EWMA):     {ewma_metric_val[-1]:.4f}")
            print()
        
            # Make training plot
            make_train_plot(model_path, ewma_losses_train, ewma_losses_val, ewma_metric_train, ewma_metric_val, args.is_classification)
    
    print(f"Training finished.\n")
    
    # Make training plot
    make_train_plot(model_path, ewma_losses_train, ewma_losses_val, ewma_metric_train, ewma_metric_val, args.is_classification)
