import os

import numpy as np

from arterial_net.train.lr_schedulers import PolyLRScheduler
from arterial_net.train.utils import make_train_plot
from arterial_net.utils.metrics import compute_accuracy, compute_rmse

import torch

def run_training(root, model, model_name, train_loader, val_loader, loss_function, total_epochs=500, optim="adam", learning_rate=0.01, lr_scheduler=True, device="cpu", fold=None, is_classification=False):
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
    total_epochs : int, optional
        Number of epochs to train. The default is 500.
    optim : string, optional
        Optimizer to use. The default is "adam".
    learning_rate : float, optional
        Initial learning rate. The default is 0.01.
    lr_scheduler : bool, optional
        Whether to use a learning rate scheduler. The default is True.
    device : string, optional
        Device to use for training. The default is "cpu".
    fold : int, optional
        Fold number. The default is None.
    classification : bool, optional
        Whether the task is classification or regression. The default is False.
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
        # Compute loss
        if is_classification:
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
        return loss, metric

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
            # Compute validation loss
            if is_classification:
                loss = loss_function(out, batch.y_class)
                # Compute accuracy for validation
                metric = compute_accuracy(out.argmax(dim=1), batch.y_class)
            else:
                loss = loss_function(out, batch.y)
                # Compute RMSE for training
                metric = compute_rmse(out, batch.y)
        return loss, metric
    
    print("\n------------------------------------------------ Training parameters")
    print(f"Batch size:                     {train_loader.batch_size}")
    print(f"Total epochs:                   {total_epochs}")
    print(f"Optimizer:                      {optim}")
    print(f"Initial learning rate:          {learning_rate}")
    print(f"Learning rate scheduler:        {lr_scheduler}")
    print(f"Loss function:                  {loss_function}")
    print(f"Running on device:              {device} \n")

    # Define model path
    model_path = os.path.join(root, "models", model_name)
    if fold is not None:
        model_path = os.path.join(model_path, f"fold_{fold}")
    os.makedirs(model_path, exist_ok=True)
    
    # Define optimizer and scheduler
    if optim == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-03
            )
    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=0.99, 
            weight_decay=1e-03)
    
    if lr_scheduler:
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min', 
        #     factor=0.1, 
        #     patience=50, 
        #     threshold=0.001, 
        #     threshold_mode='rel', 
        #     eps=1e-06,
        #     verbose=True
        #     )
        scheduler = PolyLRScheduler(optimizer, 
                                    initial_lr=learning_rate,
                                    max_steps=total_epochs,
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

    # Starts training
    for epoch in range(0, total_epochs + 1):
        # Initializes in-epoch variables
        total_epoch_loss_train, total_epoch_loss_val = 0, 0
        metric_train_epoch_list, metric_val_epoch_list = [], []
        num_graphs_train, num_graphs_val = 0, 0

        # Iterates over training DataLoader and performs a training step for each batch
        for batch in train_loader:
            # Performs training step
            loss_train, met_train = train_step(model, batch)
            # Adds loss to epoch loss
            total_epoch_loss_train += loss_train.detach()
            # Adds accuracy to list for epoch
            metric_train_epoch_list.append(met_train)
            # Update the number of graphs
            num_graphs_train += batch.segment_data.batch.max().item() + 1

        # Divides epoch accumulated training loss by number of graphs
        total_epoch_loss_train = total_epoch_loss_train / num_graphs_train
        # Appends training loss to tracking
        losses_train.append(total_epoch_loss_train)
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
            loss_val, met_val = val_step(model, batch)
            # Adds loss to epoch loss
            total_epoch_loss_val += loss_val
            # Adds accuracy to list for epoch
            metric_val_epoch_list.append(met_val)
            # Update the number of graphs
            num_graphs_val += batch.segment_data.batch.max().item() + 1

        # Divides epoch accumulated validation loss by number of graphs
        total_epoch_loss_val = total_epoch_loss_val / num_graphs_val
        # Appends validation loss to tracking
        losses_val.append(total_epoch_loss_val)
        # Computes mean validation accuracy across batches
        metric_val_epoch = np.mean(metric_val_epoch_list)
        # Appends validation accuracy to tracking
        metric_val.append(metric_val_epoch)
        # Computes exponentially weighted moving average for validation loss and accuracy
        ewma_losses_val.append(0.9 * ewma_losses_val[-1] + 0.1 * total_epoch_loss_val if len(ewma_losses_val) > 0 else total_epoch_loss_val)
        ewma_metric_val.append(0.9 * ewma_metric_val[-1] + 0.1 * metric_val_epoch if len(ewma_metric_val) > 0 else metric_val_epoch)

        # Saves best model in terms of validation loss
        if total_epoch_loss_val == np.amin(ewma_losses_val):
            torch.save(model, os.path.join(model_path, "model_best.pth"))
        # Saves best model in terms of validation accuracy
        if metric_val_epoch == np.max(ewma_metric_val):
            torch.save(model, os.path.join(model_path, "model_best_acc.pth"))

        # Updates learning rate policy if scheduler is used
        if lr_scheduler:
            # scheduler.step(total_epoch_loss_val)
            scheduler.step(current_step=epoch)
        else:
            pass

        # Prints checkpoint every 100 epochs
        if epoch % 100 == 0:
            if fold is not None:
                print(f"Epoch: {epoch:03d} (model {model_name}, fold {fold})")
            else:
                print(f"Epoch: {epoch:03d} (model {model_name})")
            print(f"Training loss (EWMA):       {ewma_losses_train[-1]:.4f}")
            if is_classification:
                print(f"Training accuracy (EWMA):   {ewma_metric_train[-1]:.4f}")
            else:
                print(f"Training RMSE (EWMA):       {ewma_metric_train[-1]:.4f}")
            print(f"Validation loss (EWMA):     {ewma_losses_val[-1]:.4f}")
            if is_classification:
                print(f"Validation accuracy (EWMA): {ewma_metric_val[-1]:.4f}")
            else:
                print(f"Validation RMSE (EWMA):     {ewma_metric_val[-1]:.4f}")
            print()
        
            # Make training plot
            make_train_plot(model_path, ewma_losses_train, ewma_losses_val, ewma_metric_train, ewma_metric_val, is_classification)
    
    print(f"Training finished.\n")
    
    # Make training plot
    make_train_plot(model_path, ewma_losses_train, ewma_losses_val, ewma_metric_train, ewma_metric_val, is_classification)
