import os, pickle

import numpy as np

from arterial_net.test.utils import draw_roc, draw_regression_plot, draw_probability_distribution, compute_interpolated_roc_curve, draw_roc_folds
from arterial_net.utils.metrics import compute_accuracy, compute_rmse, compute_classification_metrics

import torch

def run_testing(root, model_name, test_loader, model=None, device="cpu", fold=None, is_classification=False):
    """
    Performs testing of a model over a test set. If the model is not input, it loads the 
    best model (model_best.pth) from the corresponding model dir.

    Parameters
    ----------
    root : string or path-like object
        Path to root folder.
    model_name : string
        Name of the model.
    test_loader : torch_geometric.loader.DataLoader
        DataLoader for testing.
    model : torch.nn.Module, optional
        Model to test. The default is None, which means that it will 
        load the `model_best.pth` from the model_path.
    device : string, optional
        Device to use for testing. The default is "cpu".
    fold : int, optional
        Fold number. The default is None.
    is_classification : bool, optional
        Whether the task is is_classification or regression. The default is False.
    """
    def test_step(model, graph):
        """
        Performs a testing step for a single graph.

        Parameters
        ----------
        model : torch.nn.Module
            Model to test.
        graph : torch_geometric.data.Batch
            Graph to test.

        Returns
        -------
        pred : torch.tensor
            Predictions for the graph.
        label : torch.tensor
            Labels for the graph.
        metric : float
            Accuracy for the graph if is_classification is True otherwise RMSE.
        """
        # Set model in evaluation mode
        model.eval()
        # In validation we do not keep track of gradients
        with torch.no_grad():
            # Perform inference with single graph
            pred = model(graph.to(device))
        if is_classification:
            # Get label from graph
            label = graph.y_class
            # Compute accuracy for testing
            metric = compute_accuracy(pred.argmax(dim=1), graph.y_class)
        else:
            # Get label from graph
            label = graph.y
            # Compute RMSE for testing
            metric = compute_rmse(pred, graph.y)
        return pred, label, metric
    # Define model path
    model_path = os.path.join(root, "models", model_name)
    if fold is not None:
        model_path = os.path.join(model_path, f"fold_{fold}")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Performs testing with best model (minimum validation loss)
    if model is None:
        # model = torch.load(os.path.join(model_path, "model_best.pth"))
        model = torch.load(os.path.join(model_path, "model_best_acc.pth"))
    if not os.path.isdir(os.path.join(model_path, "test")): os.mkdir(os.path.join(model_path, "test"))
    if not os.path.isdir(os.path.join(model_path, "test", "labels")): os.mkdir(os.path.join(model_path, "test", "labels"))
    if not os.path.isdir(os.path.join(model_path, "test", "preds")): os.mkdir(os.path.join(model_path, "test", "preds"))
    preds, labels, metric_test, class_labels = [], [], [], []
    for graph in test_loader:
        pred, label, met = test_step(model, graph)
        if is_classification:
            preds.append(1 - pred.cpu().numpy()[0][0])
            labels.append(label.cpu().numpy()[0])
            metric_test.append(met)
            class_labels.append(graph.cpu().y_class.numpy()[0])
        else:
            preds.append(pred.cpu().numpy()[0][0])
            labels.append(label.cpu().numpy()[0])
            metric_test.append(met)
            class_labels.append(graph.cpu().y_class.numpy()[0])
        np.save(os.path.join(model_path, "test", "labels", f"{graph.id[0]}.npy"), label.cpu())
        np.save(os.path.join(model_path, "test", "preds", f"{graph.id[0]}.npy"), pred.cpu())

    # Saves testing metrics in test directory
    if is_classification:
        np.savetxt(os.path.join(model_path, "test", "accuracy.out"), metric_test)
        np.savetxt(os.path.join(model_path, "test", "accuracy_mean.out"), [np.mean(metric_test), np.std(metric_test)])
    else:
        np.savetxt(os.path.join(model_path, "test", "rmse.out"), metric_test)
        np.savetxt(os.path.join(model_path, "test", "rmse_mean.out"), [np.mean(metric_test), np.std(metric_test)])

    # Draw ROC curve
    roc_auc, optimal_threshold = draw_roc(preds, labels, class_labels, os.path.join(model_path, "test", "roc.png"))
    # Draw probability distribution
    draw_probability_distribution(preds, class_labels, optimal_threshold, os.path.join(model_path, "test", "probability_distribution.png"))
    # Draw regression plot 
    if not is_classification:
        draw_regression_plot(preds, labels, class_labels, optimal_threshold, os.path.join(model_path, "test", "regression_plot.png"))
    
    accuracy, precision, sensitivity, specificity, f1, weighted_f1, mcc = compute_classification_metrics(preds, labels, optimal_threshold)

    # Print testing metrics
    print("------------------------------------------------ Testing metrics")
    print(f"ROC AUC:     {roc_auc:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1:          {f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"MCC:         {mcc:.4f}")

    print(f"Testing metric (lowest validation loss) was {np.mean(metric_test):.4f}")
    print()

    if fold is not None:
        compute_results_over_folds(root, model_name)

def compute_results_over_folds(root, model_name):
    """
    Computes testing results over folds. Aggregates results from different folds, computing 
    a ROC curve with 95% CI bands.

    Parameters
    ----------
    root : string or path-like object
        Path to root folder.
    model_name : string
        Name of the model.
    """
    model_dir = os.path.join(root, "models", model_name)
    results_folds = {}

    aucs = []
    accuraies = []
    precisions = []
    sensitivities = []
    specificities = []
    f1s = []
    weighted_f1s = []
    mccs = []

    n_points = 100

    for fold in sorted([fold.split("_")[-1] for fold in os.listdir(model_dir) if fold.startswith("fold")]):
        results_folds[fold] = {}
        results_folds[fold]["preds"] = 1 - np.array([np.load(os.path.join(model_dir, f"fold_{fold}", "test", "preds", graph))[0][0] for graph in sorted(os.listdir(os.path.join(model_dir, f"fold_{fold}", "test", "preds")))])
        results_folds[fold]["labels"] = [np.load(os.path.join(model_dir, f"fold_{fold}", "test", "labels", graph))[0] for graph in sorted(os.listdir(os.path.join(model_dir, f"fold_{fold}", "test", "labels")))]
        results_folds[fold]["labels"] = np.where(np.array(results_folds[fold]["labels"]) > 0.5, 1, 0)

        # Compute ROC curve and interpolate in 100 points to draw tpr and fpr
        auc_score, fpr, tpr, thresholds, fpr_interp, tpr_interp, thresholds_interp, optimal_threshold = compute_interpolated_roc_curve(results_folds[fold]["preds"], results_folds[fold]["labels"], n_points=n_points)
        results_folds[fold]["auc"] = auc_score
        results_folds[fold]["fpr"] = fpr
        results_folds[fold]["tpr"] = tpr
        results_folds[fold]["thresholds"] = thresholds
        results_folds[fold]["fpr_interp"] = fpr_interp
        results_folds[fold]["tpr_interp"] = tpr_interp
        results_folds[fold]["thresholds_interp"] = thresholds_interp
        results_folds[fold]["optimal_threshold"] = optimal_threshold

        # Compute metrics
        accuracy, precision, sensitivity, specificity, f1, weighted_f1, mcc = compute_classification_metrics(results_folds[fold]["preds"], results_folds[fold]["labels"], optimal_threshold)

        aucs.append(auc_score)
        accuraies.append(accuracy)
        precisions.append(precision)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1s.append(f1)
        weighted_f1s.append(weighted_f1)
        mccs.append(mcc)

        results_folds[fold]["accuracy"] = accuracy
        results_folds[fold]["precision"] = precision
        results_folds[fold]["sensitivity"] = sensitivity
        results_folds[fold]["specificity"] = specificity
        results_folds[fold]["f1"] = f1
        results_folds[fold]["weighted_f1"] = weighted_f1
        results_folds[fold]["mcc"] = mcc

    # Draw ROC curve over folds
    draw_roc_folds(results_folds, n_points=n_points, output_path=os.path.join(model_dir, "roc_folds.png"))

    # Print final results
    print("------------------------------------------------ Final results")
    print(f"ROC AUC:     {np.mean(aucs):.4f} ({np.mean(aucs) - 1.96 * np.std(aucs) / np.sqrt(len(aucs)):.4f} - {np.mean(aucs) + 1.96 * np.std(aucs) / np.sqrt(len(aucs)):.4f})")
    print(f"Accuracy:    {np.mean(accuraies):.4f} ({np.mean(accuraies) - 1.96 * np.std(accuraies) / np.sqrt(len(accuraies)):.4f} - {np.mean(accuraies) + 1.96 * np.std(accuraies) / np.sqrt(len(accuraies)):.4f})")
    print(f"Precision:   {np.mean(precisions):.4f} ({np.mean(precisions) - 1.96 * np.std(precisions) / np.sqrt(len(precisions)):.4f} - {np.mean(precisions) + 1.96 * np.std(precisions) / np.sqrt(len(precisions)):.4f})")
    print(f"Sensitivity: {np.mean(sensitivities):.4f} ({np.mean(sensitivities) - 1.96 * np.std(sensitivities) / np.sqrt(len(sensitivities)):.4f} - {np.mean(sensitivities) + 1.96 * np.std(sensitivities) / np.sqrt(len(sensitivities)):.4f})")
    print(f"Specificity: {np.mean(specificities):.4f} ({np.mean(specificities) - 1.96 * np.std(specificities) / np.sqrt(len(specificities)):.4f} - {np.mean(specificities) + 1.96 * np.std(specificities) / np.sqrt(len(specificities)):.4f})")
    print(f"F1:          {np.mean(f1s):.4f} ({np.mean(f1s) - 1.96 * np.std(f1s) / np.sqrt(len(f1s)):.4f} - {np.mean(f1s) + 1.96 * np.std(f1s) / np.sqrt(len(f1s)):.4f})")
    print(f"Weighted F1: {np.mean(weighted_f1s):.4f} ({np.mean(weighted_f1s) - 1.96 * np.std(weighted_f1s) / np.sqrt(len(weighted_f1s)):.4f} - {np.mean(weighted_f1s) + 1.96 * np.std(weighted_f1s) / np.sqrt(len(weighted_f1s)):.4f})")
    print(f"MCC:         {np.mean(mccs):.4f} ({np.mean(mccs) - 1.96 * np.std(mccs) / np.sqrt(len(mccs)):.4f} - {np.mean(mccs) + 1.96 * np.std(mccs) / np.sqrt(len(mccs)):.4f})")
    print()

    with open(os.path.join(model_dir, "results_folds.pickle"), "wb") as f:
        pickle.dump(results_folds, f)