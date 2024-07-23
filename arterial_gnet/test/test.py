import os, pickle, json

import numpy as np

from arterial_net.test.utils import draw_roc, draw_pr_curve, draw_regression_plot, draw_probability_distribution, compute_interpolated_roc_curve, draw_roc_folds
from arterial_net.utils.metrics import compute_accuracy, compute_rmse, compute_classification_metrics

import torch

def run_testing(root, model_name, test_loader, model="best", device="cpu", fold=None, is_classification=False):
    """
    Performs testing of a model over a test set. If the model is not input, it loads the 
    best model (model_best.pth) from the corresponditestng model dir.

    Parameters
    ----------
    root : string or path-like object
        Path to root folder.
    model_name : string
        Name of the model.
    test_loader : torch_geometric.loader.DataLoader
        DataLoader for testing.
    model : str or torch.nn.Module, optional
        Model to test. The default is "best", which loads the best model from the model dir.
        If a torch.nn.Module is input, it tests the model directly.
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

    # Performs testing with best model (minimum validation loss)
    if isinstance(model, str):
        if model == "best":
            model = torch.load(os.path.join(model_path, "model_best_loss.pth"))
            test_dir = os.path.join(model_path, "test_best")
            test_dir_suffix = "test_best"
        elif model == "latest":
            model = torch.load(os.path.join(model_path, "model_latest.pth"))
            test_dir = os.path.join(model_path, "test_latest")
            test_dir_suffix = "test_latest"
        elif model == "metric":
            model = torch.load(os.path.join(model_path, "model_best_metric.pth"))
            test_dir = os.path.join(model_path, "test_metric")
            test_dir_suffix = "test_metric"
        else:
            model = torch.load(os.path.join(model_path, "model_best_loss.pth"))
            test_dir = os.path.join(model_path, "test")
            test_dir_suffix = "test"
    elif isinstance(model, torch.nn.Module):
        test_dir = os.path.join(model_path, "test")
        test_dir_suffix = "test"
    else:
        raise ValueError("Model must be a string or a torch.nn.Module.")
            
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, "preds"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "class_labels"), exist_ok=True)

    preds, labels, metric_test, class_labels = [], [], [], []
    for graph in test_loader:
        pred, label, met = test_step(model, graph)
        if is_classification:
            # There are three classes, but we are interested in the probability of the positive class (1 - the prob of the negative class)
            preds.append(1 - pred.cpu().numpy()[0][0])
            labels.append(label.cpu().numpy()[0])
            metric_test.append(met)
            class_labels.append(graph.cpu().y_class.numpy()[0])
        else:
            preds.append(pred.cpu().numpy()[0][0])
            labels.append(label.cpu().numpy()[0])
            metric_test.append(met)
            class_labels.append(graph.cpu().y_class.numpy()[0])
        np.save(os.path.join(test_dir, "labels", f"{graph.id[0]}.npy"), label.cpu())
        np.save(os.path.join(test_dir, "preds", f"{graph.id[0]}.npy"), pred.cpu())
        np.save(os.path.join(test_dir, "class_labels", f"{graph.id[0]}.npy"), graph.y_class.cpu())

    # Saves testing metrics in test directory
    if is_classification:
        np.savetxt(os.path.join(test_dir, "accuracy.out"), metric_test)
        np.savetxt(os.path.join(test_dir, "accuracy_mean.out"), [np.mean(metric_test), np.std(metric_test)])
    else:
        np.savetxt(os.path.join(test_dir, "rmse.out"), metric_test)
        np.savetxt(os.path.join(test_dir, "rmse_mean.out"), [np.mean(metric_test), np.std(metric_test)])

    # Draw ROC curve
    roc_auc, optimal_threshold = draw_roc(preds, labels, class_labels, os.path.join(test_dir, f"roc_{test_dir_suffix}.png"))
    pr_auc = draw_pr_curve(preds, labels, class_labels, os.path.join(test_dir, f"roc_{test_dir_suffix}.png"))
    if is_classification:
        # Draw probability distribution
        draw_probability_distribution(preds, class_labels, optimal_threshold, os.path.join(test_dir, f"probability_distribution_{test_dir_suffix}.png"))
    else:
        # Draw regression plot 
        draw_regression_plot(preds, labels, class_labels, optimal_threshold, os.path.join(test_dir, f"regression_plot_{test_dir_suffix}.png"))

    accuracy, precision, sensitivity, specificity, f1, weighted_f1, mcc = compute_classification_metrics(preds, class_labels, optimal_threshold, threshold_label=0.5)

    # Print testing metrics
    print(f"------------------------------------------------ Testing metrics {test_dir_suffix}")
    print(f"ROC AUC:           {roc_auc:.2f}")
    print(f"PR AUC:            {pr_auc:.2f}")
    print(f"Optimal threshold: {optimal_threshold:.2f}")
    print(f"Accuracy:          {accuracy:.2f}")
    print(f"Precision:         {precision:.2f}")
    print(f"Sensitivity:       {sensitivity:.2f}")
    print(f"Specificity:       {specificity:.2f}")
    print(f"F1:                {f1:.2f}")
    print(f"Weighted F1:       {weighted_f1:.2f}")
    print(f"MCC:               {mcc:.2f}")

    if fold is not None:
        compute_results_over_folds(root, model_name, test_dir_suffix, is_classification)

def compute_results_over_folds(root, model_name, test_dir_suffix, is_classification=False):
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
    thresholds = []
    accuracies = []
    precisions = []
    sensitivities = []
    specificities = []
    f1s = []
    weighted_f1s = []
    mccs = []

    n_points = 100

    for fold in sorted([fold.split("_")[-1] for fold in os.listdir(model_dir) if fold.startswith("fold") and os.path.exists(os.path.join(model_dir, fold, test_dir_suffix))]):
        results_folds[fold] = {}
        if is_classification:
            results_folds[fold]["preds"] = 1 - np.array([np.load(os.path.join(model_dir, f"fold_{fold}", test_dir_suffix, "preds", graph))[0][0] for graph in sorted(os.listdir(os.path.join(model_dir, f"fold_{fold}", test_dir_suffix, "preds")))])
        else:
            results_folds[fold]["preds"] = np.array([np.load(os.path.join(model_dir, f"fold_{fold}", test_dir_suffix, "preds", graph))[0][0] for graph in sorted(os.listdir(os.path.join(model_dir, f"fold_{fold}", test_dir_suffix, "preds")))])
        results_folds[fold]["labels"] = np.array([np.load(os.path.join(model_dir, f"fold_{fold}", test_dir_suffix, "labels", graph))[0] for graph in sorted(os.listdir(os.path.join(model_dir, f"fold_{fold}", test_dir_suffix, "labels")))])
        results_folds[fold]["class_labels"] = np.array([np.load(os.path.join(model_dir, f"fold_{fold}", test_dir_suffix, "class_labels", graph))[0] for graph in sorted(os.listdir(os.path.join(model_dir, f"fold_{fold}", test_dir_suffix, "class_labels")))])

        # Compute ROC curve and interpolate in 100 points to draw tpr and fpr
        auc_score, fpr, tpr, thresholds_, fpr_interp, tpr_interp, thresholds_interp, optimal_threshold = compute_interpolated_roc_curve(results_folds[fold]["preds"], results_folds[fold]["class_labels"], n_points=n_points, threshold_label=0.5)
        results_folds[fold]["roc_auc"] = auc_score
        results_folds[fold]["fpr"] = fpr
        results_folds[fold]["tpr"] = tpr
        results_folds[fold]["thresholds"] = thresholds_
        results_folds[fold]["fpr_interp"] = fpr_interp
        results_folds[fold]["tpr_interp"] = tpr_interp
        results_folds[fold]["thresholds_interp"] = thresholds_interp
        results_folds[fold]["optimal_threshold"] = optimal_threshold

        # Compute metrics
        accuracy, precision, sensitivity, specificity, f1, weighted_f1, mcc = compute_classification_metrics(results_folds[fold]["preds"], results_folds[fold]["class_labels"], optimal_threshold, threshold_label=0.5)

        aucs.append(auc_score)
        thresholds.append(optimal_threshold)
        accuracies.append(accuracy)
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
    roc_results = draw_roc_folds(results_folds, n_points=n_points, output_path=os.path.join(model_dir, f"roc_folds_{test_dir_suffix}.png"))

    if len(aucs) > 1:
        # Print final results
        print(f"------------------------------------------------ Final results ({test_dir_suffix})")
        print(f"ROC AUC:     {np.mean(aucs):.2f} ({np.mean(aucs) - 1.96 * np.std(aucs) / np.sqrt(len(aucs)):.2f} - {np.mean(aucs) + 1.96 * np.std(aucs) / np.sqrt(len(aucs)):.2f})")
        print(f"Threshold:   {np.mean(thresholds):.2f} ({np.mean(thresholds) - 1.96 * np.std(thresholds) / np.sqrt(len(thresholds)):.2f} - {np.mean(thresholds) + 1.96 * np.std(thresholds) / np.sqrt(len(thresholds)):.2f})")
        print(f"Accuracy:    {np.mean(accuracies):.2f} ({np.mean(accuracies) - 1.96 * np.std(accuracies) / np.sqrt(len(accuracies)):.2f} - {np.mean(accuracies) + 1.96 * np.std(accuracies) / np.sqrt(len(accuracies)):.2f})")
        print(f"Precision:   {np.mean(precisions):.2f} ({np.mean(precisions) - 1.96 * np.std(precisions) / np.sqrt(len(precisions)):.2f} - {np.mean(precisions) + 1.96 * np.std(precisions) / np.sqrt(len(precisions)):.2f})")
        print(f"Sensitivity: {np.mean(sensitivities):.2f} ({np.mean(sensitivities) - 1.96 * np.std(sensitivities) / np.sqrt(len(sensitivities)):.2f} - {np.mean(sensitivities) + 1.96 * np.std(sensitivities) / np.sqrt(len(sensitivities)):.2f})")
        print(f"Specificity: {np.mean(specificities):.2f} ({np.mean(specificities) - 1.96 * np.std(specificities) / np.sqrt(len(specificities)):.2f} - {np.mean(specificities) + 1.96 * np.std(specificities) / np.sqrt(len(specificities)):.2f})")
        print(f"F1:          {np.mean(f1s):.2f} ({np.mean(f1s) - 1.96 * np.std(f1s) / np.sqrt(len(f1s)):.2f} - {np.mean(f1s) + 1.96 * np.std(f1s) / np.sqrt(len(f1s)):.2f})")
        print(f"Weighted F1: {np.mean(weighted_f1s):.2f} ({np.mean(weighted_f1s) - 1.96 * np.std(weighted_f1s) / np.sqrt(len(weighted_f1s)):.2f} - {np.mean(weighted_f1s) + 1.96 * np.std(weighted_f1s) / np.sqrt(len(weighted_f1s)):.2f})")
        print(f"MCC:         {np.mean(mccs):.2f} ({np.mean(mccs) - 1.96 * np.std(mccs) / np.sqrt(len(mccs)):.2f} - {np.mean(mccs) + 1.96 * np.std(mccs) / np.sqrt(len(mccs)):.2f})")
        print()

    with open(os.path.join(model_dir, f"results_folds_{test_dir_suffix}.pickle"), "wb") as f:
        pickle.dump(results_folds, f)
    with open(os.path.join(model_dir, f"roc_results_{test_dir_suffix}.json"), "w") as f:
        json.dump(roc_results, f)