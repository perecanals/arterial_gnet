import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
plt.grid(lw = 0.5, alpha = 0.5)

import warnings
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def draw_roc(preds, labels, class_labels, output_path=None, ax=None):
    """
    Draw a ROC curve.

    Parameters
    ----------
    preds : list
        Predictions.
    labels : list
        Labels (if regression, otherwise they are the same as clas_labels).
    class_labels : list
        Class labels.
    output_path : str
        Path to save the plot. Default is None.
    """
    # Build dataframe
    df = pd.DataFrame({"predictions": preds, "labels": labels, "class": class_labels})
    df["class"] = np.where(df["class"] > 0.5, 1, 0) # Binarize class labels

    # Draw ROC
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure
    ax.grid(alpha=0.5, lw=0.5)
    plt.rcParams.update({'font.size': 14})
    fpr, tpr, thresholds = roc_curve(df["class"], df["predictions"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"Test ROC (AUC = {roc_auc:.2f})", lw=1.25, color="crimson")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize = 8)
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    return roc_auc

def draw_pr_curve(preds, labels, class_labels, output_path=None, ax=None):
    """
    Draw a precision-recall curve.

    Parameters
    ----------
    preds : list
        Predictions.
    labels : list
        Labels.
    class_labels : list
        Class labels.
    output_path : str
        Path to save the plot. Default is None.
    """
    # Build dataframe
    df = pd.DataFrame({"predictions": preds, "labels": labels, "class": class_labels})
    df["class"] = np.where(df["class"] > 0.5, 1, 0) # Binarize class labels

    # Draw PR curve
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure
    ax.grid(alpha=0.5, lw=0.5)
    plt.rcParams.update({'font.size': 14})
    precision, recall, thresholds = precision_recall_curve(df["class"], df["predictions"])
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, label=f"Test PR (AUC = {pr_auc:.2f})", lw=1.25, color="crimson")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize = 8)
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    return pr_auc

def select_optimal_threshold(preds, class_labels, criterion="youden"):
    """
    Select optimal threshold for classification.

    Parameters
    ----------
    preds : list
        Predictions.
    class_labels : list
        Class labels.
    criterion : str
        Method to select the optimal threshold. Default is "youden".

    Returns
    -------
    float
        Optimal threshold.
    """
    # Build dataframe
    df = pd.DataFrame({"predictions": preds, "class": class_labels})
    df["class"] = np.where(df["class"] > 0.5, 1, 0) # Binarize class labels

    if criterion == "youden":
        # Compute optimal threshold, maximum Youden index
        fpr, tpr, thresholds = roc_curve(df["class"], df["predictions"])
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    elif criterion == "f1":
        # Compute optimal threshold for maximum f1
        precision, recall, thresholds_pr = precision_recall_curve(df["class"], df["predictions"])
        f1 = 2 * (precision * recall) / (precision + recall)
        optimal_threshold = thresholds_pr[np.argmax(f1)]
    
    return optimal_threshold

def draw_regression_plot(preds, labels, class_labels, optimal_threshold=None, output_path=None, ax=None):
    """
    Draw a regression plot.

    Parameters
    ----------
    preds : list
        Predictions.
    labels : list
        Labels.
    class_labels : list
        Class labels.
    optimal_threshold : float
        Optimal threshold for classification. Default is None.
    output_path : str
        Path to save the plot. Default is None.
    """
    # Build dataframe
    df = pd.DataFrame({"Predictions": preds, "Labels": labels, "Class": class_labels})

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure
    ax.grid(alpha=0.5, lw=0.5)
    plt.rcParams.update({'font.size': 14})

    sns.scatterplot(data=df, x="Labels", y="Predictions", hue="Class", palette=["forestgreen", "orange", "crimson"], alpha=0.6, s = 40, ax = ax)
    
    if optimal_threshold is not None:
        x_min, x_max = ax.get_xlim()
        ax.axhline(y = np.mean(optimal_threshold), color = 'mediumblue', linestyle = '-', lw = 1, label = "Optimal threshold")
        ax.set_xlim([x_min, x_max])

    ax.set_xlabel("True T1A (min)")
    ax.set_ylabel("Predicted value")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize = 8)
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    return fig, ax

def draw_probability_distribution(preds, labels, optimal_threshold=None, output_path=None, ax=None):
    """
    Draw a probability distribution.

    Parameters
    ----------
    preds : list
        Predictions.
    labels : list
        Labels.
    threshold : float
        Threshold for classification.
    output_path : str
        Path to save the plot. Default is None.
    """
    # Build dataframe
    df = pd.DataFrame({"predictions": preds, "labels": labels})

    # Identify tp, tn, fp, fn based on predictions, labels, and threshold
    df["tp"] = np.where((df["predictions"] >= optimal_threshold) & (df["labels"] == 1), True, False)
    df["tn"] = np.where((df["predictions"] < optimal_threshold) & (df["labels"] == 0), True, False)
    df["fp"] = np.where((df["predictions"] >= optimal_threshold) & (df["labels"] == 0), True, False)
    df["fn"] = np.where((df["predictions"] < optimal_threshold) & (df["labels"] == 1), True, False)

    # Plot setup
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure
    ax.grid(alpha=0.5, lw=0.5)

    # Histogram settings
    bins = np.linspace(0, 1, 51) # 100 bins between 0 and 1
    alpha = 0.3 # Transparency for histograms

    # Plot histograms for tp, tn, fp, fn
    sns.histplot(df[df["tp"]]["predictions"], color="blue", kde=False, ax=ax, alpha=alpha, bins=bins, label="TP")
    sns.histplot(df[df["tn"]]["predictions"], color="green", kde=False, ax=ax, alpha=alpha, bins=bins, label="TN")
    sns.histplot(df[df["fp"]]["predictions"], color="red", kde=False, ax=ax, alpha=alpha, bins=bins, label="FP")
    sns.histplot(df[df["fn"]]["predictions"], color="orange", kde=False, ax=ax, alpha=alpha, bins=bins, label="FN")

    # If y_lim_max is higher than 60, set it to 60
    y_lim_max = 60
    if max(ax.get_ylim()) > y_lim_max:
        ax.set_ylim([0, y_lim_max])

    # Additional plot adjustments
    ax.axvline(x=optimal_threshold, color="red", lw=1) # Threshold line
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=10)
    plt.tight_layout()

    # Save plot if an output path is provided
    if output_path is not None:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def compute_interpolated_roc_curve(preds, labels, n_points=100, threshold_label=0.5):
    """
    Compute an interpolated ROC curve.

    Parameters
    ----------
    preds : list
        Predictions.
    labels : list
        Labels.
    n_points : int
        Number of points to interpolate. Default is 100.
    threshold_label : float
        Threshold for classification. Default is 0.5.

    Returns
    -------
    fpr : list
        False positive rate.
    tpr : list
        True positive rate.
    thresholds : list
        Thresholds.
    """
    labels = np.array(labels) >= threshold_label
    fpr, tpr, thresholds = roc_curve(labels, preds)
    auc_score = auc(fpr, tpr)

    # Get optimal threshold
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    fpr_interp = np.linspace(0, 1, n_points)
    tpr_interp = np.interp(fpr_interp, fpr, tpr)
    thresholds_interp = np.interp(fpr_interp, fpr, thresholds)

    return auc_score, fpr, tpr, thresholds, fpr_interp, tpr_interp, thresholds_interp, optimal_threshold

def compute_interpolated_pr_curve(preds, labels, n_points=100, threshold_label=0.5):
    """
    Compute an interpolated ROC curve.

    Parameters
    ----------
    preds : list
        Predictions.
    labels : list
        Labels.
    n_points : int
        Number of points to interpolate. Default is 100.
    threshold_label : float
        Threshold for classification. Default is 0.5.

    Returns
    -------
    fpr : list
        False positive rate.
    tpr : list
        True positive rate.
    thresholds : list
        Thresholds.
    """
    labels = np.array(labels) >= threshold_label
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    auc_score = auc(recall, precision)
    # IDK why, but thresholds have one less element than precision and recall
    thresholds = np.append(thresholds, 1.)

    # Get optimal threshold
    f1_score = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_score)]

    recall_interp = np.linspace(0, 1, n_points)
    precision_interp = np.interp(recall_interp, recall[::-1], precision[::-1])
    thresholds_interp = np.interp(recall_interp, recall[::-1], thresholds[::-1])

    return auc_score, recall, precision, thresholds, recall_interp, precision_interp, thresholds_interp, optimal_threshold

def draw_roc_folds(results_folds, n_points=100, output_path=None):
    """
    Draws aggergated ROC curve over folds. Plots an average curve, resulting on averaging
    the interpolated tpr and fpr, as well as confidence intervals resulting from these curves
    as bands.
    
    It also plots each individual ROC curve in the background.

    Parameters
    ----------
    results_folds : dict
        Results over folds.
    """
    aucs = []

    mean_fpr_interp = np.linspace(0, 1, n_points)
    mean_tpr_interp = np.ndarray([len(results_folds), n_points])

    for idx, fold in enumerate(results_folds.keys()):
        mean_tpr_interp[idx] = results_folds[fold]["tpr_interp"]
        aucs.append(results_folds[fold]["roc_auc"])

    # Compute mean and confidence intervals
    mean_tpr = np.mean(mean_tpr_interp, axis=0)
    mean_tpr = np.maximum(mean_tpr, 0)
    mean_tpr = np.minimum(mean_tpr, 1)
    std_tpr = np.std(mean_tpr_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr / np.sqrt(len(results_folds)), 1)
    tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr / np.sqrt(len(results_folds)), 0)

    # Draw ROC
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.grid(alpha=0.5, lw=0.5)
    plt.rcParams.update({'font.size': 14})
    ax.plot(mean_fpr_interp, mean_tpr, label=f"Test ROC (AUC = {np.mean(aucs):.2f} [{np.mean(aucs) - 1.96 * np.std(aucs) / np.sqrt(len(aucs)):.2f}-{np.mean(aucs) + 1.96 * np.std(aucs) / np.sqrt(len(aucs)):.2f}])", lw=1.25, color="crimson")
    ax.fill_between(mean_fpr_interp, tprs_lower, tprs_upper, color="crimson", alpha=0.2, lw=0)
    for idx in range(len(results_folds)):
        ax.plot(mean_fpr_interp, mean_tpr_interp[idx], color="crimson", alpha = 0.2, lw = 0.5)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize = 10)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    results = {
        "mean_fpr": mean_fpr_interp.tolist(),
        "mean_tpr": mean_tpr.tolist(),
        "std_tpr": std_tpr.tolist(),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "tprs_upper": tprs_upper.tolist(),
        "tprs_lower": tprs_lower.tolist(),
        "optimal_tpr": 0,
        "optimal_fpr": 0,
        "optimal_sensitivity": 0,
        "optimal_specificity": 0
    }

    return results

def draw_pr_folds(results_folds, n_points=100, output_path=None):
    """
    Draws aggergated ROC curve over folds. Plots an average curve, resulting on averaging
    the interpolated precision and fpr, as well as confidence intervals resulting from these curves
    as bands.
    
    It also plots each individual ROC curve in the background.

    Parameters
    ----------
    results_folds : dict
        Results over folds.
    """
    aucs = []

    mean_recall_interp = np.linspace(0, 1, n_points)
    mean_precision_interp = np.ndarray([len(results_folds), n_points])

    for idx, fold in enumerate(results_folds.keys()):
        mean_precision_interp[idx] = results_folds[fold]["precision_interp"]
        aucs.append(results_folds[fold]["pr_auc"])

    # Compute mean and confidence intervals
    mean_precision = np.mean(mean_precision_interp, axis=0)
    mean_precision = np.maximum(mean_precision, 0)
    mean_precision = np.minimum(mean_precision, 1)
    std_precision = np.std(mean_precision_interp, axis=0)
    precisions_upper = np.minimum(mean_precision + 1.96 * std_precision / np.sqrt(len(results_folds)), 1)
    precisions_lower = np.maximum(mean_precision - 1.96 * std_precision / np.sqrt(len(results_folds)), 0)

    # Draw ROC
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.grid(alpha=0.5, lw=0.5)
    plt.rcParams.update({'font.size': 14})
    ax.plot(mean_recall_interp, mean_precision, label=f"Test PR (AUC = {np.mean(aucs):.2f} [{np.mean(aucs) - 1.96 * np.std(aucs) / np.sqrt(len(aucs)):.2f}-{np.mean(aucs) + 1.96 * np.std(aucs) / np.sqrt(len(aucs)):.2f}])", lw=1.25, color="crimson")
    ax.fill_between(mean_recall_interp, precisions_lower, precisions_upper, color="crimson", alpha=0.2, lw=0)
    for idx in range(len(results_folds)):
        ax.plot(mean_recall_interp, mean_precision_interp[idx], color="crimson", alpha = 0.2, lw = 0.5)
    # ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower right", fontsize = 10)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    results = {
        "mean_recall": mean_recall_interp.tolist(),
        "mean_precision": mean_precision.tolist(),
        "std_precision": std_precision.tolist(),
        "mean_pr_auc": float(np.mean(aucs)),
        "std_pr_auc": float(np.std(aucs)),
        "precisions_upper": precisions_upper.tolist(),
        "precisions_lower": precisions_lower.tolist(),
        "optimal_precision": 0,
        "optimal_recall": 0,
        "optimal_sensitivity": 0,
        "optimal_specificity": 0
    }

    return results

def compute_classification_metrics(preds, class_labels, threshold=0.5, threshold_label=0.5, output_path=None):
    preds = np.array(preds) >= threshold
    class_labels = np.array(class_labels) >= threshold_label

    tp = np.sum((preds == 1) & (class_labels == 1))
    tn = np.sum((preds == 0) & (class_labels == 0))
    fp = np.sum((preds == 1) & (class_labels == 0))
    fn = np.sum((preds == 0) & (class_labels == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if precision + sensitivity > 0 else 0
    weighted_f1 = f1_score(class_labels, preds, average='weighted')
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if tp + fp > 0 else 0
    npv = tn / (tn + fn) if tn + fn > 0 else 0

    # Plot confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots()
    sns.heatmap(cm, cmap=plt.cm.Blues, fmt='g', ax=ax)
    ax.set_title(f'Confusion matrix (opt. threshold: {threshold:.2f})')
    ax.set_ylabel(f'True label')
    ax.set_xlabel(f'Predicted label')
    # Set text at the center of the cells (for some reason it does not print these)
    ax.text(0.33, 0.5, f"{cm[0, 0]}", fontsize=12, color="white")
    ax.text(1.38, 0.5, f"{cm[0, 1]}", fontsize=12, color="black")
    ax.text(0.38, 1.5, f"{cm[1, 0]}", fontsize=12, color="black")
    ax.text(1.38, 1.5, f"{cm[1, 1]}", fontsize=12, color="black")

    if output_path is not None:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    return accuracy, precision, sensitivity, specificity, f1, weighted_f1, mcc, ppv, npv