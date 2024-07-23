import torch
import numpy as np

from sklearn.metrics import f1_score

def compute_accuracy(pred, label):
    return torch.mean((pred == label).float()).item()

def compute_rmse(pred, label):
    return torch.sqrt(torch.mean((pred - label) ** 2)).item()

def compute_classification_metrics(preds, class_labels, threshold=0.5, threshold_label=0.5):
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

    # cm = np.array([[tp, fp], [fn, tn]])
    # Make confusion matrix plot
    # plt.figure(figsize=(5, 5))

    return accuracy, precision, sensitivity, specificity, f1, weighted_f1, mcc