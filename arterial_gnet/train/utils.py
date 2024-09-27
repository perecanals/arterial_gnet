import os

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def make_train_plot(model_path, losses_train, losses_val, metric_train, metric_val, is_classification=False):
    total_epochs = len(losses_train)
    # Saves a plot for the loss and metric evolution for training and validation as summary 
    _, ax1 = plt.subplots(figsize=[8, 7])
    if is_classification:
        plt.plot(range(total_epochs), losses_train, 'red')
        plt.plot(range(total_epochs), losses_val, 'darkred')
        plt.xlabel("epochs", fontsize=14)
        plt.ylabel("loss", fontsize=14, color="darkred")
        ax1.tick_params(axis='y', labelcolor="darkred", labelsize=14)
        plt.legend(["Loss train", "Loss val"], loc='upper center', bbox_to_anchor=(0.17, 1.1),
                ncol=2, fontsize=14)
        ax2 = ax1.twinx() 
        plt.plot(range(total_epochs), metric_train, 'b')
        plt.plot(range(total_epochs), metric_val, 'navy')
        plt.ylabel("ROC AUC", fontsize=14, color="navy")
        ax2.tick_params(axis='y', labelcolor="navy", labelsize=14)
        plt.legend(["ROC AUC train", "ROC AUC val"], loc='upper center', bbox_to_anchor=(0.73, 1.1),
                ncol=2, fontsize=14)
        plt.grid(axis = "y")
        plt.savefig(os.path.join(model_path, "training_summary.png"))
        plt.close()
    else:
        plt.plot(range(total_epochs), losses_train, 'red')
        plt.plot(range(total_epochs), losses_val, 'darkred')
        plt.xlabel("epochs", fontsize=14)
        plt.ylabel("loss", fontsize=14, color="darkred")
        ax1.tick_params(axis='y', labelcolor="darkred", labelsize=14)
        plt.legend(["Loss train", "Loss val"], loc='upper center', bbox_to_anchor=(0.17, 1.1),
                ncol=2, fontsize=14)
        ax2 = ax1.twinx() 
        plt.plot(range(total_epochs), metric_train, 'b')
        plt.plot(range(total_epochs), metric_val, 'navy')
        plt.ylabel("RMSE", fontsize=14, color="navy")
        ax2.tick_params(axis='y', labelcolor="navy", labelsize=14)
        plt.legend(["RMSE train", "RMSE val"], loc='upper center', bbox_to_anchor=(0.73, 1.1),
                ncol=2, fontsize=14)
        plt.grid(axis = "y")
        plt.savefig(os.path.join(model_path, "training_summary.png"))
        plt.close()

def compute_roc_pr_auc(preds, labels):
    fpr, tpr, thresholds = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    return roc_auc, pr_auc