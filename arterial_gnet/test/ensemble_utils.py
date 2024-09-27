import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arterial_gnet.test.utils import *

def create_arterial_maps_df_val():
    arterial_maps_df_val = pd.read_excel("/media/Disk_B/databases/ArterialMaps/data/arterial_maps_train_val_multiple_vessels_df.xlsx")
    arterial_maps_df_original_val = pd.read_excel("/media/Disk_B/databases/ArterialMaps/data/arterial_maps_pre_2023_with_supersegments.xlsx")
    arterial_maps_df_original_val = arterial_maps_df_original_val[["proces_id", "classification"]]
    arterial_maps_df_original_val.rename(columns={"classification": "classification_original"}, inplace=True)
    arterial_maps_df_val = pd.merge(arterial_maps_df_val, arterial_maps_df_original_val, on="proces_id")
    arterial_maps_df_val["proces_id"] = arterial_maps_df_val["proces_id"].astype(str)
    arterial_maps_df_val["classification"] = np.where(arterial_maps_df_val["classification_original"] == 2, 1, arterial_maps_df_val["classification"])
    arterial_maps_df_val.drop(columns=["classification_original"], inplace=True)

    return arterial_maps_df_val

def create_arterial_maps_df_test():
    arterial_maps_df_test = pd.read_excel("/media/Disk_B/databases/ArterialMaps/data/arterial_maps_test_multiple_vessels_df.xlsx")
    arterial_maps_df_original_test = pd.read_excel("/media/Disk_B/databases/ArterialMaps/data/arterial_maps_2023_with_supersegments.xlsx")
    arterial_maps_df_original_test = arterial_maps_df_original_test[["proces_id", "classification"]]
    arterial_maps_df_original_test.rename(columns={"classification": "classification_original"}, inplace=True)
    arterial_maps_df_test = pd.merge(arterial_maps_df_test, arterial_maps_df_original_test, on="proces_id")
    arterial_maps_df_test["proces_id"] = arterial_maps_df_test["proces_id"].astype(str)
    arterial_maps_df_test["classification"] = np.where(arterial_maps_df_test["classification_original"] == 2, 1, arterial_maps_df_test["classification"])
    arterial_maps_df_test.drop(columns=["classification_original"], inplace=True)

    return arterial_maps_df_test

def build_arterial_maps_df_with_regression_mean(arterial_maps_df, root_models, model_name, test_suffix="test_dataset_2023_test_latest", test_folds=5, val_folds=5):
    # For validation
    if test_suffix == "test_latest":
        arterial_maps_df["test_fold"] = np.nan
        for val_fold in range(val_folds):
            arterial_maps_df[f"val_fold_{val_fold}"] = np.nan

        for test_fold in range(test_folds):
            model_dir = os.path.join(root_models, "models", model_name, model_name + "_tf-{}".format(test_fold))
            if os.path.exists(model_dir):
                for val_fold in range(val_folds):
                    fold_dir = os.path.join(model_dir, "fold_{}".format(val_fold))
                    for proces_id in sorted([filename.split(".")[0] for filename in os.listdir(os.path.join(fold_dir, test_suffix, "preds")) if filename.endswith(".npy")]):
                        arterial_maps_df.loc[arterial_maps_df["proces_id"] == proces_id, "test_fold"] = test_fold
                        arterial_maps_df.loc[arterial_maps_df["proces_id"] == proces_id, "val_fold_{}".format(val_fold)] = 1 - np.load(os.path.join(fold_dir, test_suffix, "preds", proces_id + ".npy"))[0, 0]
    # For test
    else:
        arterial_maps_df["test_fold"] = 0
        for val_fold in range(val_folds):
            arterial_maps_df[f"val_fold_{val_fold}"] = np.nan

        for test_fold in range(test_folds):
            model_dir = os.path.join(root_models, "models", model_name, model_name + "_tf-{}".format(test_fold))
            if test_fold > 0:
                for proces_id in sorted([filename.split(".")[0] for filename in os.listdir(os.path.join(fold_dir, test_suffix, "preds")) if filename.endswith(".npy")]):
                    new_row = arterial_maps_df[(arterial_maps_df["proces_id"] == proces_id) & (arterial_maps_df["test_fold"] == 0)].copy()
                    new_row["test_fold"] = test_fold
                    arterial_maps_df = pd.concat([
                        arterial_maps_df,
                        new_row
                    ], ignore_index=True)
            if os.path.exists(model_dir):
                for val_fold in range(val_folds):
                    fold_dir = os.path.join(model_dir, "fold_{}".format(val_fold))
                    for proces_id in sorted([filename.split(".")[0] for filename in os.listdir(os.path.join(fold_dir, test_suffix, "preds")) if filename.endswith(".npy")]):
                        arterial_maps_df.loc[(arterial_maps_df["proces_id"] == proces_id) & (arterial_maps_df["test_fold"] == test_fold), "val_fold_{}".format(val_fold)] = 1 - np.load(os.path.join(fold_dir, test_suffix, "preds", proces_id + ".npy"))[0, 0]

    arterial_maps_df["test_fold"] = arterial_maps_df["test_fold"].astype(int) 

    arterial_maps_df["Regression_mean"] = arterial_maps_df[["val_fold_{}".format(i) for i in range(val_folds)]].mean(axis=1)
    arterial_maps_df["Regression_std"] = arterial_maps_df[["val_fold_{}".format(i) for i in range(val_folds)]].std(axis=1)
    arterial_maps_df["Regression_lower_bound"] = arterial_maps_df["Regression_mean"] - 1.96 * arterial_maps_df["Regression_std"] / np.sqrt(test_folds)
    arterial_maps_df["Regression_upper_bound"] = arterial_maps_df["Regression_mean"] + 1.96 * arterial_maps_df["Regression_std"] / np.sqrt(test_folds)

    return arterial_maps_df

def compute_ensemble_metrics_and_plots(arterial_maps_df, root_models, model_name, test_suffix="test_dataset_2023_test_latest", test_folds=5):
    # Where to store the plots
    model_dir = os.path.join(root_models, "models", model_name)
    plots_dir = os.path.join(model_dir, f"aggregated_ensemble_plots_{test_suffix}")
    os.makedirs(plots_dir, exist_ok=True)

    roc_aucs = []
    pr_aucs = []
    optimal_thresholds = []
    accuracies = []
    precisions = []
    sensitivities = []
    specificities = []
    f1s = []
    weighted_f1s = []
    mccs = []
    ppvs = []
    npvs = []

    criterion = "youden"
    forced_threshold = None

    # Create a subplot grid
    figure_combined, axs = plt.subplots(test_folds, 3, figsize=(15, 5*test_folds))

    results_folds = {}

    print(f"Computing ensemble metrics and plots for {test_suffix}:")

    for test_fold in range(test_folds):
        plots_dir_fold = os.path.join(model_dir, f"{model_name}_tf-{test_fold}", f"ensemble_plots_{test_suffix}")
        os.makedirs(plots_dir_fold, exist_ok=True)
        df = arterial_maps_df[arterial_maps_df["test_fold"] == test_fold]
        results_folds[test_fold] = {}
        results_folds[test_fold]["preds"] = df["Regression_mean"]
        results_folds[test_fold]["labels"] = df["classification"]
        results_folds[test_fold]["class_labels"] = df["classification"]

        # Compute ROC curve and interpolate in 100 points to draw tpr and fpr
        roc_auc_score, fpr, tpr, thresholds_, fpr_interp, tpr_interp, thresholds_interp, optimal_threshold = compute_interpolated_roc_curve(results_folds[test_fold]["preds"], results_folds[test_fold]["class_labels"], n_points=100, threshold_label=0.5)
        pr_auc_score, recall, precision, thresholds_pr, recall_interp, precision_interp, thresholds_interp_pr, optimal_threshold_pr = compute_interpolated_pr_curve(results_folds[test_fold]["preds"], results_folds[test_fold]["class_labels"], n_points=100, threshold_label=0.5)
        results_folds[test_fold]["roc_auc"] = roc_auc_score
        results_folds[test_fold]["fpr"] = fpr
        results_folds[test_fold]["tpr"] = tpr
        results_folds[test_fold]["thresholds"] = thresholds_
        results_folds[test_fold]["fpr_interp"] = fpr_interp
        results_folds[test_fold]["tpr_interp"] = tpr_interp
        results_folds[test_fold]["thresholds_interp"] = thresholds_interp
        results_folds[test_fold]["optimal_threshold"] = optimal_threshold
        results_folds[test_fold]["pr_auc"] = pr_auc_score
        results_folds[test_fold]["recall"] = recall
        results_folds[test_fold]["precision"] = precision
        results_folds[test_fold]["thresholds_pr"] = thresholds_pr
        results_folds[test_fold]["recall_interp"] = recall_interp
        results_folds[test_fold]["precision_interp"] = precision_interp
        results_folds[test_fold]["thresholds_interp_pr"] = thresholds_interp_pr
        results_folds[test_fold]["optimal_threshold_pr"] = optimal_threshold_pr

        axs[test_fold, 0].grid(alpha=0.5, lw=0.5)
        axs[test_fold, 1].grid(alpha=0.5, lw=0.5)
        axs[test_fold, 2].grid(alpha=0.5, lw=0.5)

        plt.rcParams.update({'font.size': 14})
        df = arterial_maps_df[arterial_maps_df["test_fold"] == test_fold]

        # Draw and save individual plots
        roc_auc = draw_roc(df["Regression_mean"], df["T1A"], df["classification"], output_path=os.path.join(plots_dir_fold, "roc_curve.png"))
        pr_auc = draw_pr_curve(df["Regression_mean"], df["T1A"], df["classification"], output_path=os.path.join(plots_dir_fold, "pr_curve.png"))
        optimal_threshold_ = select_optimal_threshold(df["Regression_mean"], df["classification"], criterion=criterion)
        draw_probability_distribution(df["Regression_mean"], df["classification"], optimal_threshold_, output_path=os.path.join(plots_dir_fold, "probability_distribution.png"))

        # Add plots to combined figure
        draw_roc(df["Regression_mean"], df["T1A"], df["classification"], ax=axs[test_fold, 1])
        draw_pr_curve(df["Regression_mean"], df["T1A"], df["classification"], ax=axs[test_fold, 2])
        draw_probability_distribution(df["Regression_mean"], df["classification"], optimal_threshold_, ax=axs[test_fold, 0])

        results_folds[test_fold]["roc_auc"] = roc_auc # Overwriting that from interpolated curves? Should be the same

        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)
        optimal_thresholds.append(optimal_threshold_)
        threshold = forced_threshold if forced_threshold is not None else optimal_threshold_
        accuracy, precision, sensitivity, specificity, f1, weighted_f1, mcc, ppv, npv = compute_classification_metrics(df["Regression_mean"], df["classification"], threshold, threshold_label=0.5, output_path=os.path.join(plots_dir_fold, "confusion_matrix.png"))
        accuracies.append(accuracy)
        precisions.append(precision)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1s.append(f1)
        weighted_f1s.append(weighted_f1)
        mccs.append(mcc)
        ppvs.append(ppv)
        npvs.append(npv)

    metrics_dict = {
        "roc_auc_mean": np.mean(roc_aucs),
        "roc_auc_std": np.std(roc_aucs),
        "pr_auc_mean": np.mean(pr_aucs),
        "pr_auc_std": np.std(pr_aucs),
        "optimal_threshold_mean": np.mean(optimal_thresholds),
        "optimal_threshold_std": np.std(optimal_thresholds),
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "precision_mean": np.mean(precisions),
        "precision_std": np.std(precisions),
        "sensitivity_mean": np.mean(sensitivities),
        "sensitivity_std": np.std(sensitivities),
        "specificity_mean": np.mean(specificities),
        "specificity_std": np.std(specificities),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "weighted_f1_mean": np.mean(weighted_f1s),
        "weighted_f1_std": np.std(weighted_f1s),
        "mcc_mean": np.mean(mccs),
        "mcc_std": np.std(mccs),
        "ppv_mean": np.mean(ppvs),
        "ppv_std": np.std(ppvs),
        "npv_mean": np.mean(npvs),
        "npv_std": np.std(npvs),
    }

    if forced_threshold is not None:
        print(f"Forced threshold: {forced_threshold}")
    print(f"Mean ROC AUC (95% CI): {metrics_dict['roc_auc_mean']:.2f} ({metrics_dict['roc_auc_mean'] - 1.96 * metrics_dict['roc_auc_std'] / np.sqrt(5):.2f}-{metrics_dict['roc_auc_mean'] + 1.96 * metrics_dict['roc_auc_std'] / np.sqrt(5):.2f})")
    print(f"Mean PR AUC (95% CI): {metrics_dict['pr_auc_mean']:.2f} ({metrics_dict['pr_auc_mean'] - 1.96 * metrics_dict['pr_auc_std'] / np.sqrt(5):.2f}-{metrics_dict['pr_auc_mean'] + 1.96 * metrics_dict['pr_auc_std'] / np.sqrt(5):.2f})")
    print(f"Mean optimal threshold (95% CI): {metrics_dict['optimal_threshold_mean']:.2f} ({metrics_dict['optimal_threshold_mean'] - 1.96 * metrics_dict['optimal_threshold_std'] / np.sqrt(5):.2f}-{metrics_dict['optimal_threshold_mean'] + 1.96 * metrics_dict['optimal_threshold_std'] / np.sqrt(5):.2f})")
    print(f"Mean accuracy (95% CI): {metrics_dict['accuracy_mean']:.2f} ({metrics_dict['accuracy_mean'] - 1.96 * metrics_dict['accuracy_std'] / np.sqrt(5):.2f}-{metrics_dict['accuracy_mean'] + 1.96 * metrics_dict['accuracy_std'] / np.sqrt(5):.2f})")
    print(f"Mean precision (95% CI): {metrics_dict['precision_mean']:.2f} ({metrics_dict['precision_mean'] - 1.96 * metrics_dict['precision_std'] / np.sqrt(5):.2f}-{metrics_dict['precision_mean'] + 1.96 * metrics_dict['precision_std'] / np.sqrt(5):.2f})")
    print(f"Mean sensitivity (95% CI): {metrics_dict['sensitivity_mean']:.2f} ({metrics_dict['sensitivity_mean'] - 1.96 * metrics_dict['sensitivity_std'] / np.sqrt(5):.2f}-{metrics_dict['sensitivity_mean'] + 1.96 * metrics_dict['sensitivity_std'] / np.sqrt(5):.2f})")
    print(f"Mean specificity (95% CI): {metrics_dict['specificity_mean']:.2f} ({metrics_dict['specificity_mean'] - 1.96 * metrics_dict['specificity_std'] / np.sqrt(5):.2f}-{metrics_dict['specificity_mean'] + 1.96 * metrics_dict['specificity_std'] / np.sqrt(5):.2f})")
    print(f"Mean F1 (95% CI): {metrics_dict['f1_mean']:.2f} ({metrics_dict['f1_mean'] - 1.96 * metrics_dict['f1_std'] / np.sqrt(5):.2f}-{metrics_dict['f1_mean'] + 1.96 * metrics_dict['f1_std'] / np.sqrt(5):.2f})")
    print(f"Mean weighted F1 (95% CI): {metrics_dict['weighted_f1_mean']:.2f} ({metrics_dict['weighted_f1_mean'] - 1.96 * metrics_dict['weighted_f1_std'] / np.sqrt(5):.2f}-{metrics_dict['weighted_f1_mean'] + 1.96 * metrics_dict['weighted_f1_std'] / np.sqrt(5):.2f})")
    print(f"Mean MCC (95% CI): {metrics_dict['mcc_mean']:.2f} ({metrics_dict['mcc_mean'] - 1.96 * metrics_dict['mcc_std'] / np.sqrt(5):.2f}-{metrics_dict['mcc_mean'] + 1.96 * metrics_dict['mcc_std'] / np.sqrt(5):.2f})")
    print(f"Mean PPV (95% CI): {metrics_dict['ppv_mean']:.2f} ({metrics_dict['ppv_mean'] - 1.96 * metrics_dict['ppv_std'] / np.sqrt(5):.2f}-{metrics_dict['ppv_mean'] + 1.96 * metrics_dict['ppv_std'] / np.sqrt(5):.2f})")
    print(f"Mean NPV (95% CI): {metrics_dict['npv_mean']:.2f} ({metrics_dict['npv_mean'] - 1.96 * metrics_dict['npv_std'] / np.sqrt(5):.2f}-{metrics_dict['npv_mean'] + 1.96 * metrics_dict['npv_std'] / np.sqrt(5):.2f})")

    # Draw ROC curve over folds
    roc_results = draw_roc_folds(results_folds, n_points=100, output_path=os.path.join(plots_dir, "roc_curve.png"))
    pr_results = draw_pr_folds(results_folds, n_points=100, output_path=os.path.join(plots_dir, "pr_curve.png"))

    # Save figure combined
    figure_combined.tight_layout()
    figure_combined.savefig(os.path.join(plots_dir, "combined_plots.png"), dpi=300, bbox_inches='tight')
    plt.close(figure_combined)  # Close the figure to free up memory

    # Save results to a file
    with open(os.path.join(plots_dir, "roc_results.json"), "w") as f:
        json.dump(roc_results, f, indent=4)
    with open(os.path.join(plots_dir, "pr_results.json"), "w") as f:
        json.dump(pr_results, f, indent=4)

    return metrics_dict

def create_sql_table_query(experiment_name, model_name, num_parameters, args, val_metrics_dict, test_metrics_dict):
    query = f"""
    INSERT INTO arterial_gnet_finetuning (
        experiment, model_base_name, num_parameters,
        test_size, val_size, batch_size, total_epochs, hidden_channels, hidden_channels_dense,
        optimizer, learning_rate, lr_scheduler, num_global_layers, num_segment_layers,
        num_dense_layers, num_out_layers, attn_heads, aggregation, dropout, weighted_loss,
        radius, concat, random_state, test_random_state, folds, test_folds, is_classification, oversampling,
        val_roc_auc_mean, val_roc_auc_std, val_pr_auc_mean, val_pr_auc_std,
        val_youden_threshold_mean, val_youden_threshold_std, val_accuracy_mean, val_accuracy_std,
        val_precision_mean, val_precision_std, val_sensitivity_mean, val_sensitivity_std,
        val_specificity_mean, val_specificity_std, val_f1_mean, val_f1_std,
        val_weighted_f1_mean, val_weighted_f1_std, val_mcc_mean, val_mcc_std,
        val_ppv_mean, val_ppv_std, val_npv_mean, val_npv_std,
        test_roc_auc_mean, test_roc_auc_std, test_pr_auc_mean, test_pr_auc_std,
        test_youden_threshold_mean, test_youden_threshold_std, test_accuracy_mean, test_accuracy_std,
        test_precision_mean, test_precision_std, test_sensitivity_mean, test_sensitivity_std,
        test_specificity_mean, test_specificity_std, test_f1_mean, test_f1_std,
        test_weighted_f1_mean, test_weighted_f1_std, test_mcc_mean, test_mcc_std,
        test_ppv_mean, test_ppv_std, test_npv_mean, test_npv_std
    ) VALUES (
        '{experiment_name}', '{model_name}', {num_parameters},
        {args.test_size}, {args.val_size}, {args.batch_size}, {args.total_epochs},
        {args.hidden_channels}, {args.hidden_channels_dense}, '{args.optimizer}',
        {args.learning_rate}, '{args.lr_scheduler}', {args.num_global_layers},
        {args.num_segment_layers}, {args.num_dense_layers}, {args.num_out_layers},
        {args.attn_heads}, '{args.aggregation}', {args.dropout}, '{args.weighted_loss}',
        {args.radius}, {args.concat}, {args.random_state}, {args.test_random_state},
        {args.folds}, {args.test_folds}, {args.is_classification}, {args.oversampling},
        {val_metrics_dict['roc_auc_mean']}, {val_metrics_dict['roc_auc_std']},
        {val_metrics_dict['pr_auc_mean']}, {val_metrics_dict['pr_auc_std']},
        {val_metrics_dict['optimal_threshold_mean']}, {val_metrics_dict['optimal_threshold_std']},
        {val_metrics_dict['accuracy_mean']}, {val_metrics_dict['accuracy_std']},
        {val_metrics_dict['precision_mean']}, {val_metrics_dict['precision_std']},
        {val_metrics_dict['sensitivity_mean']}, {val_metrics_dict['sensitivity_std']},
        {val_metrics_dict['specificity_mean']}, {val_metrics_dict['specificity_std']},
        {val_metrics_dict['f1_mean']}, {val_metrics_dict['f1_std']},
        {val_metrics_dict['weighted_f1_mean']}, {val_metrics_dict['weighted_f1_std']},
        {val_metrics_dict['mcc_mean']}, {val_metrics_dict['mcc_std']},
        {val_metrics_dict['ppv_mean']}, {val_metrics_dict['ppv_std']},
        {val_metrics_dict['npv_mean']}, {val_metrics_dict['npv_std']},
        {test_metrics_dict['roc_auc_mean']}, {test_metrics_dict['roc_auc_std']},
        {test_metrics_dict['pr_auc_mean']}, {test_metrics_dict['pr_auc_std']},
        {test_metrics_dict['optimal_threshold_mean']}, {test_metrics_dict['optimal_threshold_std']},
        {test_metrics_dict['accuracy_mean']}, {test_metrics_dict['accuracy_std']},
        {test_metrics_dict['precision_mean']}, {test_metrics_dict['precision_std']},
        {test_metrics_dict['sensitivity_mean']}, {test_metrics_dict['sensitivity_std']},
        {test_metrics_dict['specificity_mean']}, {test_metrics_dict['specificity_std']},
        {test_metrics_dict['f1_mean']}, {test_metrics_dict['f1_std']},
        {test_metrics_dict['weighted_f1_mean']}, {test_metrics_dict['weighted_f1_std']},
        {test_metrics_dict['mcc_mean']}, {test_metrics_dict['mcc_std']},
        {test_metrics_dict['ppv_mean']}, {test_metrics_dict['ppv_std']},
        {test_metrics_dict['npv_mean']}, {test_metrics_dict['npv_std']}
    );
    """
    return query
