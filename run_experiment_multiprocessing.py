"""
This will be the base script to run each individual experiment.

In this script, a set of parameter variations will be defined, along with the name of the experiment. At least
initially, within an experiment, all parameters will be left the same, and only the parameter under investigation
will be changed. For simplicity when adapting the script for other experiments, we will be defining all parameters as 
lists, even if they are not being varied. 

The script will perform training and validation, as well as internal testing using the test dataset from 2023.

Once these are completed, all resulting files should be stored in a folder within the experiment folder.

The script will then complete the evaluation of the models by computing the metrics resulting from the model ensemble, 
both for validaiton (in each of the validation folds) and for the test dataset.

These results will be stored in the arterial database, where a table called arterial_gnet_finetuning will be updated.

"""

import os, shutil, argparse
import torch
from itertools import product
from arterial_gnet.main_test_folds import main as training_main
from arterial_gnet.external_testing import main as testing_main
from arterial_gnet.test.ensemble_utils import *
from arterial_gnet.test.utils import *

import multiprocessing
from functools import partial
import time

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv("/media/Disk_B/databases/ArterialMaps/fine_tuning/.env")
# Load environment variables for database settings
PSQL_USER = os.getenv('PSQL_USER')
PSQL_PASSWORD = os.getenv('PSQL_PASSWORD')

# Create SQLAlchemy engine
engine = create_engine(f"postgresql://{PSQL_USER}:{PSQL_PASSWORD}@localhost:5432/arterial")

def run_single_experiment(params, device_index, experiment_name):
    """
    Run an experiment with the given parameters.

    Parameters
    ----------
    params : tuple
        Tuple of parameters for the experiment.
    device_index : int
        Index of the GPU to use.
    experiment_name : str
        Name of the experiment.
    """
    print(f"Starting experiment on device {device_index % 2} with params: {params[:5]}...")  # Print first 5 params for brevity
    root = os.environ["arterial_maps_root"]
    root_models = os.path.join("/media/Disk_B/databases/ArterialMaps/fine_tuning/experiments", experiment_name)
    os.makedirs(root_models, exist_ok=True)
    root_test = "/media/Disk_B/databases/ArterialMaps/root_test"
    external_dataset_name = "test_dataset_2023"

    device = device_index % 2 # Assuming we have 2 GPUs!

    parser = argparse.ArgumentParser(description="ArterialGNet grid search configuration")
    parser.add_argument("--base_model_name", type=str, default=params[0])
    parser.add_argument("--test_size", type=float, default=params[1])
    parser.add_argument("--val_size", type=float, default=params[2])
    parser.add_argument("--total_epochs", type=int, default=params[3])
    parser.add_argument("--batch_size", type=int, default=params[4])
    parser.add_argument("--hidden_channels", type=int, default=params[5])
    parser.add_argument("--hidden_channels_dense", type=int, default=params[6])
    parser.add_argument("--optimizer", type=str, default=params[7])
    parser.add_argument("--learning_rate", type=float, default=params[8])
    parser.add_argument("--lr_scheduler", type=str, default=params[9])
    parser.add_argument("--num_global_layers", type=int, default=params[10])
    parser.add_argument("--num_segment_layers", type=int, default=params[11])
    parser.add_argument("--num_dense_layers", type=int, default=params[12])
    parser.add_argument("--num_out_layers", type=int, default=params[13])
    parser.add_argument("--attn_heads", type=int, default=params[14])
    parser.add_argument("--aggregation", type=str, default=params[15])
    parser.add_argument("--dropout", type=float, default=params[16])
    parser.add_argument("--radius", type=int, default=params[17])
    parser.add_argument("--concat", type=str, default=params[18])
    parser.add_argument("--weighted_loss", type=str, default=params[19])
    parser.add_argument("--random_state", type=int, default=params[20])
    parser.add_argument("--test_random_state", type=int, default=params[21])
    parser.add_argument("--folds", type=int, default=params[22])
    parser.add_argument("--skip_folds", type=int, default=params[23])
    parser.add_argument("--test_folds", type=int, default=params[24])
    parser.add_argument("--train", action="store_true", default=params[25])
    parser.add_argument("--test", action="store_true", default=params[26])
    parser.add_argument("--oversampling", action="store_true", default=params[27])
    parser.add_argument("--is_classification", action="store_true", default=params[28])
    parser.add_argument("--tag", type=str, default=None if params[29] is None else params[29])
    parser.add_argument("--class_loss", type=str, default=params[30])
    parser.add_argument("--num_workers", type=int, default=params[31])

    # Parse arguments
    args = parser.parse_args()
    args.device = str(device)

    # Check if model_name with same experiment is already in the database
    model_name = "{}_bs-{}_te-{}_hc-{}_hcd-{}_op-{}_lr-{}_lrs-{}_ngl-{}_nsl-{}_ndl-{}_nol-{}_ah-{}_agg-{}_drop-{}_r-{}_concat-{}_wl-{}_os-{}_rs-{}_trs-{}".format(args.base_model_name, args.batch_size, \
                args.total_epochs, args.hidden_channels, args.hidden_channels_dense, args.optimizer, args.learning_rate, args.lr_scheduler, \
                    args.num_global_layers, args.num_segment_layers, args.num_dense_layers, args.num_out_layers, args.attn_heads, args.aggregation, \
                        args.dropout, args.radius, args.concat, args.weighted_loss, args.oversampling, args.random_state, args.test_random_state)
    if args.is_classification:
        model_name += "_class"
    if args.tag is not None:
        model_name += "_tag-{}".format(args.tag)

    query = f"SELECT * FROM arterial_gnet_finetuning WHERE experiment = '{experiment_name}' AND model_base_name = '{model_name}'"
    existing_model = pd.read_sql(query, engine)

    if not existing_model.empty:
        print(f"Model {model_name} already exists in the database. Skipping this model.")
        return

    # Run training
    model_name, model_dir = training_main(root, args, root_models) # Models will be saved in root_models/models/model_name_tf-{test_fold}

    # Get number of model parameters
    model = torch.load(os.path.join(model_dir + "_tf-0", "fold_0/model_latest.pth"))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Run "external" testing
    for test_fold in range(params[24]):
        model_name_ = model_name + "_tf-{}".format(test_fold)
        model_dir = os.path.join(root_models, "models", model_name_)
        testing_main(model_dir, root_test, external_dataset_name, device=args.device)

    # Move all tf-... folders to the model_name folder
    os.makedirs(os.path.join(root_models, "models", model_name), exist_ok=True)
    for test_fold in range(params[24]):
        model_name_ = model_name + "_tf-{}".format(test_fold)
        shutil.move(os.path.join(root_models, "models", model_name_), os.path.join(root_models, "models", model_name, model_name_))

    # Compute dfs for validaiton and test (each works a bit differently, adapt from notebooks)
    arterial_maps_df_val = create_arterial_maps_df_val()
    arterial_maps_df_test = create_arterial_maps_df_test()

    # Add prediction values
    arterial_maps_df_val = build_arterial_maps_df_with_regression_mean(arterial_maps_df_val, root_models, model_name, test_suffix="test_latest")
    arterial_maps_df_test = build_arterial_maps_df_with_regression_mean(arterial_maps_df_test, root_models, model_name, test_suffix=f"{external_dataset_name}_test_latest")

    # Compute ensemble metrics for validation and test and create plots
    val_metrics_dict = compute_ensemble_metrics_and_plots(arterial_maps_df_val, root_models, model_name, test_suffix="test_latest")
    test_metrics_dict = compute_ensemble_metrics_and_plots(arterial_maps_df_test, root_models, model_name, test_suffix=f"{external_dataset_name}_test_latest")

    # Create SQL query to insert results
    insert_query = create_sql_table_query(experiment_name, model_name, num_params, args, val_metrics_dict, test_metrics_dict)

    # Execute the query to insert the new row
    with engine.connect() as connection:
        connection.execute(text(insert_query))
        connection.commit()

    print(f"Results for model {model_name} have been inserted into the database.")

    print(f"Finished experiment on device {device_index % 2}")

def run_experiment(experiment_name, param_iterator):
    """
    Run an experiment with the given parameters.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    param_iterator : itertools.product
        Iterator of parameters for the experiment.
    """
    # Convert param_iterator to a list to avoid exhausting the iterator
    param_list = list(param_iterator)
    
    # Create a pool of workers
    num_processes = multiprocessing.cpu_count()
    num_processes = 8
    print(f"Starting pool with {num_processes} processes")
    
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Create a partial function with fixed experiment_name
    run_single_experiment_partial = partial(run_single_experiment, experiment_name=experiment_name)
    
    # Run experiments in parallel
    start_time = time.time()
    results = pool.starmap_async(run_single_experiment_partial, ((params, i) for i, params in enumerate(param_list)))
    
    # Monitor progress
    total_tasks = len(param_list)
    while not results.ready():
        completed = total_tasks - results._number_left
        print(f"Progress: {completed}/{total_tasks} tasks completed")
        time.sleep(10)  # Check progress every 10 seconds
    
    # Close the pool
    pool.close()
    pool.join()
    
    end_time = time.time()
    print(f"All experiments completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    experiment_name = "prova_mutliprocessing"
    base_model_name_list = ["ArterialGNet"]
    test_size_list = [0.2]
    val_size_list = [0.2]
    total_epochs_list = [50, 51, 52, 53]
    batch_size_list = [64]
    hidden_channels_list = [32]
    hidden_channels_dense_list = [32]
    optimizer_list = ["adam"]
    learning_rate_list = [1e-3]
    lr_scheduler_list = ["poly"]
    num_global_layers_list = [0]
    num_segment_layers_list = [0]
    num_dense_layers_list = [1]
    num_out_layers_list = [1]
    attn_heads_list = [1]
    aggregation_list = ["mean"]
    dropout_list = [0.2]
    radius_list = [0]
    concat_list = [False]
    weighted_loss_list = ["exp"]
    random_state_list = [42]
    test_random_state_list = [43]
    folds_list = [5]
    skip_folds_list = [None]
    test_folds_list = [5]
    train_list = [True]
    test_list = [True]
    oversampling_list = [False]
    is_classification_list = [True]
    tag_list = [experiment_name]
    class_loss_list = ["ce"]
    num_workers_list = [2]

    # Define the iterator
    param_iterator = product(
        base_model_name_list, test_size_list, val_size_list, total_epochs_list, batch_size_list,
        hidden_channels_list, hidden_channels_dense_list, optimizer_list, learning_rate_list,
        lr_scheduler_list, num_global_layers_list, num_segment_layers_list, num_dense_layers_list,
        num_out_layers_list, attn_heads_list, aggregation_list, dropout_list, radius_list,
        concat_list, weighted_loss_list, random_state_list, test_random_state_list,
        folds_list, skip_folds_list, test_folds_list, train_list, test_list, oversampling_list,
        is_classification_list, tag_list, class_loss_list, num_workers_list
    )

    run_experiment(experiment_name, param_iterator)