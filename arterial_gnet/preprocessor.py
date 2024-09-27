import os, json, pickle, shutil
import pandas as pd
import numpy as np
from time import time

from arterial_gnet.preprocessing.featurizer import GraphFeaturizer

import matplotlib.pyplot as plt
plt.ioff()

import multiprocessing
from multiprocessing import Pool, Manager

# tag = "train_val"
tag = "test"
base_dir = f"/media/Disk_B/databases/ArterialMaps/data/{tag}_dataset"
root = "/media/Disk_B/databases/ArterialMaps/root"
if tag == "test":
    root = "/media/Disk_B/databases/ArterialMaps/root_test"

def init_globals(manager_dict):
    global times_log
    global error_log
    global only_one_vessel_type
    times_log = manager_dict['times_log']
    error_log = manager_dict['error_log']
    only_one_vessel_type = manager_dict['only_one_vessel_type']

def preprocess_case(args):
    proces_id, df = args

    print(f"Processing case {proces_id}")

    side = df[df["proces_id"] == proces_id]["side"].values[0]
    case_dir = os.path.join(base_dir, str(proces_id))

    with open(os.path.join(case_dir, "raw_data", "supersegment_2mm.pickle"), "rb") as f:
        supersegment = pickle.load(f)

    start = time()

    try:
        featurizer = GraphFeaturizer(case_dir, side, supersegment)
        featurizer.get_global_features()
        featurizer.save_global_features()

        featurizer.build_segment_graph()
        featurizer.save_segment_graph()
        # featurizer.plot_segment_graph(feature = "tortuosity_index", cmap = "Reds")

        featurizer.build_dense_graph()
        featurizer.save_dense_graph()
        # featurizer.plot_dense_graph(feature = "curvature", cmap = "bwr")

        featurizer.save_combined_plot()

        featurizer.save_raw_pickle(regression_value=df[df["proces_id"] == proces_id]["T1A"].values[0],
                                classification_value=df[df["proces_id"] == proces_id]["classification"].values[0])
        
        plt.close()

        vessel_types = []
        for node in featurizer.segment_graph:
            vessel_types.append(featurizer.segment_graph.nodes[node]["vessel_type_name"])

        if len(vessel_types) == 1:
            print(f"Only one vessel type for case {proces_id}: {vessel_types[0]}")
            only_one_vessel_type[str(proces_id)] = vessel_types[0]

        times_log[str(proces_id)] = time() - start
    
    except Exception as e:
        error_log[str(proces_id)] = str(e)
        print(f"Error processing case {proces_id}: {e}")

if __name__ == "__main__":
    # Define the number of processes to use
    num_processes = multiprocessing.cpu_count()

    print("Number of threads:", num_processes)

    if tag == "train_val":
        arterial_maps_df = pd.read_excel("/media/Disk_B/databases/ArterialMaps/data/arterial_maps_pre_2023_with_supersegments.xlsx") # df with columns=["proces_id", "T1A", "classification", "side"]
    elif tag == "test":
        arterial_maps_df = pd.read_excel("/media/Disk_B/databases/ArterialMaps/data/arterial_maps_2023_with_supersegments.xlsx")
    # We are finishing up processing, at the moment, filter proces_ids already available in the base_dir
    arterial_maps_df = arterial_maps_df[arterial_maps_df["proces_id"].isin([int(proces_id) for proces_id in os.listdir(base_dir)])]
    
    ##### Added for classification in 1 over 60 min
    # arterial_maps_df["classification"] = np.where(arterial_maps_df["T1A"].isna(), 1, 0)

    # Impute all nans in the T1A column as a random variable following a uniform distribution between 78 (99-per) and 143 (max)
    # Set a random state
    np.random.seed(42)
    arterial_maps_df.loc[arterial_maps_df["T1A"].isna(), "T1A"] = np.random.uniform(78, 143, len(arterial_maps_df[arterial_maps_df["T1A"].isna()])).astype(int)

    manager = Manager()
    manager_dict = manager.dict()
    manager_dict['times_log'] = manager.dict()
    manager_dict['error_log'] = manager.dict()
    manager_dict['only_one_vessel_type'] = manager.dict()

    num_processes = multiprocessing.cpu_count()
    pool = Pool(processes=num_processes, initializer=init_globals, initargs=(manager_dict,))

    # Prepare arguments as a list of tuples
    tasks = [(proces_id, arterial_maps_df) for proces_id in sorted(arterial_maps_df.proces_id.unique())]

    # Process cases in parallel
    results = pool.map(preprocess_case, tasks)

    pool.close()
    pool.join()

    # Saving the updated logs after processing
    with open(f"/media/Disk_B/databases/ArterialMaps/data/preprocessing_{tag}_times_log.json", "w") as f:
        json.dump(dict(manager_dict['times_log']), f, indent=4)
    
    with open(f"/media/Disk_B/databases/ArterialMaps/data/preprocessing_{tag}_error_log.json", "w") as f:
        json.dump(dict(manager_dict['error_log']), f, indent=4)

    with open(f"/media/Disk_B/databases/ArterialMaps/data/{tag}_only_one_vessel_type.json", "w") as f:
        json.dump(dict(manager_dict['only_one_vessel_type']), f, indent=4)

    for proces_id in manager_dict['only_one_vessel_type'].keys():
        if os.path.exists(os.path.join(base_dir, str(proces_id))):
            shutil.move(os.path.join(base_dir, str(proces_id)), os.path.join("/media/Disk_B/databases/ArterialMaps/data/discarded/only_one_vessel", str(proces_id)))

    proces_ids_only_one_vessel = sorted([int(proces_id) for proces_id in manager_dict['only_one_vessel_type'].keys()])
    arterial_maps_multiple_vessels_df = arterial_maps_df[~arterial_maps_df["proces_id"].isin(proces_ids_only_one_vessel)]

    arterial_maps_multiple_vessels_df = arterial_maps_multiple_vessels_df.reset_index(drop = True)
    arterial_maps_multiple_vessels_df.to_excel(f"/media/Disk_B/databases/ArterialMaps/data/arterial_maps_{tag}_multiple_vessels_df.xlsx", index = False)

    os.makedirs(os.path.join(root, "raw"), exist_ok = True)
    for proces_id in arterial_maps_multiple_vessels_df.proces_id:
        shutil.copyfile(os.path.join(base_dir, str(proces_id), f"{str(proces_id)}.pickle"), os.path.join(root, "raw", f"{str(proces_id)}.pickle"))