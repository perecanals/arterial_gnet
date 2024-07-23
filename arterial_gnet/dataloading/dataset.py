import os, json

import numpy as np
import networkx as nx

import torch
from torch_geometric.data import InMemoryDataset, Data

from sklearn.model_selection import train_test_split

from arterial_net.dataloading.utils import load_pickle, z_score_normalization, min_max_normalization, mean_centering_normalization, normalize_vector

class ArterialMapsDataset(InMemoryDataset):
    """
    Dataset class for the Arterial Maps dataset, containing a set of objects encoding
    information for the arterial pathway (supersegment) for a stroke patient who underwent
    endovascular treatment at different scales.

    Information is encoded in the following objects:
    - Global features: features descriptive of the entire supersegment
    - Segment graph: graph of the supersegment, with each node representing a vessel segment
    - Dense graph: graph of the supersegment, with each node representing a local centerline point

    This class allows us to create the final Pytorch Geometric Data objects that will be processed 
    by the models.

    """
    def __init__(self, root, raw_file_names_list = None, processed_file_names_list = None, pre_transform = None, transform = None):
        if os.path.exists(os.path.join(root, "dataset.json")):
            pass
        else:
            print("No dataset description file found. Creating dataset description file from raw dir data.")
            generate_ArterialMaps_dataset_json(root)
        with open(os.path.join(root, "dataset.json"), "r") as f:
            self.dataset_description = json.load(f)
        self.raw_file_names_list = raw_file_names_list
        self.processed_file_names_list = processed_file_names_list
        os.makedirs(os.path.join(root, "processed"), exist_ok=True)
        super(ArterialMapsDataset, self).__init__(root=root, pre_transform=pre_transform, transform=transform)
        self.process()

    @property
    def raw_file_names(self):
        if self.raw_file_names_list is not None:
            return self.raw_file_names_list
        else:
            return sorted([f for f in sorted(os.listdir(self.raw_dir)) if f.endswith(".pickle")])

    @property
    def processed_file_names(self):
        if self.processed_file_names_list is not None:
            return self.processed_file_names_list
        else:
            return sorted([f for f in sorted(os.listdir(self.processed_dir)) if f.endswith(".pt")])
        
    def process(self): 
        # Here you would read your raw files, create Data objects, and apply any pre-transforms
        data_list = []
        for raw_file_path in self.raw_file_names:
            preprocessed_file_path = os.path.join(self.processed_dir, "{}.pt".format(raw_file_path.split(".")[0]))
            if os.path.exists(preprocessed_file_path):
                data_list.append(torch.load(preprocessed_file_path))         
            else:
                # Load the raw file
                raw_file = load_pickle(os.path.join(self.raw_dir, raw_file_path))
                raw_y = raw_file["T1A"]
                raw_y_class = raw_file["classification"]
                raw_global_features = raw_file["global_features"]
                raw_segment_graph = self.clean_node_indexing(raw_file["segment_graph"])
                raw_dense_graph = self.clean_node_indexing(raw_file["dense_graph"])

                # We will wrap the raw data into Pytorch Geometric Data objects
                data = Data()
                data.y = torch.tensor(raw_y, dtype=torch.float32)
                data.y_class = torch.tensor(raw_y_class, dtype=torch.int64)

                # For global features, we just transform it into a tensor
                data.global_data = torch.tensor(self.normalize_global_features(raw_global_features["features_list"]), dtype=torch.float32)

                # For segment features, we create a Pytorch data object
                segment_data = Data()
                segment_x, segment_edge_index, segment_egde_attr, segment_pos = [], [], [], []
                for node in raw_segment_graph.nodes:
                    segment_pos.append(raw_segment_graph.nodes[node]["pos"])
                    segment_x.append(raw_segment_graph.nodes[node]["features_list"])
                for src, dst in raw_segment_graph.edges:
                    segment_edge_index.append(np.array([src, dst]))
                    segment_egde_attr.append(raw_segment_graph.edges[src, dst]["features_list"])
                segment_data.pos = torch.tensor(np.array(segment_pos), dtype=torch.float32)
                segment_data.x = self.normalize_segment_node_features(torch.tensor(np.array(segment_x), dtype=torch.float32))
                segment_data.edge_attr = self.normalize_segment_edge_features(torch.tensor(np.array(segment_egde_attr), dtype=torch.float32))
                segment_data.edge_index = torch.transpose(torch.tensor(np.array(segment_edge_index), dtype=torch.int64), 1, 0)
                segment_data.num_nodes = len(raw_segment_graph.nodes)
                segment_data.num_edges = len(raw_segment_graph.edges)
                data.segment_data = segment_data

                # For dense features, we create a Pytorch data object
                dense_data = Data()
                dense_x, dense_edge_index, dense_pos = [], [], []
                for node in raw_dense_graph.nodes:
                    dense_pos.append(raw_dense_graph.nodes[node]["pos"])
                    dense_x.append(raw_dense_graph.nodes[node]["features_list"])
                for src, dst in raw_dense_graph.edges:
                    dense_edge_index.append(np.array([src, dst]))
                dense_data.pos = torch.tensor(np.array(dense_pos), dtype=torch.float32)
                dense_data.x = self.normalize_dense_node_features(torch.tensor(np.array(dense_x), dtype=torch.float32))
                dense_data.edge_index = torch.transpose(torch.tensor(np.array(dense_edge_index), dtype=torch.int64), 1, 0)
                dense_data.num_nodes = len(raw_dense_graph.nodes)
                dense_data.num_edges = len(raw_dense_graph.edges)
                data.dense_data = dense_data
                data.id = raw_file_path.split(".")[0]

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
                torch.save(data, preprocessed_file_path)
    
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def clean_node_indexing(self, graph_nx):
        """
        Cleans the node indexing of a networkx graph by reindexing the nodes from 0 to N-1, where N is the number of nodes.

        Parameters
        ----------
        graph_nx : networkx.Graph
            Networkx graph.

        Returns
        -------
        graph_nx : networkx.Graph
            Networkx graph with cleaned node indexing.
        """
        mapping = {node: idx for idx, node in enumerate(graph_nx.nodes)}
        return nx.relabel_nodes(graph_nx, mapping)
    
    def get_splits(self, train_files=None, test_files=None, train_idx=None, test_idx=None, test_size=0.2, random_state=42, oversample=False):
        """
        Creates train and test splits from the dataset.

        Use cases:
        1. Provide train_files and test_files: train and test splits are created from the provided file lists.
            Example:    
                train_files = ["file1.pickle", "file2.pickle", "file3.pickle"]
                test_files = ["file4.pickle", "file5.pickle", "file6.pickle"]
                train_dataset, test_dataset = dataset.get_splits(train_files=train_files, test_files=test_files)
        2. Provide train_idx and test_idx: train and test splits are created from the provided indices.
            Example:
                train_idx = [0, 1, 2]
                test_idx = [3, 4, 5]
                train_dataset, test_dataset = dataset.get_splits(train_idx=train_idx, test_idx=test_idx)
        3. Provide only test_size: train and test splits are created randomly with the provided test_size. Random state can be provided.
            Example:
                test_size = 0.2
                train_dataset, test_dataset = dataset.get_splits(test_size=test_size)

        Parameters
        ----------
        train_files : list of str, optional
            List of train files, by default None
        test_files : list of str, optional
            List of test files, by default None
        train_idx : list of int, optional
            List of train indices, by default None
        test_idx : list of int, optional
            List of test indices, by default None
        test_size : float, optional
            Test size, by default 0.2
        random_state : int, optional
            Random state, by default 42
        oversample : bool, optional
            Whether to perform oversampling of the minority classes, by default False

        Returns
        -------
        train_dataset : torch_geometric.data.InMemoryDataset
            Train dataset
        test_dataset : torch_geometric.data.InMemoryDataset
            Test dataset
        """
        if train_files is not None and test_files is not None:
            if train_files[0].endswith(".pickle") and train_files[0] in self.raw_file_names:
                train_idx = [self.raw_file_names.index(file) for file in train_files]
                test_idx = [self.raw_file_names.index(file) for file in test_files]
            elif train_files[0].endswith(".pt") and train_files[0] in self.processed_file_names:
                train_idx = [self.processed_file_names.index(file) for file in train_files]
                test_idx = [self.processed_file_names.index(file) for file in test_files]
            else:
                print("Provided file lists are not valid, defaulting to random split (test_size = {:.2f}).".format(test_size))
                pass
        elif train_idx is not None and test_idx is not None:
            pass  # Use the provided indices directly
        else:
            indices = list(range(len(self.raw_file_names)))
            train_idx, test_idx = train_test_split(indices, test_size=test_size, train_size = 1 - test_size, random_state=random_state)

        # Get the file names for the train and test splits
        train_files = [self.raw_file_names[i] for i in train_idx]
        test_files = [self.raw_file_names[i] for i in test_idx]

        # Create new dataset instances for the train and test splits
        train_dataset = ArterialMapsDataset(self.root, raw_file_names_list=train_files, pre_transform=self.pre_transform)
        test_dataset = ArterialMapsDataset(self.root, raw_file_names_list=test_files, pre_transform=self.pre_transform)

        return train_dataset, test_dataset
    
    def normalize_global_features(self, global_features):      
        if self.dataset_description is None:
            print("No dataset description file found. This will be an issue for normalization of features.")
            return global_features
        else:
            mean_features = self.dataset_description["mean_global_features"]
            std_features = self.dataset_description["std_global_features"]
            min_features = self.dataset_description["min_segment_node_features"]
            max_features = self.dataset_description["max_segment_node_features"]

            for idx, feature_name in enumerate(self.dataset_description["global_feature_names"]):
                if feature_name in ["side", "vessel_type"]:
                    pass
                elif feature_name in ["tortuosity_index", "min_polar_angle", "polar_angle", "azimuthal_angle"]:
                    global_features[idx] = min_max_normalization(global_features[idx], min_features[idx], max_features[idx])
                else:
                    global_features[idx] = z_score_normalization(global_features[idx], mean_features[idx], std_features[idx])

            return global_features

    def normalize_segment_node_features(self, segment_node_features):
        if self.dataset_description is None:
            print("No dataset description file found. This will be an issue for normalization of features.")
            return segment_node_features
        else:
            mean_features = self.dataset_description["mean_segment_node_features"]
            std_features = self.dataset_description["std_segment_node_features"]    
            min_features = self.dataset_description["min_segment_node_features"]
            max_features = self.dataset_description["max_segment_node_features"]

            for idx, feature_name in enumerate(self.dataset_description["segment_node_feature_names"]):
                if feature_name in ["vessel_type"]:
                    pass
                elif feature_name in ["tortuosity_index", "min_polar_angle", "polar_angle", "azimuthal_angle"]:
                    segment_node_features[:, idx] = min_max_normalization(segment_node_features[:, idx], min_features[idx], max_features[idx])
                else:
                    segment_node_features[:, idx] = z_score_normalization(segment_node_features[:, idx], mean_features[idx], std_features[idx])
            
            return segment_node_features
        
    def normalize_segment_edge_features(self, segment_edge_features):
        if self.dataset_description is None:
            print("No dataset description file found. This will be an issue for normalization of features.")
            return segment_edge_features
        else:
            mean_features = self.dataset_description["mean_segment_edge_features"]
            std_features = self.dataset_description["std_segment_edge_features"]
            min_features = self.dataset_description["min_segment_edge_features"]
            max_features = self.dataset_description["max_segment_edge_features"]

            for idx, feature_name in enumerate(self.dataset_description["segment_edge_feature_names"]):
                if feature_name in ["max angle difference", "max azimuthal difference", "max polar difference"]:
                    segment_edge_features[:, idx] = min_max_normalization(segment_edge_features[:, idx], min_features[idx], max_features[idx])
                else:
                    segment_edge_features[:, idx] = z_score_normalization(segment_edge_features[:, idx], mean_features[idx], std_features[idx])

            return segment_edge_features
    
    def normalize_dense_node_features(self, dense_node_features):
        if self.dataset_description is None:
            print("No dataset description file found. This will be an issue for normalization of features.")
            return dense_node_features
        else:
            mean_features = self.dataset_description["mean_dense_node_features"]
            std_features = self.dataset_description["std_dense_node_features"]
            min_features = self.dataset_description["min_dense_node_features"]
            max_features = self.dataset_description["max_dense_node_features"]

            for idx, feature_name in enumerate(self.dataset_description["dense_node_feature_names"]):
                if feature_name in ["blanking", "vessel_type"]:
                    pass
                elif feature_name in ["curvature", "torsion", "polar_angle", "azimuthal_angle", "accumulated_length_from_access"]:
                    dense_node_features[:, idx] = min_max_normalization(dense_node_features[:, idx], min_features[idx], max_features[idx])
                else:
                    dense_node_features[:, idx] = z_score_normalization(dense_node_features[:, idx], mean_features[idx], std_features[idx])

            return dense_node_features


def generate_ArterialMaps_dataset_json(
        root=os.environ["arterial_maps_root"], 
        dataset_name = "Arterial Maps",
        name_regression_label = "T1A",
        name_classification_label = "classification",
        # For global scale
        name_global_graph = "global_features",
        name_global_features = "features",
        # For segment scale
        name_segment_graph = "segment_graph",
        name_segment_node_features = "features",
        name_segment_edge_features = "features",
        # For local (2mm) scale
        name_dense_graph = "dense_graph",
        name_dense_node_features = "features",
        name_dense_edge_features = None,
        ):
    # Define raw_dir
    raw_dir = os.path.join(root, "raw")
    # Read filenames
    filenames = sorted([f for f in os.listdir(raw_dir) if f.endswith(".pickle")])
    # Load the data
    raw_data_list = [load_pickle(os.path.join(raw_dir, f)) for f in filenames]
    total_number_of_examples = len(raw_data_list)

    # Get summary of classification labels
    if name_classification_label is not None:
        classifications_list = [d[name_classification_label] for d in raw_data_list]
        classification_values, classification_counts = np.unique(classifications_list, return_counts=True)
        # Sort the classification values by value
        classification_counts = classification_counts[np.argsort(classification_values)]
        classification_values = np.sort(classification_values)
        classification_frequencies = classification_counts / np.sum(classification_counts)
    else:
        classification_values = np.zeros(0)
        classification_counts = np.zeros(0)
        classification_frequencies = np.zeros(0)

    # Some parameters can be derived from a single example
    example = raw_data_list[0]
    if name_global_features is None:
        global_feature_names = None
        num_global_features = 0
        num_global_features_list = 0
    else:
        global_feature_names = list(example[name_global_graph][name_global_features].keys())
        num_global_features = len(global_feature_names)
        num_global_features_list = len(list(example[name_global_graph]["features_list"]))
    if name_segment_node_features is None:
        segment_node_feature_names = None
        num_segment_node_features = 0
        num_segment_node_features_list = 0
    else:
        segment_node_feature_names = list(example[name_segment_graph].nodes[next(iter(example[name_segment_graph].nodes))][name_segment_node_features].keys())
        num_segment_node_features = len(segment_node_feature_names)
        num_segment_node_features_list = len(list(example[name_segment_graph].nodes[next(iter(example[name_segment_graph].nodes))]["features_list"]))
    if name_segment_edge_features is None:
        segment_edge_feature_names = None
        num_segment_edge_features = 0
        num_segment_edge_features_list = 0
    else:
        segment_edge_feature_names = list(example[name_segment_graph].edges[next(iter(example[name_segment_graph].edges))][name_segment_edge_features].keys())
        num_segment_edge_features = len(segment_edge_feature_names)
        num_segment_edge_features_list = len(list(example[name_segment_graph].edges[next(iter(example[name_segment_graph].edges))]["features_list"]))
    if name_dense_node_features is None:
        dense_node_feature_names = None
        num_dense_node_features = 0
        num_dense_node_features_list = 0
    else:
        dense_node_feature_names = list(example[name_dense_graph].nodes[next(iter(example[name_dense_graph].nodes))][name_dense_node_features].keys())
        num_dense_node_features = len(dense_node_feature_names)
        num_dense_node_features_list = len(list(example[name_dense_graph].nodes[next(iter(example[name_dense_graph].nodes))]["features_list"]))
    if name_dense_edge_features is None:
        dense_edge_feature_names = None
        num_dense_edge_features = 0
        num_dense_edge_features_list = 0
    else:
        dense_edge_feature_names = list(example[name_dense_graph].edges[next(iter(example[name_dense_graph].edges))][name_dense_edge_features].keys())
        num_dense_edge_features = len(dense_edge_feature_names)
        num_dense_edge_features_list = len(list(example[name_dense_graph].edges[next(iter(example[name_dense_graph].edges))]["features_list"]))

    # For the rest, we have to iterate over the dataset
    total_number_of_segment_nodes = 0
    total_number_of_segment_edges = 0
    total_number_of_dense_nodes = 0
    total_number_of_dense_edges = 0

    global_features_array = np.zeros((num_global_features - 1, 0))
    segment_node_features_array = np.zeros((num_segment_node_features, 0))
    segment_edge_features_array = np.zeros((num_segment_edge_features, 0))
    dense_node_features_array = np.zeros((num_dense_node_features, 0))
    dense_edge_features_array = np.zeros((num_dense_edge_features, 0))

    for example in raw_data_list:
        global_features = example[name_global_graph]
        segment_graph = example[name_segment_graph]
        dense_graph = example[name_dense_graph]

        total_number_of_segment_nodes += segment_graph.number_of_nodes()
        total_number_of_segment_edges += segment_graph.number_of_edges()
        total_number_of_dense_nodes += dense_graph.number_of_nodes()
        total_number_of_dense_edges += dense_graph.number_of_edges()

        if name_global_features is not None:
            global_features_array_ = np.array([value for value in global_features[name_global_features].values()][:-1])
            global_features_array = np.concatenate((global_features_array, np.expand_dims(global_features_array_, axis = 1)), axis=1)
        if name_segment_node_features is not None:
            for node in segment_graph.nodes:
                segment_node_features_array_ = np.array([value for value in segment_graph.nodes[node][name_segment_node_features].values()])
                segment_node_features_array = np.concatenate((segment_node_features_array, np.expand_dims(segment_node_features_array_, axis = 1)), axis=1)
        if name_segment_edge_features is not None:
            for src, dst in segment_graph.edges:
                segment_edge_features_array_ = np.array([value for value in segment_graph.edges[src, dst][name_segment_edge_features].values()])
                segment_edge_features_array = np.concatenate((segment_edge_features_array, np.expand_dims(segment_edge_features_array_, axis = 1)), axis=1)
        if name_dense_node_features is not None:
            for node in dense_graph.nodes:
                dense_node_features_array_ = np.array([value for value in dense_graph.nodes[node][name_dense_node_features].values()])
                dense_node_features_array = np.concatenate((dense_node_features_array, np.expand_dims(dense_node_features_array_, axis = 1)), axis=1)
        if name_dense_edge_features is not None:
            for src, dst in dense_graph.edges:
                dense_edge_features_array_ = np.array([value for value in dense_graph.edges[src, dst][name_dense_edge_features].values()])
                dense_edge_features_array = np.concatenate((dense_edge_features_array, np.expand_dims(dense_edge_features_array_, axis = 1)), axis=1)

    average_number_of_segment_nodes = total_number_of_segment_nodes / total_number_of_examples
    average_number_of_segment_edges = total_number_of_segment_edges / total_number_of_examples
    average_number_of_dense_nodes = total_number_of_dense_nodes / total_number_of_examples
    average_number_of_dense_edges = total_number_of_dense_edges / total_number_of_examples

    # Now create the json file
    dataset_dict = {
        "dataset_name": dataset_name,
        "total_number_of_examples": total_number_of_examples,
        "name_regression_label": name_regression_label,
        "name_classification_label": name_classification_label,
        "graph_classification_values": classification_values.tolist(),
        "graph_classification_counts": classification_counts.tolist(),
        "graph_class_frequencies": classification_frequencies.tolist(),
        "total_number_of_segment_nodes": total_number_of_segment_nodes,
        "total_number_of_segment_edges": total_number_of_segment_edges,
        "total_number_of_dense_nodes": total_number_of_dense_nodes,
        "total_number_of_dense_edges": total_number_of_dense_edges,
        "average_number_of_segment_nodes": average_number_of_segment_nodes,
        "average_number_of_segment_edges": average_number_of_segment_edges,
        "average_number_of_dense_nodes": average_number_of_dense_nodes,
        "average_number_of_dense_edges": average_number_of_dense_edges,
        "num_global_features": num_global_features_list,
        "num_segment_node_features": num_segment_node_features_list,
        "num_segment_edge_features": num_segment_edge_features_list,
        "num_dense_node_features": num_dense_node_features_list,
        "num_dense_edge_features": num_dense_edge_features_list,
        "global_feature_names": global_feature_names,
        "segment_node_feature_names": segment_node_feature_names,
        "segment_edge_feature_names": segment_edge_feature_names,
        "dense_node_feature_names": dense_node_feature_names,
        "dense_edge_feature_names": dense_edge_feature_names,
        "mean_global_features": global_features_array.mean(axis=1).tolist() if len(global_features_array) > 0 else None,
        "std_global_features": global_features_array.std(axis=1).tolist() if len(global_features_array) > 0 else None,
        "median_global_features": np.median(global_features_array, axis=1).tolist() if len(global_features_array) > 0 else None,
        "min_global_features": global_features_array.min(axis=1).tolist() if len(global_features_array) > 0 else None,
        "max_global_features": global_features_array.max(axis=1).tolist() if len(global_features_array) > 0 else None,
        "mean_segment_node_features": segment_node_features_array.mean(axis=1).tolist() if len(segment_node_features_array) > 0 else None,
        "std_segment_node_features": segment_node_features_array.std(axis=1).tolist() if len(segment_node_features_array) > 0 else None,
        "median_segment_node_features": np.median(segment_node_features_array, axis=1).tolist() if len(segment_node_features_array) > 0 else None,
        "min_segment_node_features": segment_node_features_array.min(axis=1).tolist() if len(segment_node_features_array) > 0 else None,
        "max_segment_node_features": segment_node_features_array.max(axis=1).tolist() if len(segment_node_features_array) > 0 else None,
        "mean_segment_edge_features": segment_edge_features_array.mean(axis=1).tolist() if len(segment_edge_features_array) > 0 else None,
        "std_segment_edge_features": segment_edge_features_array.std(axis=1).tolist() if len(segment_edge_features_array) > 0 else None,
        "median_segment_edge_features": np.median(segment_edge_features_array, axis=1).tolist() if len(segment_edge_features_array) > 0 else None,
        "min_segment_edge_features": segment_edge_features_array.min(axis=1).tolist() if len(segment_edge_features_array) > 0 else None,
        "max_segment_edge_features": segment_edge_features_array.max(axis=1).tolist() if len(segment_edge_features_array) > 0 else None,
        "mean_dense_node_features": dense_node_features_array.mean(axis=1).tolist() if len(dense_node_features_array) > 0 else None,
        "std_dense_node_features": dense_node_features_array.std(axis=1).tolist() if len(dense_node_features_array) > 0 else None,
        "median_dense_node_features": np.median(dense_node_features_array, axis=1).tolist() if len(dense_node_features_array) > 0 else None,
        "min_dense_node_features": dense_node_features_array.min(axis=1).tolist() if len(dense_node_features_array) > 0 else None,
        "max_dense_node_features": dense_node_features_array.max(axis=1).tolist() if len(dense_node_features_array) > 0 else None,
        "mean_dense_edge_features": dense_edge_features_array.mean(axis=1).tolist() if len(dense_edge_features_array) > 0 else None,
        "std_dense_edge_features": dense_edge_features_array.std(axis=1).tolist() if len(dense_edge_features_array) > 0 else None,
        "median_dense_edge_features": np.median(dense_edge_features_array, axis=1).tolist() if len(dense_edge_features_array) > 0 else None,
        "min_dense_edge_features": dense_edge_features_array.min(axis=1).tolist() if len(dense_edge_features_array) > 0 else None,
        "max_dense_edge_features": dense_edge_features_array.max(axis=1).tolist() if len(dense_edge_features_array) > 0 else None,
    }

     # Save the json file
    with open(os.path.join(root, "dataset.json"), "w") as f:
        json.dump(dataset_dict, f, indent=4)