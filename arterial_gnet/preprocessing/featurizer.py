import os, json, pickle

import numpy as np
import networkx as nx

from copy import deepcopy

from arterial_net.preprocessing.utils import clean_supersegment, sort_by_hierarchy, filter_vessel_types, \
    compute_segment_graph_node_features, compute_segment_graph_edge_features, get_vessel_type, \
        get_onehot_encoded_vessel_type, clean_node, get_supersegment_vessel_types
from arterial_net.visualization.utils import make_combined_plot, make_dense_graph_plot

class GraphFeaturizer():
    """
    Class to get clean featurized graphs at multiple scales from the Arterial 
    supersegment outputs. Three objects, encoding anatomical information at different
    scales, are created:
    - Global features: features of the entire supersegment. A JSON file is saved with these features
    - Segment graph: graph of the supersegment, with each node representing a vessel segment. A pickle file is saved with this graph
    - Dense graph: graph of the supersegment, with each node representing a local centerline point. A pickle file is saved with this graph

    These objects can be used to create the final Pytorch Geometric Data objects that will ultimately be used to train the models.

    Parameters
    ----------
    case_dir : str
        Path to the case directory
    side : str
        Side of the supersegment (right or left)
    supersegment : str
        Name of the supersegment

    """
    def __init__(self, case_dir, side, supersegment):
        self.case_dir = case_dir
        self.side = side
        self.supersegment = deepcopy(supersegment)

        # Global features
        self.global_features = {}
        self.global_features["features"] = {}
        self.global_features["features"]["side"] = 0 if self.side == "right" else 1
        self.total_segment = None

        # Segment graph
        self.segment_graph = nx.Graph()
        self.vessel_segments = {}

        # Dense graph
        self.dense_graph = None

    # Methods for global featurization
    def get_global_features(self):
        """
        Get global features from the total segment and save them in the global_features attribute.

        """
        self.total_segment = clean_supersegment(self.supersegment)
        vessel_type_names, vessel_types, onehot_vessel_types = get_supersegment_vessel_types(self.total_segment)
        self.total_segment.graph["vessel_type_name"] = vessel_type_names
        for node in self.total_segment:
            self.total_segment.nodes[node]["features"]["blanking"] = 0 # We remove blanking to compute features from the global segment
        self.total_segment.graph["features"] = compute_segment_graph_node_features(self.total_segment)
        self.total_segment.graph["features_list"] = np.concatenate(([self.global_features["features"]["side"]], [self.total_segment.graph["features"][key] for key in ["tortuosity_index", "length", "mean_diameter", "std_diameter", "min_polar_angle", "polar_angle", "azimuthal_angle"]], onehot_vessel_types))
        self.total_segment.graph["features"]["vessel_type"] = list(np.array(vessel_types).astype(float))
        for feature in ["tortuosity_index", "length", "mean_diameter", "std_diameter", "min_polar_angle", "polar_angle", "azimuthal_angle", "vessel_type"]:
            self.global_features["features"][feature] = self.total_segment.graph["features"][feature]
        self.global_features["features_list"] = list(self.total_segment.graph["features_list"].astype(float))

    def save_global_features(self):
        """
        Save the global features in a JSON file in the case directory.

        """
        if len(self.global_features) == 0:
            self.get_global_features()
        with open(os.path.join(self.case_dir, "global_features.json"), "w") as f:
            json.dump(self.global_features, f)

    # Methods for segment featurization
    def get_vessel_segments(self):
        """
        Generates a dictionary of vessel segments, where each key is a vessel type 
        and each value is a subgraph of the total segment.

        """
        vessel_type_counts = {}
        vessel_type_hierarchy = {}
        vessel_type_nodes = {}

        for node, data in self.total_segment.nodes(data=True):
            vessel_type = data["vessel_type_name"]
            hierarchy = data["hierarchy"]

            vessel_type_counts.setdefault(vessel_type, 0)
            vessel_type_counts[vessel_type] += 1

            vessel_type_nodes.setdefault(vessel_type, []).append(node)
            vessel_type_hierarchy.setdefault(vessel_type, []).append(hierarchy)

        for vessel_type, hierarchies in vessel_type_hierarchy.items():
            vessel_type_hierarchy[vessel_type] = np.mean(hierarchies)

        sort_by_hierarchy(vessel_type_counts, vessel_type_hierarchy, vessel_type_nodes)
        filter_vessel_types(vessel_type_counts, vessel_type_nodes, vessel_type_hierarchy)

        for vessel_type in vessel_type_nodes:
            self.vessel_segments[vessel_type] = self.total_segment.subgraph(vessel_type_nodes[vessel_type]).copy()
            self.vessel_segments[vessel_type].graph["vessel_type_name"] = vessel_type
            self.vessel_segments[vessel_type].graph["pos"] = np.mean([self.vessel_segments[vessel_type].nodes[node]["pos"] for node in self.vessel_segments[vessel_type]], axis=0)
            self.vessel_segments[vessel_type].graph["hierarchy"] = vessel_type_hierarchy[vessel_type]
            self.vessel_segments[vessel_type].graph["vesssel_type"] = get_vessel_type(vessel_type)
            self.vessel_segments[vessel_type].graph["onehot_vessel_type"] = get_onehot_encoded_vessel_type(vessel_type)
    
    def build_segment_graph(self):
        """
        Uses subgraphs of the total segment to build a segment graph, where each node 
        represents a vessel segment and each edge represents the connection between two
        segments. The graph is saved in the segment_graph attribute.

        """
        if len(self.vessel_segments) == 0:
            self.get_vessel_segments()

        node = 0
        for vessel_type in self.vessel_segments:
            self.segment_graph.add_node(node)
            self.segment_graph.nodes[node]["vessel_type_name"] = vessel_type
            self.segment_graph.nodes[node]["pos"] = self.vessel_segments[vessel_type].graph["pos"]
            self.segment_graph.nodes[node]["hierarchy"] = self.vessel_segments[vessel_type].graph["hierarchy"]
            self.segment_graph.nodes[node]["features"] = compute_segment_graph_node_features(self.vessel_segments[vessel_type])
            for feature_to_remove in ["min_diameter", "max_diameter", "proximal_diameter", "distal_diameter", "min_max_diameter_ratio", \
                                      "bending_length", "cumulative_curvature", "tortuosity_index_5_cm", "accumulated_polar_angle_differential"]:
                self.segment_graph.nodes[node]["features"].pop(feature_to_remove)
            self.segment_graph.nodes[node]["features"]["vessel_type"] = get_vessel_type(vessel_type)
            self.segment_graph.nodes[node]["vessel_type"] = get_vessel_type(vessel_type)
            self.segment_graph.nodes[node]["onehot_vessel_type"] = get_onehot_encoded_vessel_type(vessel_type).astype(int)
            self.segment_graph.nodes[node]["features_list"] = np.concatenate(([value for value in self.segment_graph.nodes[node]["features"].values()][:-1], self.segment_graph.nodes[node]["onehot_vessel_type"]))

            self.segment_graph.nodes[node]["subgraph"] = self.vessel_segments[vessel_type]

            if node > 0:
                self.segment_graph.add_edge(node - 1, node)
                self.segment_graph[node - 1][node]["features"] = compute_segment_graph_edge_features(self.segment_graph.nodes[node - 1]["subgraph"], self.segment_graph.nodes[node]["subgraph"])
                self.segment_graph[node - 1][node]["features_list"] = [value for value in self.segment_graph[node - 1][node]["features"].values()]

            node += 1

    def save_segment_graph(self):
        """
        Save the segment graph in a pickle file in the case directory.

        """
        if len(self.segment_graph) == 0:
            self.build_segment_graph()
        with open(os.path.join(self.case_dir, "segment_supersegment.pickle"), "wb") as f:
            pickle.dump(self.segment_graph, f)

    # Methods for dense featurization
    def build_dense_graph(self):
        """
        Builds a dense graph from the total segment, where each node represents a local
        centerline point. The graph is saved in the dense_graph attribute.

        Attributes from the total segment are reformatted and saved in the dense graph.
        
        """
        self.dense_graph = clean_supersegment(self.supersegment)
        for node in self.dense_graph:
            clean_node_data = clean_node(self.dense_graph.nodes[node])
            # Pop everything from the original node
            for key in self.dense_graph.nodes[node].copy():
                self.dense_graph.nodes[node].pop(key)
            for key, value in clean_node_data.items():
                nx.set_node_attributes(self.dense_graph, {node: value}, key)

    def save_dense_graph(self):
        """
        Save the dense graph in a pickle file in the case directory.
        
        """
        if self.dense_graph is None:
            self.build_supersegment()
        with open(os.path.join(self.case_dir, "dense_supersegment.pickle"), "wb") as f:
            pickle.dump(self.dense_graph, f)

    def save_raw_pickle(self, regression_value=np.nan, classification_value=np.nan):
        """
        Save everything as a single pickle file in the case directory.

        We can also add the regression and classification labels to the pickle file. This will
        be used to generate the ArterialMaps dataset, but it also makes it usable for inference.

        Parameters
        ----------
        regression_label : float
            Regression label for the case
        classification_label : int
            Classification label for the case

        """
        if len(self.global_features) == 0:
            self.get_global_features()
        if len(self.segment_graph) == 0:
            self.build_segment_graph()
        if self.dense_graph is None:
            self.build_dense_graph()
        with open(os.path.join(self.case_dir, f"{os.path.basename(self.case_dir)}.pickle"), "wb") as f:
            pickle.dump({
                "T1A": regression_value, 
                "classification": classification_value,
                "global_features": self.global_features, 
                "segment_graph": self.segment_graph, 
                "dense_graph": self.dense_graph
                }, f)
        

    # Visualization methods
    def plot_segment_graph(self, feature=None, cmap="bwr", output_path=None):
        assert len(self.segment_graph) > 0, "Segment graph not built"
        make_dense_graph_plot(self.segment_graph, feature, cmap, output_path)

    def plot_dense_graph(self, feature=None, cmap="bwr", output_path=None):
        assert self.dense_graph is not None, "Dense graph not built"
        make_dense_graph_plot(self.dense_graph, feature, cmap, output_path)

    def save_combined_plot(self):
        assert len(self.segment_graph) > 0, "Segment graph not built"
        assert self.dense_graph is not None, "Dense graph not built"
        make_combined_plot(self.side, self.segment_graph, self.dense_graph, os.path.join(self.case_dir, "combined_plot.png"))