import pickle

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp

from copy import deepcopy

def make_single_supersegment_plot(graph_path, side, output_path):
    with open(graph_path, "rb") as f:
        supersegment = pickle.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(5, 10))
    highlight_node = None
    for node in supersegment:
        if supersegment.nodes[node]["hierarchy"] == 0:
            highlight_node = node
        
    colorPalette = mcp.gen_color(cmap = "bwr", n = 2)
    color_map = [colorPalette[not supersegment.nodes[node]["is_supersegment"]] for node in supersegment] 
    
    if highlight_node is not None:
        color_map[highlight_node] = "chartreuse"

    # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_p = {}
    for n in supersegment.nodes():
        node_pos_dict_p[n] = [-supersegment.nodes(data=True)[n]["pos"][0], supersegment.nodes(data=True)[n]["pos"][2]]

    nx.draw(supersegment, node_pos_dict_p, node_size=10, node_color=color_map, ax=ax)
    ax.set_title(f"Supersegment ({side})", fontsize=12)
    ax.set_xlim([-200, 10])
    ax.set_ylim([-10, 300])

    plt.savefig(output_path)
    plt.close()

def make_dense_graph_plot(graph, feature = None, cmap = "bwr", output_path=None):
    graph_ = deepcopy(graph)
    _ = plt.figure(figsize=[6, 10], dpi=300)
    ax = plt.gca()
    if feature not in graph_.nodes[next(iter(graph.nodes))]["features"]:
        try: 
            for node in graph:
                graph_.nodes[node]["features"][feature] = graph_.nodes[node][feature]
        except:
            print("Feature not found in the graph")
            feature = None
    if feature is not None:
        # Color map
        feature_values = [graph_.nodes[node]["features"][feature] for node in graph_]
        color_palette = mcp.gen_color(cmap = cmap, n = len(np.unique(feature_values)))
        color_dict = dict(zip(np.unique(feature_values), color_palette))
        # Choose color for each node. Each node feature will have to be rounded to the nearest integer to be used as an index for the color_palette
        # color_palette indices do not have physical meaning. The whole range of the fetaure_values is divided into the n nodes of the supersegment. 
        # Each feature value should be mapped to the corresponding index of the color_palette. To do that, we have to divide the feature value of each node by the range of the feature values and multiply by the number of nodes in the supersegment.
        if max(feature_values) != min(feature_values):
            color_map = [color_dict[graph_.nodes[node]["features"][feature]] for node in graph_]
            # color_map = [color_palette[int(np.round((len(np.unique(feature_values)) - 1) * (graph.nodes[node]["features"][feature] - min(feature_values)) / (max(feature_values) - min(feature_values))))] for node in graph]
        else:
            print("All values are the same. Setting to blue")
            color_map = "blue"
    else:
        color_map = "blue"

    # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_P = {}
    for n in graph_.nodes():
        node_pos_dict_P[n] = [-graph_.nodes(data=True)[n]["pos"][0], graph_.nodes(data=True)[n]["pos"][2]]

    nx.draw(graph_, node_pos_dict_P, node_size=10, node_color=color_map)
    ax.set_xlim([-200, 10])
    ax.set_ylim([-10, 300])
    # Set title as feature
    if feature is not None:
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(feature_values), vmax=max(feature_values)))
        sm._A = []
        plt.colorbar(sm, fraction=0.046, pad=0.04, ax=ax)
        plt.title(feature.capitalize().replace("_", " "))  

    if output_path is not None:
        plt.savefig(output_path)

def make_combined_plot(side, segment_graph, dense_graph, output_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10), dpi = 300)
    # Set side as title of the whole plot
    fig.suptitle(side, fontsize=16)

    # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_P = {}
    for n in segment_graph.nodes():
        node_pos_dict_P[n] = [-segment_graph.nodes(data=True)[n]["pos"][0], segment_graph.nodes(data=True)[n]["pos"][2]]

    nx.draw(segment_graph, node_pos_dict_P, node_size=10, ax=ax[0])
    ax[0].set_xlim([-200, 10])
    ax[0].set_ylim([-10, 300])

    # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_P = {}
    for n in dense_graph.nodes():
        node_pos_dict_P[n] = [-dense_graph.nodes(data=True)[n]["pos"][0], dense_graph.nodes(data=True)[n]["pos"][2]]

    nx.draw(dense_graph, node_pos_dict_P, node_size=10, ax=ax[1])
    ax[1].set_xlim([-200, 10])
    ax[1].set_ylim([-10, 300])

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()