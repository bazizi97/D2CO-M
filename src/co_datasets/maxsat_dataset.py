"""Max2Sat Sparse dataset module."""

import glob
import os
import json

import numpy as np
import torch

from torch_geometric.data import Data as GraphData


class MAX2SATSPARSEDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for MAX2SAT Sparse problem instances stored as JSON files.

    Args:
        data_file (str): Directory path containing MAX2SAT JSON problem files.
        data_label_dir (str, optional): Directory path for data labels, if separate. Defaults to None.
    """

    def __init__(self, data_file, data_label_dir=None):
        self.data_file = data_file
        # Find all JSON problem instance files matching pattern
        self.file_lines = glob.glob(os.path.join(data_file, "*json"))
        self.data_label_dir = data_label_dir
        self.data_file = data_label_dir if data_label_dir else data_file
        # print(f'Loaded "{data_file}" with {len(self.file_lines)} examples')

    def __len__(self):
        """Returns the total number of MAX2SAT JSON instance files in dataset."""
        return len(self.file_lines)

    def get_example(self, idx):
        """
        Loads and parses a single MAX2SAT JSON instance file.

        Args:
            idx (int): Index of dataset example to fetch.

        Returns:
            tuple: (num_nodes, node_labels, edges_index, edges_attr, weights)
        """
        with open(self.file_lines[idx], "rb") as f:
            graph = json.load(f)

        num_nodes = len(graph["nodes"])

        node_labels = np.array(graph["nodes"], dtype=np.int64)
        edges_index = np.array(graph["edge_index"], dtype=np.int64)
        edges_attr = np.array(graph["edge_attr"], dtype=np.int64)
        weights = np.array(graph["weights"], dtype=np.float32)
        
        # clauses = np.array(graph["clauses"], dtype=np.int64)

        return num_nodes, node_labels, edges_index, edges_attr, weights

    def __getitem__(self, idx):
        """
        Retrieves a MAX2SAT instance by index and converts it to PyTorch Geometric Data.

        Args:
            idx (int): Dataset sample index.

        Returns:
            tuple: (index_tensor, graph_data_PyG, point_indicator_tensor)
        """
        num_nodes, node_labels, edge_index, edge_attr, weights = self.get_example(idx)
        
        # Build PyTorch Geometric Data instance with node features, transposed edge matrix, edge attributes, and weights
        graph_data = GraphData(
            x=torch.from_numpy(node_labels),
            edge_index=torch.from_numpy(edge_index.T),
            edge_attr=torch.from_numpy(edge_attr),
            weights=torch.from_numpy(weights)
        )

        # Record total node count for tracking graph size in batched pipelines
        point_indicator = np.array([num_nodes], dtype=np.int64)
        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            graph_data,
            torch.from_numpy(point_indicator).long(),
        )

