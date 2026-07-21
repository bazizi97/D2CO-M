import torch
import math
from torch.utils.data import Dataset
import numpy as np
import json
import time
import copy
import tqdm
import random
import torch.nn.functional as F
import pandas as pd
#from quaternion import *

### Data loaders ###
import glob
import os
import json

import numpy as np
import torch

from torch_geometric.data import Data as GraphData


class CATHDataset:
    """
    Loader and container class for the CATH 4.2 dataset downloaded
    from http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/.

    Has attributes `self.train`, `self.val`, `self.test`, each of which are
    JSON/dictionary-type datasets as described in README.md.

    :param path: Path to chain_set.jsonl containing protein chain structures and sequences.
    :param splits_path: Path to chain_set_splits.json containing train/validation/test split lists.
    """

    def __init__(self, path, splits_path):
        # Load dataset split definitions (chain identifiers for train, validation, and test sets)
        with open(splits_path) as f:
            dataset_splits = json.load(f)
        train_list, val_list, test_list = dataset_splits["train"], dataset_splits["validation"], dataset_splits["test"]

        self.train, self.val, self.test = [], [], []

        # Read JSONL file containing protein entries line by line
        with open(path) as f:
            lines = f.readlines()

        # Parse each line and group coordinates into (N, CA, C, O) tuples
        for line in tqdm.tqdm(lines):
            entry = json.loads(line)
            name = entry["name"]
            coords = entry["coords"]

            # Zip coordinate arrays corresponding to Backbone Nitrogen, Alpha Carbon, Carbon, and Oxygen atoms
            entry["coords"] = list(zip(coords["N"], coords["CA"], coords["C"], coords["O"]))

            # Assign entry to appropriate dataset split based on its name
            if name in train_list:
                self.train.append(entry)
            elif name in val_list:
                self.val.append(entry)
            elif name in test_list:
                self.test.append(entry)




class CFNDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for Cost Function Network (CFN) graph instances stored as PyTorch (.pt) files.

    Args:
        data_file (str): Path to the directory containing PyTorch binary dataset (.pt) files.
        data_label_dir (str, optional): Optional override directory for label files. Defaults to None.
    """

    def __init__(self, data_file, data_label_dir=None):
        self.data_file = data_file
        # Find all .pt graph files in the designated directory
        self.file_lines = glob.glob(os.path.join(data_file, '*.pt'))
        self.data_label_dir = data_label_dir
        self.data_file = data_label_dir if data_label_dir else data_file
        print(f'Loaded "{data_file}" with {len(self.file_lines)} examples')

    def __len__(self):
        """Returns the total number of graph instance files in the dataset."""
        return len(self.file_lines)

    def get_example(self, idx):
        """
        Loads a single PyTorch graph instance from disk and extracts graph attributes.

        Args:
            idx (int): Index of the instance to load.

        Returns:
            tuple: Contains (num_nodes, node_labels, edges_index, edges_attr, native, missing).
        """
        graph = torch.load(self.file_lines[idx], weights_only=False, map_location='cpu')
        
        num_nodes = graph['sequence'].shape[0]

        node_labels = graph['sequence']
        edges_index = graph["edge_index"]
        edges_attr = graph["edge_attr"].detach()
        native = graph["native"]
        missing = graph["missing"]
        # clauses = np.array(graph["clauses"], dtype=np.int64)

        return num_nodes, node_labels, edges_index, edges_attr, native, missing

    def __getitem__(self, idx):
        """
        Retrieves an item by index and constructs a PyTorch Geometric Data object.

        Args:
            idx (int): Dataset sample index.

        Returns:
            tuple: (index_tensor, graph_data_PyG, point_indicator_tensor)
        """
        num_nodes, node_labels, edge_index, edge_attr, native, missing = self.get_example(idx)
        
        # Package node, edge, and native state attributes into PyG GraphData
        graph_data = GraphData(x=node_labels,
                              edge_index=edge_index,
                              edge_attr=edge_attr,
                              nat=native,
                              missing=missing)

        # Store node count indicator for batching / tracking purposes
        point_indicator = np.array([num_nodes], dtype=np.int64)
        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            graph_data,
            torch.from_numpy(point_indicator).long(),
        )