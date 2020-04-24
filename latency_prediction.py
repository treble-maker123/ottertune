"""
This module prunes the metrics according to the paper.

Example:
    python3 prune.py --dataset=offline_workload
"""
from argparse import Namespace, ArgumentParser
from pdb import set_trace
from time import time

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans

from modules.dataset import Dataset, DATASET_PATHS
from modules.logger import build_logger
from typing import List, Union, Tuple, Dict

def build_config() -> Namespace:
    """
    Builds a Namespace object containing configurations from argparser.
    """
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default='offline_workload',
                        help='The dataset to use, one of "offline_workload", '
                             '"online_worload_B", "online_workload_C", or '
                             '"test"')

    parser.add_argument('--pruned-metrics', type=str, default='outputs/pruned_metrics.txt',
                        help='The file containing the pruned metrics.')

    known_args, _ = parser.parse_known_args()

    return known_args


def remove_pruned_metrics(dataset) -> pd.DataFrame:
    with open(CONFIG.pruned_metrics) as f:
        kept_cols = [line.strip() for line in f]
    cols = ['workload id'] + dataset.get_tuning_knob_headers() + kept_cols
    data = {col: dataset.get_column_values(col) for col in cols}
    return pd.DataFrame(data=data)


def create_s_matrix(dataset) -> Dict[str, Dict[str, List[float]]]:
    """
    Method that generates S matrix, as defined on pg. 7 of the original paper.
    """
    s = {}

    # Gets the list of all unique DBMS configurations
    config_cols = np.swapaxes([dataset.get_column_values(t) for t in dataset.get_tuning_knob_headers()], 0, 1)
    unique_configurations = list(set(tuple(i) for i in [list(t) for t in list(config_cols)]))

    # Gets list of metrics
    metric_headers = dataset.get_metric_headers()

    # Generates list of unique worklaod IDs
    unique_workloads = list(set(dataset.get_column_values(dataset.get_headers()[0])))

    # Forms a matrix for each metric, putting them all in a dict called s (representing matrix S in paper)
    for metric in metric_headers:
        curr_matrix = {}
        for wl in unique_workloads:
            # Indices of the dataset that are of the current workload id: wl
            wl_indices = dataset.get_dataframe()[dataset.get_dataframe()['workload id'] == wl][metric].index

            # knob configurations of each of the configurations
            wl_configs = [list(i) for i in list(np.array(config_cols)[wl_indices])]

            # setup current row of matrix as empty, len(unique_configurations) columns since columns are each each configuration
            curr_row = [0 for t in range(len(unique_configurations))]

            # gets current metric's value at each of the configurations of the current workload
            wl_metric_values = dataset.get_dataframe()[dataset.get_dataframe()['workload id'] == wl][metric].values
            for idx, u_c in enumerate(unique_configurations):
                u_c = list(u_c)
                if (u_c in wl_configs) and (curr_row[idx] == 0):
                    wl_idx = wl_configs.index(u_c)
                    curr_row[idx] = wl_metric_values[wl_idx]
                # paper mentions median of values if spot in row is already filled, doesn't seem to occur
            curr_matrix[wl] = curr_row
            if max(curr_row) > 0: print(wl, max(curr_row))
        s[metric] = curr_matrix
        for t in s.keys():
            print(t)
            print(s[t].keys(), '\n')
    return s


def workload_mapping(s_matrix):
    """
    Method that computes the workload mapping, as described on pg. 7 of the original paper.
    """
    

    return 0


def main():
    """
    Main method for the script.
    """
    dataset = Dataset(file_path=DATASET_PATHS[CONFIG.dataset])
    # print(dataset.get_headers())

    # build the training and validation sets from the partitions
    train, val = dataset.build_dataset_from_partition(part_idx=1)

    # make copies of train/val but only with non-pruned columns
    train = Dataset(dataframe=remove_pruned_metrics(train))
    val = Dataset(dataframe=remove_pruned_metrics(val))

    # create S matrix for training set
    train_s = create_s_matrix(train)
    val_s = create_s_matrix(val)








if __name__ == "__main__":
    CONFIG = build_config()
    LOG = build_logger()

    main()
