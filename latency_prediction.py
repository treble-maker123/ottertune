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
    unique_configurations = sorted(list(set(tuple(i) for i in [list(t) for t in list(config_cols)])))

    # Gets list of metrics
    metric_headers = dataset.get_metric_headers()

    # Generates list of unique worklaod IDs
    unique_workloads = sorted(list(set(dataset.get_column_values(dataset.get_headers()[0]))))

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
        s[metric] = curr_matrix
    return s


def workload_mapping(input_workload, s_matrix, dataset) -> int:
    """
    Method that computes the workload mapping, as described on pg. 7 of the original paper.
    Finds the workload in the dataset that is closest to the input workload.
    """
    def s_matrix_to_numpy_array(s_mat):
        output_s_mat = []
        for X in s_mat.keys():
            curr_mat = []
            for ij in s_mat[X].keys():
                curr_mat.append(s_mat[X][ij])
            output_s_mat.append(curr_mat)
        return np.array(output_s_mat)

    def workload_to_s_row(workload, metric_idx):
        if len(workload.shape) == 1:
            s_idx = unique_configurations.index(tuple(workload[1:13]))
            s_row = [0] * len(unique_configurations)
            s_row[s_idx] = workload[13+metric_idx]

        else:
            s_row = [0] * len(unique_configurations)
            for i in range(workload.shape[0]):
                s_idx = unique_configurations.index(tuple(workload[i, 1:13]))
                s_row[s_idx] = workload[13+metric_idx]
        return np.array(s_row)

    # Convert S matrix to np array
    np_s_matrix = s_matrix_to_numpy_array(s_matrix)

    # Make a copy of the np_s_matrix for making bins
    np_s_matrix_binned = np.array(np_s_matrix)

    # Helpful information -- unique configurations & workloads
    config_cols = np.swapaxes([dataset.get_column_values(t) for t in dataset.get_tuning_knob_headers()], 0, 1)
    unique_configurations = sorted(list(set(tuple(i) for i in [list(t) for t in list(config_cols)])))
    unique_workloads = sorted(list(set(dataset.get_column_values(dataset.get_headers()[0]))))
    num_unique_workloads = len(unique_workloads)

    scores = [0 for i in range(num_unique_workloads)]
    for i in range(np_s_matrix_binned.shape[0]):
        # Convert current matrix to decile bins
        bins = np.percentile(np_s_matrix_binned[i], [10 * (j+1) for j in range(10)])
        curr_met_matrix = np.digitize(np_s_matrix_binned[i], bins)

        # Convert the input workload to the format of the S matrix
        workload_s_row = workload_to_s_row(input_workload, i)

        # Calculate euclidean distance from input workload to each given workload (for current metric)
        for wl_idx in range(num_unique_workloads):
            scores[wl_idx] += np.linalg.norm(curr_met_matrix[wl_idx]-workload_s_row)

    return np.argmax(np.array(scores))


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

    # Compute workload mapping
    # Example of function usage, need to create actual pipeline
    input_workload = train.get_dataframe().values[0]
    workload_mapping(input_workload, val_s, val)
    workload_mapping(input_workload, train_s, train)


if __name__ == "__main__":
    CONFIG = build_config()
    LOG = build_logger()

    main()
