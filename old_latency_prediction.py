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
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

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
    if 'latency' not in cols:
        cols.append('latency')
    # print('cols', cols, '\n')
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
    # metric_headers = dataset.get_metric_headers()
    metric_headers = dataset.get_metric_headers()+['latency']

    # Generates list of unique workload IDs
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
            curr_row = [-1 for t in range(len(unique_configurations))]

            # gets current metric's value at each of the configurations of the current workload
            wl_metric_values = dataset.get_dataframe()[dataset.get_dataframe()['workload id'] == wl][metric].values
            for idx, u_c in enumerate(unique_configurations):
                u_c = list(u_c)
                if (u_c in wl_configs) and (curr_row[idx] == -1):
                    wl_idx = wl_configs.index(u_c)
                    curr_row[idx] = wl_metric_values[wl_idx]
                elif (u_c in wl_configs) and (curr_row[idx] != -1):
                    print('\n\n\nALREADY FILLED\n\n\n')
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
            s_row = [-1] * len(unique_configurations)
            s_row[s_idx] = workload[13+metric_idx]

        else:
            s_row = [-1] * len(unique_configurations)
            for j in range(workload.shape[0]):
                if tuple(workload[j, 1:13]) in unique_configurations:
                    s_idx = unique_configurations.index(tuple(workload[j, 1:13]))
                    s_row[s_idx] = workload[j, 13+metric_idx]
                else:
                    print('ERROR -- NEW CONFIGURATION')
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

    scores = np.array([0 for i in range(num_unique_workloads)])
    for i in range(np_s_matrix_binned.shape[0]):
        # Convert current matrix to decile bins
        # bins = np.percentile(np_s_matrix_binned[i], [10 * (j+1) for j in range(10)])
        ss = StandardScaler()
        ss.fit_transform(np_s_matrix_binned[i])
        bins = np.percentile(np_s_matrix_binned[i], np.arange(10, 101, 10))
        # curr_met_matrix = np.digitize(np_s_matrix_binned[i], bins)
        # https://github.com/cmu-db/ottertune/blob/37e0d58ce0adf1f9211a1fa8b107a5773f75353f/server/analysis/preprocessing.py#L109
        curr_met_matrix = np.zeros_like(np_s_matrix_binned[i])
        for j in range(10)[::-1]:
            decile = bins[j]
            curr_met_matrix[np_s_matrix_binned[i] <= decile] = j

        # Convert the input workload to the format of the S matrix
        # workload_s_row = workload_to_s_row(input_workload, i)
        # workload_s_row = np.digitize(workload_s_row, bins)
        og_workload_s_row = np.array([workload_to_s_row(input_workload, i)])
        ss.transform(og_workload_s_row)
        og_workload_s_row = og_workload_s_row[0]
        # https://github.com/cmu-db/ottertune/blob/37e0d58ce0adf1f9211a1fa8b107a5773f75353f/server/analysis/preprocessing.py#L109
        workload_s_row = np.zeros_like(og_workload_s_row)
        for j in range(10)[::-1]:
            decile = bins[j]
            workload_s_row[og_workload_s_row <= decile] = j


        # Calculate euclidean distance from input workload to each given workload (for current metric)
        for wl_idx in range(num_unique_workloads):
            scores[wl_idx] += np.linalg.norm(curr_met_matrix[wl_idx]-workload_s_row)

    # print(np.argwhere(scores==np.min(scores))[:,0].shape)
    return np.random.choice(np.argwhere(scores==np.min(scores))[:,0])


def latency_pred(input_workload, dataset, mapped_idx) -> int:
    """
    Method that creates a Gaussian Process (GP) model for latency prediction.
    """
    def get_wl_data(wl_id):
        most_similar_workload_data = dataset.get_dataframe()[dataset.get_dataframe()['workload id'] == wl_id]
        X = most_similar_workload_data.drop(dataset.get_non_metric_headers(), axis=1).values
        y = most_similar_workload_data['latency'].values
        return X, y

    # Trains model based on most-similar workload from mapping part
    unique_wls = sorted(list(set(dataset.get_column_values(dataset.get_headers()[0]))))
    map_wl = unique_wls[mapped_idx]
    map_X, map_y = get_wl_data(map_wl)
    gpr = GaussianProcessRegressor(kernel=RBF()).fit(map_X, map_y)

    # Trains model based on existing target workload examples from training data
    # NOTE: OVERWRITES PARAMETERS FROM BEFORE
    own_wl = np.unique(input_workload['workload id'].values)[0]
    own_X, own_y = get_wl_data(own_wl)
    gpr.fit(own_X, own_y)

    return 0


def main():
    """
    Main method for the script.
    """
    dataset = Dataset(file_path=DATASET_PATHS[CONFIG.dataset])

    # build the training and validation sets from the partitions
    train, val = dataset.build_dataset_from_partition(part_idx=1)

    # make copies of train/val but only with non-pruned columns
    # train = Dataset(dataframe=remove_pruned_metrics(train))
    # val = Dataset(dataframe=remove_pruned_metrics(val))
    val = Dataset(dataframe=remove_pruned_metrics(val))
    # Just using val dataset as example

    # create S matrix for training set
    # train_s = create_s_matrix(train)
    val_s = create_s_matrix(val)

    # Compute workload mapping
    # Example of function usage, need to create actual pipeline but should be simple
    input_workload = val.get_dataframe()
    here_un_wl = sorted(list(set(val.get_column_values(val.get_headers()[0]))))

    # Example: Workload 15-4 is the workload we're trying to predict latency for
    ex_wl_id = '15-4'
    # Gets example workload for 15-4, all values that are 15-4
    curr_input_workload = input_workload[input_workload['workload id'] == ex_wl_id]
    # Gets unique_workloads (here_un_wl) index of closest workload
    # Should be 15-4 in this example case since we're using same data
    here_idx = workload_mapping(curr_input_workload.values, val_s, val)

    # Predicts latency
    latency_pred(curr_input_workload, val, here_idx)


if __name__ == "__main__":
    CONFIG = build_config()
    LOG = build_logger()

    main()
