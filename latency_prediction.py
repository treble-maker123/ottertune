"""
This module prunes the metrics according to the paper.

Example:
    python3 prune.py --dataset=offline_workload
"""
from argparse import Namespace, ArgumentParser
from pdb import set_trace
from time import time
import pickle

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score, mean_squared_error

from modules.dataset import Dataset, DATASET_PATHS
from modules.workload_gpr import WorkloadGPR
from modules.logger import build_logger
from typing import List, Union, Tuple, Dict
from tqdm import tqdm

from os import path

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

def make_s_matrix(target_wl, observed_data, primer_data, fn='outputs/s_matrix.npy'):
    if path.exists(fn):
        S = np.load(fn, allow_pickle=True)
    else:
        primer_wl = primer_data.get_specific_workload(target_wl)
        primer_wl_configs = [tuple(i) for i in primer_wl.get_tuning_knobs()]
        S = []
        workload_gpr = WorkloadGPR()
        for met in tqdm(observed_data.load_pruned_metrics()):
            curr_matrix = []
            for observed_wl_id in observed_data.get_workload_ids():
                observed_wl_configs = observed_data.get_specific_workload(observed_wl_id)
                observed_wl_configs = [tuple(i) for i in observed_data.get_tuning_knobs()]
                curr_row = []
                for primer_wl_config in primer_wl_configs:
                    if primer_wl_config in observed_wl_configs:
                        observed_wl_idx = observed_wl_configs.index(primer_wl_config)
                        curr_row.append(observed_data.get_dataframe().iloc[observed_wl_idx][met])
                    else:
                        curr_model = workload_gpr.get_model(observed_wl_id, met)
                        curr_row.append(curr_model.predict([[i for i in primer_wl_config]])[0])
                curr_matrix.append(curr_row)
            S.append(curr_matrix)
        S = np.array(S)
        np.save(fn, S)
    return S

def find_closest_observed_wl(target_wl, observed_data, primer_data, S):
    def target_to_s_format():
        primer_wl = primer_data.get_specific_workload(target_wl)
        primer_wl_configs = [tuple(i) for i in primer_wl.get_tuning_knobs()]
        S_target = []
        for met in primer_data.load_pruned_metrics():
            curr_vector = []
            for idx in range(len(primer_wl_configs)):
                curr_vector.append(primer_wl.get_dataframe().iloc[idx][met])
            S_target.append(curr_vector)
        return np.array(S_target)

    scores = np.array([0 for i in range(S.shape[1])])
    S_target = target_to_s_format()
    binned_S = np.zeros_like(S)
    binned_S_target = np.zeros_like(S_target)
    for met_idx in range(S.shape[0]):
        binned_S[met_idx] = observed_data.bin_metrics(S[met_idx], 10)
        binned_S_target[met_idx] = observed_data.bin_metrics(np.expand_dims(S_target[met_idx],1), 10)[:,0]
        for wl_idx in range(S.shape[1]):
            scores[wl_idx] += np.linalg.norm(binned_S[met_idx][wl_idx]-binned_S_target[met_idx])
    return observed_data.get_workload_ids()[np.argmin(scores)]
    # return observed_data.get_workload_ids()[np.random.choice(np.argwhere(scores==np.min(scores))[:,0])]

def train_concat_model(target_wl, observed_data, primer_data, closest_wl):
    def remove_duplicate_knobs(xd, yd):
        unique_knobs = [tuple(xd[i,:]) for i in range(xd.shape[0])]
        idxs_to_remove = []
        for i in range(xd.shape[0]):
            following_knobs = [tuple(xd[j,:]) for j in range(i+1,xd.shape[0])]
            if tuple(xd[i]) in following_knobs:
                idxs_to_remove.append(i)
        xd = np.delete(xd, idxs_to_remove, 0)
        yd = np.delete(yd, idxs_to_remove, 0)
        return xd, yd

    closest_data = observed_data.prune_columns(
        ['workload id'] + observed_data.get_tuning_knob_headers()+['latency']).get_specific_workload(closest_wl).get_dataframe()
    target_data = primer_data.prune_columns(
        ['workload id'] + primer_data.get_tuning_knob_headers()+['latency']).get_specific_workload(target_wl).get_dataframe()
    concat_data = pd.concat([closest_data, target_data], ignore_index=True).values
    X, y = remove_duplicate_knobs(concat_data[:, 1:-1], concat_data[:, -1])
    # alpha = np.array([1e-5 for i in range(X.shape[0]-5)] + [1e-5 for i in range(5)])
    # model = GaussianProcessRegressor(kernel=RBF())
    model = GaussianProcessRegressor(RBF())
    ss_x = StandardScaler()
    ss_y = StandardScaler()
    # X = ss_x.fit_transform(X)
    # y = ss_y.fit_transform(np.expand_dims(y,1))
    model.fit(X, y)
    return model, ss_x, ss_y

def eval_model(target_wl, model, eval_data, ss_x, ss_y):
    eval_X = eval_data.prune_columns(
        ['workload id'] + eval_data.get_tuning_knob_headers()).get_specific_workload(target_wl).get_dataframe()
    eval_X = eval_X.values[:,1:13]
    # eval_X = ss_x.transform(eval_X)
    result = model.predict(eval_X)[0]
    # result = ss_y.inverse_transform([[result]])[0][0][0]
    return result


def split_online_b(online_b_data):
    primer = pd.DataFrame(columns=online_b_data.get_dataframe().columns)
    eval = pd.DataFrame(columns=online_b_data.get_dataframe().columns)
    for wl_id in online_b_data.get_workload_ids():
        curr_ds = online_b_data.get_specific_workload(wl_id)
        for idx in range(curr_ds.get_dataframe().values.shape[0]):
            if idx == 0:
                eval = eval.append(curr_ds.get_dataframe().iloc[idx:idx+1], ignore_index=True)
            else:
                primer = primer.append(curr_ds.get_dataframe().iloc[idx:idx+1], ignore_index=True)
    primer = Dataset(dataframe=primer)
    eval = Dataset(dataframe=eval)
    latency_gt = eval.get_column_values('latency')
    eval = eval.prune_columns(['workload id'] + eval.get_tuning_knob_headers())

    return primer, eval, latency_gt

def main():
    """
    Main method for the script.
    """
    def run_on_b_data():
        b_primer, b_test, b_gt = split_online_b(online_b_data)
        b_pred = []
        for curr_wl in tqdm(b_test.get_workload_ids()):
            S = make_s_matrix(curr_wl, offline_data, b_primer, 'outputs/b_s_matrix.npy')
            closest_observed_wl = find_closest_observed_wl(curr_wl, offline_data, b_primer, S)
            model, ss_x, ss_y = train_concat_model(curr_wl, offline_data, b_primer, closest_observed_wl)
            b_pred.append(eval_model(curr_wl, model, b_test, ss_x, ss_y))
        print('r2:\t', r2_score(b_gt, b_pred))
        print('mse:\t', mean_squared_error(b_gt, b_pred))

    def run_on_test_data():
        for curr_wl in tqdm(test_data.get_workload_ids()):
            S = make_s_matrix(curr_wl, offline_data, online_c_data)
            closest_observed_wl = find_closest_observed_wl(curr_wl, offline_data, online_c_data, S)
            model, ss = train_concat_model(curr_wl, offline_data, online_c_data, closest_observed_wl)
            pred_latency = eval_model(curr_wl, model, test_data, ss)

    offline_data = Dataset(file_path=DATASET_PATHS['offline_workload'])
    online_b_data = Dataset(file_path=DATASET_PATHS['online_workload_B'])
    online_c_data = Dataset(file_path=DATASET_PATHS['online_workload_C'])
    test_data = Dataset(file_path=DATASET_PATHS['test'])

    run_on_b_data()
    # run_on_test_data()

if __name__ == "__main__":
    CONFIG = build_config()
    LOG = build_logger()
    print()
    main()
    print()
