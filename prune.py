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
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.random_projection import GaussianRandomProjection

from modules.dataset import Dataset, DATASET_PATHS
from modules.logger import build_logger


def build_config() -> Namespace:
    """
    Builds a Namespace object containing configurations from argparser.
    """
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default='offline_workload',
                        help='The dataset to use, one of "offline_workload", '
                             '"online_workload_B", "online_workload_C", or '
                             '"test"')
    parser.add_argument('--model', type=str, default='kmeans',
                        help='Which model to run, either "kmeans" or "kmedoid"')
    parser.add_argument('--output-path', type=str,
                        default='outputs/pruned_metrics.txt',
                        help='The output path and file name of this script.')

    parser.add_argument('--num-factors', type=int, default=5,
                        help='The number of factors to reduce the metrics.')
    # paper picks one between 0 and 20, we will use 10 per project requirement
    parser.add_argument('--max-clusters', type=int, default=10,
                        help='The maximum number of clusters to generate for '
                             'K-means.')
    parser.add_argument('--use-k', dest='use_k', action='store_true',
                        default=False,
                        help='Whether to use a specific k for clustering, if '
                             ' so, num-clusters argument will be used.')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of clusters to use, only when use-k '
                             ' is true.')

    known_args, _ = parser.parse_known_args()

    return known_args


def build_k_means(factors: np.ndarray) -> KMeans:
    """
    Builds a KMeans model from the given factors.
    """
    if CONFIG.use_k:
        k = CONFIG.k
        LOG.debug('Running K-means with k=%s clusters...', k)
        model = KMeans(n_clusters=k, n_init=50, max_iter=500).fit(factors)
    else:
        best_model, best_score = None, float('-inf')
        scores = []

        for i in range(1, CONFIG.max_clusters):
            k = i + 1
            LOG.debug('Starting K-means with %s clusters...', k)
            start = time()
            model = KMeans(n_clusters=k, n_init=50, max_iter=500).fit(factors)
            score = silhouette_score(factors, model.labels_)
            scores.append(score)
            LOG.info('Finished K-means with %s clusters in %s seconds, score: %s',
                     k, round(time()-start), score)
            if score > best_score:
                best_model = model
                best_score = score
                LOG.debug('Better score! Saving model with k=%s.', score)

        scores = pd.DataFrame(scores, index=np.arange(1, len(scores) + 1))
        scores.to_csv('outputs/k_means_silhouette.csv')

        model = best_model

    return model


def build_k_medoids(factors: np.ndarray):
    """
    Builds a KMedoids model from the given factors
    """
    if CONFIG.use_k:
        k = CONFIG.k
        LOG.debug('Running K-Medoids with k=%s clusters...', k)
        model = KMedoids(n_clusters=k, max_iter=500).fit(factors)
    else:
        best_model, best_score = None, float('-inf')
        scores = []

        for i in range(1, CONFIG.max_clusters):
            k = i + 1
            LOG.debug('Starting K-medoids with %s clusters...', k)
            start = time()
            model = KMedoids(n_clusters=k, max_iter=500).fit(factors)
            score = silhouette_score(factors, model.labels_)
            scores.append(score)
            LOG.info('Finished K-medoids with %s clusters in %s seconds, score: %s',
                     k, round(time()-start), score)
            if score > best_score:
                best_model = model
                best_score = score
                LOG.debug('Better score! Saving model with k=%s.', score)

        model = best_model

        scores = pd.DataFrame(scores, index=np.arange(1, len(scores) + 1))
        scores.to_csv('outputs/k_medoids_silhouette.csv')

    return model


def main():
    """
    Main method for the script.
    """
    dataset = Dataset(file_path=DATASET_PATHS[CONFIG.dataset])
    df = dataset.get_dataframe()

    # remove columns that are constant values
    metric_headers = dataset.get_metric_headers()
    constant_headers = []
    variable_headers = []
    for header in metric_headers:
        if np.unique(df[header].values).size > 1:
            variable_headers.append(header)
        else:
            constant_headers.append(header)

    metric_headers = variable_headers
    dataset = Dataset(dataframe=df.drop(constant_headers, axis=1))
    raw_metrics = dataset.get_metrics()
    metrics = raw_metrics.T

    # factor analysis
    LOG.info('Starting factor analysis with %s factors...', CONFIG.num_factors)
    start = time()
    # model = FactorAnalysis(n_components=CONFIG.num_factors)
    # factors = model.fit_transform(metrics)  # num_metrics * num_factors
    rng = np.random.RandomState(74)
    model = GaussianRandomProjection(eps=0.999, random_state=rng)
    factors = model.fit_transform(metrics)
    LOG.debug('Dimension before factor analysis: %s', metrics.shape)
    LOG.debug('Dimension after factor analysis: %s', factors.shape)
    LOG.info('Finished factor analysis in %s seconds.', round(time()-start))

    # clustering
    if CONFIG.model == 'kmeans':
        model = build_k_means(factors)
    elif CONFIG.model == 'kmedoids':
        model = build_k_medoids(factors)
    else:
        raise ValueError('Unrecognized model: %s', CONFIG.model)

    # find cluster center
    labels = model.labels_
    # each dimension in transformed_data is the distance to the cluster
    # centers.
    transformed_data = model.transform(factors)
    leftover_metrics = []
    for i in np.unique(labels):
        # index of the points for the ith cluster
        cluster_member_idx = np.argwhere(labels == i).squeeze(1)
        cluster_members = transformed_data[cluster_member_idx]
        # find the index of the minimum-distance point to the center
        closest_member = cluster_member_idx[np.argmin(cluster_members[:, i])]
        leftover_metrics.append(metric_headers[closest_member])

    # latency needs to be in the metrics
    if 'latency' not in leftover_metrics:
        leftover_metrics += ['latency']

    with open(CONFIG.output_path, 'w') as file:
        file.writelines('\n'.join(leftover_metrics))


if __name__ == "__main__":
    CONFIG = build_config()
    LOG = build_logger()

    main()
