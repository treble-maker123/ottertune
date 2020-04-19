"""
This module prunes the metrics according to the paper.

Example:
    python3 prune.py --dataset=offline_workload
"""
from argparse import Namespace, ArgumentParser
from pdb import set_trace
from time import time

from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans

from modules.dataset import Dataset, DATASET_PATHS
from modules.logger import build_logger


def build_config() -> Namespace:
    """
    Builds a Namespace object containing configurations from argparser.
    """
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default='offline_workload',
                        help='The dataset to use, one of "offline_workload", '
                             '"online_worload_B", "online_workload_C", or '
                             '"test"')
    parser.add_argument('--output-path', type=str, default='pruned_metrics.csv',
                        help='The output path and file name of this script.')

    # https://github.com/cmu-db/ottertune/blob/d47b09b0c096312e26336fd38d4c76ccc02adda3/server/website/website/tasks/periodic_tasks.py#L297
    parser.add_argument('--num-factors', type=int, default=5,
                        help='The number of factors to reduce the metrics.')
    # paper picks one between 0 and 20, we will use 10 per project requirement
    parser.add_argument('--num-clusters', type=int, default=10,
                        help='The number of clusters to generate for K-means.')

    known_args, _ = parser.parse_known_args()

    return known_args


def main():
    """
    Main method for the script.
    """
    dataset = Dataset(file_path=DATASET_PATHS[CONFIG.dataset])
    metrics = dataset.get_metrics()
    metrics = Dataset.bin_metrics(metrics)

    # factor analysis
    LOG.info('Starting factor analysis with %s factors...', CONFIG.num_factors)
    start = time()
    model = FactorAnalysis(n_components=CONFIG.num_factors)
    factors = model.fit_transform(metrics)  # num_worloads * num_factors
    LOG.debug('Dimension before factor analysis: %s', metrics.shape)
    LOG.debug('Dimension after factor analysis: %s', factors.shape)
    LOG.info('Finished factor analysis in %s seconds.', round(time()-start))

    # k-means clustering
    LOG.info('Starting K-means with %s clusters...', CONFIG.num_clusters)
    start = time()
    model = KMeans(n_clusters=CONFIG.num_clusters, n_init=50, max_iter=500)
    model = model.fit(factors)
    LOG.info('Finished K-means clusteringin %s seconds.', round(time()-start))

    # find cluster center
    # TODO: Finish code to pick out cluster centers
    #  labels = model.labels_
    #  cluster_centers = model.cluster_centers_


if __name__ == "__main__":
    CONFIG = build_config()
    LOG = build_logger()

    main()
