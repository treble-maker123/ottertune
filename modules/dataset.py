"""
This module supplies classes and methods for interacting with the dataset.
"""
from pdb import set_trace
from typing import List

import pandas as pd
import numpy as np

DATA_PATH_PREFIX = 'project3_dataset'

DATASET_PATHS = {
    'offline_workload': f"{DATA_PATH_PREFIX}/offline_workload.CSV",
    'online_workload_B': f"{DATA_PATH_PREFIX}/online_workload_B.CSV",
    'online_workload_C': f"{DATA_PATH_PREFIX}/online_workload_C.CSV",
    'test': f"{DATA_PATH_PREFIX}/test.CSV"
}


class Dataset:
    """
    This class contains methods to manipulate the data files.
    """

    def __init__(self, file_path: str):
        self._file_path = file_path
        self._dataframe = None

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas dataframe that contains the specified data file.
        """
        if not self._dataframe:
            self._dataframe = pd.read_csv(self._file_path)

        return self._dataframe

    def get_headers(self) -> List[str]:
        """
        Returns a list of headers from the data file. This method excludes the
        first header 'workdload id'
        """
        return self.get_dataframe().columns.values.tolist()[1:]

    def get_metrics(self, unique: bool = True) -> np.ndarray:
        """
        Returns the data in a numpy array. Note that the first columnm
        ('workload id') is discarded, and the resulting array is converted
        from an object array into a float array, i.e. True and False values are
        converted into 1.0 and 0.0.

        Args:
            unique: whether to trim columns with values that do not change.

        Returns:
            An array of dimension [NUM_WORKLOADS * NUM_METRICS].
        """
        metrics = self.get_dataframe().values[:, 1:].astype(float)

        if unique:
            the_same = metrics.max(axis=1) == metrics.min(axis=1)
            idx_to_remove = np.argwhere(the_same).squeeze()
            metrics = np.delete(metrics, idx_to_remove, axis=1)

        return metrics

    @classmethod
    def bin_metrics(cls, metrics: np.ndarray,
                    num_bins: int = 10, axis=0) -> np.ndarray:
        """
        Bin the values of the input metrics, a two-dimensional array, in to
        num_bins bins, the values of the bins evenly spaced between 0.0 and
        1.0.

        Args:
            metrics: the two-dimensional array on which to bin the values
            num_bins: number of bins to create
            axis: along which axis should the values be binned

        Returned:
            An array of the same dimension as the input metrics array, where
            the maximum value is less than or equal to 1, and the minimum is
            greater than 0.
        """
        # i.e. [10, 20, ..., 100] for num_bins = 10
        percentiles = np.arange(0, 101, 100 / num_bins)[1:].astype(int)
        bin_values = percentiles.astype(float) / 100.0

        # NOTE: axis=0 to get percentiles across the column row
        metrics_pct = np.percentile(metrics, percentiles, axis=axis)
        assert metrics_pct.shape[0] == num_bins
        assert metrics_pct.shape[1] == metrics.shape[1]

        binned_metrics = np.zeros_like(metrics).astype(float)
        for i in reversed(range(10)):
            ith_pct_values = metrics_pct[i][np.newaxis, :]
            binned_metrics[metrics <= ith_pct_values] = bin_values[i]

        assert binned_metrics.max() <= 1.0, f"Max is {binned_metrics.max()}"
        assert binned_metrics.min() > 0, f"Min is {binned_metrics.min()}"

        return binned_metrics
