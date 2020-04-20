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
        if self._dataframe is None:
            self._dataframe = pd.read_csv(self._file_path)

        return self._dataframe

    def get_headers(self) -> List[str]:
        """
        Returns a list of headers from the data file.
        """
        return self.get_dataframe().columns.values.tolist()

    def get_tuning_knob_headers(self) -> List[str]:
        """
        Returns a list of headers for the tuning knobs.
        """
        return [f"k{i}" for i in range(1, 9, 1)] \
            + [f"s{i}" for i in range(1, 5, 1)]

    def get_non_metric_headers(self) -> List[str]:
        """
        Returns a list of headers that include the knobs, 'workload id', and
        'latency'.
        """
        return self.get_tuning_knob_headers() + ['workload id', 'latency']

    def get_metric_headers(self) -> List[str]:
        """
        Returns a list of headers that exclude 'workdload id', the tuning
        knobs, as well as 'latency'.
        """
        non_metric_headers = self.get_non_metric_headers()
        all_headers = self.get_headers()
        return [h for h in all_headers if h not in non_metric_headers]

    def get_metrics(self) -> np.ndarray:
        """
        Returns the metrics in a numpy array. Note that this method discards
        the columns 'workload id', 'k1' to 'k8', 's1' to 's4', as well as the
        'latency' column.

        Returns:
            An array of dimension [NUM_WORKLOADS * NUM_METRICS].
        """
        dataframe = self.get_dataframe().copy(deep=True)
        dataframe.drop(self.get_non_metric_headers(), axis=1, inplace=True)
        return dataframe.values.astype(float)

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
