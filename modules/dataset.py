"""
This module supplies classes and methods for interacting with the dataset.
"""
from pdb import set_trace
from typing import List, Union

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

    def __init__(self, file_path: str = None, dataframe: pd.DataFrame = None):
        if file_path is not None:
            self._dataframe = pd.read_csv(file_path)
        elif dataframe is not None:
            self._dataframe = dataframe
        else:
            raise ValueError("Either file_path or dataframe must be passed "
                             "to Dataset constructor.")

    def __add__(self, other: 'Dataset') -> 'Dataset':
        """
        Overloads the '+' operator so that `dataset1 + dataset2` gives a
        dataset that is the union of both datasets.
        """
        this_headers = self.get_headers()
        other_headers = other.get_headers()

        # make sure the headers match
        for header in other_headers:
            assert header in this_headers

        dataframes = [self.get_dataframe(), other.get_dataframe()]
        return Dataset(dataframe=pd.concat(dataframes, ignore_index=True))

    def __len__(self) -> int:
        """
        Overloads len() to return the size of the dataframe.
        """
        return len(self.get_dataframe())

    def __getitem__(self, index: int) -> pd.Series:
        """
        Overloads dataset[0] to return the corresponding series in dataframe.
        """
        return self.get_dataframe().iloc[index]

    def get_items(self, indices: List[int]) -> pd.DataFrame:
        """
        Return a new DataFrame with the specified rows.
        """
        return self.get_dataframe().iloc[indices].reset_index(drop=True)

    def get_column_values(self, col_header: str) -> List[Union[str, int, float]]:
        """
        Returns all of the values of the given column as a list of strings.
        """
        return self.get_dataframe()[col_header].tolist()

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas dataframe that contains the specified data file.
        """
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
        return [f"k{i}" for i in range(1, 9, 1)]
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
