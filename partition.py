"""
This script will take the workloads and partition it into n sets.
"""
from argparse import Namespace, ArgumentParser
from pdb import set_trace

import pandas as pd
import numpy as np

from modules.dataset import Dataset, DATASET_PATHS, \
    DATA_PATH_PREFIX as DATA_PATH
from modules.logger import build_logger


def build_config() -> Namespace:
    """
    Builds a Namespace object containing configurations from argparser.
    """
    parser = ArgumentParser()

    parser.add_argument('--num-sets', type=int, default=5,
                        help='Number of sets to partition the data into.')

    known_args, _ = parser.parse_known_args()

    return known_args


def main():
    """
    Main method for the script.
    """
    offline_workload = Dataset(file_path=DATASET_PATHS['offline_workload'])
    online_workload_b = Dataset(file_path=DATASET_PATHS['online_workload_B'])
    online_workload_c = Dataset(file_path=DATASET_PATHS['online_workload_C'])
    online_workload = online_workload_b + online_workload_c

    offline_ids = np.arange(0, len(offline_workload))
    online_ids = np.arange(0, len(online_workload))

    for i in range(0, CONFIG.num_sets):
        # randomly sample ids from offline workloads to be part of partition i
        offline_part_ids = np.random.choice(offline_ids,
                                            size=len(offline_ids) // 5,
                                            replace=False)
        offline_part_df = offline_workload.get_items(offline_part_ids)

        # do the same for the online workloads
        online_part_ids = np.random.choice(online_ids,
                                           size=len(online_ids) // 5,
                                           replace=False)
        online_part_df = online_workload.get_items(online_part_ids)

        # combine the online and offline workloads and save it
        combined_df = pd.concat([offline_part_df, online_part_df],
                                ignore_index=True)
        combined_df.to_csv(f"{DATA_PATH}/workload_partition_{i+1}.csv",
                           index=False)

        # remove the ids from the pool
        offline_ids = np.array(
            [j for j in offline_ids if j not in offline_part_ids])
        online_ids = np.array(
            [j for j in online_ids if j not in online_part_ids])

        LOG.info('Finished writing partition %s.', i)


if __name__ == '__main__':
    LOG = build_logger()
    CONFIG = build_config()

    main()
