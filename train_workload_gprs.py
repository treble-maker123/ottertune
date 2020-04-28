"""
This script will run the training of workload mapping GPRs.
"""
from time import time

from modules.dataset import Dataset, DATASET_PATHS
from modules.workload_gpr import WorkloadGPR
from modules.logger import build_logger


def main():
    # only training GPRs for offline loads
    dataset = Dataset(file_path=DATASET_PATHS['offline_workload'])
    # load the pruned metric headers
    pruned_metrics = Dataset.load_pruned_metrics()
    # prune the dataset
    dataset = dataset.prune_columns(pruned_metrics
                                    + ['workload id']
                                    + dataset.get_tuning_knob_headers())

    # build the GPRs
    start = time()
    gprs = WorkloadGPR(dataset=dataset)
    LOG.info(f"Finished building GPRs in {round(time() - start)} seconds.")

    # pickle 'em
    LOG.info("Pickling GPRs...")
    start = time()
    gprs.pickle_models()
    LOG.info(f"Finished pickling models in {round(time() - start)} seconds.")


if __name__ == '__main__':
    LOG = build_logger()

    main()
