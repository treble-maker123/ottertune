from pdb import set_trace

from modules.logger import build_logger
from modules.workload_gpr import WorkloadGPR
from modules.dataset import Dataset, DATASET_PATHS


def main():
    gprs = WorkloadGPR()
    dataset = Dataset(file_path=DATASET_PATHS['offline_workload'])
    pruned_metrics = Dataset.load_pruned_metrics()
    dataset = dataset.prune_columns(pruned_metrics
                                    + ['workload id']
                                    + dataset.get_tuning_knob_headers())
    df = dataset.get_dataframe()

    workload_ids = dataset.get_workload_ids()

    wlid = workload_ids[1]
    pm = pruned_metrics[1]

    gpr = gprs.get_model(wlid, pm)

    workload = df[df['workload id'] == wlid]
    X = workload[dataset.get_tuning_knob_headers()].values
    y = workload[pm].values

    y_hat = gpr.predict(X)
    LOG.info(f"MSE: {((y - y_hat) ** 2).sum()}")


if __name__ == '__main__':
    LOG = build_logger()
    main()
