"""
This script is used for cross-validation of the workload metric-prediction
GPRs.
"""
from pdb import set_trace

import pandas as pd
import numpy as np

from modules.logger import build_logger
from modules.workload_gpr import WorkloadGPR
from modules.dataset import Dataset, DATASET_PATHS
from modules.util import clear_wl_models


def main():
    LOG.debug('Clearing out all of the workload models.')
    clear_wl_models()

    dataset = Dataset(file_path=DATASET_PATHS['offline_workload'])
    pruned_metrics = Dataset.load_pruned_metrics()
    pruned_dataset = dataset.prune_columns(pruned_metrics
                                           + ['workload id']
                                           + dataset.get_tuning_knob_headers())
    df = pruned_dataset.get_dataframe()

    # pick the ith data to use as validation
    i = [1, 3, 5]
    workload_ids = pruned_dataset.get_workload_ids()
    validation_df = pd.concat([df[df['workload id'] == wid].iloc[i]
                               for wid in workload_ids])
    validation_idx = validation_df.index
    valid_dataset = Dataset(dataframe=validation_df)

    diff_idx = df.index.difference(validation_df.index)

    train_df = df.iloc[diff_idx]
    train_dataset = Dataset(dataframe=train_df)

    LOG.info("Training workload GPRs...")
    gprs = WorkloadGPR(dataset=train_dataset)

    LOG.info("Validating GPRs...")
    train = {}
    result = {}
    for pm in pruned_metrics:
        for wid in workload_ids:
            name = f"{pm}|{wid}"
            model = gprs.get_model(wid, pm)

            # train
            X = train_df[dataset.get_tuning_knob_headers()].values
            y = train_df[pm].values
            y_hat = model.predict(X)
            mape = np.mean(np.abs((y - y_hat) / y)) * 100
            train[name] = mape

            # validation
            X = validation_df[dataset.get_tuning_knob_headers()].values
            y = validation_df[pm].values
            y_hat = model.predict(X)
            mape = np.mean(np.abs((y - y_hat) / y)) * 100
            result[name] = mape
            #  LOG.info('%s: %s', name, mape)

    LOG.info('Training average MAPE: %s',
             np.array(list(train.values())).mean())
    LOG.info('Validation average MAPE: %s',
             np.array(list(result.values())).mean())


if __name__ == '__main__':
    LOG = build_logger()
    main()
