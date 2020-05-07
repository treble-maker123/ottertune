'''
This class provides an interface for interacting with the workload Gaussian
Process Regressors (GPR)
'''
from typing import Dict
from pdb import set_trace
import os
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from tqdm import tqdm

from modules.dataset import Dataset
from modules.logger import build_logger

LOG = build_logger()


class WorkloadGPR:
    """
    Trains a set of GPRs for workload matching
    """

    def __init__(self, dataset: Dataset = None, scaler=None):
        # { 'workload-id_metric-ident': GaussianProcessRegressor }
        self.models: Dict[str, GaussianProcessRegressor] = {}

        if dataset is not None:
            # train new GPRs
            LOG.info("Building GPRs from dataset.")
            self._build_models_from_dataset(dataset, scaler)
        else:
            # load the models from the ./models directory
            models_fnames = os.listdir('./models')
            # filter out the non-workload GPRs, marked all worload GPRs with wl
            models_fnames = [m for m in models_fnames if m[:2] == 'wl']

            for n in models_fnames:
                self.models[n] = None

    def get_model(self, workload_id: str, metric_name: str) \
            -> GaussianProcessRegressor:
        """
        Returns a GPR model for the given workload id and metric.
        """
        # converts the underscores in the metric name to dashes because that's
        # how it's saved.
        metric_name = metric_name.replace('_', '-')
        model_fname = f"wl_{workload_id}_{metric_name}.pickle"

        # lazy loading, load only if needed, since each file is close to 1MB
        if self.models[model_fname] is None:
            with open(f"./models/{model_fname}", 'rb') as f:
                self.models[model_fname] = pickle.load(f)

        return self.models[model_fname]

    def pickle_models(self):
        """
        This will pickle all of the GPR models in self.model into individual
        files and save them under ./models.
        """
        for name, model in self.models.items():
            with open(f"./models/{name}", 'wb') as f:
                pickle.dump(model, f)

    def _build_models_from_dataset(self, dataset: Dataset, scaler=None):
        """
        Build all of the GPR models from scratch
        """
        df = dataset.get_dataframe()
        metrics = dataset.get_metric_headers()
        workload_ids = dataset.get_workload_ids()
        knob_headers = dataset.get_tuning_knob_headers()
        total_gprs = len(workload_ids) * len(metrics)

        with tqdm(total=total_gprs) as pbar:
            for w in workload_ids:
                workloads = df[df['workload id'] == w]
                for m in metrics:
                    X = workloads[knob_headers].values

                    if scaler is not None:
                        X = scaler.transform(X)

                    y = workloads[m].values
                    m_file_name = m.replace('_', '-')

                    # krasserm.github.io/2018/03/19/gaussian-processes#effect-of-kernel-parameters-and-noise-parameter
                    restarts = 5
                    # sigma_f, l
                    kernel = ConstantKernel(10.0) * RBF(y.std())
                    # sigma_y
                    alpha = 0.1
                    model = GaussianProcessRegressor(kernel=kernel,
                                                     n_restarts_optimizer=restarts,
                                                     alpha=alpha,
                                                     normalize_y=True)
                    model.fit(X, y)
                    self.models[f"wl_{w}_{m_file_name}.pickle"] = model
                    pbar.update(1)
