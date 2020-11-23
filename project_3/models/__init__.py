import logging
import torch

from skorch import NeuralNetClassifier
from sklearn.base import BaseEstimator, TransformerMixin

import models.simpleCNN

logger = logging.getLogger(__name__)


class TorchModel(BaseEstimator, TransformerMixin):
    def __init__(self, model_params, training_params, model_cls, force_cpu: bool = False):
        self.device = torch.device('cpu')
        self.model_params = model_params
        self.training_params = training_params
        self.model_cls = model_cls
        self.net = None

        # if torch.cuda.is_available():
        #     self.device = torch.device('cpu') if force_cpu else torch.device('cuda')
        #     logger.info("The following GPUs are available: "
        #                 f"{', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])}")

    def fit(self, X, y=None):
        self.model_params['input_shape'] = X.shape
        model = self.model_cls(**self.model_params)
        model.to(self.device)
        self.training_params['module'] = model
        self.net = NeuralNetClassifier(**self.training_params)
        # import pdb; pdb.set_trace()
        self.net.fit(torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.long))
        return self

    def predict(self, X):
        return self.net.predict(torch.tensor(X, dtype=torch.float))




