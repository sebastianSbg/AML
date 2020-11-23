# The skorch library triggers a future warning in the sklearn library.
# These warnings spam the output so they are disabled here
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)

import os
import logging
import argparse

import ray
import torch
import numpy as np

from tqdm import tqdm
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from skorch.callbacks import EarlyStopping

from preprocessing import Preprocessor_v1, Scaler
from utils import get_data, dump_pipe
from models import TorchModel
from models.simpleCNN import simpleCNN

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument("--cpu", action='count', help="Force the training to be done on the CPU")
    parser.add_argument("-d", "--debug", action='count',
                        help="Turn on debug mode. This will make sure that " +
                             "potential CUDA errors are synchronized with the python stack trace.")
    parser.add_argument("-i", "--iterations", type=int, default=50,
                        help="Number of iterations to look for optimimal parameters")
    parser.add_argument("-s", "--small", action='count',
                        help="Run with a reduced dataset size. To generate a smaller version of the original "
                             "training set, call 'utils.py' first. Note this is only meant to be used for "
                             "testing purposes")
    parser.add_argument("-v", "--verbose", action='count')

    args, unparsed = parser.parse_known_args()
    if unparsed: logger.warning(f"The following argument were not understood: {', '.join(unparsed)}")
    return args


def evaluate(config):
    x, y = config['data']
    preprocessor_params = config['preprocessor_params']
    model_params = config['model_params']
    training_params = config['training_params']
    training_params['callbacks'] = list(training_params['callbacks'])
    training_params['criterion__weight'] = torch.tensor(list(training_params['criterion__weight']), dtype=torch.float)

    # import pdb; pdb.set_trace()
    pipe = Pipeline([
        ('preprocessor', Preprocessor_v1(spacing=75)),
        ('scaler', Scaler()),
        ('model', TorchModel(model_params, training_params, model_cls=simpleCNN))])

    cv = StratifiedShuffleSplit(n_splits=config.get('cv', 5), test_size=0.2)
    scores = []
    for train_idx, val_idx in tqdm(cv.split(x, y), total=config.get('cv', 5)):
        x_t, x_v = x[train_idx], x[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]
        pipe_t = clone(pipe)
        pipe_t.fit(x_t, y_t)
        y_pred = pipe_t.predict(x_v)
        scores.append(f1_score(y_v, y_pred, average='micro'))

    print(f"Average F1 score {np.mean(scores)} (std {np.std(scores):.3f})")
    ray.tune.report(mean_loss=np.mean(scores))

    # res = cross_validate(pipe, x, y, scoring='f1_micro', cv=config.get('cv', 5), verbose=0)
    #
    # print(f"Average F1 score {np.mean(res['test_score'])} (std {np.std(res['test_score']):.3f})")
    # ray.tune.report(mean_loss=np.mean(res['test_score']))


def tune(x, y, tune_kwargs, current_best_guess):
    ray.init()
    # Search algorithm
    algo = HyperOptSearch()  # points_to_evaluate=current_best_guess)
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    # Scheduler
    scheduler = AsyncHyperBandScheduler()
    tune_kwargs['config']['data'] = x, y
    analysis = ray.tune.run(evaluate,
                            search_alg=algo,
                            scheduler=scheduler,
                            metric="mean_loss",
                            mode="min",
                            **tune_kwargs)

    return analysis


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose mode set")

    if args.debug:
        logger.info("Running in debug mode!")
        # Synchronize CUDA with python
        os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)

    x_train, _, y_train, _, _ = get_data(reduced_size=bool(args.small))

    print(x_train.shape)
    preprocessor_params = {}
    model_params = {}
    training_params = {'max_epochs': 200, 'lr': ray.tune.choice([1e-2, 1e-3, 1e-4, 1e-5]),
                       'optimizer': torch.optim.Adam, 'criterion': torch.nn.NLLLoss,
                       'criterion__weight': [1,
                                             ray.tune.uniform(4, 8.5),
                                             ray.tune.uniform(1, 3.5),
                                             ray.tune.uniform(14, 21)],
                       # 'criterion__weight': [ray.tune.uniform(1, 25)] * 4,
                       'iterator_train__shuffle': True, 'verbose': args.verbose,
                       'callbacks': [EarlyStopping(patience=10)]}

    tune_kwargs = {'num_samples': args.iterations,
                   'resources_per_trial': {'cpu': 23, 'gpu': 1},
                   'config':
                       {'preprocessor_params': preprocessor_params,
                        'model_params': model_params,
                        'training_params': training_params}
                   }

    current_best_guess = [{'criterion__weight': (1., 6.83972912, 2.05563094, 17.82352941)}]

    analysis = tune(x_train, y_train, tune_kwargs, current_best_guess)

    print("Best config: ", analysis.get_best_config(
        metric="mean_loss", mode="min"))
    print("Tuning finished successfully")
