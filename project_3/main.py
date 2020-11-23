import os
import logging
import argparse
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm
from skorch.callbacks import EarlyStopping
from sklearn.base import clone
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from preprocessing import Preprocessor_v1, Scaler

from utils import get_data, dump_pipe, print_cm
from models import TorchModel
from models.simpleCNN import simpleCNN

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument("--cpu", action='count', help="Force the training to be done on the CPU")
    parser.add_argument("-d", "--debug", action='count',
                        help="Turn on debug mode. This will make sure that " +
                             "potential CUDA errors are synchronized with the python stack trace.")
    parser.add_argument("-s", "--small", action='count',
                        help="Run with a reduced dataset size. To generate a smaller version of the original "
                             "training set, call 'utils.py' first. Note this is only meant to be used for "
                             "testing purposes")
    parser.add_argument("-t", "--test_only", action='count', help="Does not run the evaluation procedure")
    parser.add_argument("-v", "--verbose", action='count')

    args, unparsed = parser.parse_known_args()
    if unparsed: logger.warning(f"The following argument were not understood: {', '.join(unparsed)}")
    return args


def test(pipe, X, y, n_splits=5):
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2)
    scores = []
    cm_list = []
    for train_idx, val_idx in tqdm(cv.split(X, y), total=n_splits):
        x_t, x_v = X[train_idx], X[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]
        pipe_t = clone(pipe)
        pipe_t.fit(x_t, y_t)
        y_pred = pipe_t.predict(x_v)
        scores.append(f1_score(y_v, y_pred, average='micro'))
        cm_list.append(confusion_matrix(y_v, y_pred))

    print(f"Average F1 score {np.mean(scores)} (std {np.std(scores):.3f})")
    print("Average Confusion Matrix")
    # import pdb; pdb.set_trace()
    # cm = np.mean(cm_list, axis=0)
    print_cm(np.mean(cm_list, axis=0), labels=list(map(str, range(4))))
    return scores

    # res = cross_validate(pipe, X, y, scoring='f1_micro', cv=5, verbose=2)
    # print(f"Average F1 score {np.mean(res['test_score'])} (std {np.std(res['test_score']):.3f})")
    # print(res)


def eval(pipe, X, y, X_test, y_test=None, save: bool = True):
    pipe.fit(X, y)
    y_pred = pipe.predict(X_test)

    # Remember the test score for later comparisons
    if y_test is not None:
        score = f1_score(y_test, y_pred, average='micro')
        file_path, _ = dump_pipe(pipe)
        with open("eval_hist.txt", 'a') as f:
            f.write(f"Score: {score:.4f} || File: {file_path}\n")

    if save:
        df_pred = pd.DataFrame(y_pred, columns=["y"])
        df_pred["id"] = df_pred.index
        df_pred[["id", "y"]].to_csv("y_pred.csv", index=False)

    return y_pred


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


    x_train, x_test, y_train, y_test, x_eval = get_data(reduced_size=bool(args.small))

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("The following GPUs are available: "
                    f"{', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])}")

    print(x_train.shape)
    model_params = {}
    training_params = {'max_epochs': 200, 'lr': .001,
                       'optimizer': torch.optim.Adam, 'criterion': torch.nn.NLLLoss,
                       'criterion__weight': torch.tensor([1., 8, 4, 17], dtype=torch.float),
                       'iterator_train__shuffle': True, 'verbose': args.verbose,
                       'callbacks': [EarlyStopping(patience=15)]}

    pipe = Pipeline([
        ('preprocessor', Preprocessor_v1(spacing=75)),
        ('scaler', Scaler()),
        ('model', TorchModel(model_params, training_params, model_cls=simpleCNN))])

    test(pipe, x_train, y_train)
    if not args.test_only:
        eval(pipe, x_train, y_train, x_test, y_test)
