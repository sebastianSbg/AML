import os
import pickle
import logging
import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PIPE_DUMP_DIR = "dumped_pipes"

logger = logging.getLogger(__name__)

def get_data(test_size: float = .2, reduced_size: bool = False):
    if reduced_size:
        logger.warning("Using a reduced size training set!")

    try:
        df_x = pd.read_csv("X_train_small.csv" if reduced_size else "X_train.csv", dtype=np.float)
        df_y = pd.read_csv("y_train_small.csv" if reduced_size else "y_train.csv", dtype=np.float)
        df_eval = pd.read_csv("X_test.csv", dtype=np.float)

    except ValueError as ve:
        ValueError(f"The following error occurred while reading in the data: '{ve}' "
                   f"You may encounter this issue because the data has a malformed new line character in the end")

    df_x = df_x.set_index("id")
    df_y = df_y.set_index("id")
    df_eval = df_eval.set_index("id")

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df_x[:].describe())

    x = df_x.to_numpy()
    y = df_y.to_numpy().squeeze()
    eval = df_eval.to_numpy()

    labels, counts = np.unique(y, return_counts=True)
    logger.info(f"Labels: {', '.join(map(str, labels))}, counts: {', '.join(map(str, counts))}")

    weights_suggestion = np.sum(counts)/(len(labels)*counts)
    weights_suggestion /= np.amin(weights_suggestion)
    logger.info(f"A first guess for class weights in the loss function could be: {weights_suggestion}")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)

    # np.random.seed(42)
    # perm = np.random.permutation(len(x_train))
    # x_train = x_train[perm]
    # y_train = y_train[perm]
    #
    # np.random.seed(42)
    # perm = np.random.permutation(len(x_test))
    # x_test = x_test[perm]
    # y_test = y_test[perm]

    logger.info("Loaded dataset")
    return x_train, x_test, y_train, y_test, eval

def create_smaller_dataset(size:int = 32):
    df_x = pd.read_csv("X_train.csv")
    df_y = pd.read_csv("y_train.csv")

    df_x = df_x.set_index("id")
    df_y = df_y.set_index("id")

    selection_idx = np.random.choice(np.arange(len(df_x)), size)

    df_x.iloc[selection_idx].to_csv("X_train_small.csv")
    df_y.iloc[selection_idx].to_csv("y_train_small.csv")

def dump_pipe(pipe):
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    file_path = os.path.join(PIPE_DUMP_DIR, f"{timestamp}.pkl")

    logger.info(f"Dumping pipe to {file_path}")
    os.makedirs(PIPE_DUMP_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(pipe,f)

    return file_path, timestamp

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

if __name__ == "__main__":
    create_smaller_dataset()
