"""
Prepare data
"""

import numpy as np
import csv

# Import test data
test = np.loadtxt("test.csv", delimiter=",", skiprows=1)
ID_test = test[:, 0].astype(int)
x_test = test[:, 1:830]

# Import training data
traindata = np.loadtxt("train.csv", delimiter=",", skiprows=1)
ID_train = traindata[:, 0].astype(int)
y_train = traindata[:, 1]
x_train = traindata[:, 2:830]

# Split training data into training and validation sets
seed = 28
p = 0.2 # percentage of traindata used for validation
entries = int(p*len(x_train))
num_row  = y_train.shape[0]
np.random.seed(seed)
indices = np.random.permutation(num_row)
x_val = x_train[indices[:entries]]
y_val = y_train[indices[:entries]]
x_tr = x_train[indices[entries:]]
y_tr = y_train[indices[entries:]]