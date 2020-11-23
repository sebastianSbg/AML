import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class simpleCNN(torch.nn.Module):
    def __init__(
            self,
            input_shape: Tuple,
            layer1_features: int = 16,
            layer2_features: int = 32,
            layer2_kernel_size: int = 5,
            layer2_max_pool_size: int = 7,
            layer2_max_pool_stride: int = 6,
            layer3_kernel_size: int = 5,
            **kwargs

    ):
        super(simpleCNN, self).__init__()
        self.input_shape = input_shape
        self.conv1 = torch.nn.Conv1d(self.input_shape[1], layer1_features, 1, 1)
        self.conv2 = torch.nn.Conv1d(layer1_features, layer2_features, kernel_size=layer2_kernel_size)
        self.maxpool = torch.nn.MaxPool1d(layer2_max_pool_size, stride=layer2_max_pool_stride)
        self.conv3 = torch.nn.Conv1d(layer2_features, 4, kernel_size=layer3_kernel_size)
        self.activation = torch.nn.ReLU()
        # Ahh! LogSoftmax or Softmax that is the question! See:
        # https://github.com/skorch-dev/skorch/issues/519
        self.softmax = torch.nn.Softmax(dim=1)
        self.batchnorm = torch.nn.BatchNorm1d(4)

    def forward(self, x, **kwargs):
        # import pdb; pdb.set_trace()
        X = self.activation(self.conv1(x))
        X = self.maxpool(self.activation(self.conv2(X)))
        X = self.activation(self.conv3(X))
        X = self.batchnorm(torch.mean(X.squeeze(1), dim=-1))
        output = self.softmax(X)
        # print(torch.exp(output))
        return output