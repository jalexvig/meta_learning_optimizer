import os

import torch
from torch import nn
from torch.optim import SGD


class MetaOptimizer(nn.Module):

    def forward(self, *input):
        pass

    def __init__(self):

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=3,
            batch_first=True
        )

        self.fc1 = torch.nn.Linear(3, 1)

        self.fc_gradients = torch.nn.Linear(1, 1)

        self.log_stds = torch.nn.Parameter(torch.tensor(-20.0))

        self.params = list(self.parameters())
        self.optimizer = SGD(self.params, lr=0.1)


def get_lstm_kernel_bias_torch(fpath_checkpoint):

    d = torch.load(fpath_checkpoint)

    kernel_ih = d['rnn.weight_ih_l0']
    kernel_hh = d['rnn.weight_hh_l0']

    bias_ih = d['rnn.bias_ih_l0']
    bias_hh = d['rnn.bias_hh_l0']

    kernel = torch.cat([kernel_ih, kernel_hh], dim=1).numpy().T
    bias = (bias_ih + bias_hh).numpy()

    return kernel, bias


if __name__ == '__main__':

    fpath = '/Users/alex/ml/lstm_learn_optimizer/saved/multivargauss_binary_adam_sgd_1/checkpoint'
    kernel, bias = get_lstm_kernel_bias_torch(fpath)
    print(kernel.shape)
    print(bias.shape)
