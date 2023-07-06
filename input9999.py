import numpy as np
import pandas as pd
import torch
from numpy import array
from torch.nn.utils.rnn import pad_sequence

dtype = torch.float
PAD_VALUE = -999


def create_window_seqs(
    X: np.array,
    y: np.array,
    min_sequence_length: int,
):
    """
    Creates windows of fixed size with appended zeros
    @param X: features
    @param y: targets, in synchrony with features (i.e. x[t] and y[t] correspond to the same time)
    """
    # convert to small sequences for training, starting with length 10
    seqs = []
    targets = []
    mask_ys = []

    # starts at sequence_length and goes until the end
    # for idx in range(min_sequence_length, X.shape[0]+1, 7): # last in range is step
    for idx in range(min_sequence_length, X.shape[0] + 1, 1):
        # Sequences
        seqs.append(torch.from_numpy(X[:idx, :]))
        # Targets
        y_ = y[:idx]
        mask_y = torch.ones(len(y_))
        targets.append(torch.from_numpy(y_))
        mask_ys.append(mask_y)
    seqs = pad_sequence(seqs, batch_first=True, padding_value=0).type(dtype)
    ys = pad_sequence(targets, batch_first=True, padding_value=PAD_VALUE).type(dtype)
    mask_ys = pad_sequence(mask_ys, batch_first=True, padding_value=0).type(dtype)

    return seqs, ys, mask_ys


class SeqData(torch.utils.data.Dataset):
    def __init__(self, region=None, meta=None, X=None, y=None, mask_y=None):
        self.region = region
        self.meta = meta
        self.X = X
        self.y = y
        self.mask_y = mask_y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.region[idx], self.meta[idx], self.X[idx, :, :], self.y[idx])
