from torchvision import datasets, transforms
import torch.utils.data as data_utils
import torch
from base import BaseDataLoader
import pandas as pd
import numpy as np


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)

        print("TYPE DATASET:", type(self.dataset))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MnistKaggleDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        df = pd.read_csv(data_dir)
        self.is_train = training

        if training:
            X = df.iloc[:, 1:].values.reshape(-1, 1, 28, 28) / 255
            Y = np.double(df.iloc[:, 0].values)
            self.dataset = data_utils.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).long())
        else:
            X = df.iloc[:, :].values.reshape(-1, 1, 28, 28) / 255
            Y = np.zeros(X.shape[0])
            self.dataset = data_utils.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).long())

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
