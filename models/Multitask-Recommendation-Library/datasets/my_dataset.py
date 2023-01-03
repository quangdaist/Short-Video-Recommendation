import numpy as np
import pandas as pd
import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        data = pd.read_csv(dataset_path).to_numpy()[:, 1:]

        # drop nan values
        data = data.dropna()

        self.categorical_data = data[:, 4:54:5].astype(np.int)
        self.numerical_data = data[:, [i for i in range(2, data.shape[1]-3) if not ((4 <= i <= 49) and i%5 == 4)]].astype(np.float32)
        self.labels = data[:, -3:].astype(np.float32)
        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]