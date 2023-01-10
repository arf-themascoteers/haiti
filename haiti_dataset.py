import torch
from torch.utils.data import Dataset


class HaitiDataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor(data[:, :-1], dtype=torch.float32)
        self.y = torch.tensor(data[:, -1], dtype=torch.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]