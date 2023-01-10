import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset


class HaitiDataset(Dataset):
    def __init__(self, is_train=True, ctype="rgb"):
        filename = "rgb.csv"
        if ctype == "hsv":
            filename = "hsv.csv"
        elif ctype == "mod_hsv":
            filename = "mod_hsv.csv"
        sm = pd.read_csv(filename).to_numpy()
        X = sm[:, :-1]
        y = sm[:, -1]

        self.X, X_test, self.y, y_test = train_test_split(X, y, random_state=5)

        if not is_train:
            self.X = X_test
            self.y = y_test

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]