from haiti_dataset import HaitiDataset
from sklearn.model_selection import KFold
import pandas as pd


def get_ds(ctype="rgb"):
        filename = "rgb.csv"
        if ctype == "hsv":
            filename = "hsv.csv"
        elif ctype == "mod_hsv":
            filename = "mod_hsv.csv"
        full_data = pd.read_csv(filename).to_numpy()
        kf = KFold(n_splits=10)
        for i, (train_index, test_index) in enumerate(kf.split(full_data)):
            train_data = full_data[train_index]
            test_data = full_data[test_index]
            yield HaitiDataset(train_data), HaitiDataset(test_data)