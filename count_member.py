import pandas as pd
import colorsys
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import colour
import numpy as np

df = pd.read_csv("rgb.csv")
npdf = df.to_numpy()

for i in range(5):
    members =  len(npdf[np.where(npdf[:,3] == i)])
    print(members)
