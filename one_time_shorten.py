import pandas as pd
import colorsys
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import colour
import numpy as np

df = pd.read_csv("haiti.csv")
cc = pd.Categorical(df.Class)
klass = np.array(cc.codes).reshape(-1,1)
num_class = klass.max() + 1
df = df.drop("Class", axis='columns')
npdf = df.to_numpy()
npdf = npdf / 255
npdf = np.concatenate((npdf, klass), axis=1)

member_per_class = 10000000

for i in range(num_class):
    lowest =  len(npdf[np.where(npdf[:,3] == i)])
    if lowest < member_per_class:
        member_per_class = lowest

filtered_data = np.zeros((member_per_class * num_class, npdf.shape[1]))

for i in range(num_class):
    data = npdf[np.where(npdf[:,3] == i)]
    np.random.shuffle(data)
    start = i*member_per_class
    end = start + member_per_class
    filtered_data[start:end] = data[0:member_per_class]

df = pd.DataFrame(filtered_data, columns = ['R','G','B','Class'])
df.to_csv("rgb.csv", index=False)
print("done")