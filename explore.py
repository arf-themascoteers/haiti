import pandas as pd

df = pd.read_csv("hsv.csv")
npdf = df.to_numpy()
print(npdf[:,0].min())
print(npdf[:,1].min())
print(npdf[:,2].min())

print(npdf[:,0].max())
print(npdf[:,1].max())
print(npdf[:,2].max())
print("done")