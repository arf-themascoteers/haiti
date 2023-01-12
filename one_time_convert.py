import pandas as pd
import colour


df = pd.read_csv("rgb.csv")
npdf = df.to_numpy()
for i in range(len(npdf)):
    npdf[i,0:3] = colour.RGB_to_HSV(npdf[i,0:3])
df = pd.DataFrame(npdf, columns = ['H','S','V','Class'])
df.to_csv("hsv.csv", index=False)
print("done")