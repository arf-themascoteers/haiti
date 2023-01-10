import pandas as pd
import colour
import numpy as np
import math


def get_x_y(h):
    angle = 2 * math.pi * h
    x = math.cos(angle)
    y = math.sin(angle)
    return x,y


df = pd.read_csv("hsv.csv")
npdf = df.to_numpy()
mydata = np.zeros((npdf.shape[0],npdf.shape[1]+1))
mydata[:,2:] = npdf[:,1:]
for i in range(len(npdf)):
    x,y = get_x_y(npdf[i,0])
    mydata[i,0] = x
    mydata[i,1] = y
df = pd.DataFrame(mydata, columns = ['HX', 'HY', 'S','V','Class'])
df.to_csv("mod_hsv.csv", index=False)
print("done")



