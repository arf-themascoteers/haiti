import torch
from dataset_manager import get_ds
from torch.utils.data import DataLoader
import torch.nn.functional as F

model = torch.load(f"hsv.h5")


for train_ds, test_ds in get_ds(ctype="hsv"):
    x = train_ds.X
    x.requires_grad = True
    y_hat = model(train_ds.X)
    loss = F.cross_entropy(y_hat, train_ds.y)
    loss.backward()
    grads = torch.abs(x.grad).sum(axis=0)
    grads = grads/torch.sum(grads) * 100
    print(grads)
    break
