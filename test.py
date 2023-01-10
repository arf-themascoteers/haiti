import torch
from torch.utils.data import DataLoader


def test(dataset, model):
    BATCH_SIZE = 10000
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    print(f"Test started ...")
    total = 0
    correct = 0
    with torch.no_grad():
        for data, y_true in dataloader:
            y_pred = model(data)
            pred = torch.argmax(y_pred, dim=1, keepdim=True)
            correct = correct + pred.eq(y_true.data.view_as(pred)).sum()
            total = total + len(y_true)

    return correct/total*100

