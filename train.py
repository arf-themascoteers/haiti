from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


def train(dataset, model):
    NUM_EPOCHS = 1000
    BATCH_SIZE = 10000

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)

    for epoch in range(0, NUM_EPOCHS):
        for data, y_true in dataloader:
            optimizer.zero_grad()
            y_pred = model(data)
            loss = F.cross_entropy(y_pred, y_true)
            loss.backward()
            optimizer.step()
            #print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')

    return model
