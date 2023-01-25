from torch import nn


class HaitiMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3,15)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(15,5)

    def forward(self, x):
        self.x1 = self.fc1(x)
        self.x2 = x = self.relu(self.x1)
        self.x3 = self.fc2(self.x2)
        return self.x3