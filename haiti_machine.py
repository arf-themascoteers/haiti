from torch import nn


class HaitiMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3,15),
            nn.LeakyReLU(),
            nn.Linear(15,5)
        )

    def forward(self, x):
        x = self.fc(x)
        return x