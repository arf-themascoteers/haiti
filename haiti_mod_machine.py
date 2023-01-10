from torch import nn


class HaitiModMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4,15),
            nn.LeakyReLU(),
            nn.Linear(15,5)
        )

    def forward(self, x):
        x = self.fc(x)
        return x