from torch import nn


class SynthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 3)
        )

    def forward(self, inp):
        return self.net(inp)
