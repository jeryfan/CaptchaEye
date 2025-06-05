from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 1, 28, 28] → [B, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # [B, 32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),# [B, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # [B, 64, 7, 7]

            nn.Flatten(),                               # [B, 64*7*7]
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)