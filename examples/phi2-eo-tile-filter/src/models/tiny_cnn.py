import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 2, base: int = 16):
        super().__init__()
        c1, c2, c3 = base, base*2, base*4
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)
