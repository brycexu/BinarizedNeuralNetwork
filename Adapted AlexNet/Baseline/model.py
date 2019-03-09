import torch.nn as nn
import torch.nn.functional as F

class AAlexNet(nn.Module):
    def __init__(self):
        super(AAlexNet, self).__init__()
        self.convolutions = nn.Sequential(

            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            F.relu(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            F.relu(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.3),
            nn.BatchNorm2d(256),
            F.relu(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            F.relu(),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.3),
            nn.BatchNorm2d(512),
            F.relu(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            F.relu(),
        )
        self.linears = nn.Sequential(

            nn.Linear(512 * 4 * 4, 1024),
            nn.Dropout(0.3),
            nn.BatchNorm1d(1024),
            F.relu(),

            nn.Linear(1024, 1024),
            nn.Dropout(0.3),
            nn.BatchNorm1d(1024),
            F.relu(),

            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(-1, 512 * 4 * 4)
        out = self.linears(x)
        return out