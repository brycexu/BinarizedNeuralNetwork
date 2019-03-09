import torch.nn as nn
from utils import BinarizeConv2d,BinarizeLinear,BinaryHardtanh

class BinarizedVgg16Net(nn.Module):
    def __init__(self):
        super(BinarizedVgg16Net, self).__init__()
        self.convolutions = nn.Sequential(

            BinarizeConv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            BinaryHardtanh(),

            BinarizeConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            BinaryHardtanh(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            BinarizeConv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            BinaryHardtanh(),

            BinarizeConv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            BinaryHardtanh(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            BinarizeConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            BinaryHardtanh(),

            BinarizeConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            BinaryHardtanh(),

            BinarizeConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            BinaryHardtanh(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            BinarizeConv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            BinaryHardtanh(),

            BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            BinaryHardtanh(),

            BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            BinaryHardtanh(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            BinaryHardtanh(),

            BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            BinaryHardtanh(),

            BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            BinaryHardtanh(),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linears = nn.Sequential(

            BinarizeLinear(512, 10),
            nn.BatchNorm1d(10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(-1, 512)
        out = self.linears(x)
        return out