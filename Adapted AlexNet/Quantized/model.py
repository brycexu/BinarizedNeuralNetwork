import torch.nn as nn
from utils import QLinear, QConv2d, RangeBN

NUM_BITS = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD = 32
BIPRECISION = True

class AAlexNet(nn.Module):
    def __init__(self):
        super(AAlexNet, self).__init__()
        self.convolutions = nn.Sequential(

            QConv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=True, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_GRAD, biprecision=BIPRECISION),
            RangeBN(128, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD),
            nn.ReLU(inplace=True),

            QConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True, num_bits=NUM_BITS,
                    num_bits_weight=NUM_BITS_GRAD, biprecision=BIPRECISION),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RangeBN(128, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD),
            nn.ReLU(inplace=True),

            QConv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True, num_bits=NUM_BITS,
                    num_bits_weight=NUM_BITS_GRAD, biprecision=BIPRECISION),
            RangeBN(256, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD),
            nn.ReLU(inplace=True),

            QConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True, num_bits=NUM_BITS,
                    num_bits_weight=NUM_BITS_GRAD, biprecision=BIPRECISION),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RangeBN(256, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD),
            nn.ReLU(inplace=True),

            QConv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True, num_bits=NUM_BITS,
                    num_bits_weight=NUM_BITS_GRAD, biprecision=BIPRECISION),
            RangeBN(512, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD),
            nn.ReLU(inplace=True),

            QConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True, num_bits=NUM_BITS,
                    num_bits_weight=NUM_BITS_GRAD, biprecision=BIPRECISION),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RangeBN(512, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD),
            nn.ReLU(inplace=True),

        )
        self.linears = nn.Sequential(

            QLinear(512 * 4 * 4, 1024, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD),
            RangeBN(1024, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD),
            nn.ReLU(inplace=True),

            QLinear(1024, 1024, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD),
            RangeBN(1024, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD),
            nn.ReLU(inplace=True),

            QLinear(1024, 10, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD),
            RangeBN(10, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(-1, 512 * 4 * 4)
        out = self.linears(x)
        return out