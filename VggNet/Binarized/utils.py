import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class BinarizeF(Function):
    @staticmethod
    def forward(ctx, input):
        out = input.new(input.size())
        out[input >= 0] = 1
        out[input < 0] = -1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

Binarize = BinarizeF.apply

class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        weight = Binarize(self.weight)
        out = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.zero_()
        nn.init.xavier_normal(self.weight)

class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        weight = Binarize(self.weight)
        if self.bias is None:
            out = F.linear(input, weight)
        else:
            out = F.linear(input, weight, self.bias)
        return out

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.zero_()
        nn.init.xavier_normal(self.weight)

class BinaryHardtanh(nn.Module):
    def __init__(self):
        super(BinaryHardtanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, input):
        out = self.hardtanh(input)
        out = Binarize(out)
        return out
