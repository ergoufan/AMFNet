from torch import nn

from .att import ChannelAttention, Pi, PAttention


class MAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(MAFusion, self).__init__()
        self.pa = PAttention(dim, reduction)
        self.ca = ChannelAttention(dim, reduction)
        self.pi = Pi(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.pa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pi(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result