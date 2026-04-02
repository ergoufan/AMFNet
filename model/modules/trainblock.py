from torch import nn
import torch
import torch.functional as F
from .deconv import DEConv
from .MSGP import MSGP
from .fft import AM


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class AFRMtrain(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(AFRMtrain, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.fre = AM(dim, expand = 2)
        self.conv1 = DEConv(dim)
        self.act1 = nn.GELU()
        self.conv2 = conv(dim, dim, kernel_size,  bias=True)

    def forward(self, x):
        y = self.norm1(x)
        y = self.fre(y)
        y = x + y

        res = self.norm2(y)
        res = self.conv1(res)
        res = self.act1(res)
        res = self.conv2(res)
        res = res + y
        return res


class GMPFMtrain(nn.Module):    
    def __init__(self, conv, dim, kernel_size):
        super(GMPFMtrain, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.msgc = MSGP(dim, n_levels=4)
        self.conv1 = DEConv(dim)
        self.act1 = nn.GELU()
        self.conv2 = conv(dim, dim, kernel_size,  bias=True)

    def forward(self, x):
        y = self.norm1(x)
        y = self.msgc(y)
        y = x + y

        res = self.norm2(y)
        res = self.conv1(res)
        res = self.act1(res)
        res = self.conv2(res)
        res = res + y
        return res