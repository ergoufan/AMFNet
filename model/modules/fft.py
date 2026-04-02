
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class AM(nn.Module):
    
    def __init__(self, nc, expand = 2):
        super(AM, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')     
        mag = torch.abs(x_freq)     
        pha = torch.angle(x_freq)   
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')       
        return x_out