
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSGP(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim*2, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels - 1)])
        self.mfr0 = nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU() 

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)      
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**i, w//2**i)
                s1 = F.adaptive_max_pool2d(xc[i], p_size)
                s2 = F.adaptive_avg_pool2d(xc[i], p_size)
                
                s = torch.cat([s1, s2], dim=1)

                s = self.mfr[i-1](s)   
      
                s = F.interpolate(s, size=(h, w), mode='nearest')   
               
            else:
                s = self.mfr0(xc[i])
          
            out.append(s)
          

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


