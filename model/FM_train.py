import torch.nn as nn
import torch.nn.functional as F
import torch


from .modules import FMMtrain, MSFFMtrain, MAFusion


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class FM(nn.Module):
    def __init__(self, base_dim=32):
        super(FM, self).__init__()
        # down-sample
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride = 1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        self.down_level1_block1 = FMMtrain(default_conv, base_dim, 3)
        self.down_level1_block2 = FMMtrain(default_conv, base_dim, 3)
        self.down_level1_block3 = FMMtrain(default_conv, base_dim, 3)
        self.down_level1_block4 = FMMtrain(default_conv, base_dim, 3)
        self.up_level1_block1 = MSFFMtrain(default_conv, base_dim, 3)
        self.up_level1_block2 = MSFFMtrain(default_conv, base_dim, 3)
        self.up_level1_block3 = MSFFMtrain(default_conv, base_dim, 3)
        self.up_level1_block4 = MSFFMtrain(default_conv, base_dim, 3)

        self.down_level2_block1 = FMMtrain(default_conv, base_dim * 2, 3)
        self.down_level2_block2 = FMMtrain(default_conv, base_dim * 2, 3)
        self.down_level2_block3 = FMMtrain(default_conv, base_dim * 2, 3)
        self.down_level2_block4 = FMMtrain(default_conv, base_dim * 2, 3)
        self.up_level2_block1 = MSFFMtrain(default_conv, base_dim * 2, 3)
        self.up_level2_block2 = MSFFMtrain(default_conv, base_dim * 2, 3)
        self.up_level2_block3 = MSFFMtrain(default_conv, base_dim * 2, 3)
        self.up_level2_block4 = MSFFMtrain(default_conv, base_dim * 2, 3)

        self.level3_block1 = MSFFMtrain(default_conv, base_dim * 4, 3)
        self.level3_block2 = MSFFMtrain(default_conv, base_dim * 4, 3)
        self.level3_block3 = MSFFMtrain(default_conv, base_dim * 4, 3)
        self.level3_block4 = MSFFMtrain(default_conv, base_dim * 4, 3)
        self.level3_block5 = MSFFMtrain(default_conv, base_dim * 4, 3)
        self.level3_block6 = MSFFMtrain(default_conv, base_dim * 4, 3)
        self.level3_block7 = MSFFMtrain(default_conv, base_dim * 4, 3)
        self.level3_block8 = MSFFMtrain(default_conv, base_dim * 4, 3)
        # up-sample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))

        self.mix1 = MAFusion(base_dim * 4, reduction=8)
        self.mix2 = MAFusion(base_dim * 2, reduction=4)

    def forward(self, x):

        x_down1 = self.down1(x)     # input: [B, 3, H, W] → 输出: [B, 32, H, W]    3*3conv
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)

        x_down2 = self.down2(x_down1)      # down: [B, 32, H, W] → [B, 64, H/2, W/2]   
        x_down2_init = self.down_level2_block1(x_down2)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)

        x_down3 = self.down3(x_down2_init)    # down: [B, 64, H/2, W/2] → [B, 128, H/4, W/4]
        x1 = self.level3_block1(x_down3)
        x2 = self.level3_block2(x1)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x6 = self.level3_block6(x5)    
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)
        x_level3_mix = self.mix1(x_down3, x8)

        x_up1 = self.up1(x_level3_mix)      #  up:[B, 128, H/4, W/4] → [B, 64, H/2, W/2]
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        x_level2_mix = self.mix2(x_down2, x_up1)

        x_up2 = self.up2(x_level2_mix)        #  up:[B, 64, H/2, W/2] → [B, 32, H, W]
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        out = self.up3(x_up2)     # [B, 32, H, W] → [B, 3, H, W]     3*3conv

        out = out + x
        return out
    


if __name__ == "__main__":
    model = FM()
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output.shape)