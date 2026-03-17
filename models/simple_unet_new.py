import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding='same')
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1) #maybe change the dimension, check.
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class BlockCond(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding='same')
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
            self.conv1_cond = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            self.conv1_cond = nn.Conv2d(in_ch, out_ch, 3, padding='same')

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.bnorm1_cond = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        self.relu_cond = nn.ReLU()
        
    def forward(self, x, cond, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        hcond = self.bnorm1_cond(self.relu_cond(self.conv1_cond(cond)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel

        h = h + time_emb
        h = h + hcond

        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, 
                 in_channels = 50, 
                 cond_channels = 5,  
                 out_channels = 50, 
                 down_channels = (64, 64, 64, 64, 64), 
                 up_channels = (64, 64, 64, 64, 64), 
                 time_emb_dim = 64):

        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        # down_channels = (64, 64, 64, 64, 64)
        # up_channels = (64, 64, 64, 64, 64)
        # down_channels = (128, 128, 128, 128, 128)
        # up_channels = (128, 128, 128, 128, 128)
        # # time_emb_dim = 32
        # time_emb_dim = 64
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding='same')
        self.conv0_cond = nn.Conv2d(cond_channels, down_channels[0], 3, padding='same')

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])


        self.output = nn.Conv2d(up_channels[-1], out_channels, 1)

    # def forward(self, x, timestep):
    #     # Embedd time
    #     t = self.time_mlp(timestep)
    #     # Initial conv
    #     x = self.conv0(x)
    #     # Unet
    #     residual_inputs = []
    #     for down in self.downs:
    #         x = down(x, t)
    #         residual_inputs.append(x)

    #     for up in self.ups:
    #         residual_x = residual_inputs.pop()
    #         # Add residual x as additional channels
    #         x = torch.cat((x, residual_x), dim=1)
    #         x = up(x, t)
    #     return self.output(x)

    def forward(self, x, timestep):
        # Embed time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        num_blocks = len(self.down_channels)-1
        xpad =  ((x.shape[2]//(2**num_blocks))+1)*(2**num_blocks)-x.shape[2]
        ypad = ((x.shape[3]//(2**num_blocks))+1)*(2**num_blocks)-x.shape[3]
        x = F.pad(x, (0,ypad,0,xpad))


        # U-Net
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)            # each Block downsamples (stride=2)
            residual_inputs.append(x)  # store feature at this resolution

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # shapes must match here; padding above ensures it
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)              # each Block upsamples (transpose stride=2)

        x = x[:,:,:-xpad:, :-ypad]
        outmatrix= self.output(x)
        return outmatrix


class SimpleUnetCond(nn.Module):
    def __init__(self, in_channels=27, cond_channels=5, out_channels=27,
                 down_channels=(128,128,128,128,128),
                 up_channels  =(128,128,128,128,128),
                 time_emb_dim=64):
        super().__init__()
        self.down_channels = down_channels
        self.up_channels   = up_channels

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # single stem after concat
        self.stem = nn.Conv2d(in_channels + cond_channels, down_channels[0], 3, padding="same")

        # down / up
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim)
            for i in range(len(down_channels)-1)
        ])
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True)
            for i in range(len(up_channels)-1)
        ])

        self.output = nn.Conv2d(up_channels[-1], out_channels, 1)

    def forward(self, x, cond, timestep):
        # time embedding
        t = self.time_mlp(timestep.float())

        # match spatial size if needed
        if x.shape[-2:] != cond.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # remember original H,W for final crop
        H0, W0 = x.shape[-2], x.shape[-1]

        # concat condition and project once
        x = torch.cat([x, cond], dim=1)         # [B, in+cond, H, W]
        x = self.stem(x)                         # [B, down_channels[0], H, W]

        # pad to multiples of 2**num_blocks
        num_blocks = len(self.down_channels) - 1
        pow2 = 2 ** num_blocks
        xpad = (( (H0 + pow2 - 1) // pow2) * pow2) - H0
        ypad = (( (W0 + pow2 - 1) // pow2) * pow2) - W0
        if xpad or ypad:
            x = F.pad(x, (0, ypad, 0, xpad))

        # U-Net
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)

        for up in self.ups:
            skip = residuals.pop()
            x = torch.cat([x, skip], dim=1)
            x = up(x, t)

        # crop back to original H0, W0 (safe crop; no ':-0')
        x = x[:, :, :H0, :W0]
        return self.output(x)



