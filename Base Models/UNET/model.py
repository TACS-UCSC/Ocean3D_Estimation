import torch
import torch.nn as nn


torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=4):
        super(UNet, self).__init__()
        
        # ---- Encoder ----
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.down1 = conv_block(in_channels, 64)   # ✅ now configurable
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(128, 256)

        # Decoder
        self.up2 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.upconv2 = conv_block(256, 128)

        # IMPORTANT: odd spatial size fix
        self.up1 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2, output_padding=(1, 0)
        )
        self.upconv1 = conv_block(128, 64)

        # Output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)  # ✅ configurable

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        # Bottleneck
        bn = self.bottleneck(p2)

        # Decoder
        u2 = self.up2(bn)
        u2 = self.upconv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)
        u1 = self.upconv1(torch.cat([u1, d1], dim=1))

        return self.final(u1)


model = UNet(in_channels=6, out_channels=4).to(device)
