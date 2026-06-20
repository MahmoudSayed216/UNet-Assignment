import torch.nn as nn
import torch


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.non_linearity1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.non_linearity2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.non_linearity1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.non_linearity2(x)
        x = self.pool(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.non_linearity1 = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.non_linearity2 = nn.ReLU()

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn1(x)
        x = self.non_linearity1(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.non_linearity2(x)   # fixed: was non_linearity1 twice
        return x


class UNet(nn.Module):
    def __init__(self, level=-1, n_classes=3):
        super().__init__()

        self.level = level
        self.downsample1 = Downsample(3, 64)
        self.downsample2 = Downsample(64, 128)
        self.downsample3 = Downsample(128, 256)

        self.mid_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )

        # Bottom-most level (right above mid_layer) -> skip connection always on
        self.upsample1 = Upsample(512, 128)   # concat(lvl3d[256] + m[256]) = 512

        # Level 2
        if self.level == -1 or self.level == 2:
            self.upsample2 = Upsample(256, 64)   # concat(lvl2d[128] + lvl3u[128])
        else:
            self.upsample2 = Upsample(128, 64)   # lvl3u[128] only

        # Level 1
        if self.level == -1 or self.level == 1:
            self.upsample3 = Upsample(128, 64)   # concat(lvl1d[64] + lvl2u[64])
        else:
            self.upsample3 = Upsample(64, 64)    # lvl2u[64] only

        self.output_layer = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        lvl1d = self.downsample1(x)
        lvl2d = self.downsample2(lvl1d)
        lvl3d = self.downsample3(lvl2d)
        m     = self.mid_layer(lvl3d)

        # Bottom level: always a skip connection
        lvl3u = self.upsample1(torch.concat((lvl3d, m), dim=1))

        # Level 2
        if self.level == -1 or self.level == 2:
            lvl2u = self.upsample2(torch.concat((lvl2d, lvl3u), dim=1))
        else:
            lvl2u = self.upsample2(lvl3u)

        # Level 1
        if self.level == -1 or self.level == 1:
            lvl1u = self.upsample3(torch.concat((lvl1d, lvl2u), dim=1))
        else:
            lvl1u = self.upsample3(lvl2u)

        return self.output_layer(lvl1u)