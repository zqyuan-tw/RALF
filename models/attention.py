import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        mid_channels = in_channels // reduction
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gmp = self.gmp(x)
        gap = self.gap(x)        
        x  = self.sigmoid(self.mlp(gap) + self.mlp(gmp))

        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        m, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.mean(x, dim=1, keepdim=True)        
        x = self.conv(torch.cat([m, a], dim=1))

        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels) -> None:
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x

        return x
    