import torch
import torch.nn as nn
import torch_dct as dct
from .reconstructor import Reconstructor
from .xception import Xception
from .attention import SpatialAttention
     

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False):
        super(ResidualBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if out_channels != in_channels else nn.Identity()
        
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class SRMFilter(nn.Module):
    def __init__(self, learnable=False) -> None:
        super(SRMFilter, self).__init__()

        self.filter = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        for param in self.parameters():
            param.requires_grad = learnable

        self.apply(self.init_weights)

    def init_weights(self, layer):

        filter1 = torch.FloatTensor([[0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0], 
        [0, 1,-2, 1, 0], 
        [0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0]]).expand(3, 5, 5) / 2
                
        filter2 = torch.FloatTensor([[0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0]]).expand(3, 5, 5) / 4
        
        filter3 = torch.FloatTensor([[-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]]).expand(3, 5, 5) / 12

        if isinstance(layer, nn.Conv2d):
            with torch.no_grad():
                layer.weight.data = torch.stack([filter1, filter2, filter3])
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, x):
        noise = torch.round(self.filter(x if isinstance(x, torch.ByteTensor) else x.mul(255)))
        noise[noise < -2] = -2.
        noise[noise > 2] = 2.

        return noise


class FrequencyDestructor(nn.Module):
    def __init__(self, input_size=256) -> None:
        super(FrequencyDestructor, self).__init__()
        
        self.freq_mask = nn.Parameter(torch.zeros(input_size, input_size))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        filtered_x = dct.idct_2d(self.sigmoid(self.freq_mask) * dct.dct_2d(x))

        return filtered_x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim, kernel_size=4):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(8, 16, kernel_size=4),
            conv_bn_lrelu(16, 32, kernel_size=4),
            conv_bn_lrelu(32, 64, kernel_size=4),
            conv_bn_lrelu(64, 128, kernel_size=4),
            conv_bn_lrelu(128, 256, kernel_size=4),
            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Flatten()
        )
        
        self.apply(self.weights_init)
        
    def weights_init(self, layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
            nn.init.constant_(layer.bias.data, 0)

    def forward(self, input):
        return self.discriminator(input)


class Classifier(nn.Module):
    def __init__(self) -> None:
        super(Classifier, self).__init__()
        
        xception = Xception()
        
        self.stage1 = nn.Sequential(
            xception.conv1,
            xception.bn1,
            xception.relu,
            xception.conv2,
            xception.bn2,
            xception.relu,
        )
        
        self.attn_map = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SpatialAttention()
        )
        
        self.stage2 = nn.Sequential(
            xception.block1,
            xception.block2,
            xception.block3,
            xception.block4,
            xception.block5,
            xception.block6,
            xception.block7,
            xception.block8,
            xception.block9,
            xception.block10,
            xception.block11,
            xception.block12,

            xception.conv3,
            xception.bn3,
            xception.relu,

            xception.conv4,
            xception.bn4,
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            xception.fc
        )
        
    def forward(self, x, residual):
        out1 = self.stage1(x)
        attn_map = self.attn_map(residual)
        out = self.stage2(out1 * attn_map)
        
        return out


class RALF(nn.Module):
    def __init__(self, input_size=256):
        super(RALF, self).__init__()

        self.destructor = FrequencyDestructor(input_size)

        self.reconstructor = Reconstructor()

        self.classifier = Classifier()
        

    def forward(self, x): 
        filtered_x = self.destructor(x)
        residual_x = torch.abs(x - filtered_x)
        
        reconstruction = self.reconstructor(filtered_x)

        logits = self.classifier(torch.abs(reconstruction - x), residual_x)

        return logits
    