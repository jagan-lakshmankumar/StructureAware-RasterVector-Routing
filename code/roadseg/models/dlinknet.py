"""D-LinkNet (ResNet-34 encoder + central dilated cascade)."""

import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet34 as tv_resnet34
except Exception:  # pragma: no cover
    tv_resnet34 = None


class DBlock(nn.Module):
    """Cascade of dilated convolutions (1, 2, 4, 8, 16) with residual."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(channels, channels, 3, padding=8, dilation=8)
        self.conv5 = nn.Conv2d(channels, channels, 3, padding=16, dilation=16)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        d = self.relu(self.conv1(x))
        d = self.relu(self.conv2(d))
        d = self.relu(self.conv3(d))
        d = self.relu(self.conv4(d))
        d = self.relu(self.conv5(d))
        return x + d


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DLinkNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        if tv_resnet34 is None:
            raise ImportError("torchvision is required for D-LinkNet")
        backbone = tv_resnet34(weights=None)
        self.firstconv = backbone.conv1
        self.firstbn = backbone.bn1
        self.firstrelu = backbone.relu
        self.firstmaxpool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4
        self.dblock = DBlock(512)
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)
        self.finaldeconv = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.finalconv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        x0 = self.firstrelu(self.firstbn(self.firstconv(x)))
        x = self.firstmaxpool(x0)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        center = self.dblock(e4)
        d4 = self.decoder4(center)
        d4 = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False) + e3
        d3 = self.decoder3(d4)
        d3 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False) + e2
        d2 = self.decoder2(d3)
        d2 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False) + e1
        d1 = self.decoder1(d2)
        d1 = F.interpolate(d1, size=x0.shape[-2:], mode="bilinear", align_corners=False) + x0
        out = self.finaldeconv(d1)
        out = self.finalconv(out)
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out
