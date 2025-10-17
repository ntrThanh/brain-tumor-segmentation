from torch import nn, F
import torch


class TumorSegmentationModel(nn.Module):
    def __init__(self):
        super(TumorSegmentationModel, self).__init__()

        # Encoder
        self.encode1 = Encoder(1, 64)
        self.encode2 = Encoder(64, 128)
        self.encode3 = Encoder(128, 256)
        self.encode4 = Encoder(256, 512)

        # Bottleneck
        self.conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        # Encoder

        self.decode1 = Decoder(1024, 512)
        self.decode2 = Decoder(512, 256)
        self.decode3 = Decoder(256, 128)
        self.decode4 = Decoder(128, 64)

        #final
        self.final = nn.Conv2d(64, 1, kernel_size=3, padding=1)



    def forward(self, x):
        s1, p1 = self.encode1(x)
        s2, p2 = self.encode2(p1)
        s3, p3 = self.encode3(p2)
        s4, p4 = self.encode4(p3)

        b = self.conv(p4)

        d4 = self.decode1(b, s4)
        d3 = self.decode2(d4, s3)
        d2 = self.decode3(d3, s2)
        d1 = self.decode4(d2, s1)

        out = self.final(d1)

        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.encoder(x)
        x_skip = self.pool(x)
        return x, x_skip


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_skip):
        x = self.up(x)

        if x_skip.size() != x.size():
            diffY = x_skip.size()[2] - x.size()[2]
            diffX = x_skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, x], dim=1)

        x = self.conv(x)
        return x
