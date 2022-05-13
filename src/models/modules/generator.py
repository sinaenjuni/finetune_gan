import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def linear(in_features, out_features, bias=True):
    return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias), eps=1e-6)

def batchnorm_2d(in_features, eps=1e-4, momentum=0.1, affine=True):
    return nn.BatchNorm2d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=True):
    return nn.ConvTranspose2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)

def sndeconv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=groups,
                                            bias=bias),
                         eps=1e-6)


class Generator(nn.Module):
    def getLayer(self, deconv, num_input, num_outout, kernel_size, stride, padding, bn):
        layer = []
        layer.append(deconv(in_channels=num_input,
                            out_channels=num_outout,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding))
        if bn:
            layer.append(nn.BatchNorm2d(num_outout))

        layer.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layer = nn.Sequential(*layer)
        return layer

    def __init__(self, linear, deconv, image_size, image_channel, std_channel, latent_dim, num_classes, bn):
        super(Generator, self).__init__()

        self.image_size = image_size // 2 ** 4
        self.std_channel = std_channel
        self.embedding = nn.Sequential(nn.Embedding(num_classes, latent_dim),
                                       nn.Flatten(start_dim=1))

        self.layer1 = nn.Sequential(linear(in_features=latent_dim,
                                          out_features = self.image_size * self.image_size * std_channel * 4),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))  # 2*2*256

        self.layer2 = self.getLayer(deconv, std_channel * 4, std_channel * 2, kernel_size=4, stride=2, padding=1, bn=bn)   # 4*4*128
        self.layer3 = self.getLayer(deconv, std_channel * 2, std_channel * 2, kernel_size=4, stride=2, padding=1, bn=bn)   # 8*8*128
        self.layer4 = self.getLayer(deconv, std_channel * 2, std_channel * 1, kernel_size=4, stride=2, padding=1, bn=bn)   # 16*16*64
        self.layer5 = deconv(std_channel*1, image_channel, kernel_size=4, stride=2, padding=1)                             # 32*32*3

    def forward(self, x, y):
        y = self.embedding(y)
        x = torch.multiply(x, y)
        x = self.layer1(x)
        x = x.view(-1, self.std_channel * 4, self.image_size, self.image_size)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.tanh_(x)
        # x = torch.sigmoid_(x)
        return x


if __name__ == "__main__":
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.02)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, std=0.02)

    G = Generator(linear=snlinear, deconv=sndeconv2d, image_size=32, image_channel=3, std_channel=64, latent_dim=128, num_classes=10, bn=True)
    G.apply(initialize_weights)

    inputs = torch.randn((32, 128))
    label = (torch.rand(32)*10).long()

    outputs = G(inputs, label)
    print(outputs.size())

    for i in G.named_parameters():
        print(i[0])