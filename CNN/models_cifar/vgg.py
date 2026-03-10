'''VGG for CIFAR10.
(c) YANG, Wei
'''
import torch.nn as nn
import math
import torch.nn.init as init
from modules import *
import torch.nn.functional as F


__all__ = ['vgg_small_1w1a']

class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class VGG(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.nonlinear0 = nn.Hardtanh(inplace=True)
        self.conv1 = BinarizeConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.conv2 = BinarizeConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.conv3 = BinarizeConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.nonlinear3 = nn.Hardtanh(inplace=True)
        self.conv4 = BinarizeConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.nonlinear4 = nn.Hardtanh(inplace=True)
        self.conv5 = BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.nonlinear5 = nn.Hardtanh(inplace=True)
        self.fc = nn.Linear(512*4*4, num_classes)

        self.auxiliary0 = nn.Sequential(
            SepConv(channel_in=128, channel_out=256),
            SepConv(channel_in=256, channel_out=512),
            SepConv(channel_in=512, channel_out=1024),
            nn.AvgPool2d(4, 4)
        )

        self.auxiliary1 = nn.Sequential(
            SepConv(channel_in=128, channel_out=256),
            SepConv(channel_in=256, channel_out=512),
            nn.AvgPool2d(4, 4)
            # SepConv(channel_in=512, channel_out=1024)
        )

        self.auxiliary2 = nn.Sequential(
            SepConv(channel_in=256, channel_out=512),
            SepConv(channel_in=512, channel_out=1024),
            nn.AvgPool2d(4, 4)
        )

        self.auxiliary3 = nn.Sequential(
            SepConv(channel_in=256, channel_out=512),
            nn.AvgPool2d(4, 4)
            # SepConv(channel_in=512, channel_out=1024)
        )

        self.auxiliary4 = nn.Sequential(
            SepConv(channel_in=512, channel_out=1024),
            nn.AvgPool2d(4, 4)
        )

        # self.auxiliary4 = nn.AvgPool2d(4, 4)

        # self.auxiliary5 = nn.AvgPool2d(4, 4)

        self.auxiliary5 = nn.Sequential(
            SepConv(channel_in=512, channel_out=1024),
            nn.AvgPool2d(4, 4)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BinarizeConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        feature_list = []

        x = self.conv0(x) # [1024, 128, 32, 32]
        x = self.bn0(x) # [1024, 128, 32, 32]
        x = self.nonlinear0(x) # [1024, 128, 32, 32]
        feature_list.append(x)

        x = self.conv1(x) # [1024, 128, 32, 32]
        x = self.pooling(x) # [1024, 128, 16, 16]
        x = self.bn1(x) # [1024, 128, 16, 16]
        x = self.nonlinear1(x) # [1024, 128, 16, 16]
        feature_list.append(x)

        x = self.conv2(x) # [1024, 256, 16, 16]
        x = self.bn2(x) # [1024, 256, 16, 16]
        x = self.nonlinear2(x) # [1024, 256, 16, 16]
        feature_list.append(x)

        x = self.conv3(x) # [1024, 256, 16, 16]
        x = self.pooling(x) # [1024, 256, 8, 8]
        x = self.bn3(x) # [1024, 256, 8, 8]
        x = self.nonlinear3(x) # [1024, 256, 8, 8]
        feature_list.append(x)

        x = self.conv4(x) # [1024, 512, 8, 8]
        x = self.bn4(x) # [1024, 512, 8, 8]
        x = self.nonlinear4(x) # [1024, 512, 8, 8]
        feature_list.append(x)

        x = self.conv5(x) # [1024, 512, 8, 8]
        x = self.pooling(x) # [1024, 512, 4, 4]
        x = self.bn5(x) # [1024, 512, 4, 4]
        x = self.nonlinear5(x) # [1024, 512, 4, 4]
        feature_list.append(x)

        x = x.reshape(x.size(0), -1) # [1024, 8192]
        x = self.fc(x) # [1024, 10]

        out0_feature = self.auxiliary0(feature_list[0]).view(x.size(0), -1)
        out1_feature = self.auxiliary1(feature_list[1]).view(x.size(0), -1)
        out2_feature = self.auxiliary2(feature_list[2]).view(x.size(0), -1)
        out3_feature = self.auxiliary3(feature_list[3]).view(x.size(0), -1)
        out4_feature = self.auxiliary4(feature_list[4]).view(x.size(0), -1)
        out5_feature = self.auxiliary4(feature_list[4]).view(x.size(0), -1)

        # feat_list = [out0_feature]
        feat_list = [out5_feature, out4_feature, out3_feature, out2_feature, out1_feature, out0_feature]

        for index in range(len(feat_list)):
            feat_list[index] = F.normalize(feat_list[index], dim=1)
        if self.training:
            return x, feat_list
        else:
            return x

        # return x

def vgg_small_1w1a(**kwargs):
    model = VGG(**kwargs)
    return model


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda name: 'conv' in name or 'fc' in name, [name[0] for name in list(net.named_modules())]))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('vgg'):
            print(net_name)
            test(globals()[net_name]())
            print()