'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import *
from torch.autograd import Function,Variable

__all__ = ['resnet20_1w1a']

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

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

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        pad = 0 if planes == self.expansion*in_planes else planes // 4
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                        nn.AvgPool2d((2,2)),
                        LambdaLayer(lambda x:
                        F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0)))


    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = F.hardtanh(out, inplace=True)
        x1 = out
        out = self.bn2(self.conv2(out))
        out += x1
        out = F.hardtanh(out, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)

        # self.auxiliary0 = nn.Sequential(
        #     SepConv(
        #         channel_in=16 * block.expansion,
        #         channel_out=32 * block.expansion
        #     ),
        #     SepConv(
        #         channel_in=32 * block.expansion,
        #         channel_out=64 * block.expansion
        #     ),
        #     SepConv(
        #         channel_in=64 * block.expansion,
        #         channel_out=128 * block.expansion
        #     ),
        #     nn.AvgPool2d(4, 4)
        # )

        self.auxiliary1 = nn.Sequential(
            SepConv(
                channel_in=16 * block.expansion,
                channel_out=32 * block.expansion
            ),
            SepConv(
                channel_in=32 * block.expansion,
                channel_out=64 * block.expansion
            ),
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        self.auxiliary2 = nn.Sequential(
            SepConv(
                channel_in=32 * block.expansion,
                channel_out=64 * block.expansion
            ),
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        self.auxiliary3 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        # self.auxiliary3 = nn.AvgPool2d(4, 4)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        x = F.hardtanh(self.bn1(self.conv1(x)), inplace=True)
        # feature_list.append(x)
        x = self.layer1(x)
        feature_list.append(x)
        x = self.layer2(x)
        feature_list.append(x)
        x = self.layer3(x)
        feature_list.append(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.reshape(x.size(0), -1)
        x = self.bn2(x)
        x = self.linear(x)

        # out0_feature = self.auxiliary0(feature_list[0]).view(x.size(0), -1) # 即 .view(batchsize, -1)，将多维tensor展平成一维
        out1_feature = self.auxiliary1(feature_list[0]).view(x.size(0), -1)  # 即 .view(batchsize, -1)，将多维tensor展平成一维
        out2_feature = self.auxiliary2(feature_list[1]).view(x.size(0), -1)
        out3_feature = self.auxiliary3(feature_list[2]).view(x.size(0), -1)

        # feat_list = [out3_feature]
        feat_list = [out3_feature, out2_feature, out1_feature]

        for index in range(len(feat_list)):
            feat_list[index] = F.normalize(feat_list[index], dim=1)
        if self.training:
            return x, feat_list
        else:
            return x


def resnet20_1w1a(num_classes=10):
    return ResNet(BasicBlock_1w1a, [3, 3, 3],num_classes=num_classes)



def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda name: 'conv' in name or 'linear' in name, [name[0] for name in list(net.named_modules())]))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()