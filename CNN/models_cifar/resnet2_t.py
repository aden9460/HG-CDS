'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import *


__all__ =['resnet18A_t','resnet18B_t','resnet18C_t','resnet18_t']

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

class BasicBlock_t(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_t, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        pad = 0 if planes == self.expansion*in_planes else planes // 4
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                        nn.AvgPool2d((2,2)),
                        LambdaLayer(lambda x:
                        F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0)))

    def forward(self, x):
        # out = self.relu(self.bn1(self.conv1(x)))
        out = F.hardtanh(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = self.relu(out)
        out = F.hardtanh(out, inplace=True)
        return out


class Bottleneck_t(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_t, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = nn.ReLU(self.bn1(self.conv1(x)))
        out = nn.ReLU(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = nn.ReLU(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = num_channel[0]

        self.conv1 = nn.Conv2d(3, num_channel[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel[0])
        self.layer1 = self._make_layer(block, num_channel[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channel[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channel[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channel[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(num_channel[3]*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(num_channel[3]*block.expansion)

        # self.auxiliary0 = nn.Sequential(
        #     SepConv(
        #         channel_in=64 * block.expansion,
        #         channel_out=128 * block.expansion
        #     ),
        #     SepConv(
        #         channel_in=128 * block.expansion,
        #         channel_out=256 * block.expansion
        #     ),
        #     SepConv(
        #         channel_in=256 * block.expansion,
        #         channel_out=512 * block.expansion
        #     ),
        #     nn.AvgPool2d(4, 4)
        # )

        self.auxiliary1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        self.auxiliary2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )

        self.auxiliary3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )

        self.auxiliary4 = nn.AvgPool2d(4, 4)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x): # [1024, 3, 32, 32]
        feature_list = []
        x = self.bn1(self.conv1(x)) # [1024, 64, 32, 32]
        # feature_list.append(x)
        x = self.layer1(x) # [1024, 64, 32, 32]
        feature_list.append(x)
        x = self.layer2(x) # [1024, 128, 16, 16]
        feature_list.append(x)
        x = self.layer3(x) # [1024, 256, 8, 8]
        feature_list.append(x)
        x = self.layer4(x) # [1024, 512, 4, 4]
        feature_list.append(x)
        x = F.avg_pool2d(x, 4) # [1024, 512, 1, 1]
        x = x.reshape(x.size(0), -1) # [1024, 512]
        x = self.bn2(x) # [1024, 512]
        x = self.linear(x) # [1024, 10]
        # out0_feature = self.auxiliary0(feature_list[0]).view(x.size(0), -1)
        out1_feature = self.auxiliary1(feature_list[0]).view(x.size(0), -1)
        out2_feature = self.auxiliary2(feature_list[1]).view(x.size(0), -1)
        out3_feature = self.auxiliary3(feature_list[2]).view(x.size(0), -1)
        out4_feature = self.auxiliary4(feature_list[3]).view(x.size(0), -1)
        # out = self.fc(out4_feature)

        feat_list = [out4_feature, out3_feature, out2_feature, out1_feature]
        # feat_list = [out4_feature, out3_feature, out2_feature, out1_feature, out0_feature]

        # x = out4_feature
        # x = self.bn2(x)  # [1024, 512]
        # x = self.linear(x)

        for index in range(len(feat_list)):
            feat_list[index] = F.normalize(feat_list[index], dim=1)
        if self.training:
            return x, feat_list
        else:
            return x


def resnet18A_t(**kwargs):
    return ResNet(BasicBlock_t, [2,2,2,2],[32,32,64,128],**kwargs)

def resnet18B_t(**kwargs):
    return ResNet(BasicBlock_t, [2,2,2,2],[32,64,128,256],**kwargs)

def resnet18C_t(**kwargs):
    return ResNet(BasicBlock_t, [2,2,2,2],[64,64,128,256],**kwargs)

def resnet18_t(**kwargs):
    return ResNet(BasicBlock_t, [2,2,2,2],[64,128,256,512],**kwargs)

def resnet34_t(**kwargs):
    return ResNet(BasicBlock_t, [3,4,6,3],[64,128,256,512],**kwargs)

def resnet50_t(**kwargs):
    return ResNet(Bottleneck_t, [3,4,6,3],[64,128,256,512],**kwargs)

def resnet101_t(**kwargs):
    return ResNet(Bottleneck_t, [3,4,23,3],[64,128,256,512],**kwargs)

def resnet152_t(**kwargs):
    return ResNet(Bottleneck_t, [3,8,36,3],[64,128,256,512],**kwargs)


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
