import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import models
from collections import OrderedDict
from .dyconv2d import DyConv2d
affine_par = True


class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 affine=True, inplace=True):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels, affine=affine)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, InputChannels, dilation=[1, 1], layers=[3, 4, 6, 2]):
        super(ResNet, self).__init__()
        feats = list(models.resnet18(pretrained=True).children())
        feats[0] = nn.Conv2d(InputChannels, 64, kernel_size=7, stride=2, padding=3)
        self.layer1 = nn.Sequential(*feats[0:3])
        self.layer2 = nn.Sequential(*feats[3:5])
        self.layer3 = feats[5]
        self.layer4 = feats[6]
        self.layer5 = feats[7]

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x_low = self.layer3(x)
        x = self.layer4(x_low)
        x = self.layer5(x)

        return x, x_low


class ASPP(nn.Module):
    def __init__(self, channels=2048, mid_channels=256):
        super(ASPP, self).__init__()
        self.conv_1x1 = nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1)
        self.bn_1x1 = nn.BatchNorm2d(mid_channels)
        self.relu_1x1 = nn.ReLU(inplace=True)
        self.conv_3x3_6 = Conv_BN_ReLU(channels, mid_channels, kernel_size=3, stride=1, dilation=6, padding=6)
        self.conv_3x3_12 = Conv_BN_ReLU(channels, mid_channels, kernel_size=3, stride=1, dilation=12, padding=12)
        self.conv_3x3_18 = Conv_BN_ReLU(channels, mid_channels, kernel_size=3, stride=1, dilation=18, padding=18)
        self.image_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_1x1_reduce = Conv_BN_ReLU(channels, mid_channels, kernel_size=1, stride=1)
        self.conv_1x1_cat = Conv_BN_ReLU(mid_channels*5, mid_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        out_pool = self.conv_1x1_reduce(self.image_pool(x))
        out_pool = f.interpolate(input=out_pool, size=(h, w), mode='bilinear')
        out_1 = self.conv_1x1(x)
        out_6 = self.conv_3x3_6(x)
        out_12 = self.conv_3x3_12(x)
        out_18 = self.conv_3x3_18(x)
        out = torch.cat((out_pool, out_1, out_6, out_12, out_18), dim=1)
        out = self.conv_1x1_cat(out)

        return out


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(2048, 8, kernel_size=1)
        self.fc = nn.Linear(32, 4)
        self.activate = nn.Softmax(dim=1)

    def forward(self, x):
        tem_features = self.conv(self.pool(x[0]))
        hum_features = self.conv(self.pool(x[1]))
        x_wind_features = self.conv(self.pool(x[2]))
        y_wind_features = self.conv(self.pool(x[3]))
        features = torch.cat((tem_features, hum_features, x_wind_features, y_wind_features), dim=1)
        features = features.view(-1, 32)
        out = self.activate(self.fc(features))

        return out


class GroupFusion(nn.Module):
    def __init__(self, channels, midchannels=4):
        super(GroupFusion, self).__init__()
        self.conv_reduce = Conv_BN_ReLU(channels*4, midchannels, kernel_size=1)
        self.conv_cat = Conv_BN_ReLU(midchannels, 4, kernel_size=1)
        self.conv = Conv_BN_ReLU(4*channels, channels, kernel_size=1, groups=channels)

    def forward(self, x):
        n, c, h, w = x[0].shape
        multiplier = torch.cat([x[0], x[1], x[2], x[3]], dim=1)
        multiplier = self.conv_reduce(multiplier)
        multiplier = self.conv_cat(multiplier)
        features = [x[i]*multiplier[:, i:i+1, :, :] for i in range(4)]
        x_ = torch.stack(features, dim=2)
        x__ = x_.view(n, c*4, h, w)
        x__ = self.conv(x__)

        return x__


class C3D_Fusion(nn.Module):
    def __init__(self, channels):
        super(C3D_Fusion, self).__init__()
        self.c3d = nn.Conv3d(channels, channels, (4, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1))
        self.bn = nn.BatchNorm2d(num_features=channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_ = torch.stack(x, dim=2)
        x_ = self.relu(self.bn(self.c3d(x_)))
        x__ = torch.squeeze(x_)

        return x__


class Add_Fusion(nn.Module):
    def __init__(self):
        super(Add_Fusion, self).__init__()

    def forward(self, x):
        x_ = x[0] + x[1] + x[2] + x[3]
        return x_


class Cat_Fusion(nn.Module):
    def __init__(self, channels):
        super(Cat_Fusion, self).__init__()
        self.conv_cat = Conv_BN_ReLU(channels*4, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_ = torch.cat([x[0], x[1], x[2], x[3]], dim=1)
        return self.conv_cat(x_)


class Classify(nn.Module):
    def __init__(self, num_class, mid_channels=256):
        super(Classify, self).__init__()
        self.conv = nn.Conv2d(mid_channels*3, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels, affine=affine_par)
        # for i in self.bn.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(mid_channels, num_class, kernel_size=1, stride=1)
        # self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')

    def forward(self, x, hist):
        x = torch.cat(x, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        # x = self.upsample(x)

        return f.interpolate(x, scale_factor=8, mode='bilinear')


class IntegralClassify(nn.Module):
    def __init__(self, num_class, mid_channels=256):
        super(IntegralClassify, self).__init__()
        self.conv1 = Conv_BN_ReLU(mid_channels, mid_channels//2, kernel_size=1)
        self.conv2 = Conv_BN_ReLU(mid_channels, mid_channels // 2, kernel_size=1)
        self.conv_x1 = nn.Conv2d(mid_channels//2, num_class, kernel_size=1)
        self.conv_x2 = nn.Conv2d(mid_channels//2, num_class, kernel_size=1)
        # self.conv_x0 = nn.Conv2d(mid_channels, num_class, kernel_size=1)
        self.conv_x1s = nn.Conv2d(mid_channels//2, num_class, kernel_size=1)
        self.conv_x2s = nn.Conv2d(mid_channels//2, num_class, kernel_size=1)
        # self.conv_x0s = nn.Conv2d(mid_channels, num_class, kernel_size=1)
        # self.conv_x1ss = nn.Conv2d(mid_channels // 2, num_class, kernel_size=1)
        # self.conv_x2ss = nn.Conv2d(mid_channels // 2, num_class, kernel_size=1)

    def forward(self, x, hist):
        temp = self.conv_x1(self.conv1(x[1]-x[0])) + self.conv_x2(self.conv2(x[2]-x[1])) + \
               self.conv_x1s(torch.pow(self.conv1(x[1]-x[0]), 2)) + \
               self.conv_x2s(torch.pow(self.conv2(x[2]-x[1]), 2))
        return f.interpolate(temp, scale_factor=8, mode='bilinear') + hist


class DeepLab(nn.Module):
    def __init__(self, input_channels, dilations, high_channels=2048, low_channels=512):
        super(DeepLab, self).__init__()
        self.high_channels = high_channels
        self.low_channels = low_channels
        self.tem_module = ResNet(input_channels, dilations)
        self.hum_module = ResNet(input_channels, dilations)
        self.x_wind_module = ResNet(input_channels, dilations)
        self.y_wind_module = ResNet(input_channels, dilations)
        self.fusion = Cat_Fusion(self.high_channels)
        self.fusion_low = Cat_Fusion(self.low_channels)
        # self.fusion = C3D_Fusion(2048)
        # self.fusion_low = C3D_Fusion(512)
        self.aspp = ASPP(self.high_channels, self.low_channels)

    def forward(self, x):
        channels = int(x.shape[1]/4)
        tem_features, tem_features_low = self.tem_module(x[:, 0:channels, :, :])
        hum_features, hum_features_low = self.hum_module(x[:, channels:2*channels, :, :])
        x_wind_features, x_wind_features_low = self.x_wind_module(x[:, 2*channels:3*channels, :, :])
        y_wind_features, y_wind_features_low = self.y_wind_module(x[:, 3*channels:4*channels, :, :])
        features = [tem_features, hum_features, x_wind_features, y_wind_features]
        features_low = [tem_features_low, hum_features_low, x_wind_features_low, y_wind_features_low]

        character = self.fusion(features)
        character_low = self.fusion_low(features_low)

        return self.aspp(character), character_low


class StateTransfer(nn.Module):
    def __init__(self, input_length=2, output_length=1, mid_channles=256):
        super(StateTransfer, self).__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.transfer = Conv_BN_ReLU(mid_channles, mid_channles, kernel_size=1)

    def forward(self, input):
        x = 0
        for i in range(self.input_length):
            x += self.transfer(input[i])
            if i != 0:
                x /= 2
        out = [x]
        for j in range(self.output_length):
            out.append(self.transfer(out[j]))
        return out[1:]


class StateTransfer_C3D(nn.Module):
    def __init__(self, input_length=2, output_length=1, mid_channles=256):
        super(StateTransfer_C3D, self).__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.transfer = Conv_BN_ReLU(mid_channles, mid_channles, kernel_size=1)

    def forward(self, input):
        x = 0
        for i in range(self.input_length):
            x += self.transfer(input[i])
            if i != 0:
                x /= 2
        out = [x]
        for j in range(self.output_length):
            out.append(self.transfer(out[j]))
        return out[1:]


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'dyconv' in layer_name:
            dyconv2d = DyConv2d(in_channels=v[0], out_channels=v[1],
                                kernel_size=v[2], stride=v[3],
                                padding=v[4])
            layers.append((layer_name, dyconv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))

        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))
