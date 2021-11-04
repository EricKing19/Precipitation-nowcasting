# from models.TCN import TemporalConvNet
from .utils import *
import torch
import torch.nn as nn
import torch.nn.functional as f


# class SFNet(nn.Module):
#     def __init__(self, input_channels, dilations, num_class, input_length=3, output_length=3, base='resnet50',
#                  high_channels=2048, low_channels=512):
#         super(SFNet, self).__init__()
#         self.deeplab = DeepLab(input_channels, dilations, base, high_channels, low_channels)
#         self.classifier = IntegralClassify(num_class, mid_channels=low_channels)
#         self.conv_reduce = Conv_BN_ReLU(low_channels, 16, kernel_size=1, stride=1)
#         self.conv_cat_1 = Conv_BN_ReLU(low_channels+16, low_channels, kernel_size=3, stride=1, padding=1)
#         self.conv_cat_2 = Conv_BN_ReLU(low_channels, low_channels, kernel_size=3, stride=1, padding=1)
#         self.state_transfer = TemporalConvNet(low_channels, [low_channels, low_channels],
#         input_length=input_length, output_length=output_length)
#         self.non_linear = nn.Softmax()
#
#     def forward(self, x):
#         # basic0, basic1, hist = x[0][0], x[0][1], x[1]
#         _, hist = x[0], x[1]
#
#         basic = {}
#         for i, j in enumerate(x[0]):
#             basic[i] = j
#
#         features = {}
#         features_low = {}
#         for i in basic:
#             features[i], features_low[i] = self.deeplab(basic[i])
#             features_low[i] = self.conv_reduce(features_low[i])
#             features[i] = f.interpolate(features[i], scale_factor=4, mode='bilinear', align_corners=False)
#
#         features_ = {}
#         for l in range(len(features)):
#             i = features[l]
#             j = features_low[l]
#             features_[l] = torch.cat([i, j], dim=1)
#             features_[l] = self.conv_cat_2(self.conv_cat_1(features_[l]))
#
#         features_length = torch.stack(list(features_.values()), dim=2)
#         features_3 = self.state_transfer(features_length)
#         features_3.insert(0, features_[l])
#         features_3.insert(0, features_[l-1])
#
#         pred = {}
#         hist_list = [hist]
#         for i in range(len(features_3)):
#             if i+3 > len(features_3):
#                 break
#             pred[i] = self.classifier(features_3[i:i+3], hist_list[i])
#             hist_list.append(pred[i])
#
#         return pred


class SFNet(nn.Module):
    def __init__(self, input_channels, dilations, num_class, input_length=3, output_length=3, base='resnet50',
                 high_channels=2048, low_channels=512):
        super(SFNet, self).__init__()
        self.deeplab = DeepLab(input_channels, dilations, high_channels, low_channels)
        self.classifier = Classify(num_class, mid_channels=low_channels)
        self.conv_reduce = Conv_BN_ReLU(low_channels, 16, kernel_size=1, stride=1)
        self.conv_cat_1 = Conv_BN_ReLU(low_channels+16, low_channels, kernel_size=3, stride=1, padding=1)
        self.conv_cat_2 = Conv_BN_ReLU(low_channels, low_channels, kernel_size=3, stride=1, padding=1)
        self.non_linear = nn.Softmax()

    def forward(self, x):
        x = x[0]
        features, features_low = self.deeplab(x)
        features_low = self.conv_reduce(features_low)
        features = f.interpolate(features, scale_factor=4, mode='bilinear', align_corners=False)
        features_ = torch.cat([features, features_low], dim=1)
        features_ = self.conv_cat_2(self.conv_cat_1(features_))

        pred = self.classifier(features_)
        return [pred]


if __name__ == '__main__':
    input_data = [torch.randn([2, 112, 256, 256])]
    # reference = torch.randn(2, 5, 256, 256)
    x = SFNet(input_channels=28, dilations=[2, 4], num_class=5, high_channels=2048, low_channels=512, input_length=4,
             output_length=4)
    pred = x(input_data)
    print('finish')
