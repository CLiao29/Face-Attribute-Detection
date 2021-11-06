from typing import Optional
import torch
from torch import nn 

'''
This model is implemented based on Pytorch MobileNetV2 repo, which can be found in 
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
'''


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

'''
ConvSet:
A container which involves 3 layers in the structure of : 
    +-----------------------------------------------+
    |Conv layer | -> Batch Normalization | -> ReLU6 |
    +-----------------------------------------------+
Conv layer can be either the Standard 3x3 conv layer or Depthwise and Pointwise layers
'''
class ConvSet(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, groups = 1):
        # padding is adjusted according to the kernel size, by default =1 when kernel =3, but =1 when kernel =1 for the pointwise
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups = groups, bias = False),
            nn.BatchNorm2d(out_channels, affine= True,),
            nn.ReLU6(inplace=True),
            )


'''
BottleneckResidual:
As described in MobileNetV2 paper, more info on its #Table 1#
'''
class BottleneckResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super().__init__()
        hidden_channel = in_channels * expansion_factor
        self.use_shortcut = stride == 1 and in_channels == out_channels

        layers = []
        if expansion_factor != 1:
            # 1x1 pointwise conv
            layers.append(ConvSet(in_channels, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvSet(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

'''
MobileNetV2:

t: expansion factor
c: output channels
n: Each layers are repeated n times
s: stride = s for the first layer, and 1 for the rest
'''
class MobileNetV2(nn.Module):
    # alpha is the hyper-parameter width multiplier, more info on MobileNetV1 paper
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = BottleneckResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvSet(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, expansion_factor=t, stride = stride))
                input_channel = output_channel
        # building last several layers
        features.append(ConvSet(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(last_channel, num_classes)
        )
    

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        
        return x


        



