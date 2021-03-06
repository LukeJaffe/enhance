'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG11_48': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M3'],
    #'VGG11_48': [64, 'M', 'D5', 128, 'M', 'D3', 256, 256, 'M', 'D2', 512, 512, 'M', 'D1', 512, 512, 'M3'],
    'VGG11_128': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG11_96': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M3'],
    'VGG11_64': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 'D5', 128, 128, 'M', 'D3', 256, 256, 256, 'M', 'D2', 512, 512, 512, 'M', 'D1', 512, 512, 512, 'M'],
    'VGG16_48': [64, 64, 'M', 'D5', 128, 128, 'M', 'D3', 256, 256, 256, 'M', 'D2', 512, 512, 512, 'M', 'D1', 512, 512, 512, 'M'],
    'VGG16_64': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_128': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_96': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M3'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_64': [64, 64, 'M', 'D5', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 1024, 1024, 1024, 1024, 'M', 1024, 1024, 1024, 1024, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 101)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'M3':
                layers += [nn.MaxPool2d(kernel_size=3, stride=3)]
            elif x == 'D5':
                layers += [nn.Dropout(0.75)]
            elif x == 'D3':
                layers += [nn.Dropout(0.5)]
            elif x == 'D2':
                layers += [nn.Dropout(0.25)]
            elif x == 'D1':
                layers += [nn.Dropout(0.1)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
