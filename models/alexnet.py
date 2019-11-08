#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch.utils.model_zoo as model_zoo

import pdb

__all__ = ['AlexNet', 'alexnet']


class ContConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
        kernel_size=-1, stride=-1, padding=-1, groups=1, Linear=False):
        super(ContConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
    
        self.Linear = Linear
        if not self.Linear:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=self.kernel_size,
                    stride=self.stride, padding=self.padding, groups=self.groups)
            self.bn = nn.BatchNorm2d(output_channels, eps=1e-3)
        else:
            self.linear = nn.Linear(input_channels, output_channels)
            self.bn = nn.BatchNorm1d(output_channels, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features_0 = nn.Sequential(
            ContConv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features_1 = nn.Sequential(
            ContConv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features_2 = nn.Sequential(
            ContConv2d(256, 384, kernel_size=3, stride=1, padding=1),
            ContConv2d(384, 384, kernel_size=3, stride=1, padding=1),
            ContConv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            ContConv2d(256*6*6, 4096, Linear=True),
            nn.Dropout(p=0.1),
            ContConv2d(4096, 4096, Linear=True),
            nn.Linear(4096, self.num_classes),
        )

        self.reset_params()

    def forward(self, x):
        x = self.features_0(x)
        x = self.features_1(x)
        x = self.features_2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_in')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight, mode='fan_in')
                if m.bias is not None:
                    init.constant(m.bias, 0)

def alexnet(pretrained=False, **kwargs):
    model=AlexNet(**kwargs)
    if pretrained:
        model_path='model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model

