#!/usr/bin/env python
# -*- coding: utf-8 -*-
#resnet18_all.py is used to quantize the weight and activation of ResNet-18.
from __future__ import print_function, absolute_import

import torch.nn as nn
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
from .quantization import *
import pdb

__all__ = ['ResNet_Q', 'resnet18_q']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ac_quan_values, ac_quan_bias, ac_init_beta, count,
                 stride=1, downsample=None, QA_flag=True):
        super(BasicBlock, self).__init__()
        self.QA_flag = QA_flag
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if self.QA_flag:
            self.quan1 = Quantization(quant_values=ac_quan_values, quan_bias=ac_quan_bias[count], init_beta=ac_init_beta[count])

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        if self.QA_flag:
            self.quan2 = Quantization(quant_values=ac_quan_values, quan_bias=ac_quan_bias[count+1], init_beta=ac_init_beta[count+1])

        self.stride = stride
        self.ac_T = 1

    def set_activation_T(self, activation_T):
        self.ac_T = activation_T

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # quantization 

        if self.QA_flag:
            out = self.quan1(out, self.ac_T)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        if self.QA_flag:
            out = self.quan2(out, self.ac_T)

        return out

class ResNet_Q(nn.Module):

    def __init__(self, block, layers, num_classes=1000, QA_flag=True, ac_quan_bias=None, ac_quan_values=None, ac_beta=None):
        self.inplanes = 64
        self.ac_quan_values = ac_quan_values
        self.ac_quan_bias = ac_quan_bias
        self.ac_beta = ac_beta
        self.count=0
        super(ResNet_Q, self).__init__()
        self.QA_flag = QA_flag       

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.QA_flag:
            #print(self.count)
            self.quan0 = Quantization(quant_values=self.ac_quan_values, quan_bias=self.ac_quan_bias[self.count],
                                      init_beta=self.ac_beta[self.count])
            self.count += 1

        self.layer1 = self._make_layer(block, 64, layers[0], QA_flag=self.QA_flag)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, QA_flag=self.QA_flag)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, QA_flag=self.QA_flag)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, QA_flag=self.QA_flag)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        self.set_params()
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, QA_flag=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.ac_quan_values, self.ac_quan_bias, self.ac_beta,
                            self.count, stride, downsample, QA_flag=QA_flag))
        self.inplanes = planes * block.expansion
        self.count += 2
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.ac_quan_values, \
                                self.ac_quan_bias, self.ac_beta, self.count, QA_flag=QA_flag))
            self.count += 2

        return nn.Sequential(*layers)

    def set_resnet_ac_T(self, input_ac_T):
        for m in self.layer1:
            if isinstance(m, BasicBlock):
                m.set_activation_T(input_ac_T)
        for m in self.layer2:
            if isinstance(m, BasicBlock):
                m.set_activation_T(input_ac_T)
        for m in self.layer3:
            if isinstance(m, BasicBlock):
                m.set_activation_T(input_ac_T)
        for m in self.layer4:
            if isinstance(m, BasicBlock):
                m.set_activation_T(input_ac_T)

    
    def set_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_in')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def forward(self, x, input_ac_T=0):
        if self.QA_flag:
            self.set_resnet_ac_T(input_ac_T)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
    
        if self.QA_flag:
            l1 = self.quan0(x, input_ac_T)        

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_q(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Q(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


