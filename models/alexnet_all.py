#!/usr/bin/env python
# -*- coding: utf-8 -*-
# alexnet_all.py is used to quantize the weight and activation of AlexNet.
from __future__ import print_function, absolute_import

import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch.utils.model_zoo as model_zoo
from .quantization import *
import pdb

__all__ = ['AlexNet_Q', 'alexnet_q']


class ContConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, ac_quan_values, ac_quan_bias, ac_init_beta, count,
        kernel_size=-1, stride=-1, padding=-1, groups=1, QA_flag=True, Linear=False):
        super(ContConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.QA_flag = QA_flag
    
        self.Linear = Linear
        if not self.Linear:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=self.kernel_size,
                    stride=self.stride, padding=self.padding, groups=self.groups)
            self.bn = nn.BatchNorm2d(output_channels, eps=1e-3)
        else:
            self.linear = nn.Linear(input_channels, output_channels)
            self.bn = nn.BatchNorm1d(output_channels, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
        
        if self.QA_flag:
            self.quan = Quantization(quant_values=ac_quan_values, quan_bias=ac_quan_bias[count], init_beta=ac_init_beta)

        self.ac_T = 1

    def set_activation_T(self, activation_T):
        self.ac_T = activation_T
     

    def forward(self, x):
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        #quantization
        if self.QA_flag:
            x = self.quan(x, self.ac_T) 

        return x


class AlexNet_Q(nn.Module):
    def __init__(self, QA_flag=True, ac_quan_bias=None, ac_quan_values=None, ac_beta=None, num_classes=1000):
        self.ac_quan_values = ac_quan_values
        self.ac_quan_bias = ac_quan_bias
        self.ac_beta = ac_beta
        self.count = 0
        super(AlexNet_Q, self).__init__()
        self.num_classes = num_classes

        self.QA_flag = QA_flag

        self.features_0 = nn.Sequential(
            ContConv2d(3, 96, ac_quan_values=None, ac_quan_bias=None, ac_init_beta=None, count=0, 
                       kernel_size=11, stride=4, padding=2, QA_flag=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if self.QA_flag:
            #print(self.count)
            self.quan0 = Quantization(quant_values=self.ac_quan_values, quan_bias=self.ac_quan_bias[self.count], 
                                      init_beta=self.ac_beta[self.count])
            self.count += 1

        self.features_1 = nn.Sequential(
            ContConv2d(96, 256, ac_quan_values=None, ac_quan_bias=None, ac_init_beta=None, count=0, 
                       kernel_size=5, stride=1, padding=2, QA_flag=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if self.QA_flag:
            #print(self.count)
            self.quan1 = Quantization(quant_values=self.ac_quan_values, quan_bias=self.ac_quan_bias[self.count],
                                      init_beta=self.ac_beta[self.count])
            self.count += 1

        self.features_2 = nn.Sequential(
            ContConv2d(256, 384, ac_quan_values=self.ac_quan_values, ac_quan_bias=self.ac_quan_bias, ac_init_beta=self.ac_beta, 
                       count=2, kernel_size=3, stride=1, padding=1, QA_flag=self.QA_flag),
            ContConv2d(384, 384, ac_quan_values=self.ac_quan_values, ac_quan_bias=self.ac_quan_bias, ac_init_beta=self.ac_beta,
                       count=3, kernel_size=3, stride=1, padding=1, QA_flag=self.QA_flag),
            ContConv2d(384, 256, ac_quan_values=None, ac_quan_bias=None, ac_init_beta=None, count=0,
                       kernel_size=3, stride=1, padding=1, QA_flag=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if self.QA_flag:
            #print(self.count)
            self.quan2 = Quantization(quant_values=self.ac_quan_values, quan_bias=self.ac_quan_bias[4],
                                      init_beta=self.ac_beta[4])
       
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            ContConv2d(256*6*6, 4096, ac_quan_values=self.ac_quan_values, ac_quan_bias=self.ac_quan_bias, ac_init_beta=self.ac_beta,
                       count=5, Linear=True, QA_flag=self.QA_flag),
            nn.Dropout(p=0.1),
            ContConv2d(4096, 4096, ac_quan_values=None, ac_quan_bias=None, ac_init_beta=None, count=0, Linear=True, QA_flag=False),
            nn.Linear(4096, self.num_classes),
        )

        self.reset_params()
        
    def set_ac_T(self, input_ac_T):
        for m in self.features_0:
            if isinstance(m, ContConv2d):
                m.set_activation_T(input_ac_T)
        for m in self.features_1:
            if isinstance(m, ContConv2d):
                m.set_activation_T(input_ac_T)
        for m in self.features_2:
            if isinstance(m, ContConv2d):
                m.set_activation_T(input_ac_T)
        for m in self.classifier:
            if isinstance(m, ContConv2d):
                m.set_activation_T(input_ac_T)

    def forward(self, x, input_ac_T=1):
        if self.QA_flag:
            self.set_ac_T(input_ac_T)

        x = self.features_0(x)
        if self.QA_flag:
            x = self.quan0(x, input_ac_T)

        x = self.features_1(x)
        if self.QA_flag:
            x = self.quan1(x, input_ac_T)

        x = self.features_2(x)
        if self.QA_flag:
            x = self.quan2(x, input_ac_T)

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

def alexnet_q(pretrained=False, **kwargs):
    model=AlexNet_Q(**kwargs)
    if pretrained:
        model_path='model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model

