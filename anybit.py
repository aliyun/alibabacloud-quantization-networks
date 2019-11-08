#!/usr/bin/env python
# -*- coding: utf-8 -*-
# anybit.py is used to quantize the weight of model.

from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy
import pdb

def sigmoid_t(x, b=0, t=1):
    """
    The sigmoid function with T for soft quantization function.
    Args:
        x: input
        b: the bias
        t: the temperature
    Returns:
        y = sigmoid(t(x-b))
    """
    temp = -1 * t * (x - b)
    temp = torch.clamp(temp, min=-10.0, max=10.0)
    return 1.0 / (1.0 + torch.exp(temp))

def step(x, bias):
    """ 
    The step function for ideal quantization function in test stage.
    """
    y = torch.zeros_like(x) 
    mask = torch.gt(x - bias,  0.0)
    y[mask] = 1.0
    return y

class QuaOp(object):
    """
    Quantize weight.
    Args:
        model: the model to be quantified.
        QW_biases (list): the bias of quantization function.
                          QW_biases is a list with m*n shape, m is the number of layers,
                          n is the number of sigmoid_t.
        QW_values (list): the list of quantization values, 
                          such as [-1, 0, 1], [-2, -1, 0, 1, 2].
    Returns:
        Quantized model.
    """
    def __init__(self, model, QW_biases, QW_values=[]):
        # Count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1
        # Omit the first conv layer and the last linear layer
        start_range = 1
        end_range = count_targets - 2
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

        print('target_modules number: ', len(self.target_modules))
        self.QW_biases = QW_biases
        self.QW_values = QW_values
        # the number of sigmoid_t 
        self.n = len(self.QW_values) - 1
        self.threshold = self.QW_values[-1] * 5 / 4.0
        # the gap between two quantization values
        self.scales = []
        offset = 0.
        for i in range(self.n):
            gap = self.QW_values[i + 1] - self.QW_values[i]
            self.scales.append(gap)
            offset += gap
        self.offset = offset / 2.

    def forward(self, x, T, quan_bias, train=True):
        if train:
            y = sigmoid_t(x, b=quan_bias[0], t=T)*self.scales[0]
            for j in range(1, self.n):
                y += sigmoid_t(x, b=quan_bias[j], t=T)*self.scales[j]
        else:
            y = step(x, bias=quan_bias[0])*self.scales[0] 
            for j in range(1, self.n):
                y += step(x, bias=quan_bias[j])*self.scales[j]
        y = y - self.offset

        return y

    def backward(self, x, T, quan_bias):
        y_1 = sigmoid_t(x, b=quan_bias[0], t=T)*self.scales[0]
        y_grad = (y_1.mul(self.scales[0] - y_1)).div(self.scales[0])
        for j in range(1, self.n):
            y_temp = sigmoid_t(x, b=quan_bias[j], t=T)*self.scales[j]
            y_grad += (y_temp.mul(self.scales[j] - y_temp)).div(self.scales[j])

        return y_grad

    def quantization(self, T, alpha, beta, init, train_phase=True):
        """
        The operation of network quantization.
        Args:
            T: the temperature, a single number. 
            alpha: the scale factor of the output, a list.
            beta: the scale factor of the input, a list. 
            init: a flag represents the first loading of the quantization function.
            train_phase: a flag represents the quantization 
                  operation in the training stage.
        """
        self.save_params()
        self.quantizeConvParams(T, alpha, beta, init, train_phase=train_phase)

    def save_params(self):
        """
        save the float parameters for backward
        """
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def restore_params(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


    def quantizeConvParams(self, T, alpha, beta, init, train_phase):
        """
        quantize the parameters in forward
        """
        T = (T > 2000)*2000 + (T <= 2000)*T
        for index in range(self.num_of_params):
            if init:
                beta[index].data = torch.Tensor([self.threshold / self.target_modules[index].data.abs().max()]).cuda()
                alpha[index].data = torch.reciprocal(beta[index].data)
            # scale w
            x = self.target_modules[index].data.mul(beta[index].data)
            
            y = self.forward(x, T, self.QW_biases[index], train=train_phase)
            #scale w^hat
            self.target_modules[index].data = y.mul(alpha[index].data)


    def updateQuaGradWeight(self, T, alpha, beta, init):
        """
        Calculate the gradients of all the parameters.
        The gradients of model parameters are saved in the [Variable].grad.data.
        Args:
            T: the temperature, a single number. 
            alpha: the scale factor of the output, a list.
            beta: the scale factor of the input, a list. 
            init: a flag represents the first loading of the quantization function.
        Returns:
            alpha_grad: the gradient of alpha.
            beta_grad: the gradient of beta.
        """
        beta_grad = [0.0] * len(beta)
        alpha_grad = [0.0] * len(alpha)
        T = (T > 2000)*2000 + (T <= 2000)*T 
        for index in range(self.num_of_params):
            if init:
                beta[index].data = torch.Tensor([self.threshold / self.target_modules[index].data.abs().max()]).cuda()
                alpha[index].data = torch.reciprocal(beta[index].data)
            x = self.target_modules[index].data.mul(beta[index].data)

            # set T = 1 when train binary model
            y_grad = self.backward(x, 1, self.QW_biases[index]).mul(T)
            # set T = T when train the other quantization model
            #y_grad = self.backward(x, T, self.QW_biases[index]).mul(T)
            
        
            beta_grad[index] = y_grad.mul(self.target_modules[index].data).mul(alpha[index].data).\
                               mul(self.target_modules[index].grad.data).sum()
            alpha_grad[index] = self.forward(x, T, self.QW_biases[index]).\
                                mul(self.target_modules[index].grad.data).sum()

            self.target_modules[index].grad.data = y_grad.mul(beta[index].data).mul(alpha[index].data).\
                                                   mul(self.target_modules[index].grad.data)
        return alpha_grad, beta_grad

