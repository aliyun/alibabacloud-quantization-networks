#!/usr/bin/env python
# -*- coding: utf-8 -*-
# quantization.py is used to quantize the activation of model.
from __future__ import print_function, absolute_import

import torch
import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
import pickle
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
import pdb

class SigmoidT(torch.autograd.Function):
    """ sigmoid with temperature T for training
        we need the gradients for input and bias
        for customization of function, refer to https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(self, input, scales, n, b, T):        
        self.save_for_backward(input)
        self.T = T
        self.b = b
        self.scales = scales
        self.n = n

        buf = torch.clamp(self.T * (input - self.b[0]), min=-10.0, max=10.0)
        output = self.scales[0] / (1.0 + torch.exp(-buf))
        for k in range(1, self.n):
            buf = torch.clamp(self.T * (input - self.b[k]), min=-10.0, max=10.0)
            output += self.scales[k] / (1.0 + torch.exp(-buf)) 
        return output

    @staticmethod
    def backward(self, grad_output):
        # set T = 1 when train binary model in the backward.
        #self.T = 1
        input, = self.saved_tensors
        b_buf = torch.clamp(self.T * (input - self.b[0]), min=-10.0, max=10.0)
        b_output = self.scales[0] / (1.0 + torch.exp(-b_buf))
        temp = b_output * (1 - b_output) * self.T
        for j in range(1, self.n):        
            b_buf = torch.clamp(self.T * (input - self.b[j]), min=-10.0, max=10.0)
            b_output = self.scales[j] / (1.0 + torch.exp(-b_buf))
            temp += b_output * (1 - b_output) * self.T
        grad_input = Variable(temp) * grad_output      
        # corresponding to grad_input
        return grad_input, None, None, None, None

sigmoidT = SigmoidT.apply

def step(x, b):
    """ 
    The step function for ideal quantization function in test stage.
    """
    y = torch.zeros_like(x)
    mask = torch.gt(x - b,  0.0)
    y[mask] = 1.0
    return y


class Quantization(nn.Module):
    """ Quantization Activation
    Args:
       quant_values: the target quantized values, like [-4, -2, -1, 0, 1 , 2, 4]
       quan_bias and init_beta: the data for initialization of quantization parameters (biases, beta)
                  - for activations, format as `N x 1` for biases and `1x1` for (beta)
                    we need to obtain the intialization values for biases and beta offline

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Usage:
        - for activations, just pending this module to the activations when build the graph
    """

    def __init__(self, quant_values=[-1, 0, 1], quan_bias=[0], init_beta=0.0):
        super(Quantization, self).__init__()
        """register_parameter: params w/ grad, and need to be learned
            register_buffer: params w/o grad, do not need to be learned
            example shown in: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        self.values = quant_values
        # number of sigmoids
        self.n = len(self.values) - 1 
        self.alpha = Parameter(torch.Tensor([1]))
        self.beta = Parameter(torch.Tensor([1]))
        self.register_buffer('biases', torch.zeros(self.n))
        self.register_buffer('scales', torch.zeros(self.n))
          
        boundary = np.array(quan_bias)
        self.init_scale_and_offset()
        self.bias_inited = False
        self.alpha_beta_inited = False
        self.init_biases(boundary)
        self.init_alpha_and_beta(init_beta)

    def init_scale_and_offset(self):
        """
        Initialize the scale and offset of quantization function.
        """
        for i in range(self.n):
            gap = self.values[i + 1] - self.values[i]
            self.scales[i] = gap

    def init_biases(self, init_data):
        """
        Initialize the bias of quantization function.
        init_data in numpy format.
        """                    
        # activations initialization (obtained offline)
        assert init_data.size == self.n
        self.biases.copy_(torch.from_numpy(init_data))
        self.bias_inited = True
        #print('baises inited!!!')

    def init_alpha_and_beta(self, init_beta):
        """
        Initialize the alpha and beta of quantization function.
        init_data in numpy format.
        """
        # activations initialization (obtained offline)
        self.beta.data = torch.Tensor([init_beta]).cuda()
        self.alpha.data = torch.reciprocal(self.beta.data)
        self.alpha_beta_inited = True

    def forward(self, input, T=1):
        assert self.bias_inited 
        input = input.mul(self.beta)
        if self.training:
            assert self.alpha_beta_inited
            output = sigmoidT(input, self.scales, self.n, self.biases, T)
        else:
            output = step(input, b=self.biases[0])*self.scales[0]       
            for i in range(1, self.n):
                output += step(input, b=self.biases[i])*self.scales[i]
            
        output = output.mul(self.alpha)
        return output


