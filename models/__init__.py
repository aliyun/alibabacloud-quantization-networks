#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .alexnet import *
from .alexnet_all import *
from .resnet import *
from .resnet18_all import *

__factory = {
    'alexnet': alexnet,
    'alexnet_q': alexnet_q,
    'resnet18': resnet18,
    'resnet18_q': resnet18_q,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
