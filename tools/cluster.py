#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cluster.py is used to get the bias(b_i) of quantization function.

from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import os

import sys
import pdb

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def params_cluster(params, Q_values):
    # print("The max and min values of params: ", params.max(), params.min())
    # print("The shape of params: ", params.shape)

    max_value = abs(params).max().tolist()
    # print("max_abs_value: ", max_value)

    quan_values = Q_values
    threshold = quan_values[-1]*5/4.0
    # print("scale threshold: ", threshold)
    pre_params = np.sort(params.reshape(-1, 1), axis = 0)
    pre_params = pre_params* (threshold/max_value)

    #  cluster 
    n_clusters = len(quan_values)
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(pre_params)
    label_pred = estimator.labels_ 
    centroids = estimator.cluster_centers_ 

    #print("cluster_centers: ", centroids)
    #print("label_pred: ", label_pred)

    temp = label_pred[0]
    saved_index = [0]*(n_clusters - 1)
    j = 0
    for index, i in enumerate(label_pred):
        if i != temp:
            saved_index[j] = index
            j += 1
            temp = i
            
    # print("boundary_index: ", saved_index)

    # print(pre_params[saved_index[0]-1], pre_params[saved_index[0]])
    # print(pre_params[saved_index[1]-1], pre_params[saved_index[1]])

    boundary = [0]*(n_clusters - 1)
    for i in range(n_clusters - 1):
        temp = (pre_params[saved_index[i] - 1] + pre_params[saved_index[i]]) / 2
        boundary[i] = temp.tolist()[0]
    # print("boundary: ", boundary)
    return boundary

def main(args):
    Q_values = [-4, -2, -1, 0, 1, 2, 4]
    #Q_values = [-2, -1, 0, 1, 2]
    #Q_values = [-1, 0, 1]
    

    all_file = sorted(os.listdir(args.root))
    for filename in all_file:
        if '.npy' in filename:
            params_road = osp.join(args.root, filename)
            params = np.load(params_road)
            boundary = params_cluster(params, Q_values)
            print(filename, boundary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameter cluster")
    #file road
    parser.add_argument('-r', '--root', type=str, default=".")

    main(parser.parse_args())












