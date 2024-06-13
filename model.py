#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx



def get_graph_feature(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(x[:, :3, :], k=k)  # Perform KNN using XYZ data only

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.reshape(batch_size * num_points, -1)
    feature = torch.index_select(x, 0, idx)
    feature = feature.reshape(batch_size, num_points, k, num_dims)

    x_self = x.reshape(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature_diff = feature - x_self

    feature_combined = torch.cat(
        (feature[:, :, :, :3],  # Original XYZ
         feature_diff[:, :, :, :3],  # Difference XYZ
         feature[:, :, :, 3:4],  # Original intensity
         feature_diff[:, :, :, 3:4]),  # Difference intensity
        dim=3
    )
    feature = feature_combined.permute(0, 3, 1, 2)

    return feature


def get_graph_feature2(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(x[:, :3, :], k=k)  # Perform KNN using XYZ data only

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.reshape(batch_size * num_points, -1)
    feature = torch.index_select(x, 0, idx)
    feature = feature.reshape(batch_size, num_points, k, num_dims)

    x_self = x.reshape(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature_diff = feature - x_self

    feature_combined = torch.cat(
        (feature[:, :, :, :3],  # Original XYZ
         feature_diff[:, :, :, :3],  # Difference XYZ
         feature[:, :, :, 3:4],  # Original intensity
         feature_diff[:, :, :, 3:4]),  # Difference intensity
        dim=3
    )

    feature_combined = feature_combined.repeat(1, 1, 1, 16)[:, :, :, :128]

    feature = feature_combined.permute(0, 3, 1, 2)

    return feature


"""

def get_graph_feature(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.size()
    #print("Input feature shape:", x.shape)

    if idx is None:
        idx = knn(x, k=k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.reshape(batch_size * num_points, -1)
    feature = torch.index_select(x, 0, idx)
    feature = feature.reshape(batch_size, num_points, k, num_dims)
    #print("Feature shape after knn and selection:", feature.shape)

    x_self = x.reshape(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature_diff = feature - x_self

    feature_combined = torch.cat((feature, feature_diff), dim=3)
    #print("Feature shape after combining with self-feature:", feature_combined.shape)

    # Select the first 8 channels to form the shape [batch_size, num_points, k, 8]
    feature = feature_combined.permute(0, 3, 1, 2)  # [batch_size, num_dims * 2, num_points, k]
    feature = feature[:, :8, :, :]  # Adjust the number of channels by selecting only the first 8 channels
    #print("Final feature map shape:", feature.shape)

    return feature


def get_graph_feature2(x, k=20, idx=None): # Original function
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.reshape(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.reshape(batch_size * num_points, -1)[idx, :]
    feature = feature.reshape(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

"""

class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(4, 64, kernel_size=1, bias=False)  # Change the input channels from 3 to 4
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        #self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        #self.conv1 = nn.Sequential(nn.Conv2d(8, 64, kernel_size=1, bias=False),
                                   #self.bn1,
                                   #nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(8, 128, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        #self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   #self.bn4,
                                   #nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        #self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   #self.bn5,
                                   #nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(576, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        #x = checkpoint(get_graph_feature, x, self.k)
        #print("After get_graph_feature:", x.shape)
        x = self.conv1(x)
        #print("After conv1:", x.shape)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature2(x1, k=self.k)
        #print("After get_graph_feature:", x.shape)
        #x = checkpoint(get_graph_feature2, x1, self.k)
        x = self.conv2(x)
        #print("After conv2:", x.shape)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature2(x2, k=self.k)
        #print("After get_graph_feature:", x.shape)
        #x = checkpoint(get_graph_feature2, x2, self.k)
        x = self.conv3(x)
        #print("After conv3:", x.shape)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature2(x3, k=self.k)
        #print("After get_graph_feature:", x.shape)
        #x = checkpoint(get_graph_feature2, x3, self.k)
        x = self.conv4(x)
        #print("After conv4:", x.shape)
        x4 = x.max(dim=-1, keepdim=False)[0]
        #print("Before torch.cat:", x.shape)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        #print("After torch.cat:", x.shape)
        x = self.conv5(x)
        #print("After conv5:", x.shape)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        #print("Before final linear layer:", x.shape)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        # print("Final output:", x.shape)
        return x