#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
# 自己写的，可以参考a4的nmt_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    """
    Highway layer
    """
    def __init__(self, embed_size):
        """
        @param embed_size (int): Embedding size(dimensionality), e_word in pdf.
        """
        super(Highway, self).__init__()

        # 看了Linear的文档，Shape：Input(N, *, H_in), Output(N, *, H_out)
        # 说明Linear对有batch_size的Input同样只对最后一维做线性变换。
        self.x_projection = nn.Linear(embed_size, embed_size)
        self.x_gate = nn.Linear(embed_size, embed_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_conv_out):
        """
        @param X_conv_out(tensor(batch_size, embed_size)): X_conv_out in the pdf.
        @returns X_highway(tensor(batch_size, embed_size)): X_highway in the pdf.
        """

        X_proj = F.relu(self.x_projection(x_conv_out))
        X_gate = self.sigmoid(self.x_gate(x_conv_out))
        X_highway = X_gate * X_proj + (1 - X_gate) * x_conv_out
        return X_highway

### END YOUR CODE 

