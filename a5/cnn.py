#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    """
    CNN layer.
    """
    def __init__(self, e_char, m_word, filters, kernel_size=5):
        """
        @param e_char: character embedding dimensions, e_char in the pdf.
        @param m_word: maximum word length, m_word in the pdf.
        @param filters: the number of filters, which is set to be e_word in the pdf.
        @param kernel_size: the kernel size, which is set to be 5 in the pdf.
        """
        super(CNN, self).__init__()
        self.filters = filters

        # default values
        self.x_convolution = None

        self.x_convolution = nn.Conv1d(in_channels=e_char, out_channels=filters, kernel_size=kernel_size)
        self.x_maxpooling = nn.MaxPool1d(m_word - kernel_size + 1)

    def forward(self, x_reshaped):
        """
        @param x_reshaped: input of the CNN layer, X_reshaped in the pdf.
        @returns x_conv_out: output of the CNN layer, X_conv_out in the pdf.
        """
        # x_conv: (-1, embed_size, max_word_length - kernel_size + 1)
        x_conv = self.x_convolution(x_reshaped)
        # x_conv_out: (-1, embed_size, 1)
        x_conv_out = self.x_maxpooling(F.relu(x_conv))
        # 改变shape，使x_conv_out: (-1, embed_size)
        x_conv_out = x_conv_out.reshape(-1, self.filters)

        return x_conv_out

### END YOUR CODE

