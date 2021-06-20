#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, embed_size, kernel_size=5):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=embed_size, out_channels=embed_size, kernel_size=kernel_size, bias=True)
    
    def forward(self, X_reshape):
        """
        :param X_reshape: (bz, embed_size, max_word_length)
        :return:
        """
        X_conv = self.conv1d(X_reshape)     # (bz, embed_size, max_word_length - kernel_size + 1)
        X_conv = F.relu(X_conv)
        X_conv_out, _ = torch.max(X_conv, dim=2)
        return X_conv_out                   # (bz, embed_size)

### END YOUR CODE

if __name__ == '__main__':
    embed_size = 64
    kernel_size = 5
    batch_size = 16
    max_word_length = 21
    X_reshape = torch.ones([batch_size, embed_size, max_word_length])
    model = CNN(embed_size, kernel_size)
    res = model(X_reshape)
    print(res.shape)

