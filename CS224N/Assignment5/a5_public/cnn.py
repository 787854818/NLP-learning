#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, e_char, e_word, kernel_size=5):
        super().__init__()
        self.conv1d =nn.Conv1d(in_channels=e_char, out_channels=e_word, kernel_size=kernel_size)

    def forward(self, X_reshaped):          # X_reshaped: batch_size * e_char * m_word
        X_conv = self.conv1d(X_reshaped)    # batch_size * e_word * (m_word-k+1)
        X_conv = F.relu(X_conv)
        X_conv_out = torch.max(X_conv, dim=2)[0]   # batch_size * e_word ; 之所以要取[0]是因为max返回的是(values, indices)
        return X_conv_out
### END YOUR CODE


if __name__ == '__main__':
    e_char = 56
    e_word = 256
    m_word = 21
    k = 5
    batch_size = 64
    X_reshaped = torch.randn(batch_size, e_char, m_word )
    # test
    cnn_model = CNN(e_char, e_word, k)
    X_conv_out = cnn_model(X_reshaped)
    shape = X_conv_out.shape
    assert shape[0] == batch_size and shape[1] == e_word
    print("----------pass shape check-----------")
    print(X_conv_out.shape)

