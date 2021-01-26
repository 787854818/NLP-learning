#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
from torch.nn.functional import relu
### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, e_word):
        super().__init__()
        self.w_projection = nn.Linear(e_word, e_word, bias=True)
        self.gate_projection = nn.Linear(e_word, e_word, bias=True)
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X_conv_out):
        X_proj = relu(self.w_projection(X_conv_out))        # batch_size * e_word
        X_gate = torch.sigmoid(self.gate_projection(X_proj))
        X_highway = X_gate*X_proj + (1-X_gate)*X_conv_out
        # X_word_emb = self.dropout(X_highway)
        return X_highway


### END YOUR CODE 

