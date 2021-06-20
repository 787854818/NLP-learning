#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch
### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, word_embed_size):
        super(Highway, self).__init__()
        self.projection = nn.Linear(in_features=word_embed_size, out_features=word_embed_size, bias=True)
        self.gate = nn.Linear(in_features=word_embed_size, out_features=word_embed_size, bias=True)

    def forward(self, X_conv_out):
        """
        :param X_conv_out: (bz, word_embed_size)
        :return: X_highway: (bz, word_embed_size
        """
        X_proj = torch.relu(self.projection(X_conv_out))    # (bz, word_embed_size)
        X_gate = torch.sigmoid(self.gate(X_conv_out))       # (bz, word_embed_size)
        X_highway = torch.mul(X_gate, X_proj) + torch.mul(torch.add(-X_gate, 1), X_conv_out)        # (bz, word_embed_size)

        return X_highway


### END YOUR CODE

if __name__ == '__main__':
    embed_size = 64
    batch_size = 16
    X_test = torch.ones([batch_size, embed_size])

    model = Highway(embed_size)
    X_highway = model(X_test)
    print(X_highway.shape)



