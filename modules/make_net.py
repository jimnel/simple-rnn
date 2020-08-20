#!/usr/bin/env python3
import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    Class for a simple RNN with hidden state size hidden_size
    The RNN is deep with a 32 layer for both the hidden state and the output
    """

    def __init__(self, hidden_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.h_fc1 = nn.Linear(1 + hidden_size, 32)
        self.h_fc2 = nn.Linear(32, hidden_size)

        self.o_fc1 = nn.Linear(1 + hidden_size, 32)
        self.o_fc2 = nn.Linear(32, 3)

        self.act = nn.ReLU()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)

        hidden = self.act(self.h_fc1(combined))
        hidden = self.h_fc2(hidden)

        output = self.act(self.o_fc1(combined))
        output = self.o_fc2(output)

        return output, hidden

    def initHidden(self, p):
        return torch.zeros(p, self.hidden_size)
