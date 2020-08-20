#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("modules")
import make_net
import make_data

rnn = make_net.RNN(16)
op = optim.SGD(rnn.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

"""
We train on samples with length 4, 5 and 6
And validation on samles with length 7
"""

L_range = [4, 6]
n_it = 300  # number of iterations
P_mini = 50
b3 = 32

y = np.concatenate([i*np.ones(b3, dtype=int) for i in range(3)])
y = torch.from_numpy(y).long()

P_val = 100
L_val = 7
x_val = np.row_stack([[make_data.make_sample(L_val, j) for _ in range(P_val)] for j in range(3)])
y_val = np.concatenate([i*np.ones(P_val, dtype=int) for i in range(3)])

"""
As we are always generating new data there is no need to use early stopping or any
kind of regularization
"""

hist = np.zeros((n_it, 2))  # first component is training loss, second is accuracy on L=7
for i in range(n_it):
    loss_mini = np.zeros(P_mini)
    for ii in range(P_mini):
        L = np.random.randint(L_range[0], L_range[1]+1)
        X = np.row_stack([[make_data.make_sample(L, j) for _ in range(b3)] for j in range(3)])

        rnn.zero_grad()
        hidden = rnn.initHidden(3*b3)
        for l in range(L):
            xi = torch.tensor(X[:, l]).view(-1, 1).float()
            output, hidden = rnn(xi, hidden)

        loss = criterion(output, y)
        loss.backward()
        op.step()
        loss_mini[ii] = loss.item()

    hist[i, 0] = loss_mini.mean()

    hidden = rnn.initHidden(3 * P_val)
    for l in range(L_val):
        xi = torch.tensor(x_val[:, l]).view(-1, 1).float()
        output_val, hidden = rnn(xi, hidden)

    pred_val = output_val.detach().numpy()
    pred_val = np.argmax(pred_val, 1)
    hist[i, 1] = np.mean(pred_val == y_val)
    print(i, L, hist[i])


torch.save(rnn.state_dict(), "net")
np.save("hist.npy", hist)

