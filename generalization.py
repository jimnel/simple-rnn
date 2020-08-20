#!/usr/bin/env python3
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("modules")
import make_net
import make_data

plt.rcParams.update({'font.size': 13})

rnn = make_net.RNN(16)
rnn.load_state_dict(torch.load("net"))

L_range = np.arange(4, 20)
P = 100  # number of each class to generate
acc = []

for L in L_range:
    x = np.row_stack([[make_data.make_sample(L, j) for _ in range(P)] for j in range(3)])
    y = np.concatenate([i*np.ones(P, dtype=int) for i in range(3)])

    hidden = rnn.initHidden(3 * P)
    for l in range(L):
        xi = torch.tensor(x[:, l]).view(-1, 1).float()
        output, hidden = rnn(xi, hidden)
    pred = output.detach().numpy()
    pred = np.argmax(pred, 1)
    acc.append(np.mean(pred == y))

plt.figure("generalization")
plt.plot(L_range, acc)
plt.xlabel("L")
plt.ylabel("Acc.")
plt.axvline(x=6, ls='dashed', c='k')  # show's limit of where model was trained
plt.tight_layout()
plt.show()

