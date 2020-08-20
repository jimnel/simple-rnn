#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 13})

hist = np.load("hist.npy")
P = hist.shape[0]

fig, axs = plt.subplots(nrows=2, sharex='col', num='history')
axs[0].plot(np.arange(1, 1+P), hist[:, 0])
axs[0].set_ylabel('Train Error')

axs[1].plot(np.arange(1, 1+P), hist[:, 1])
axs[1].set_ylabel('Val Acc.')

axs[1].axhline(y=1./3., linestyle='dashed', color='k')  # this line corresponds to random guessing acc.
axs[1].set_xlabel('Epochs')

axs[1].set_ylim(0.2, 1.0)
fig.tight_layout()

plt.show()

