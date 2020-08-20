#!/usr/bin/env python3
import sys
sys.path.append("modules")
import make_data
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

L = 10
M = 5000
comp = 2

x0 = []
labels=["A", "D", "N"]
for c in range(3):
    x = [make_data.make_sample(L, c) for _ in range(M)]
    x0.append([q[comp-1] for q in x])

plt.figure("eda")
plt.hist(x0, histtype='step', bins=40, label=labels)
plt.legend(loc="upper center")
plt.ylabel("Counts")
plt.xlabel("Value of %i component (L=%i)" % (comp, L))

plt.tight_layout()

plt.show()
