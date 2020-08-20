#!/usr/bin/env python3
import numpy as np

def make_sample(L, C):
    """
    Input:
        L - the length of the desired series
        C - the class
            0 = ascending
            1 = descending
            2 = neither
    Output:
        A single data point of class C with length L
    """

    tmp = np.random.rand(L)

    if C == 0:
        # if ascending, sort
        return np.sort(tmp)
    elif C == 1:
        # if descending sort
        return np.sort(tmp)[::-1]
    else:
        # if neither randomly generate it and check it isn't class 0 or 1
        a1 = np.sort(tmp)
        d1 = ((tmp - a1)**2).mean()

        a2 = np.sort(tmp)[::-1]
        d2 = ((tmp - a2)**2).mean()
        if (d1 > 1e-6) and (d2 > 1e-6):
            return tmp
        else:
            return make_sample(L, C)
