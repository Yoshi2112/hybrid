# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import numba as nb


@nb.njit()
def position_advance(P, V, DT):
    P += V * DT
    return P


if __name__ == '__main__':
    N = 1000000

    x_min = 0
    x_max = 10
    dt    = 0.1

    pos = np.random.uniform(x_min, x_max, N)
    vel = np.random.normal(0, 0.2, N)
    
    pos = position_advance(pos, vel, dt) 
