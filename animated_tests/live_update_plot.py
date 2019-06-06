# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:02:07 2019

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

n_vals  = 10
n_times = 20
x       = np.arange(n_vals)
fig, ax = plt.subplots()

for ii in range(n_times):
    ax.clear()
    y = np.random.rand(n_vals)
    ax.scatter(x, y)
    ax.set_ylim(0, 1)
    plt.pause(0.2)
    