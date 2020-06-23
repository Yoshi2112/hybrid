# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:58:47 2020

@author: Yoshi
"""

import matplotlib.pyplot as plt
import numpy as np

n_lines = 150
colors  = plt.cm.plasma(np.linspace(0, 1, n_lines))
x       = np.arange(10)

for ii in range(n_lines):
    plt.plot(x, x*ii, c=colors[ii])
