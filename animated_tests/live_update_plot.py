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
y       = np.zeros(n_vals)
fig, ax = plt.subplots()
line,   = ax.plot(x, y)
ax.set_ylim(0, 1)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

for ii in range(n_times):
    y = np.random.rand(n_vals)
    line.set_ydata(y)
    plt.pause(0.2)
    