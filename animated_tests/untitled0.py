# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:07:31 2019

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.axis([0, 10, 0, 1])

for i in range(10):
    y = np.random.random()
    #ax.scatter(i, y)
    fig.set_xdata(i)
    fig.set_ydata(y)
    plt.pause(0.5)
    

plt.show()