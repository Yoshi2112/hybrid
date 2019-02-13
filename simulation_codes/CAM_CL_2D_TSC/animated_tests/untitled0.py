# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:07:31 2019

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 10, 0, 1])

for i in range(10):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.5)

plt.show()