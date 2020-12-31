# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 21:39:40 2020

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

times = np.arange(np.datetime64('2020-12-01'), np.datetime64('2020-12-31'))
yaxis = np.arange(0, 20)
caxis = np.random.normal(0.0, 1.0, (times.shape[0], yaxis.shape[0]))

plt.pcolormesh(times, yaxis, caxis.T, shading='nearest')
plt.show()