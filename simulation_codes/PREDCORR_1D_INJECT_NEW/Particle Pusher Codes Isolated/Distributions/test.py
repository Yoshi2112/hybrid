# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:38:16 2020

@author: Yoshi
"""
import numpy  as np
import matplotlib.pyplot as plt

# =============================================================================
# mu, sigma = 0, 0.1
# s = np.random.normal(mu, sigma, 1000)
# 
# count, bins, ignored = plt.hist(s, 30, density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
#           linewidth=2, color='r')
# plt.show()
# =============================================================================

lim       = 10
angle_deg = 90 - 26.6
angle_rad = angle_deg * np.pi / 180.



m = np.tan(angle_rad)
x = np.linspace(-lim, lim, 10*lim, endpoint=True)
y = m*x

plt.plot(x, y)

plt.axvline(0, c='k', alpha=0.5)
plt.axhline(0, c='k', alpha=0.5)

plt.xlim(-lim, lim)
plt.ylim(-lim, lim)