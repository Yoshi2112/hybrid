# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:47:44 2020

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# full_N  = 1000000
# half_N  = 500000
# sigma   = 10
# 
# vbins = np.linspace(-3*sigma, 3*sigma, 101, endpoint=True)
# 
# vx_full =        np.random.normal(0, sigma, full_N)
# vx_half = np.abs(np.random.normal(0, sigma, half_N))
# 
# fig, ax = plt.subplots(1, sharex=True)
# ax.hist(vx_full, bins=vbins)
# ax.hist(vx_half, bins=vbins)   
# 
# =============================================================================
mu, sigma = 0, 0.1 # mean and standard deviation
Ns        = 10000000
s         = np.random.normal(mu, sigma, Ns)

count, bins, ignored = plt.hist(s, 30, density=True)
fs                   = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( -0.5* (bins - mu)**2 / sigma**2)

plt.plot(bins, fs, linewidth=2, color='r')
plt.show()