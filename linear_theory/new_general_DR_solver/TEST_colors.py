# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:58:47 2020

@author: Yoshi
"""

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# n_waves = 10
# cvals   = np.arange(10)
# cmax    = 10.0
# 
# cvals[0] = 10.0
# 
# theta     = np.linspace(0, np.pi*8, 1000)
# sine_wave = np.sin(theta)
# 
# for ii, cval in zip(range(n_waves), cvals):
#     plt.plot(theta, sine_wave + 0.5*ii, c=plt.cm.viridis(cval/cmax))
# 
# sm   = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=cmax))
# cbar = plt.colorbar(sm)
# plt.show()
# =============================================================================

w_r = 2 * np.pi * 1.0
w_i = 3e-2
B0  = 1.0 

Nt    = 1000
t_max = 20.0
t     = np.linspace(0, t_max, Nt)


B = B0 * np.exp(1j * w_r * t) * np.exp(-1.0*w_i * t)

plt.plot(t, B)
plt.show()

