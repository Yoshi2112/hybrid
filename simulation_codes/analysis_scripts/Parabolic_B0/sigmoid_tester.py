# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:40:07 2022

@author: Yoshi
"""

''' Test to see if I can use a sigmoid to change the background field from
some value B0 to some other value B0 +- A0

Just need some way to change/control the rate at which the function varies,
so it is comparable to the dB/dt of a ULF wave of between 2-30 mHz

Sigmoid function:

    w(t) = w_max / (1 + exp(-k(t - t_max)))
    
k is a proxy for the growth rate
w_max is the final value for w
t_max is the end time (symmetric around zero?)
'''
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# # Time variables in seconds
# dt    = 3e-3
# t_max = 160.
# t     = np.arange(0.0, t_max, dt)
# 
# B_max = 200.0
# k     = 10e-1
# Bt    = B_max / (1 + np.exp(-k*(t - t_max)))
# 
# plt.figure()
# plt.plot(t, Bt)
# =============================================================================

# Online example
# For f(x) = L / (1 + e ** (-k*(x - x0)))

L  = -5.0   # Maximum value, proxy for amplitude. Sign determines if it's an increase or decrease
k  = 1.0   # Steepness of curve, higher is steeper
x0 = 10.0    # x-value of midpoint. Approximate point at which the change occurs. (Symmetry)
x  = np.linspace(0.0, 20.0, 1000)

offset = 200.0   # Proxy for B0

func = L / (1 + np.exp(-k*(x - x0))) + offset

plt.plot(x, func)

