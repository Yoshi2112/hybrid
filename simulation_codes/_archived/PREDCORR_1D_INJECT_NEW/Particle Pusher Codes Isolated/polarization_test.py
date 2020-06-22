# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:33:06 2020

@author: Yoshi
"""

import numpy as np

t_max = 1000        # Seconds
x_max = 1000        # Meters

dx = 0.1
dt = 0.1
k  = 1.
w  = 1.

x = np.arange(0, x_max, dx)
t = np.arange(0, t_max, dt)



