# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 15:41:38 2021

@author: Yoshi
"""
import numpy as np

N = 5

old_particles = np.zeros((9, N), dtype=np.float64)
pos           = np.random.normal(loc = 0.0, scale = 10., size=N)

old_particles[0] = pos; print(pos)       # Store positions
pos *= 0.0; print(pos)                       # Do operation on position array
pos[:] = old_particles[0]; print(pos)        # Restore positions