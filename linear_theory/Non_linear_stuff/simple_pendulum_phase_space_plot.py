# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:56:57 2021

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt

# To Do:
# Create basic phase space plot for pendulum
# Then try to create the one from Omura et al. (2010)

dt      = 1.0
th_max  = 8.0
dth_max = 3.0
Nth     = 150
Ndth    = 75

theta   = np.linspace(-th_max, th_max, Nth)  
dtheta  = np.linspace(-dth_max, dth_max, Ndth)

fig, ax = plt.subplots()

th, dth = np.meshgrid(theta, dtheta)

# Calculate arrow directions and energy of each configuration
U = np.zeros(th.shape)
V = np.zeros(th.shape)
E = np.zeros(th.shape)

for ii in range(th.shape[0]):
    for jj in range(th.shape[1]):
        # Calculate point after dt
        new_x = th[ ii, jj] + dt * dth[ii, jj]
        new_y = dth[ii, jj] - dt * np.sin(th[ii, jj])
        
        # Calculate arrow vector (difference between current, next points)
        U[ii, jj] = new_x - th[ii, jj]
        V[ii, jj] = new_y - dth[ii, jj]
        
        # Calculate energy (Hamiltonian)
        E[ii, jj] = 0.5 * dth[ii, jj] ** 2 - np.cos(th[ii, jj])


# Plot points, arrows and contours of constant energy
norm_U = U / np.sqrt(U ** 2 + V ** 2)
norm_V = V / np.sqrt(U ** 2 + V ** 2)

ax.quiver(th, dth, norm_U, norm_V, 
          headlength=6.0, headwidth=2.0)
    
ax.set_xlim(-th_max, th_max)
ax.set_ylim(-dth_max, dth_max)

ax.contour(th, dth, E, levels=15)