# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:26:31 2020

@author: iarey
"""
'''
Try making a loss cone distribution using Maxwellians (just to see if I can)
THEN
Just use rejection method on alpha
'''
import numpy as np
import matplotlib.pyplot as plt

## Constants ##
c      = 2.998925e+08               # Speed of light (m/s)
mp     = 1.672622e-27               # Mass of proton (kg)
kB     = 1.380649e-23               # Boltzmann's Constant (J/K)
mu0    = (4e-7) * np.pi             # Magnetic Permeability of Free Space (SI units)

A      = 0                          # Anisotropy : T_perp / T_parallel - 1
T_para = 100.0                      # Parallel temperature in eV
ni     = 200e6                      # Ion density
B      = 200e-9                     # Local magnetic field
N      = 100000                     # Number of particles to simulate
delta  = 1.0                        # Loss Cone Fullness Parameter (0 = Empty, 1 = Full/Maxwellian)

## CALCULATED QUANTITES ## 
T_perp = T_para*(A + 1)                             # Perpendicular temperature in eV
vth_para = np.sqrt(kB * T_para * 11603. / mp)       # Thermal velocity
vth_perp = np.sqrt(kB * T_perp * 11603. / mp)       # Thermal velocity
va       = B / np.sqrt(mu0 * ni * mp)               # Alfven speed (m/s)
T        = (T_perp + T_para) * 11603.               # Total plasma temperature
beta     = 2 * mu0 * ni * kB * T / B ** 2           # Plasma beta

v_parallel = np.random.normal(0., vth_para, N)      #
v_y        = np.random.normal(0., vth_perp**2, N)   # Should this be squared? Cuz its M(vy)M(vz). 
v_z        = np.random.normal(0., vth_perp**2, N)   # Also, should I generate again? Or use twice? Try both
v_ybet     = beta*np.random.normal(0., beta*vth_perp**2, N)
v_zbet     = beta*np.random.normal(0., beta*vth_perp**2, N)
    
coeff      = (1. - delta) / (1. - beta) 
psd_perp   = delta * v_y*v_z + coeff * (v_y*v_z - v_ybet*v_zbet)

plt.scatter(psd_perp/va, v_parallel/va, s=1)
    