# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:26:31 2020

@author: iarey
"""
import numpy as np
import matplotlib.pyplot as plt

'''
1D Maxwellian at equilibrium is equivalent to a normal distribution (i.e. varying only with positive/negative v)
'''

## Constants ##
c      = 2.998925e+08               # Speed of light (m/s)
mp     = 1.672622e-27               # Mass of proton (kg)
kB     = 1.380649e-23               # Boltzmann's Constant (J/K)
mu0    = (4e-7) * np.pi             # Magnetic Permeability of Free Space (SI units)


## INPUTS ##
T      = 1.0                        # Temperature in eV
ni     = 200e6                      # Ion density
B      = 200e-9                     # Local magnetic field
N      = 1000

v_max  =  1
v_min  = -1
v      = np.linspace(v_min, v_max, N)


## CALCULATED QUANTITES ## 
vth = np.sqrt(kB * T * 11603. / mp)        # Thermal velocity
va  = B / np.sqrt(mu0 * ni * mp)           # Alfven speed

## CALCULATE PHASE SPACE ##
psd_value = np.zeros(v.shape[0], dtype=np.float64)

for ii in range(N):
    vi = v[ii] * va
    
    outer = (ni / vth) * (1 / (2*np.pi)) ** (3/2)
    expon = np.exp(- 0.5 * (vi / vth) ** 2)
    
    psd_value[ii]  = outer * expon
    
        
plt.figure()
plt.plot(v, psd_value)
