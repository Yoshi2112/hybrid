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
mp     = 1.672622e-27               # Mass of proton (kg)
kB     = 1.380649e-23               # Boltzmann's Constant (J/K)
mu0    = (4e-7) * np.pi             # Magnetic Permeability of Free Space (SI units)

## INPUTS ##
T      = 100.0                      # Temperature in eV
ni     = 200e6                      # Ion density
B      = 200e-9                     # Local magnetic field
N      = 500                        # Number of solutions/samples
v_mag  = 2                          # Limits on magnitude of speed (as multiples of Alfven speed)


## CALCULATED QUANTITES ## 
vth = np.sqrt(kB * T * 11603. / mp)        # Thermal velocity (m/s)
va  = B / np.sqrt(mu0 * ni * mp)           # Alfven speed     (m/s)
v   = np.linspace(-v_mag, v_mag, N)*va     # Velocity space   (m/s)

## Calculate analytic distribution ##
psd_value  = np.zeros(v.shape[0], dtype=np.float64)
outer_term = ni / (vth * np.sqrt(2*np.pi))
for ii in range(N):
    psd_value[ii] = outer_term * np.exp(- 0.5 * (v[ii] / vth) ** 2)

## Do plot ##
plt.figure()
plt.plot(v, psd_value, label='Analytic')

if True:
    # Set up and plot random distribution      
    psd_random        = np.random.normal(0, vth, int(ni))
    counts, bin_edges = np.histogram(psd_random, bins=N)
    bin_centers       = 0.5 * (bin_edges[1:] + bin_edges[:-1])   
    plt.plot(bin_centers, counts, '-', drawstyle='steps', label='Random')

plt.xlabel(r'$v$')
plt.ylabel(r'$n_i$')
plt.legend()
plt.show()
