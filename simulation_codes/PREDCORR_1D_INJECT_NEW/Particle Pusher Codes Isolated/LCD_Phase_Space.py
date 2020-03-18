# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:26:31 2020

@author: iarey
"""
import numpy as np
import matplotlib.pyplot as plt

## Constants ##
q      = 1.602177e-19                       # Elementary charge (C)
c      = 2.998925e+08                       # Speed of light (m/s)
mp     = 1.672622e-27                       # Mass of proton (kg)
me     = 9.109384e-31                       # Mass of electron (kg)
kB     = 1.380649e-23                       # Boltzmann's Constant (J/K)
e0     = 8.854188e-12                       # Epsilon naught - permittivity of free space
mu0    = (4e-7) * np.pi                     # Magnetic Permeability of Free Space (SI units)
RE     = 6.371e6                            # Earth radius in metres
B_surf = 3.12e-5                            # Magnetic field strength at Earth surface



## INPUTS ##
T_perp = 1.                         # Temperatures in eV
T_para = 1.                         #
ni     = 1e6                        # Ion density
delta  = 0.0                        # Loss cone filling parameter
B      = 200e-9                     # Local magnetic field
va     = B / np.sqrt(mu0 * ni * mp) # Alfven speed 

v_para_max =  10
v_para_min = -10
v_para_N   = 100
v_para     = np.linspace(v_para_min, v_para_max, v_para_N)

v_perp_max =  10
v_perp_min = -10
v_perp_N   = 100
v_perp     = np.linspace(v_perp_min, v_perp_max, v_perp_N)


## CALCULATED QUANTITES ## 
T        = T_perp + T_para
beta     = 2 * mu0 * ni * kB * T / B**2
vth_perp = np.sqrt(kB * T_perp * 11603. / mp)
vth_para = np.sqrt(kB * T_para * 11603. / mp)

## CALCULATE PHASE SPACE ##
psd_value = np.zeros((v_para.shape[0], v_perp.shape[0]), dtype=np.float64)

for ii in range(v_para_N):
    for jj in range(v_perp_N):
        vx = v_para[ii]
        vy = v_perp[jj]
        
        f_outer = ni / ((2*np.pi) ** 3 * vth_para * vth_perp ** 2)
        f_expon = np.exp(- vx ** 2 / (2 * vth_para))
        f_term  = f_outer * f_expon
        
        g_first = delta * np.exp(- vy ** 2 / (2 * vth_perp**2))
        g_outer = (1. - delta) / (1 - beta)
        g_secA  = np.exp(- vy ** 2 / (2 * vth_perp ** 2))
        g_secB  = np.exp(- vy ** 2 / (2 * vth_perp ** 2 * beta))
        g_term  = g_first + g_outer * (g_secA - g_secB)
        
        psd_value[ii, jj] = f_term * g_term