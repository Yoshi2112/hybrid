# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:26:31 2020

@author: iarey
"""
import numpy as np
import matplotlib.pyplot as plt

## Constants ##
c      = 2.998925e+08               # Speed of light (m/s)
mp     = 1.672622e-27               # Mass of proton (kg)
kB     = 1.380649e-23               # Boltzmann's Constant (J/K)
mu0    = (4e-7) * np.pi             # Magnetic Permeability of Free Space (SI units)


## INPUTS ##
T_para = 100.0
T_perp = 100.0                      # Temperature in eV
ni     = 200e6                      # Ion density
B      = 200e-9                     # Local magnetic field
N      = 500

v_drift_para = 0.0                  # Parallel      Drift in multiples of alfven velocity
v_drift_perp = 0.0                  # Perpendicular Drift in multiples of alfven velocity

v_para_max  =  2
v_para_min  = -2
v_para      = np.linspace(v_para_min, v_para_max, N)

v_perp_max  =  2
v_perp_min  = -2
v_perp      = np.linspace(v_perp_min, v_perp_max, N)


## CALCULATED QUANTITES ## 
vth_para = np.sqrt(kB * T_para * 11603. / mp)        # Thermal velocity
vth_perp = np.sqrt(kB * T_perp * 11603. / mp)        # Thermal velocity
va       = B / np.sqrt(mu0 * ni * mp)                # Alfven speed (m/s)
vdx       = v_drift_para * va                        # Para drift  speed (m/s)
vdy       = v_drift_perp * va                        # Perp drift  speed (m/s)

## CALCULATE PHASE SPACE ##
psd_value = np.zeros((v_para.shape[0], v_perp.shape[0]), dtype=np.float64)

for ii in range(N):
    for jj in range(N):
        vx = v_para[ii] * va
        vy = v_perp[jj] * va
        
        factor = ni / np.sqrt(np.pi ** 3)
        
        outer_para = 1. / np.sqrt(np.sqrt(2) * vth_perp)
        expon_para = np.exp(- 0.5 * (vx - vdx)**2 / vth_para**2)
        
        outer_perp = 1. / (np.sqrt(2) * vth_perp)
        expon_perp = np.exp(- 0.5 * (vy - vdy)**2 / vth_perp**2)

        psd_value[ii, jj] = factor * outer_para * expon_para * outer_perp * expon_perp
    
        
plt.figure()
plt.pcolormesh(v_para, v_perp, psd_value.T, cmap='jet')
plt.axis('equal')
plt.title('Anisotropic Bimaxwellian Distribution')
plt.ylabel('$v_\perp (v_A^{-1})$',     rotation=0, fontsize=14, labelpad=30)
plt.xlabel('$v_\parallel (v_A^{-1})$', rotation=0, fontsize=14)
plt.xlim(v_para_min, v_para_max)
plt.ylim(v_perp_min, v_perp_max)
plt.colorbar().set_label('Number density ($m^{-3}$')
