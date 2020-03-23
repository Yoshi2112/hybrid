# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:26:31 2020

@author: iarey
"""
'''
Use rejection method on alpha
'''
import numpy as np
import matplotlib.pyplot as plt
import pdb

def calc_losses(vx, vy, vz):
    N_loss = 0; loss_idx = []
    
    v_perp = np.sqrt(vy ** 2 + vz ** 2)
    alpha  = np.arctan(v_perp / vx) * 180. / np.pi
    
    in_loss_cone = (abs(alpha) < loss_cone)
    N_loss       = in_loss_cone.sum()
    loss_idx     = np.where(in_loss_cone == True)[0]

    print('{} particles in loss cone'.format(N_loss))
    return N_loss, loss_idx


def init_LCD():
    # Initial guesses
    vx      = np.random.normal(0., vth_para, N)        # Velocity components (normals)
    vy      = np.random.normal(0., vth_perp, N)        # Velocity components (normals)
    vz      = np.random.normal(0., vth_perp, N)        # Velocity components (normals)
    
    Nl, idxs = calc_losses(vx, vy, vz)
    
    # Remove any not in loss cone and reinitialize
    while Nl > 0:
        vx[idxs] = np.random.normal(0., vth_para, Nl)
        vy[idxs] = np.random.normal(0., vth_perp, Nl)
        vz[idxs] = np.random.normal(0., vth_perp, Nl)
        
        Nl, idxs = calc_losses(vx, vy, vz)
    return vx, vy, vz


## Constants ##
c      = 2.998925e+08               # Speed of light (m/s)
mp     = 1.672622e-27               # Mass of proton (kg)
kB     = 1.380649e-23               # Boltzmann's Constant (J/K)
mu0    = (4e-7) * np.pi             # Magnetic Permeability of Free Space (SI units)

A      = 1                          # Anisotropy : T_perp / T_parallel - 1
T_para = 30000.0                    # Parallel temperature in eV
ni     = 200e6                      # Ion density
B_eq   = 200e-9                     # Local magnetic field
B_xmax = 800e-9

N      = 1000000                    # Number of particles to simulate
delta  = 1.0                        # Loss Cone Fullness Parameter (0 = Empty, 1 = Full/Maxwellian)



## CALCULATED QUANTITES ## 
T_perp   = T_para*(A + 1)                           # Perpendicular temperature in eV
vth_para = np.sqrt(kB * T_para * 11603. / mp)       # Thermal velocity
vth_perp = np.sqrt(kB * T_perp * 11603. / mp)       # Thermal velocity
va       = B_eq / np.sqrt(mu0 * ni * mp)            # Alfven speed (m/s)
loss_cone= np.arcsin(np.sqrt(B_eq / B_xmax))*180 / np.pi
print(loss_cone)
# =============================================================================
# vx, vy, vz = init_LCD()
# 
# vperp      = np.sign(vz) * np.sqrt(vy ** 2 + vz ** 2)
# 
# plt.scatter(vperp/va, vx/va, s=1)
# plt.title('Loss Cone Distribution, sorta')
# plt.xlabel('$v_\perp$')
# plt.ylabel('$v_\parallel$')
# plt.axis('equal')
# =============================================================================
    
