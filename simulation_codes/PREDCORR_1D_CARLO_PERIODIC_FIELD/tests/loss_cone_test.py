# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:51:20 2020

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

RE     = 6371e3                                     # Earth radius in m
B_E    = 3.12e-5                                    # Magnetic field strength at Earth equatorial surface

NX     = 4096                                       # Number of cells
dx     = 16107.                                     # Cell width in m
xmax   = NX * dx / 2                                # Distance from equator to simulation boundary, along field

anchor = 150e3                                      # Anchor point (Ionospheric) altitude in meters
L      = 5.35                                       # L value of field line to solve along

B_eq   = B_E / (L ** 3)                             # Magnetic field at equator, based on L value
lat_A  = np.arccos(np.sqrt((RE + anchor)/(RE*L)))   # Anchor latitude in radians
lat_L  = np.arccos(np.sqrt( 1.0 / L))               # Latitude of Earth's surface at this L
B_A    = B_eq * np.sqrt(4 - 3*np.cos(lat_A) ** 2) / (np.cos(lat_A) ** 6)
LC_eq  = np.arcsin(np.sqrt(B_eq / B_A))

dlam   = 1e-6                                       # Latitude increment in radians
fx_len = 0                                          # Length of field from equator to ii*dlam
lam    = 0.0
ii     = 1
while fx_len < xmax:
    lam_i   = dlam * ii                                                             # Current latitude
    d_len   = L * RE * np.cos(lam_i) * np.sqrt(4 - 3*np.cos(lam_i) ** 2) * dlam     # Length increment
    fx_len += d_len                                                                 # Accrue arclength
    ii     += 1                                                                     # Increment count
   
theta_x = lam_i
B_xmax  = (B_E / L ** 3) * np.sqrt(4 - 3*np.cos(theta_x) ** 2) / np.cos(theta_x) ** 6
    
    
print('Length of simulation : {} m  '.format(xmax))
print('Length along field   : {} m  '.format(fx_len))
print('Latitude at sim end  : {} deg'.format(lam_i*180./np.pi))
print('B at simulation end  : {} nT'.format(B_xmax*1e9))


# =============================================================================
# # Look at loss cone shift through latitude along L
# dl     = 0.000001
# lmb    = np.arange(0, lat_A+dl, dl)
# lmb_B  = (B_E / L ** 3) * np.sqrt(4 - 3*np.cos(lmb) ** 2) / np.cos(lmb) ** 6
# lmb_LC = np.arcsin(np.sqrt(lmb_B / B_A))
# 
# plt.plot(lmb*180./np.pi, lmb_LC*180./np.pi, marker='o')
# plt.xlabel('Latitude')
# plt.ylabel('Loss Cone')
# 
# =============================================================================
    
