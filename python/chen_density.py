# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:38:12 2017

@author: c3134027
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

def Y(x, x0, dx):
    return (0.5 * np.tanh(3.4535 * ((x - x0) / dx)) + 0.5)

## ------ DEFINE THINGS ------ ##
MLAT = 10 * pi / 180            # Magnetic Latitude in Radians
MLT  = 0                        # Magnetic Local Time (longitude)

RE   = 6371000.                 # Earth radius in metres

a_ps = 0.8                      # Plasmasphere latitudinal variation parameter (Denton et al, 2006)
a_tr = 2.0                      # Trough       latitudinal variation parameter

P = 9.0                         # Plume enhancement factor relative to trough density
B = 3.0                         # Plume magnitude parameter
N = 5.0                         # Number of density fluctuations

L_ppi = 3.5                     # Plasmapause inner
L_ppo = 3.6                     # Plasmapause outer
L_pp  = (L_ppi + L_ppo) / 2     # Plasmapause centre

L0 = 4.5                        # Plume inner edge
L1 = 5.5                        # Plume outer edge 

Ri = 3.4                        # Initial domain L
Rf = 7.0                        # Final domain L

n = 1000                              # Number of points
r  = np.linspace(Ri*RE, Rf*RE, n)     # Radial distance from Earth centre (m)
L  = r / RE                           # L parameter corresponding to r

## ------ PLASMASPHERE (ps) & PLASMATROUGH (tr) DENSITY MODELS ------ ##
R = (L * RE / r)

                       
n_ps = 1390. * ((3 / L)**(4.83)) * (R ** a_ps)       

n_tr1 = 124. * ((3. / L) ** 4.0)
n_tr2 = 36. * ((3./L) ** 3.5)
n_tr3 = (MLT - 7.7 * ((3. / L) ** 2.0) - 12.) * (pi / 12.)
    
n_tr =  n_tr1 + n_tr2 * np.cos(n_tr3) * (R ** a_tr)

L_frac = (L - L0) / (L1 - L0)
n_pl = (P - B * (1. - np.cos(2.*pi*N*L_frac))) * n_tr

## ------ DENSITY PROFILE ------ ##
dL = 0.05               # Transition width
Y1 = Y(L, L_pp, dL)     # ps/tr transition
Y2 = Y(L, L0,   dL)     # tr/pl transition (inner)
Y3 = Y(L, L1,   dL)     # pl/tr transition (outer)

ne = np.zeros(n)

# Locate transition positions
val1 = (L_pp + L0) / 2.     ;   idx1 = min(range(len(L)), key=lambda i: abs(L[i]-val1))
val2 = (L0 + L1)   / 2.     ;   idx2 = min(range(len(L)), key=lambda i: abs(L[i]-val2))

# Construct electron density profile
for ii in range(n):
    if ii < idx1:
        ne[ii] = n_ps[ii] * (1. - Y1[ii]) + n_tr[ii] * Y1[ii]
        
    elif ((ii >= idx1) and (ii <= idx2)):
        ne[ii] = n_tr[ii] * (1. - Y2[ii]) + n_pl[ii] * Y2[ii]

    elif ii > idx2:
        ne[ii] = n_pl[ii] * (1. - Y3[ii]) + n_tr[ii] * Y3[ii]
        
## ------ PLOT THINGS ------ ##       

fig = plt.figure(1)
ax  = fig.add_subplot(111)

ax.plot(r/RE, ne * 1e6)
ax.set_yscale('log')
ax.set_ylim(1e6, 1e9)

fig2 = plt.figure(2, figsize = (18,9))
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=3, sharex=ax1)
ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3, sharex=ax1)

ax1.plot(r/RE, n_ps * 1e6)
ax2.plot(r/RE, n_tr * 1e6)
ax3.plot(r/RE, n_pl * 1e6)

ax3.set_xlabel(r'r $(R_E)$', fontsize=18)
ax1.set_ylabel(r'$n_{ps}$', rotation=0, labelpad=15, fontsize=18)
ax2.set_ylabel(r'$n_{tr}$', rotation=0, labelpad=15, fontsize=18)
ax3.set_ylabel(r'$n_{pl}$', rotation=0, labelpad=15, fontsize=18)

for ax in [ax1, ax2, ax3]:
    ax.set_yscale('log')
    ax.set_xlim(Ri, Rf)

#==============================================================================
# plt.plot(r/RE, (1 - Y3) * n_pl * 1e6)
# plt.plot(r/RE, Y3 * n_tr * 1e6)
# 
# plt.yscale('log')
#==============================================================================
