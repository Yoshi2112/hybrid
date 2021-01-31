# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:22:34 2021

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt

# Constants
BE     = 0.31*1e5   # Equatorial field strength in nT
r_A    = 120e3      # Ionospheric height
RE     = 6.371e6    # Earth radii

def new_trace_fieldline(L):
    '''
    Traces field line position and B-intensity for given L. 
    Traces half, flips and copies the values, and then
    reverses sign of s since the integration is done from the southern hemisphere
    but we normally think of traces from the northern.
    
    Validated: Ionospheric B0 goes to surface B0 for rA = 0.
    '''
    

    Np       = int(1e5+1)
    iono_lat = np.arccos(np.sqrt((RE + r_A)/(RE*L))) 
    mlat     = np.linspace(-iono_lat, iono_lat, Np, endpoint=True)
    dlat     = mlat[1] - mlat[0]
    mid_idx  = (Np-1) // 2
    
    Bs = np.zeros(Np, dtype=float)
    s  = np.zeros(Np, dtype=float)
    
    # Step through MLAT starting at equator. Calculate B at each point
    current_s = 0.0 
    for ii in range(mid_idx, Np):
        ds     = L*RE*np.cos(mlat[ii])*np.sqrt(4.0 - 3.0*np.cos(mlat[ii]) ** 2) * dlat
        s[ii]  = current_s
        Bs[ii] = (BE/L**3)*np.sqrt(4-3*np.cos(mlat[ii])**2) / np.cos(mlat[ii])**6
        current_s += ds
        
    # Flip around equator for 1st half
    s[ :mid_idx] = -1.0*np.flip( s[mid_idx + 1:])
    Bs[:mid_idx] =      np.flip(Bs[mid_idx + 1:])
        
    # Check surface field measurement
    #BL = BE * np.sqrt(4 - 3/L)
    
    # Reverse s sign, convert to T and degrees respectively
    return -s, Bs*1e-9, -mlat*180./np.pi


if __name__ == '__main__':
    S, BS, MLAT = new_trace_fieldline(4.27)
    
    fig, ax = plt.subplots()
    ax.plot(S/RE, BS*1e9, c='k', lw=0.5)
