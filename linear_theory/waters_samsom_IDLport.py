#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:04:18 2018

@author: yoshi
"""

# This program calculates the growth rate for the cyclotron instability for compositions of
# singly ionized H, He, and O, including both warm and cold species. It also allows variable
# anisotropy ratios, perpendicular temperatures (ev), and magnetic flux densities.
# NOTE: All values of the growth rate are multiplied by 1.0E9,  unless the amplitudes
# are normalized, and then the maximum value is 1.0.
# Original Pascal code by John C. Samson
#
# Converted to IDL : Colin L. Waters
# July, 2009
#
# Converted to Python : Joshua S. Williams
# October, 2018

#%%
import numpy as np
import matplotlib.pyplot as plt


def geomagnetic_magnitude(L_shell, lat=0.):
    '''Returns the magnetic field magnitude (intensity) on the specified L shell at the given colatitude, in Tesla
    '''
    RE         = 6371000.
    B_surf     = 3.12e-5

    r_loc = L_shell*RE*(np.cos(lat*np.pi / 180.) ** 2)
    B_tot = B_surf * ((RE / r_loc) ** 3) * np.sqrt(1 + 3*(np.sin(lat*np.pi / 180.) ** 2))
    return B_tot

if __name__ == '__main__':
    path      = 'E:/python/jwilliams/'
    out_fname = path + 'pc1_tst.dat'

    normalize = 0                       # Normalize amplitude? (0: No, 1: Yes)
    normal    = 1
    maxfreq   = 0.5

    L    = 4.0                          # L-shell for magnetic field
    B0   = geomagnetic_magnitude(L)     # In Tesla
    NPTS = 1000                         # Number of points to solve growth rate for
    N    = 3                            # Number of species

    # Index 1 denotes hydrogen ; 2 denotes helium; 3 denotes oxygen etc.
    M    = np.zeros(N)
    M[0] = 1.0     # Hydrogen
    M[1] = 4.0     # Helium
    M[2] = 16.0    # Oxygen

    # Density of cold species (number/cc)
    ndensc    = np.zeros(N, dtype=float)
    ndensc[0] = 10
    ndensc[1] = 0
    ndensc[2] = 0

    # Density of warm species (same order as cold) (number/cc)
    ndensw    = np.zeros(N, dtype=float)
    ndensw[0] = 5
    ndensw[1] = 0
    ndensw[2] = 0

    # Input the perpendicular temperature (eV)
    temperp = np.zeros(N, dtype=float)
    temperp[0]=50000.
    temperp[1]=00000.
    temperp[2]=00000.

    # Input the temperature anisotropy
    A = np.zeros(N, dtype=float)
    
    A[0] = 1
    A[1] = 1.
    A[2] = 1.
    
#######################################
######## MAIN SCRIPT ##################
#######################################

    PMASS   = 1.67E-27
    MUNOT   = 1.25660E-6
    EVJOULE = 6.242E18
    CHARGE  = 1.60E-19

    smallarray = np.zeros(N)
    etac       = np.zeros(N)
    etaw       = np.zeros(N)
    ratioc     = np.zeros(N)
    ratiow     = np.zeros(N)
    alpha      = np.zeros(N)
    bet        = np.zeros(N)
    tempar     = np.zeros(N)
    numer      = np.zeros(N)
    denom      = np.zeros(N)

    growth     = np.zeros(NPTS)               # Convective growth rate value
    stop       = np.zeros(NPTS)               # Stop band flag
    x          = np.zeros(NPTS)               # X-axis values

    if maxfreq > 1.0:
        maxfreq = 1.0

    step = maxfreq / float(NPTS)

    if normal == 1:
        cyclotron = 1.0
    else:
        cyclotron = CHARGE*B0/(2.0*np.pi*PMASS)

    for i in range(0,N):           # Loop over 3 species of ions
        ndensc[i]  = ndensc[i]*1.0E6
        ndensw[i]  = ndensw[i]*1.0E6
        etac[i]    = ndensc[i]  / ndensw[0]
        etaw[i]    = ndensw[i]  / ndensw[0]
        temperp[i] = temperp[i] / EVJOULE
        tempar[i]  = temperp[i] / (1.0 + A[i])
        alpha[i]   = np.sqrt(2.0*tempar[i] / PMASS)
        bet[i]     = ndensw[i]*tempar[i] / (B0*B0 / (2.0*MUNOT))
        numer[i]   = M[i]*(etac[i] + etaw[i])

        if np.isnan(etac[i]) == True:
            etac[i] = 0.0

        if np.isnan(etaw[i]) == True:
            etaw[i] = 0.0

    themax = 0.0

    for k in range(NPTS):
        x[k] = (k + 1)*step

        for i in range(N):
            denom[i] = 1.0 - M[i]*x[k]

        sum1  = 0.0
        prod2 = 1.0

        for i in range(N):
           prod2   *= denom[i]
           prod     = 1.0
           temp     = denom[i]
           denom[i] = numer[i]

           for j in range(N):
               prod *=denom[j]

           sum1     += prod
           denom[i]  = temp

        sum2 = 0.0
        arg4 = prod2 / sum1

        # Check for stop band.
        if (arg4 < 0.0 and x[k] > 1.0 / M[N - 1]):
            growth[k] = 0.0
            stop[k]   = 1
        else:
            arg3 = arg4 / (x[k]*x[k])
            for i in range(N):
                if ndensw[i] > 1.0E-3:
                    arg1 = np.sqrt(np.pi)*etaw[i] / ((M[i]) ** 2 * alpha[i])
                    arg1 = arg1*((A[i]+1.0)*(1.0-M[i]*x[k])-1.0)
                    arg2 = (-etaw[i] / M[i]) * (M[i]*x[k] - 1.0) ** 2 / bet[i]*arg3

                    if arg2 > 200.0:
                        arg2 = 200.0

                    if arg2 < -200.0:
                        arg2 = -200.0

                    sum2 += arg1 * np.exp(arg2)

        growth[k] = sum2*arg3 / 2.0

        if growth[k] < 0.0:
            growth[k] = 0.0

        x[k] *= cyclotron

        if growth[k] > themax:
            themax = growth[k]


    if normalize == 1:
        for k in range(NPTS):
            growth[k] = growth[k] / themax
    else:
        for k in range(NPTS):
            growth[k] = growth[k]*1.0E9

    #fig = plt.figure(figsize=(18, 10))
    #ax  = fig.add_subplot(111)
    
    plt.plot(x, growth)
    
    for ii in range(NPTS - 1):
        if stop[ii] == 1:
            plt.axvspan(x[ii], x[ii + 1], color='k')
    
    #plt.xlim(0, 2.0)
    #plt.ylim(0, 100)

    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Growth Rate ($\omega / V_g 10^{-7} cm^{-1}$)', rotation=90, fontsize=14)
    plt.title('Growth Rate: Kozyra parameters (Fig 1a)', fontsize=16)
    
# =============================================================================
#     ax.text(0.82, 0.99, 'Cold', transform=ax.transAxes, fontsize=10, color='b')
#     ax.text(0.89, 0.99, 'Hot', transform=ax.transAxes, fontsize=10, color='r')
#     ax.text(0.95, 0.99, '$A_i$', transform=ax.transAxes, fontsize=10, color='k')
#     
#     ax.text(0.80, 0.95, '$H^{+}$:'   , transform=ax.transAxes, fontsize=10)
#     ax.text(0.82, 0.95, '%.3f$cm^{-3}$' % (ndensc[0]/1e6), transform=ax.transAxes, fontsize=10, color='b')
#     ax.text(0.89, 0.95, '%.3f$cm^{-3}$' % (ndensw[0]/1e6), transform=ax.transAxes, fontsize=10, color='r')
# 
#     ax.text(0.79, 0.90, '$He^{+}$:'   , transform=ax.transAxes, fontsize=10)
#     ax.text(0.82, 0.90, '%.3f$cm^{-3}$' % (ndensc[1]/1e6), transform=ax.transAxes, fontsize=10, color='b')
#     ax.text(0.89, 0.90, '%.3f$cm^{-3}$' % (ndensw[1]/1e6), transform=ax.transAxes, fontsize=10, color='r')
# 
#     ax.text(0.80, 0.85, '$O^{+}$:'   , transform=ax.transAxes, fontsize=10)
#     ax.text(0.82, 0.85, '%.3f$cm^{-3}$' % (ndensc[2]/1e6), transform=ax.transAxes, fontsize=10, color='b')
#     ax.text(0.89, 0.85, '%.3f$cm^{-3}$' % (ndensw[2]/1e6), transform=ax.transAxes, fontsize=10, color='r')
# =============================================================================
