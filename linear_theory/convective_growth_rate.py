# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:56:09 2019

@author: Yoshi
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
#
# Minimal changes to make program Python-compatible. Default parameters are those used 
# in Fraser et al. (1989)  Figure 7/Table 1.
#
# Made callable as a function: March, 2019

import numpy as np
import pdb


def plot_growth_rate(F, GR, ST, ax):
    ax.plot(F, GR)
    for ii in range(ST.shape[0] - 1):
        if ST[ii] == 1:
            ax.axvspan(F[ii], F[ii + 1], color='k')            # PLOT STOP BAND
    return

def calculate_growth_rate(field, ndensc, ndensw, ANI, temperp=None, beta=None, norm_ampl=0, norm_freq=0, NPTS=1000, maxfreq=1.0):
    '''
    Calculates the convective growth rate S as per eqn. 6 of Kozyra (1984). Plasma parameters passed as 
    length 3 numpy.ndarrays, one for each H+, He+, O+
    
    INPUT:
        field  -- Magnetic field intensity, nT
        ndensc -- Cold plasma density, /cm3
        ndensw -- Warm plasma density, /cm3
        ANI    -- Temperature anisotropy of each species
        
    OPTIONAL:
        temperp   -- Perpendicular temperature of species warm component, eV
        beta      -- Parallel plasma beta of species warm component
        norm_ampl -- Flag to normalize growth rate to max value (0: No, 1: Yes). Default 0
        norm_ampl -- Flag to normalize frequency to proton cyclotron units. Default 0
        NPTS      -- Number of sample points up to maxfreq. Default 500
        maxfreq   -- Maximum frequency to calculate for in proton cyclotron units. Default 1.0
        
    NOTE: At least one of temperp or beta must be defined.
    '''
    # Perform input checks 
    N   = ndensc.shape[0]
    
    if temperp is None and beta is None:
        raise ValueError('Either beta or tempperp must be defined.')
    elif temperp is not None and beta is not None:
        pdb.set_trace()
        print('Both temperp and beta arrays defined, defaulting to temperp array values...')
        beta = None
        
    # CONSTANTS
    PMASS   = 1.673E-27
    MUNOT   = 1.25660E-6
    EVJOULE = 6.242E18
    CHARGE  = 1.602E-19
    
    # OUTPUT PARAMS
    growth = np.zeros(NPTS)                         # Output growth rate variable
    x      = np.zeros(NPTS)                         # Input normalized frequency
    stop   = np.zeros(NPTS)                         # Stop band flag (0, 1)
    
    # Set here since these are constant (and wrong?)
    M    = np.zeros(N)
    M[0] = 1.0 
    M[1] = 4.0  
    M[2] = 16.0 
    
    # LOOP PARAMS
    step  = maxfreq / float(NPTS)
    FIELD = field*1.0E-9                 # convert to Tesla
        
    NCOLD   = ndensc * 1.0E6
    NHOT    = ndensw * 1.0E6
    etac    = NCOLD / NHOT[0]       
    etaw    = NHOT  / NHOT[0] 
    numer   = M * (etac+etaw)
    
    if beta is None:
        TPERP   = temperp / EVJOULE
        TPAR    = TPERP / (1.0 + ANI)
        bet     = NHOT*TPAR / (FIELD*FIELD/(2.0*MUNOT))
    else:
        bet     = beta
        TPAR    = np.zeros(N)
        for ii in range(N):
            if NHOT[ii] != 0:
                TPAR[ii]    = (FIELD*FIELD/(2.0*MUNOT)) * bet[ii] / NHOT[ii]
        
    alpha = np.sqrt(2.0 * TPAR /  (M * PMASS))      # M added in denominator

    for k in range(1, NPTS):
          x[k]   = k*step
          denom  = 1.0 - M*x[k]
          
          sum1  = 0.0
          prod2 = 1.0
          
          for i in range(N):
               prod2   *= denom[i]
               prod     = 1.0
               temp     = denom[i]
               denom[i] = numer[i]
               
               for j in range(N):
                   prod *= denom[j]
               
               sum1    += prod
               denom[i] = temp
          
          sum2 = 0.0
          arg4 = prod2 / sum1
    
          # Check for stop band.
          if (arg4 < 0.0) and (x[k] > 1.0/M[N-1]):
              growth[k] = 0.0
              stop[k]   = 1
          else:
             arg3 = arg4 / (x[k] ** 2)
             
             
             for i in range(N):
                if (NHOT[i] > 1.0E-3):
                     arg1  = np.sqrt(np.pi) * etaw[i] / ((M[i]) ** 2 * alpha[i])       # Outside term
                     arg1 *= ((ANI[i] + 1.0) * (1.0 - M[i]*x[k]) - 1.0)                # Inside square brackets (multiply by outside)
                     arg2  = (-etaw[i] / M[i]) * (M[i]*x[k] - 1.0) ** 2 / bet[i]*arg3
                     
                     sum2 += arg1*np.exp(arg2)
             
    
             growth[k] = sum2*arg3/2.0
    
    ###########################
    ### NORMALIZE AND CLEAN ###
    ###########################    
    for ii in range(NPTS):
        if (growth[ii] < 0.0):
            growth[ii] = 0.0
            
        if np.isnan(growth[ii]) == True:
            growth[ii] = np.inf
          
    if (norm_freq == 0):
        cyclotron  = CHARGE*FIELD/(2.0*np.pi*PMASS)
        x         *= cyclotron
          
    if (norm_ampl == 1):
        growth /= growth.max()
    else:
        growth *= 1e9
    return x, growth, stop



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PMASS   = 1.673E-27
    MUNOT   = 1.25660E-6
    EVJOULE = 6.242E18
    CHARGE  = 1.602E-19
    
    test_FIELD   = 487.5#300.
    
    cyclotron  = CHARGE*test_FIELD*1e-9/(2.0*np.pi*PMASS) # SI units

    test_c    = np.zeros(3)
    test_c[0] = 10.
    test_c[1] = 0.
    test_c[2] = 0.
    
    test_w    = np.zeros(3)
    test_w[0] = 5.
    test_w[1] = 0.
    test_w[2] = 0.0
    
    test_tp    = np.zeros(3)
    test_tp[0] = 50000.         # Perpendicular temperature (ev)
    test_tp[1] = 0.
    test_tp[2] = 0.

    test_A    = np.zeros(3)
    test_A[0] = 1.              # Temperature anisotropy
    test_A[1] = 1.
    test_A[2] = 1.
    
# =============================================================================
#     test_c = np.zeros(3)
#     test_c[0] = 196.
#     test_c[1] = 22.
#     test_c[2] = 2.
#     
#     test_w = np.zeros(3)
#     test_w[0] = 5.1
#     test_w[1] = 0.05
#     test_w[2] = 0.13
#     
#     test_tp = np.zeros(3)
#     test_tp[0] = 30000.         # Perpendicular temperature (ev)
#     test_tp[1] = 10000.
#     test_tp[2] = 10000.
# 
#     test_A = np.zeros(3)
#     test_A[0] = 1.              # Temperature anisotropy
#     test_A[1] = 1.
#     test_A[2] = 1.
# =============================================================================
    
    fr, gr, st = calculate_growth_rate(test_FIELD, test_c, test_w, test_A, temperp=test_tp, norm_freq=1, maxfreq=0.5)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot_growth_rate(fr, gr, st, ax1)
    plt.show()