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
import sys

def plot_growth_rate(F, GR, ST, ax):
    ax.plot(F, GR)
    for ii in range(ST.shape[0] - 1):
        if ST[ii] == 1:
            ax.axvspan(F[ii], F[ii + 1], color='k')            # PLOT STOP BAND
    plt.show()
    return


def calculate_growth_rate(field, ndensc, ndensw, ANI, beta, norm_ampl=0, norm_freq=0, NPTS=500, maxfreq=1.0):
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
    
    for ii in np.arange(N):
        if ndensw[ii] != 0:
            if beta[ii] == 0:
                raise ValueError('Zero beta not allowed for warm components.')
        
    # CONSTANTS
    PMASS   = 1.673E-27
    MUNOT   = 1.25660E-6
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
    etac    = NCOLD / NHOT[0]        # Needs a factor of Z ** 2?
    etaw    = NHOT  / NHOT[0]        # Here too
    numer   = M * (etac+etaw)
    
    TPAR    = np.zeros(N)
    for ii in range(N):
        if NHOT[ii] != 0:
            TPAR[ii]    = (FIELD*FIELD/(2.0*MUNOT)) * beta[ii] / NHOT[ii]
        
    alpha = np.sqrt(2.0 * TPAR / PMASS)

    for k in np.arange(1, NPTS):
          x[k]   = k*step
          denom  = 1.0 - M*x[k]
          
          sum1  = 0.0
          prod2 = 1.0
          
          for i in np.arange(N):
               prod2   *= denom[i]
               prod     = 1.0
               temp     = denom[i]
               denom[i] = numer[i]
               
               for j in np.arange(N):
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
             
             
             for i in np.arange(N):
                if (NHOT[i] > 1.0E-3):
                     arg1  = np.sqrt(np.pi) * etaw[i] / ((M[i]) ** 2 * alpha[i])       # Outside term
                     arg1 *= ((ANI[i] + 1.0) * (1.0 - M[i]*x[k]) - 1.0)                # Inside square brackets (multiply by outside)
                     arg2  = (-etaw[i] / M[i]) * (M[i]*x[k] - 1.0) ** 2 / beta[i]*arg3
                     
                     sum2 += arg1*np.exp(arg2)
             
             growth[k] = sum2*arg3/2.0
    
    ###########################
    ### NORMALIZE AND CLEAN ###
    ###########################    
    for ii in np.arange(NPTS):
        if (growth[ii] < 0.0):
            growth[ii] = 0.0
          
    if (norm_freq == 0):
        cyclotron  = CHARGE*FIELD/(2.0*np.pi*PMASS)
        x         *= cyclotron
          
    if (norm_ampl == 1):
        growth /= growth.max()
    else:
        growth *= 1e9
    return x, growth, stop


if __name__ == '__main__':

    
    N         = 3   # Number of species
    
    n_min   = 20.  ; n_max   = 420.   ; dn   = 5.
    B_min   = 100. ; B_max   = 300.   ; db   = 5.
    nhe_min = 0.0  ; nhe_max = 0.999  ; dnhe = 0.001
    
    hot_ion_frac = 0.05
    
    B0           = np.arange(B_min ,  B_max   + db  , db )
    n0           = np.arange(n_min ,  n_max   + dn  , dn )
    nhe          = np.arange(nhe_min, nhe_max + dnhe, dnhe)

    growth_max   = np.zeros((B0.shape[0], n0.shape[0]))
    
    count   = 0
    max_val = 0.0
    for kk in np.arange(1, nhe.shape[0]):
        print('Doing He = {}%'.format(100.*nhe[kk]))
        for ii in np.arange(B0.shape[0]):
            for jj in np.arange(n0.shape[0]):
                # Magnetic field flux in nT
                field   = B0[ii]     
                
                n_cold  = n0[jj]*(1 - hot_ion_frac)                 # Split density into cold
                n_warm  = n0[jj]*hot_ion_frac                       # and warm components
                
                # Density of cold/warm species (number/cc)
                ndensc     = np.zeros(N)
                ndensc[0]  = (1 - nhe[kk]) * n_cold                          
                ndensc[1]  = nhe[kk]       * n_cold 
                ndensc[2]  = 0.0  
                
                ndensw     = np.zeros(N)
                ndensw[0]  = (1 - nhe[kk]) * n_warm 
                ndensw[1]  = nhe[kk]       * n_warm  
                ndensw[2]  = 0.0  
    
                H_frac  = 100. * np.array([ndensc[0], ndensw[0]]) / n0[jj]
                He_frac = 100. * np.array([ndensc[1], ndensw[1]]) / n0[jj]
                O_frac  = 100. * np.array([ndensc[2], ndensw[2]]) / n0[jj]
                
                # Warm species parallel beta
                betapar = np.zeros(N)
                betapar[0] = 10.
                betapar[1] = 10.
                betapar[2] = 0.
                
                # Temperature anisotropy
                A    = np.zeros(N)
                A[0] = 2.                                         
                A[1] = 2.
                A[2] = 0.
                
                FREQ, GR_RATE, STFLAG = calculate_growth_rate(field, ndensc, ndensw, A, beta=betapar, norm_freq=0, maxfreq=1.0)
                growth_max[ii, jj]    = GR_RATE.max()
                
                if GR_RATE.max() > max_val:
                    max_val = GR_RATE.max()
