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

def calculate_growth_rate(field, ndensc, ndensw, ANI, temperp=None, beta=None, norm_ampl=0, norm_freq=0, NPTS=500, maxfreq=1.0):
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
    
    SEEMS TO EXPLODE WHEN NOT ALL COMPONENTS ARE PRESENT. NEED TO CODE MORE CAREFULLY TO CATCH EXCEPTIONS.
    '''
    # Perform input checks 
    N   = ndensc.shape[0]
    
    if temperp is None and beta is None:
        raise ValueError('Either beta or tempperp must be defined.')
    elif temperp is not None and beta is not None:
        print 'Both temperp and beta arrays defined, defaulting to temperp array values...'
        beta = None
        
    if temperp is None:
        if ndensw.shape[0] != N or ANI.shape[0] != N or beta.shape[0] != N:
            raise IndexError('Plasma parameter arrays not of equal length, aborting...')
    elif beta is None:
        if ndensw.shape[0] != N or ANI.shape[0] != N or temperp.shape[0] != N:
            raise IndexError('Plasma parameter arrays not of equal length, aborting...')
    else:
        raise Exception('Unknown Error, aborting...')
        
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
    etac    = NCOLD / NHOT[0]        # Needs a factor of Z ** 2?
    etaw    = NHOT  / NHOT[0]        # Here too
    numer   = M * (etac+etaw)
    delta   = NCOLD[0] / NHOT[0]
    
    
    if beta is None:
        TPERP   = temperp / EVJOULE
        TPAR    = TPERP / (1.0 + ANI)
        bet     = NHOT*TPAR / (FIELD*FIELD/(2.0*MUNOT))
    else:
        bet     = beta
        TPAR    = (FIELD*FIELD/(2.0*MUNOT)) * bet / NHOT
        
    alpha = np.sqrt(2.0 * TPAR / PMASS)
    
    for kk in range(1, NPTS):
          x[kk]   = kk*step
          denom   = 1.0 - M*x[kk]
              
          inner_sum = 0.0
          for ii in range(1, N):
              if denom[ii] == 0:
                  inner_sum += np.sign(numer[ii]) * np.inf
              else:              
                  inner_sum += numer[ii] / denom[ii]
          inner_sum += (1 + delta) / (1. - x[kk])
          
          if inner_sum < 0.0 and x[kk] > 1.0/M[N-1]:    # Not sure about this
              growth[kk] = 0.0
              stop[kk]   = 1
          else:
              
              outer_sum = 0.0
              for ll in range(N):
                  arg1  = np.sqrt(np.pi) * etaw[ll] / ((M[ll]) ** 2 * alpha[ll])
                  arg1 *= ((ANI[ll] + 1.0) * (1.0 - M[ll]*x[kk]) - 1.0)
                  
                  arg2  = (-etaw[ll] / M[ll]) * (M[ll]*x[kk] - 1.0) ** 2 / (bet[ll] * x[kk] ** 2)
                  
                  outer_sum += arg1 * np.exp(arg2 / inner_sum)
              
              growth[kk] = outer_sum / (2.0 * x[kk] ** 2 * inner_sum)
          
    ###########################
    ### NORMALIZE AND CLEAN ###
    ###########################    
    for ii in range(NPTS):
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