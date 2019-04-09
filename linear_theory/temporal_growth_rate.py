# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:23:57 2019

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb


def get_k(X):
    '''
    Calculates k for given normalized frequency X
    
    CHECKED against equation. Appears correct. Gives reasonable results.
    '''
    outside = w_pw2[0] / (SPLIGHT ** 2)
    first   = ((1 + delta)*X**2) / (1 - X)
    
    second = 0
    for ii in range(1, N):
        second += (etaw[ii] + etac[ii]) * (M[ii] * X ** 2) / (1 - M[ii]*X)
    
    k_arg    = outside * (first + second)
    
    if k_arg < 0:
        k_output = np.nan
    else:
        k_output = np.sqrt(k_arg)
    return k_output


def calculate_group_velocity(X):
    '''
    Use this plus the temporal growth rate to validate the CGR. Or the other
    way around
    
    CHECKED against equation. Appears correct. Gives reasonable results (vg -> 0 at resonances?).
    '''
    first_term = 2 * pcyc * SPLIGHT / np.sqrt(w_pw2[0])
    
    first_sum  = (1 + delta) / (1 - X)
    for ii in range(1, N):
        first_sum += (etac[ii] + etaw[ii]) * M[ii] / (1 + M[ii] * X)
        
    first_sum  = np.sqrt(first_sum)
    first_term*= first_sum
    
    second_term = (1. + delta) * (2. - X) / ((1. - X) ** 2)
    for jj in range(1, N):
        second_term += (etac[jj] + etaw[jj]) * M[jj] * (2. - M[jj] * X) / ((1 - M[jj] * X) ** 2)
    
    result = first_term / second_term
    return result


def calculate_temporal_growth_rate(X):
    '''
    Found a renegade minus sign in 'n_second' with the A + 1 term
    '''
    
    this_k    = get_k(X)
    
    ## Check for stop band ##
    stop = 0
    if np.isnan(this_k) == True:
        '''
        Goes negative -> nan when in a stop band
        '''
        stop        = 1
        temporal_gr = 0
    else:
        numer_sum = 0
        for ii in range(N):
            n_first      = pcyc * etaw[ii] * np.sqrt(np.pi) / (M[ii] ** 2 * alpha[ii] * this_k)
            n_second     = (A[ii] + 1) * (1 - M[ii] * X) - 1
            n_exponent   = (- pcyc ** 2 / (M[ii] ** 2)) * ((M[ii] * X - 1) ** 2) / (alpha[ii] ** 2 * this_k ** 2)
            numer_sum   += n_first * n_second * np.exp(n_exponent)
        
        denom_sum = X * (delta + 1.) * (2. - X) / ((X - 1.) ** 2)
        for ii in range(1, N):
            denom_sum += (etac[ii] + etaw[ii]) * M[ii] * X * (2. - M[ii] * X) / ((M[ii] * X - 1.) ** 2)
            
        temporal_gr = numer_sum / denom_sum
    return temporal_gr, stop




def call_functions():
    k           = np.zeros(NPTS)                         # Input normalized frequency
    x           = np.zeros(NPTS)                         # Input normalized frequency
    temp_growth = np.zeros(NPTS)                         # Output growth rate variable
    group_vel   = np.zeros(NPTS)                         # Stop band flag (0, 1)
    stop        = np.zeros(NPTS)
    
    step = 1.0 / NPTS
                                      
    for ii in range(1, NPTS):
        x[ii]                     = ii*step
        k[ii]                     = get_k(x[ii])
        temp_growth[ii], stop[ii] = calculate_temporal_growth_rate(x[ii])
        group_vel[ii]             = calculate_group_velocity(x[ii])
    return x, k, temp_growth*1e9, group_vel, stop


if __name__ == '__main__':
    '''
    Equations taken straight from Kozyra et al. (1984) without modification. Assumed Gaussian units throughout.
    
    Bits I'm unsure on:
        -- Units of charge... Fr or abC? Is this Gaussian or EMU?
        -- Value for Boltmann's constant? Using eV/K. Only used to calculate alpha
        -- Units for energy? Using eV. Only used to calculate alpha and beta
    '''
    PMASS    = 1.673E-24         # g
    CHARGE   = 4.80326e-10       # StatC (Fr)
    #CHARGE   = 1.602e-20        # abC
    SPLIGHT  = 3E10              # cm/s
    NPTS     = 500
    N        = 3
    
    field_SI = 300.0e-9          # T
    FIELD    = field_SI * 1e4    # G
    
    
    # Index 1 denotes hydrogen ; 2 denotes helium; 3 denotes oxygen etc.
    M    = np.zeros(N)
    M[0] = 1.0     #; Hydrogen
    M[1] = 4.0     #; Helium
    M[2] = 16.0    #; Oxygen
    
    # Print,' Input densities of cold species (number/cc) [3]'
    ndensc    = np.zeros(N)
    ndensc[0] = 196.
    ndensc[1] = 22.
    ndensc[2] = 2.
    
    # Density of warm species (same order as cold) (number/cc)
    ndensw    = np.zeros(N)
    ndensw[0] = 5.1
    ndensw[1] = 0.05
    ndensw[2] = 0.13
    
    # Input the perpendicular temperature (ev)
    temperp    = np.zeros(N)
    temperp[0] = 30000.
    temperp[1] = 10000.
    temperp[2] = 10000.
    
    # Input the temperature anisotropy
    A    = np.zeros(N)
    A[0] = 1.0
    A[1] = 1.
    A[2] = 1.
    
    
    ##################################
    #### CALCULATE SOME CONSTANTS ####
    ##################################
    mi      = M * PMASS
            
    w_pc2   = 4 * np.pi * ndensc * CHARGE ** 2 / mi
    w_pw2   = 4 * np.pi * ndensw * CHARGE ** 2 / mi
    
    ion_cyclotron = CHARGE * FIELD / (2 * np.pi * mi * SPLIGHT)
    pcyc          = ion_cyclotron[0]
    
    etac    = M * (w_pc2 / w_pw2[0])
    etaw    = M * (w_pw2 / w_pw2[0])
    delta   = w_pc2[0] / w_pw2[0]
    
    
    if True:
        EVJOULE = 1 / CHARGE                                      # Convert eV to energy (CGS)
        tper    = temperp / EVJOULE                               # BMANN   = 1.381e-16                                      # erg/K
        tpar    = tper / (1.0 + A)                                # 
        alpha   = np.sqrt(tpar / mi)                              # NOT RIGHT... alpha must end up as cm/s
    elif False:
        EVJOULE = 6.242E18                                        # 'Stolen' from CGR code.. SI to CGS conversion
        tper    = temperp / EVJOULE                             
        tpar    = tper / (1.0 + A)                                # Temps converted from eV to J?
        alpha   = np.sqrt(2.0 * tpar / PMASS) * 100               # in m/s, converted to cm/s ? Gives far more valid results
    
    

    #############################
    #### CALL/TEST FUNCTIONS ####
    #############################
    freq, wnum, mu, vg, stop_band = call_functions()
    
    freq         *= ion_cyclotron[0]
    
    plt.figure()
    plt.plot(freq, wnum)
    plt.xlabel('X')
    plt.ylabel('k')
    plt.title('Wave Number (k)')
    
    for f in ion_cyclotron:
        plt.axvline(f, c='k')
        
    for ii in range(stop_band.shape[0] - 1):
        if stop_band[ii] == 1:
            plt.axvspan(freq[ii], freq[ii + 1], color='k')            # PLOT STOP BAND
    
    plt.figure()
    plt.plot(freq, mu)
    plt.xlabel('X')
    plt.ylabel('Growth Rate')
    plt.title('Temporal Growth Rate ($\mu$)')
    
    for f in ion_cyclotron:
        plt.axvline(f, c='k')
        
    for ii in range(stop_band.shape[0] - 1):
        if stop_band[ii] == 1:
            plt.axvspan(freq[ii], freq[ii + 1], color='k')            # PLOT STOP BAND
    
    plt.figure()
    plt.plot(freq, vg)
    plt.xlabel('X')
    plt.ylabel('Group Velocity (cm/s)')
    plt.title('Wave Group Velocity ($V_g$)')
    
    for f in ion_cyclotron:
        plt.axvline(f, c='k')
        
    plt.figure()
    plt.plot(freq, mu/vg)
    plt.xlabel('X')
    plt.ylabel('Convective Growth Rate')
    plt.title('Convective Growth Rate ($\mu / V_g$)')
    
    for f in ion_cyclotron:
        plt.axvline(f, c='k')
        
    for ii in range(stop_band.shape[0] - 1):
        if stop_band[ii] == 1:
            plt.axvspan(freq[ii], freq[ii + 1], color='k')            # PLOT STOP BAND
        
        