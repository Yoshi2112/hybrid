# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:30:20 2020

@author: Yoshi

Note: This script just copies the functions related to calculating the cold
dispersion/growth rates since the 'omura play' source script isn't a final
product
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from   scipy.special     import wofz

import extract_parameters_from_data   as data
from growth_rates_from_RBSP         import extract_species_arrays
from dispersion_solver_multispecies import create_species_array


def Z(arg):
    '''Return Plasma Dispersion Function : Normalized Fadeeva function'''
    return 1j*np.sqrt(np.pi)*wofz(arg)


def get_k_cold(w, Species):
    '''
    Calculate the k of a specific angular frequency w in a cold
    multicomponent plasma. Assumes a cold plasma (i.e. negates 
    thermal effects). Hot species cast to cold by including their
    plasma frequencies.
    
    This will give the cold plasma dispersion relation for the Species array
    specified, since the CPDR is surjective in w (i.e. only one possible k for each w)
    
    Omura et al. (2010)
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cold_sum = 0.0
        for ii in range(Species.shape[0]):
            cold_sum += Species[ii]['plasma_freq_sq'] / (w * (w - Species[ii]['gyrofreq']))
    
        k = np.sqrt(1 - cold_sum) * w / c
    return k


def linear_growth_rates(w, Species):
    '''
    Calculates the temporal and convective linear growth rates for a plasma
    composition contained in Species for each frequency w. Assumes a cold
    dispersion relation is valid for k.
    
    Equations adapted from Chen et al. (2011)
    '''
    # Get k for each frequency to evaluate
    k  = get_k_cold(w, Species)
    
    # Calculate Dr/k_para
    w_der_sum = 0.0
    k_der_sum = 0.0
    Di        = 0.0
    for ii in range(Species.shape[0]):
        sp = Species[ii]
        
        # If cold
        if sp['tper'] == 0:
            w_der_sum += sp['plasma_freq_sq'] * sp['gyrofreq'] / (w - sp['gyrofreq'])**2
            k_der_sum += 0.0
            Di        += 0.0
        
        # If hot
        else:
            zs           = (w - sp['gyrofreq']) / (sp['vth_par']*k)
            Yz           = np.real(Z(zs))
            dYz          = -2*(1 + zs*Yz)
            A_bit        = (sp['anisotropy'] + 1) * w / sp['gyrofreq']
            
            # Calculate frequency derivative of Dr (sums bit)
            w_der_outsd  = sp['plasma_freq_sq']*sp['gyrofreq'] / (w*k*sp['vth_par'])
            w_der_first  = A_bit * Yz
            w_der_second = (A_bit - sp['anisotropy']) * w * dYz / (k * sp['vth_par']) 
            w_der_sum   += w_der_outsd * (w_der_first + w_der_second)
    
            # Calculate Di (sums bit)
            Di_bracket = 1 + (sp['anisotropy'] + 1) * (w - sp['gyrofreq']) / sp['gyrofreq']
            Di_after   = sp['gyrofreq'] / (k * sp['vth_par']) * np.sqrt(np.pi) * np.exp(- zs ** 2)
            Di        += sp['plasma_freq_sq'] * Di_bracket * Di_after
    
            # Calculate wavenumber derivative of Dr (sums bit)
            k_der_outsd  = sp['plasma_freq_sq']*sp['gyrofreq'] / (w*k*k*sp['vth_par'])
            k_der_first  = A_bit - sp['anisotropy']
            k_der_second = Yz + zs * dYz
            k_der_sum   += k_der_outsd * k_der_first * k_der_second
    
    # Get and return ratio
    Dr_wder = 2*w + w_der_sum
    Dr_kder = -2*k*c**2 - k_der_sum

    temporal_growth_rate   = - Di / Dr_wder
    group_velocity         = - Dr_kder / Dr_wder
    convective_growth_rate = - temporal_growth_rate / np.abs(group_velocity)
    return temporal_growth_rate, convective_growth_rate


if __name__ == '__main__':
    c = 3e8
    
    rbsp_path = 'G://DATA//RBSP//'
    save_drive= 'G://'
    
    time_start  = np.datetime64('2013-07-25T21:00:00')
    time_end    = np.datetime64('2013-07-25T22:00:00')
    probe       = 'a'
    pad         = 0
    
    date_string = time_start.astype(object).strftime('%Y%m%d')
    save_string = time_start.astype(object).strftime('%Y%m%d_%H%M_') + time_end.astype(object).strftime('%H%M')
    save_dir    = '{}NEW_LT//EVENT_{}//NEW_FIXED_DISPERSION_RESULTS//'.format(save_drive, date_string)

    # NEEDS TO BE STUCK IN A FUNCTION
    # Set frequency range to look at (from 0-5Hz)
    Nf          = 1000
    f_max       = 5.0
    f_min       = 0.0
    freqs       = np.linspace(f_max, f_min, Nf) * 2 * np.pi
    
    # Create species array for each time
    times, B0, name, mass, charge, density, tper, ani, cold_dens = \
    extract_species_arrays(rbsp_path, time_start, time_end, probe, pad,
                           return_raw_ne=True)
    # Does this need to be in the time loop?
    Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)