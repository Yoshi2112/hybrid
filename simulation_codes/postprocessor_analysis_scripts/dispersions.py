# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:14:56 2019

@author: Yoshi
"""
import analysis_config as cf
import numpy as np
import sys
data_scripts_dir = 'C://Users//iarey//Documents//GitHub//hybrid//linear_theory//'
sys.path.append(data_scripts_dir)


def get_cgr_from_sim(norm_flag=0):
    from convective_growth_rate import calculate_growth_rate
    from analysis_config import species_lbl, density, temp_type, Tper, Tpar, Nj, B0
    
    cold_density = np.zeros(3)
    warm_density = np.zeros(3)
    cgr_ani      = np.zeros(3)
    tempperp     = np.zeros(3)
    anisotropies = Tper / Tpar - 1
    
    for ii in range(Nj):
        if temp_type[ii] == 0:
            if 'H^+'    in species_lbl[ii]:
                cold_density[0] = density[ii] / 1e6
            elif 'He^+' in species_lbl[ii]:
                cold_density[1] = density[ii] / 1e6
            elif 'O^+'  in species_lbl[ii]:
                cold_density[2] = density[ii] / 1e6
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')
                
        if temp_type[ii] == 1:
            if 'H^+'    in species_lbl[ii]:
                warm_density[0] = density[ii] / 1e6
                cgr_ani[0]      = anisotropies[ii]
                tempperp[0]     = Tper[ii] / 11603.
            elif 'He^+' in species_lbl[ii]:
                warm_density[1] = density[ii] / 1e6
                cgr_ani[1]      = anisotropies[ii]
                tempperp[1]     = Tper[ii] / 11603.
            elif 'O^+'  in species_lbl[ii]:
                warm_density[2] = density[ii] / 1e6
                cgr_ani[2]      = anisotropies[ii]
                tempperp[2]     = Tper[ii] / 11603.
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')
    
    freqs, cgr, stop = calculate_growth_rate(B0*1e9, cold_density, warm_density, cgr_ani, temperp=tempperp, norm_freq=norm_flag)
    return freqs, cgr, stop


def get_linear_dispersion_from_sim(k=None, plot=False, save=False):
    '''
    Still not sure how this will work for a H+, O+ mix, but H+-He+ should be fine
    '''
    from chen_warm_dispersion   import get_dispersion_relation
    
    from analysis_config import species_present, NX, dx, Tper, Tpar, temp_type,\
                                species_lbl, density, Nj, B0, anal_dir
                   
    if k is None:
        k         = np.fft.fftfreq(NX, dx)
        k         = k[k>=0]
    
    N_present    = species_present.count(True)
    cold_density = np.zeros(N_present)
    warm_density = np.zeros(N_present)
    cgr_ani      = np.zeros(N_present)
    tempperp     = np.zeros(N_present)
    anisotropies = Tper / Tpar - 1
    
    for ii in range(Nj):
        if temp_type[ii] == 0:
            if 'H^+'    in species_lbl[ii]:
                cold_density[0] = density[ii]
            elif 'He^+' in species_lbl[ii]:
                cold_density[1] = density[ii]
            elif 'O^+'  in species_lbl[ii]:
                cold_density[2] = density[ii]
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')
                
        if temp_type[ii] == 1:
            if 'H^+'    in species_lbl[ii]:
                warm_density[0] = density[ii]
                cgr_ani[0]      = anisotropies[ii]
                tempperp[0]     = Tper[ii] / 11603.
            elif 'He^+' in species_lbl[ii]:
                warm_density[1] = density[ii]
                cgr_ani[1]      = anisotropies[ii]
                tempperp[1]     = Tper[ii] / 11603
            elif 'O^+'  in species_lbl[ii]:
                warm_density[2] = density[ii]
                cgr_ani[2]      = anisotropies[ii]
                tempperp[2]     = Tper[ii] / 11603
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')

    if save == True:
        savepath = anal_dir
    else:
        savepath = None
    
    k_vals, CPDR_solns, warm_solns = get_dispersion_relation(B0, cold_density, warm_density, cgr_ani, tempperp,
               norm_k=False, norm_w=False, kmin=k[0], kmax=k[-1], k_input_norm=0, plot=plot, save=save, savepath=savepath)

    return k_vals, CPDR_solns, warm_solns


def get_wx(component):
    arr = cf.get_array(component)
    
    fft_matrix  = np.zeros(arr.shape, dtype='complex128')
    for ii in range(arr.shape[1]):
        fft_matrix[:, ii] = np.fft.fft(arr[:, ii] - arr[:, ii].mean())
    
    wx = (fft_matrix[:arr.shape[0] // 2, :] * np.conj(fft_matrix[:arr.shape[0] // 2, :])).real
    return wx


def get_kt(component):
    arr = cf.get_array(component)
    
    fft_matrix  = np.zeros(arr.shape, dtype='complex128')
    for ii in range(arr.shape[0]): # Take spatial FFT at each time
        fft_matrix[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())

    kt = (fft_matrix[:, :arr.shape[1] // 2] * np.conj(fft_matrix[:, :arr.shape[1] // 2])).real
    return kt


def get_wk(component):
    arr = cf.get_array(component)
    
    num_times = arr.shape[0]

    df = 1. / (num_times * cf.dt_field)
    dk = 1. / (cf.NX * cf.dx)

    f  = np.arange(0, 1. / (2*cf.dt_field), df)
    k  = np.arange(0, 1. / (2*cf.dx), dk)
    
    fft_matrix  = np.zeros(arr.shape, dtype='complex128')
    fft_matrix2 = np.zeros(arr.shape, dtype='complex128')

    for ii in range(arr.shape[0]): # Take spatial FFT at each time
        fft_matrix[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())

    for ii in range(arr.shape[1]):
        fft_matrix2[:, ii] = np.fft.fft(fft_matrix[:, ii] - fft_matrix[:, ii].mean())

    wk = fft_matrix2[:f.shape[0], :k.shape[0]] * np.conj(fft_matrix2[:f.shape[0], :k.shape[0]])
    return k, f, wk