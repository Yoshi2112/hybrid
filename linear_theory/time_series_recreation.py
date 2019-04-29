# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:20:40 2019

@author: iarey
"""

import numpy as np
import matplotlib.pyplot as plt
from convective_growth_rate import calculate_growth_rate

def reverse_array(arr):
    narr_pts = arr.shape[0]
    new_arr  = np.zeros(narr_pts, dtype=arr.dtype)
    
    for ii in range(narr_pts):
        new_arr[narr_pts - ii - 1] = arr[ii]
    return new_arr


def fft_method():
    '''
    Maybe try using the Hermitian FFT method for this?
    Real time-series/frequency components (nice symmetry), 
    less stuffing around.
    '''
    fft_array       = np.zeros(2*(N-1) + 1)          # Maybe make +2?
    fft_array[0]    = conv_growth.sum()
    fft_array[1:N ] = conv_growth[1:] 
    fft_array[N:]   = reverse_array(conv_growth[1:]) # Maybe make N+1?

    timeseries      = np.fft.ifft(fft_array)
    plt.figure()
    plt.plot(timeseries)
    return

def integral_method():
    # After leaving the generation region, the amplitudes of the frequencies will be 
    # given by the product of the CGR growth amplitudes and the size of the region.
    RE              = 6371000.                              # Earth radius in m 
    gen_region_size = 0.1*RE                                # Size of generation region (m)
    amplitudes      = conv_growth * gen_region_size
    
    A_max = amplitudes.max()
    df    = freq[-1] / N
    
    t_max = 1000.
    dt    = 0.01
    time  = np.arange(0., t_max, dt)
    
    wave = np.zeros(time.shape[0])
    for ii in range(N):
        wave += df * amplitudes[ii] / A_max * np.sin(2 * np.pi * freq[ii] * time)
        
    plt.figure()
    plt.plot(time, wave)
    plt.title('Wave Timeseries')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    return


if __name__ == '__main__':
    plot            = True
    N               = 500          # Discretization of frequencies
    magnetic_field  = 487.5           # nT

    cold_dens    = np.zeros(3)      # Cold plasma density (/cm3)
    cold_dens[0] = 10.
    cold_dens[1] = 0.
    cold_dens[2] = 0.
    
    warm_dens    = np.zeros(3)      # Warm plasma density (/cm3)
    warm_dens[0] = 5.0
    warm_dens[1] = 0.00
    warm_dens[2] = 0.00
    
    perp_temp    = np.zeros(3)      # Perpendicular temperature (eV)
    perp_temp[0] = 50000.
    perp_temp[1] = 00000.
    perp_temp[2] = 00000.

    anisotropy    = np.zeros(3)     # Anisotropy
    anisotropy[0] = 1.
    anisotropy[1] = 0.
    anisotropy[2] = 0.
    
    freq, conv_growth, stop_band = calculate_growth_rate(magnetic_field, cold_dens, warm_dens, 
                              anisotropy, temperp=perp_temp, norm_freq=0, maxfreq=1.0, NPTS=N)
    
    # Get rid of random infs
    for ii in range(N):
        if conv_growth[ii] == np.inf:
            conv_growth[ii] = 0.5*(conv_growth[ii - 1] + conv_growth[ii + 1])
    
    if plot == True:
        plt.figure()
        plt.plot(freq, conv_growth)
        plt.title('Frequency Space')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(r'Growth Rate ($\times 10^-7$ m^{-1})')
    
    integral_method()
    
        