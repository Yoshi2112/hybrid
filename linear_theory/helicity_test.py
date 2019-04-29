# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:09:18 2019

@author: iarey
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_helicity(x, By, Bz):
    '''
    Could potentially contain a few signage issues, need to double check
    the maths of this when I have internet. But basic structure is there.
    
    Test on Left and Right-Hand polarised waves travelling in each +x and -x
    directions.
    (How to construct that from 2 transverse series?)
    '''
    dx      = x[1] - x[0]
    k_modes = np.fft.rfftfreq(x.shape[0], d=dx)
    By_fft  = np.fft.rfft(By)
    Bz_fft  = np.fft.rfft(Bz)
    
    # Four fourier coefficients from FFT (since real inputs give symmetric outputs)
    # Check this is correct. Also, potential signage issue?
    By_cos = By_fft.real
    By_sin = By_fft.imag
    Bz_cos = Bz_fft.real
    Bz_sin = Bz_fft.imag
    
    # Construct spiral mode k-coefficiencts
    Bk_pos = 0.5 * ( (By_cos + Bz_sin) + 1j * (Bz_cos - By_sin ) )
    Bk_neg = 0.5 * ( (By_cos - Bz_sin) + 1j * (Bz_cos + By_sin ) )
    
    # Construct spiral mode timeseries
    Bt_pos = np.zeros(x.shape[0])
    Bt_neg = np.zeros(x.shape[0])
    
    for ii in range(k_modes.shape[0]):
        Bt_pos += Bk_pos[ii] * np.exp(-1j*k_modes[ii]*x)
        Bt_neg += Bk_neg[ii] * np.exp( 1j*k_modes[ii]*x)
    return Bt_pos, Bt_neg


if __name__ == '__main__':
    A0 = 1.0
    f0 = 1.0 # Hz

    t_min = 0.0     # Seconds
    t_max = 1000.
    dt    = 0.015625
    t     = np.arange(t_min, t_max, dt)
    N     = t.shape[0]
    
    ## Direction of propagation of a wave depends on the signs of kx and wt
    

    
    