# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 20:16:30 2020

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    t_max = 30.0        # Max time (s)
    dt    =  0.001      # Timestep (s)
    t_arr = np.arange(0, t_max, dt)
    
    b_amp   = 20e-9     # Driven wave frequency (T)
    b_wnum  = 0.0       # Driven wave wavenumber (not used yet)
    b_freq  = 1.2       # Driven wave linear frequency (Hz)
    b_phase = 0.0       # Driven wave phase
    
    sine = b_amp * np.sin(2 * np.pi * b_freq * t_arr + b_phase * np.pi / 180.)
    
    if False:
        plt.plot(t_arr, sine)
    
    A = 1.0    # Envelope amplitude
    B = 5.0    # Time for maximum peak (Mean)
    C = 1.0    # Width of peak (Doubled SD)
    
    # At B        : 100%     of A
    #    B +/- C  : 36%
    #    B +/- 2C : 1.8%
    #    B +/- 3C : 0.0123%
    
    gaussian = A * np.exp(- ((t_arr - B)/ C) ** 2 )
    
    if False:
        plt.plot(t_arr, gaussian)
        plt.axvline(B, c='k', alpha=1.0)
        
        for ii in range(1, 4):
            for jj in [-1.0, 1.0]:
                plt.axvline(B + jj*ii*C, c='k', alpha=0.25)
    
    enveloped_sine = sine * gaussian
    
    if True:
        plt.plot(t_arr, enveloped_sine)
    
            
        
    
        
        
        