# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:26:59 2019

@author: iarey
"""
import sys
data_scripts_dir = 'D://Google Drive//Uni//PhD 2017//Data//Scripts//'
sys.path.append(data_scripts_dir)

import numpy as np
import matplotlib.pyplot as plt
from analysis_scripts import analytic_signal
from peak_detectors_tests import new_peak_finder

if __name__ == '__main__':
    fig, ax = plt.subplots()
    
    k_freqs = [1e-4, 1.01e-4]
    t_freqs = [1.0, 1.50]
    amps    = [1.0, 1.0]
    phase   = [0.0, 0.0]
    
    Ns      = len(k_freqs)
    RE      = 6371000.
    
    x_min = 0
    x_max = 3. * RE
    dx    = 1e-3*RE
    x     = np.arange(x_min, x_max, dx)
    
    t_min = 0
    t_max = 10.
    dt    = 0.05
    t     = np.arange(t_min, t_max, dt)
    
    for time in [0]:
        signal = np.zeros(x.shape[0], dtype=np.complex128)
        for ii in range(Ns):
            exponent = k_freqs[ii] * x - 2 * np.pi * t_freqs[ii] * time
            signal  += np.exp(1j * (exponent + phase[ii]))
            #signal += amps[ii] * np.sin(k_freqs[ii]* - 2. * np.pi * t_freqs[ii] * time)
        
        signal_envelope = analytic_signal(signal, dt=dx)
        
        ax.plot(x / RE, signal.real)
        ax.plot(x / RE, signal_envelope, c='r')
        ax.set_xlabel('x ($R_E$)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(x_min/RE, x_max/RE)
        
        #plt.pause(0.1)
        #ax.clear()
        
        #figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()    
    plt.show()
    
    