# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:56:34 2019

@author: c3134027
"""

from scipy.optimize import curve_fit
import lmfit as lmf
import matplotlib.pyplot as plt
import numpy as np

def field_line_length():
    sizen  = 401                                      # number of spatial grid points
    L      = 5.0                                      # L value of field line to solve along
    Re     = 6371000.                                 # Earth radius in m
    LRe    = L*Re
    anchor = 150000                                   # Altitude of Ionosphere anchor point (m)
    lat0   = np.arccos(np.sqrt((Re + anchor)/(Re*L))) # Latitude for this L value (at ionosphere height)
    h      = 2.0 * lat0 / (sizen - 1)                 # Step size of lambda (latitude)
    f_len  = 0
    
    for ii in range(sizen):
        lda    = -lat0 + ii*h
        f_len += LRe * np.cos(lda) * np.sqrt(4.0 - 3.0*np.cos(lda) ** 2) * h
    
    f_len /= Re
    print('Field line length: {}R_E'.format(round(f_len, 2)))
    return


def exponential_sine(t, amp, freq, growth, phase):
    return amp * np.sin(2*np.pi*freq*t + phase) * np.exp(growth*t)


def extract_growth_rate(arr, fi):
    
    gmodel = lmf.Model(exponential_sine, nan_policy='propagate')
    
    gmodel.set_param_hint('amp',    value=1.0, min=0.0,     max=abs(mode_matrix).max())
    gmodel.set_param_hint('freq',   value=fi, min=-2*fi,    max=2*fi)
    gmodel.set_param_hint('growth', value=0.05, min=0.0,    max=0.5*fi)
    gmodel.set_param_hint('phase',  value=0.0, vary=False)
    
    for mode_num in [1]:#range(1, k.shape[0]):
        data_to_fit = mode_matrix[:cut_idx, mode_num].real
    
        result      = gmodel.fit(data_to_fit, t=time_fit, method='leastsq')

        plt.plot(time_fit, data_to_fit, 'ko', label='data')
        plt.plot(time_fit, result.best_fit, 'r-', label='lmfit')

        popt, pcov = curve_fit(exponential_sine, time_fit, data_to_fit, maxfev=1000000)
        plt.plot(time_fit, exponential_sine(time_fit, *popt), label='curve_fit')
        plt.legend()
        print(popt)

    plt.show()
    return

if __name__ == '__main__':
    
    t_min = 0.0
    t_max = 15.0
    dt    = 0.1
    time  = np.arange(t_min, t_max, dt)
    
    a0    = 0.02
    f0    = 0.51
    phi   = 0.0
    gamma = 0.2

# =============================================================================
#     test_data = exponential_sine(time, a0, f0, gamma, phi)
#     plt.plot(time, test_data)
# =============================================================================
    
    field_line_length()