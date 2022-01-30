# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:51:03 2022

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt

def logistic_function(t):
    '''
    Should return a single value at some time t
    based on global parameters
    '''
    arg = -g_rate*(t - t_mid)
    return B0 + B_ULF / (1 + np.exp(arg))

def get_derivative(arr):
    '''
    Assume first and last derivatives are zero for the sigmoid
    '''
    deriv = np.zeros(arr.shape, arr.dtype)
    for ii in range(1, arr.shape[0]-1):
        deriv[ii] = (arr[ii + 1] - arr[ii - 1]) / (2*dt)
    return deriv

t_max  = 200.
t_arr  = np.linspace(0.0, t_max, 1000)
dt     = t_arr[1] - t_arr[0]

t_mid  = 100.       # Midpoint of slope
g_rate = 5e-1       # Larger value is steeper/faster change
B_ULF  = -5.0e-9    # Amplitude of slope. Negative for decrease
B0     = 200e-9     # Background field

B   = logistic_function(t_arr)
dB  = get_derivative(B)

plt.ioff()
fig, ax = plt.subplots(2)
ax[0].set_title(f'Comparison of Logistic Function Gradient with 20 mHz ULF :: k = {g_rate}')
ax[0].plot(t_arr, B*1e9, c='k')
ax[0].set_ylabel('nT')
ax1 = ax[0].twinx()
ax1.plot(t_arr, dB*1e9, c='k', ls='--')
ax1.set_ylabel('nT/s')


# Sample sine wave for GR comparison, frequency in Hz
f0    = 0.020
sine  = B0 + B_ULF * np.sin(2*np.pi*f0*t_arr)
dsine = get_derivative(sine)
ax[1].plot(t_arr, sine*1e9, c='b')
ax[1].set_ylabel('nT')
ax2 = ax[1].twinx()
ax2.plot(t_arr[1:-1], dsine[1:-1]*1e9, c='b', ls='--')
ax2.set_ylabel('nT/s')





fig.show()