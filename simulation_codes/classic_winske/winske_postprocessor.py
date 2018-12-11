# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:13:37 2018

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

field_path = 'F://runs//winske_anisotropy_test//vanilla_winske//fields//'
arr        = np.load(field_path + 'BYS' + '.npy')

arr_kt     = np.zeros(arr.shape, dtype=complex)
arr_wk     = np.zeros(arr.shape, dtype=complex)

# For each time (spatial FFT)
for ii in range(arr.shape[0]):
    arr_kt[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())
    
# =============================================================================
# # For each gridpoint (temporal FFT)
# for jj in range(arr.shape[1]):
#     arr_wk[:, jj] = np.fft.fft(arr_kt[:, jj] - arr_kt[:, jj].mean())
#     
# power = (arr_wk[:arr.shape[0]/2, :arr.shape[1]/2] * np.conj(arr_wk[:arr.shape[0]/2, :arr.shape[1]/2])).real
#     
# plt.pcolormesh(np.log10(power[1:, 1:]), cmap='jet')
# plt.ylim(0, 100)
# plt.show()
# =============================================================================

power_k = (arr_kt[:, :arr.shape[1]/2] * np.conj(arr_kt[:, :arr.shape[1]/2])).real

plt.pcolormesh(power_k, cmap='jet')
plt.xlim(None, 32)
plt.show()