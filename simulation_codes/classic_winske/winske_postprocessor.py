# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:13:37 2018

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_fourier_analyses(arr):
    arr_kt     = np.zeros(arr.shape, dtype=complex)
    arr_wk     = np.zeros(arr.shape, dtype=complex)
    
    t = np.arange(ntimes) * dt
    k = np.arange(nx)
    
    # For each time (spatial FFT)
    for ii in range(arr.shape[0]):
        arr_kt[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())
        
    # For each gridpoint (temporal FFT)
    for jj in range(arr.shape[1]):
        arr_wk[:, jj] = np.fft.fft(arr_kt[:, jj] - arr_kt[:, jj].mean())
        
    # Conjugates
    power_k = (arr_kt[:, :arr.shape[1]/2] * np.conj(arr_kt[:, :arr.shape[1]/2])).real
    power   = (arr_wk[:arr.shape[0]/2, :arr.shape[1]/2] * np.conj(arr_wk[:arr.shape[0]/2, :arr.shape[1]/2])).real
    
    # Reals only
    # power_k = arr_kt[:, :arr.shape[1]/2].real
    # power   = arr_wk[:arr.shape[0]/2, :arr.shape[1]/2].real
    
    plt.ioff()
    fig = plt.figure(1, figsize=(12, 8))
    ax  = fig.add_subplot(111)
    
    ax.pcolormesh(np.log10(power[1:, 1:]), cmap='jet')
    ax.set_ylim(0, 100)
    ax.set_title(r'w-k Plot: $\omega/k$ (Winske code)')
    ax.set_ylabel(r'$\omega$', rotation=0)
    ax.set_xlabel('k (m-number?)')
    
    fullpath = plot_path + 'wk' + '.png'
    fig.savefig(fullpath, edgecolor='none', bbox_inches='tight')
    plt.close()
    print 'w-k Plot saved'
    
    fig2 = plt.figure(2, figsize=(12, 8))
    ax2  = fig2.add_subplot(111)
    
    ax2.pcolormesh(k[:arr.shape[1]/2], t, power_k, cmap='jet')
    ax2.set_xlim(None, 32)
    ax2.set_title(r'k-t Plot: $\omega/k$ (Winske code)')
    ax2.set_ylabel(r'$\Omega_i t$', rotation=0)
    ax2.set_xlabel('k (m-number?)')
    
    fullpath = plot_path + 'kt' + '.png'
    fig2.savefig(fullpath, edgecolor='none', bbox_inches='tight')
    plt.close()
    print 'k-t Plot saved'
    return


def waterfall_plot(field):
    plt.ioff()
    
    amp   = 100.                 # Amplitude multiplier of waves: 
    
    cells  = np.arange(nx)
    
    for (ii, t) in zip(np.arange(ntimes), np.arange(0, ntimes*dt, dt)):
        #if round(t, 2)%0.5 == 0:
        plt.plot(cells, amp*(arr[ii] / arr.max()) + ii, c='k', alpha=0.05)
        
    plt.xlim(0, nx)
    plt.show()
    return


if __name__ == '__main__':
    data_path = 'E://runs//winske_anisotropy_test//vanilla_winske//save_data//'
    plot_path = 'E://runs//winske_anisotropy_test//vanilla_winske//plots//'
    
    arr        = np.load(data_path + 'BYS' + '.npy')
    
    ntimes = arr.shape[0]
    nx     = arr.shape[1]
    dt     = 0.05
    
    generate_fourier_analyses(arr)
    #waterfall_plot(arr)