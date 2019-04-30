# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:09:18 2019

@author: iarey
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

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
    A0 = 1.0   # Arb.
    f0 = 1.0   # Hz
    k0 = 1e-2  # /m
    
    t_min = 0.0     # Seconds
    t_max = 1000.
    dt    = 0.015625
    t     = np.arange(t_min, t_max, dt)
    Nt    = t.shape[0]
    
    x_min = 0.0
    x_max = 2000.
    dx    = 25.0
    x     = np.arange(x_min, x_max, dx)
    Nx    = x.shape[0]
    
    y     = np.zeros(Nx)
    z     = np.zeros(Nx)
    u     = np.zeros(Nx)
    
    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    
    ## Direction of propagation of a wave depends on the signs of kx and wt
    for ii in range(Nt):
        time         = t[ii]
        pos_exponent = k0*x - 2 * np.pi * f0 * time
        positive_LH  = A0 * np.exp( 1j * pos_exponent)
        positive_RH  = A0 * np.exp(-1j * pos_exponent)
        
        neg_exponent = - k0*x - 2 * np.pi * f0 * time
        negative_LH = A0 * np.exp( 1j * neg_exponent)
        negative_RH = A0 * np.exp(-1j * neg_exponent)
        
        total_wave = positive_LH + positive_RH + negative_LH + negative_RH
        
        ## Plot/Update quiver plot
        ax.quiver(x, y, z, u, total_wave.real, total_wave.imag, length=0.5, color='b')
# =============================================================================
#         ax.quiver(x, y, z, u, positive_LH.real, positive_LH.imag, length=0.5, color='b')
#         ax.quiver(x, y, z, u, positive_RH.real, positive_RH.imag, length=0.5, color='r')
#         
#         ax.quiver(x, y, z, u, negative_LH.real, negative_LH.imag, length=0.5, color='c')
#         ax.quiver(x, y, z, u, negative_RH.real, negative_RH.imag, length=0.5, color='m')
# =============================================================================
        
        ax.set_xlim(0, x_max)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
    
        plt.pause(0.1)
        ax.clear()
        
        ## Do Helicity Thing ## 
        
        
        
        
    plt.show()
    
    
# =============================================================================
#     fig, ax = plt.subplots()
#     for ii in range(N):
#         ax.axis([-2, 2, -2, 2])
#         ax.quiver(0, 0, positive_RH[ii].real, positive_RH[ii].imag, scale=1, units='xy', color='b')
#         ax.quiver(0, 0, positive_LH[ii].real, positive_LH[ii].imag, scale=1, units='xy', color='r')
#         plt.pause(0.1)
#         ax.clear()
#     plt.show()
#     
# =============================================================================
    