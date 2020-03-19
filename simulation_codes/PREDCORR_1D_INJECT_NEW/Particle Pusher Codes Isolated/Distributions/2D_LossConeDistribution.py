# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:26:31 2020

@author: iarey
"""
import numpy as np
import matplotlib.pyplot as plt

## Constants ##
c      = 2.998925e+08               # Speed of light (m/s)
mp     = 1.672622e-27               # Mass of proton (kg)
kB     = 1.380649e-23               # Boltzmann's Constant (J/K)
mu0    = (4e-7) * np.pi             # Magnetic Permeability of Free Space (SI units)

A_list = np.arange(0, 10, 0.01)
for xx in range(A_list.shape[0]):
    ## INPUTS ##
    A      = A_list[xx]                 # Anisotropy : T_perp / T_parallel - 1
    T_para = 100.0                      # Parallel temperature in eV
    ni     = 200e6                      # Ion density
    B      = 200e-9                     # Local magnetic field
    N      = 500
    delta  = 0.5                        # Loss Cone Fullness Parameter (0 = Empty, 1 = Full/Maxwellian)
    
    v_para_max  =  2
    v_para_min  = -2
    v_para      = np.linspace(v_para_min, v_para_max, N)
    
    v_perp_max  =  2
    v_perp_min  = -2
    v_perp      = np.linspace(v_perp_min, v_perp_max, N)
    
    
    ## CALCULATED QUANTITES ## 
    T_perp = T_para*(A + 1)                             # Perpendicular temperature in eV
    vth_para = np.sqrt(kB * T_para * 11603. / mp)       # Thermal velocity
    vth_perp = np.sqrt(kB * T_perp * 11603. / mp)       # Thermal velocity
    va       = B / np.sqrt(mu0 * ni * mp)               # Alfven speed (m/s)
    T        = (T_perp + T_para) * 11603.               # Total plasma temperature
    beta     = 2 * mu0 * ni * kB * T / B ** 2           # Plasma beta
    
    
    ## CALCULATE PHASE SPACE ##
    psd_para  = np.zeros(v_para.shape[0], dtype=np.float64)
    psd_perp  = np.zeros(v_perp.shape[0], dtype=np.float64)
    psd_value = np.zeros((v_para.shape[0], v_perp.shape[0]), dtype=np.float64)
    
    para_outer   = 1. / (np.sqrt(2 * np.pi) * vth_para)
    for ii in range(N):
        vx = v_para[ii] * va
        
        para_expon   = np.exp(- 0.5 * vx ** 2 / vth_para ** 2)
        psd_para[ii] = para_outer*para_expon
        
        
    perp_outer = 1. / (2 * np.pi * vth_perp ** 2)
    exp_outer  = ((1 - delta) / (1 - beta))
    for jj in range(N):
        vy = v_perp[jj] * va
        
        exp1      = delta * np.exp(-0.5 * vy**2 / vth_perp**2)
        
        exp2      = np.exp(-0.5 * vy**2 /         vth_perp**2)
        exp3      = np.exp(-0.5 * vy**2 / (beta * vth_perp**2))
    
        psd_perp[jj] = perp_outer * (exp1 + exp_outer * (exp2 - exp3))
    
    for ii in range(N):
        for jj in range(N):
            psd_value[ii, jj] = ni * psd_para[ii] * psd_perp[jj]
          
    plt.ioff()
    plt.figure(figsize=(20, 10))
    plt.pcolormesh(v_para, v_perp, psd_value.T, cmap='jet')
    plt.axis('equal')
    plt.title('Loss Cone Distribution')
    plt.ylabel('$v_\perp (v_A^{-1})$',     rotation=0, fontsize=14, labelpad=30)
    plt.xlabel('$v_\parallel (v_A^{-1})$', rotation=0, fontsize=14)
    
    plt.xlim(v_para_min, v_para_max)
    plt.ylim(v_perp_min, v_perp_max)
    plt.colorbar().set_label('Number density ($m^{-3}$', fontname='monospace', fontsize=14, labelpad=20)
    
    fontname='monospace'
    fontsize=14
    
    plt.figtext(0.63, 0.86, '$T_\parallel$ = %d eV' % T_para, fontname=fontname, fontsize=fontsize)
    plt.figtext(0.63, 0.83, '$T_\perp$ = %d eV' % T_perp, fontname=fontname, fontsize=fontsize)
    plt.figtext(0.63, 0.80, '$A$ = %.2f' % A, fontname=fontname, fontsize=fontsize)
    plt.figtext(0.63, 0.77, '$B_0$ = %.1f nT' % (B*1e9), fontname=fontname, fontsize=fontsize)
    plt.figtext(0.63, 0.74, '$n_0$ = %.1f cc' % (ni/1e6), fontname=fontname, fontsize=fontsize)
    plt.figtext(0.63, 0.71, '$\Delta$ = %d' % delta, fontname=fontname, fontsize=fontsize)
    
    if True:
        filedir  = 'F://runs//LCD//'
        filename = 'LCD_{:05d}'.format(xx)
        plt.savefig(filedir + filename)
        print('Plot saved as {}'.format(filedir + filename))
        plt.close('all')
    else:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()