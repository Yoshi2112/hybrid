# -*- coding: utf-8 -*-
"""
Created on Sat May 23 14:24:12 2020

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

def g_func(VPERP):
    exp1 = np.exp(- VPERP ** 2 / (2*     vth_perp**2))
    exp2 = np.exp(- VPERP ** 2 / (2*beta*vth_perp**2))
    return (1. / (1. - beta)) * (exp1 - exp2)

def f_func(VPAR, VPERP):
    C    = n_eq / np.sqrt(np.pi ** 3 * 8 * vth_par ** 2 * vth_perp ** 4)
    exp0 = np.exp(- VPAR ** 2 / (2 * vth_par**2))
    G    = g_func(VPERP)
    return C * exp0 * G

if __name__ == '__main__':
    kB  = 1.38065e-23
    mu0 = 4e-7 * np.pi
    mp  = 1.673e-27
    
    vpar_lim  = 7
    vperp_lim = 7
    
    t_perp = 1e3 * 11603.
    t_par  = 1e3 * 11603.
    
    vth_par  = np.sqrt(kB * t_par  / mp)
    vth_perp = np.sqrt(kB * t_perp / mp)
    
    B_eq = 200e-9
    n_eq = 200e6
    va   = B_eq / np.sqrt(mu0 * n_eq * mp)
    
    Nv   = 200
    Nb   = 100
    
    bet    = np.linspace(0.0, 1.0, Nb)
    v_par  = np.linspace(- vpar_lim,  vpar_lim, Nv) * va
    v_perp = np.linspace(-vperp_lim, vperp_lim, Nv) * va
    
    plt.ioff()
    beta = 0.2
    LCD = np.zeros((Nv, Nv), dtype=np.float64)
    for ii in range(Nv):
        for jj in range(Nv):
            LCD[ii, jj] = f_func(v_par[ii], v_perp[jj])
    
    plt.figure(figsize=(15,10))
    
    if True:
        plt.pcolormesh(np.log10(LCD), vmin=-15, vmax=-10)
    else:
        plt.pcolormesh(LCD)
    
    
    plt.colorbar()
    plt.axis('equal')
    plt.title('$\\beta = %.2f$' % beta)
    
# =============================================================================
#     if False:
#         plt.savefig('F://runs//teset_LCD//' + 'LCD_{:04}.png'.format(mm))
#         plt.close('all')
#     else:
# =============================================================================
    plt.show()
        
    