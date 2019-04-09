# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:29:15 2019

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt
from   scipy.optimize    import fsolve

'''
Equations from Wang et al. 2016
'''

def cold_plasma_dispersion_function(w, k):
    cold_sum = w_pe2 / (w * (w - elec_cyc))                # Add electrons
    for ii in range(N):
        cold_sum += w_pi2[ii] / (w * (w - ion_cyc[ii]))     # Add each ion species
    return 1 - cold_sum - (c * k / w) ** 2



    

if __name__ == '__main__':
    mp    = 1.673E-27         # kg
    me    = 9.109E-31         # kg
    q     = 1.602e-19         # C
    c     = 3E8               # m/s
    e0    = 8.854e-12
    mu0   = (4e-7) * np.pi

    H_frac   = 0.70
    He_frac  = 0.20
    O_frac   = 0.10
    
    N        = 3               # Number of species
    n0       = 200e6           # /m3
    field    = 487.5e-9        # T
    
    ndens    = np.zeros(N)
    ndens[0] = H_frac  * n0
    ndens[1] = He_frac * n0
    ndens[2] = O_frac  * n0

    mi    = np.zeros(N)
    mi[0] = 1.  * mp
    mi[1] = 4.  * mp
    mi[2] = 16. * mp

    w_pi2   = ndens       * q ** 2 / (mi * e0)
    w_pe2   = ndens.sum() * q ** 2 / (me * e0)
    
    ion_cyc  = q * field / mi
    elec_cyc = q * field / me
    
    rho       = (ndens * mi).sum()
    alfven    = field / np.sqrt(mu0 * rho)
    
    
    ######################
    ### SOLVE AND PLOT ###
    ######################
    plt.ioff()
    normalize_frequency = True
    species_colors      = ['r', 'b', 'g']
    
    # Initialize k space: Normalized by va/pcyc
    Nk     = 1000
    k_min  = 0.0  * ion_cyc[0] / alfven
    k_max  = 2.0  * ion_cyc[0] / alfven
    k_vals = np.linspace(k_min, k_max, Nk, endpoint=False)
    
    solns           = np.zeros((Nk, N)) 
    solns[0]        = np.array([ion_cyc[1], ion_cyc[2], 1e-10])*1.01   # Initial guess
    
    for ii in range(1, Nk):
        for jj in range(N):
            solns[ii, jj] = fsolve(cold_plasma_dispersion_function, x0=solns[ii - 1, jj], args=(k_vals[ii]))
    
    
    k_min  *= alfven / (ion_cyc[0])
    k_max  *= alfven / (ion_cyc[0]) 
    k_vals *= alfven / (ion_cyc[0])
    
    for ii in range(N):
        if normalize_frequency == True:
            plt.plot(k_vals[1:], solns[1:, ii] / ion_cyc[0], c=species_colors[ii], linestyle='--', label='Norm. Cold')
            plt.axhline(ion_cyc[ii] / ion_cyc[0], c='k', linestyle=':')
        else:
            plt.plot(k_vals[1:], solns[1:, ii] / (2 * np.pi), c=species_colors[ii], linestyle='--', label='Cold')
            plt.axhline(ion_cyc[ii] / (2 * np.pi), c='k', linestyle=':')
        
    plt.legend()
    plt.xlim(k_min, k_max)
    plt.xlabel(r'$kv_A / \Omega_p$')
    plt.ylabel(r'$\omega/\Omega_p$')
    plt.ylim(0, 1.0)
    plt.show()
