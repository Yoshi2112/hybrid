# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:29:15 2019

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt
from   scipy.optimize    import fsolve
from   scipy.special     import wofz
import pdb

'''
Equations from Wang et al. 2016. Can't seem to work out how eq. 6 applies for T = 0... 
How to code this?
'''
def wofz_vs_linear():
    '''
    Test function to see what the last term of equation (6) tends to as T -> 0
    
    It tends to ion_cyc / (ion_cyc - w)
    '''
    N_test     = 500
    test_w     = 0.75 * ion_cyc[0]
    test_k     = 0.001
    amin       = -20
    amax       = 5
    
    test_alpha = np.logspace(amin, amax, N_test)
    inverse    = np.zeros(N_test)
    pdisp      = np.zeros(N_test, dtype=np.complex128)
    
    for ii in range(N_test):
        inverse[ii]= ion_cyc[0] / (test_k * test_alpha[ii])
        pdisp_arg  = (test_w - ion_cyc[0]) / (test_alpha[ii]*test_k)
        pdisp[ii]  = 1j*np.sqrt(np.pi)*wofz(pdisp_arg)
    
    product = inverse*pdisp
    print(product[0])
    print('pcyc: {}'.format(ion_cyc[0]))
    print('tstw: {}'.format(test_w))
    print('arg: {}'.format(ion_cyc[0] / (ion_cyc[0] - test_w)))
    return


def cold_plasma_dispersion_function(w, k):
    
    cold_sum = w_pe2 / (w * (w - elec_cyc))                 # Add electrons
    for ii in range(N):
        cold_sum += w_pi2[ii] / (w * (w - ion_cyc[ii]))     # Add each ion species
    return 1 - cold_sum - (c * k / w) ** 2


def full_maxwellian_dispersion_function(w, k, type_out='real'):
    '''
    Accounts for the anisotropy and temperature of the warm proton component
    '''
    # Sum over energetic ion components
    hot_components = 0
    for ii in range(N):
        pdisp_arg  = (w - ion_cyc[ii]) / (alpha[ii]*k)
        pdisp_func = 1j*np.sqrt(np.pi)*wofz(pdisp_arg) / (alpha[ii]*k)
        brackets   = (A[ii] + 1) * (ion_cyc[ii] - w) - ion_cyc[ii]
        
        hot_components += w_pw2[ii] * (A[ii] - pdisp_func * brackets)
    
    # Sum over cold ion components
    cold_components = 0
    for ii in range(N):
        cold_components += w_pc2[ii] * w / (w - ion_cyc[ii])
    
    # Add cold electron component
    cold_components += w_pe2 * w / (w - elec_cyc)
    
    # This equation should equal zero
    solution = (c ** 2) * (k ** 2) - hot_components + cold_components - (w ** 2)
        
    if type_out == 'real':
        return solution.real
    elif type_out == 'imag':
        return solution.imag
    else:
        return solution
    

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
    
    ndensc    = np.zeros(N)    # Solve same equation but with T = 0?
    ndensc[0] = H_frac  * n0
    ndensc[1] = He_frac * n0
    ndensc[2] = O_frac  * n0
    
    ndensw    = np.zeros(N)
    ndensw[0] = H_frac  * n0
    ndensw[1] = He_frac * n0
    ndensw[2] = O_frac  * n0


    # Input the perpendicular temperature (ev)
    t_perp    = np.zeros(N)
    t_perp[0] = 50000.
    t_perp[1] = 10000.
    t_perp[2] = 10000.
    
    
    # Input the parallel temperature (ev)
    t_par    = np.zeros(N)
    t_par[0] = 25000.
    t_par[1] = 10000.
    t_par[2] = 10000.
    A = t_perp / t_par - 1


    mi    = np.zeros(N)
    mi[0] = 1.  * mp
    mi[1] = 4.  * mp
    mi[2] = 16. * mp

    w_pc2   = ndensc                  * q ** 2 / (mi * e0)   # Cold ion plasma frequencies
    w_pw2   = ndensw                  * q ** 2 / (mi * e0)   # Warm ion plasma frequency
    w_pe2   = (ndensc + ndensw).sum() * q ** 2 / (me * e0)   # Electron plasma frequency
    w_pi2   = (ndensc + ndensw).sum() * q ** 2 / (mi * e0)   # Cold approx plasma frequencies (sum of densities)
    
    ion_cyc  = q * field / mi
    elec_cyc = q * field / me
    
    rho       = ((ndensc + ndensw) * mi).sum()
    alfven    = field / np.sqrt(mu0 * rho)
    
    t_par    *= q                                            # Convert temperatures in eV to Joules
    t_perp   *= q
    alpha     = np.sqrt(2.0 * t_par / mi)                    # in m/s, converted to cm/s ? Gives far more valid results

# =============================================================================
#     ######################
#     ### SOLVE AND PLOT ###
#     ######################
#     plt.ioff()
#     normalize_frequency = True
#     species_colors      = ['r', 'b', 'g']
#     
#     # Initialize k space: Normalized by va/pcyc
#     Nk     = 1000
#     k_min  = 0.0  * ion_cyc[0] / alfven
#     k_max  = 2.0  * ion_cyc[0] / alfven
#     k_vals = np.linspace(k_min, k_max, Nk, endpoint=False)
#     
#     solns           = np.zeros((Nk, N)) 
#     solns[0]        = np.array([ion_cyc[1], ion_cyc[2], 1e-10])*1.01   # Initial guess
#     
#     for ii in range(1, Nk):
#         for jj in range(N):
#             solns[ii, jj] = fsolve(cold_plasma_dispersion_function, x0=solns[ii - 1, jj], args=(k_vals[ii]))
#     
#     k_min  *= alfven / (ion_cyc[0])
#     k_max  *= alfven / (ion_cyc[0]) 
#     k_vals *= alfven / (ion_cyc[0])
#     
#     for ii in range(N):
#         if normalize_frequency == True:
#             plt.plot(k_vals[1:], solns[1:, ii] / ion_cyc[0], c=species_colors[ii], linestyle='--', label='Norm. Cold')
#             plt.axhline(ion_cyc[ii] / ion_cyc[0], c='k', linestyle=':')
#         else:
#             plt.plot(k_vals[1:], solns[1:, ii] / (2 * np.pi), c=species_colors[ii], linestyle='--', label='Cold')
#             plt.axhline(ion_cyc[ii] / (2 * np.pi), c='k', linestyle=':')
#         
#     plt.legend()
#     plt.xlim(k_min, k_max)
#     plt.xlabel(r'$kv_A / \Omega_p$')
#     plt.ylabel(r'$\omega/\Omega_p$')
#     plt.ylim(0, 1.0)
#     plt.show()
# =============================================================================
