# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy             as np
from   scipy.special     import wofz
from   scipy.optimize    import fsolve
import matplotlib.pyplot as plt
import cxroots           as cx
import pdb


def dispersion_function(w, k, type_out='real'):
    '''
    Function used in scipy.fsolve minimizer to find roots of dispersion relation.
    Iterates over each k to find values of w that minimize 'solution'.
    
    May not work properly? Does having a purely real w negate the benefits of even
    using wofz? This works for now, but need to use complex root finder to take full
    advantage of this code.
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
    solution = (SPLIGHT ** 2) * (k ** 2) - hot_components + cold_components - (w ** 2)
        
    if type_out == 'real':
        return solution.real
    elif type_out == 'imag':
        return solution.imag
    else:
        return solution
  



if __name__ == '__main__':
    PMASS    = 1.673E-24         # g
    EMASS    = 9.109E-28         # g
    CHARGE   = 4.80326e-10       # StatC (Fr)
    SPLIGHT  = 3E10              # cm/s
    BMANN    = 8.617e-5          # eV/K
    NPTS     = 500
    N        = 3
    
    field_nT = 487.5e-9          # nT
    FIELD    = field_nT * 1e4    # G
        
    H_frac   = 0.70
    He_frac  = 0.20
    O_frac   = 0.10
    RCH_frac = 0.60
    
    n0       = 200.
    
    # Index 1 denotes hydrogen ; 2 denotes helium; 3 denotes oxygen etc.
    M    = np.zeros(N)
    M[0] = 1.0     #; Hydrogen
    M[1] = 4.0     #; Helium
    M[2] = 16.0    #; Oxygen
    
    # Print,' Input densities of cold species (number/cc) [3]'
    ndensc    = np.zeros(N)
    ndensc[0] = H_frac  * n0 * (1. - RCH_frac)
    ndensc[1] = He_frac * n0
    ndensc[2] = O_frac  * n0
    
    # Density of warm species (same order as cold) (number/cc)
    ndensw    = np.zeros(N)
    ndensw[0] = H_frac * n0 * RCH_frac
    
    # Input the perpendicular temperature (ev)
    temperp    = np.zeros(N)
    temperp[0] = 25000.
    temperp[1] = 10000.
    temperp[2] = 10000.
    
    # Input the temperature anisotropy
    A    = np.zeros(N)
    A[0] = 1.0
    A[1] = 1.
    A[2] = 1.
    
    ###########################
    ### COMPUTE SOME THINGS ###
    ###########################
    mi      = M * PMASS
    ndense  = (ndensc + ndensw).sum()
    
    w_pc2   = 4 * np.pi * ndensc * CHARGE ** 2 / mi
    w_pw2   = 4 * np.pi * ndensw * CHARGE ** 2 / mi
    w_pe2   = 4 * np.pi * ndense * CHARGE ** 2 / EMASS
    
    ion_cyc  = CHARGE * FIELD / (2 * np.pi * mi    * SPLIGHT)
    elec_cyc = CHARGE * FIELD / (2 * np.pi * EMASS * SPLIGHT)
    
    etac    = M * (w_pc2 / w_pw2[0])
    etaw    = M * (w_pw2 / w_pw2[0])
    delta   = w_pc2[0] / w_pw2[0]
    
    TPAR    = temperp / (1.0 + A)
    
    if False:
        alpha   = np.sqrt(BMANN * TPAR / mi)                      # NOT RIGHT... alpha must end up as cm/s
    elif True:
        EVJOULE = 6.242E18
        tper    = temperp / EVJOULE                             
        tpar    = tper / (1.0 + A)                                # Temps converted from eV to J?
        alpha   = np.sqrt(2.0 * tpar / mi) * 100                  # in m/s, converted to cm/s ? Gives far more valid results
    
    
    ######################
    ### SOLVE AND PLOT ###
    ######################
    plt.ioff()
    
    # Number of active species (solutions)
    rho       = ((ndensc + ndensw) * mi).sum()
    alfven    = FIELD / np.sqrt(4 * np.pi * rho)
        
    # Initialize k space: Normalized by va/pcyc. 
    k_min  = 0.   * 2 * np.pi * ion_cyc[0] / alfven
    k_max  = 2.   * 2 * np.pi * ion_cyc[0] / alfven 
    Nk     = 1000
    k_vals = np.linspace(k_min, k_max, Nk, endpoint=False)
    
    species_colors = ['r', 'b', 'g']
    initial_guesses = np.array([ion_cyc[1], ion_cyc[2], 0.])*1.01
    
    solns          = np.zeros((Nk, initial_guesses.shape[0])) 
    solns[0]       = initial_guesses
    
    for ii in range(1, Nk):
        for jj in range(N):
            solns[ii, jj] = fsolve(dispersion_function, x0=solns[ii - 1, jj], args=(k_vals[ii], 'real'))
    
    # Multiply k-values by 100 to convert from /cm to /m
    k_min  *= alfven / (2 * np.pi * ion_cyc[0])
    k_max  *= alfven / (2 * np.pi * ion_cyc[0])
    k_vals *= alfven / (2 * np.pi * ion_cyc[0])
    
    norm = True
    if norm == True:
        solns /= ion_cyc[0]
        
        for ii in range(N):
            plt.plot(k_vals[1:], solns[1:, ii], c=species_colors[ii], label='Norm. Kozyra')
            plt.axhline(ion_cyc[ii] / ion_cyc[0], c='k', linestyle=':')
    else:
        for ii in range(N):
            plt.plot(k_vals[1:], solns[1:, ii], c=species_colors[ii], label='Kozyra')
            plt.axhline(ion_cyc[ii], c='k', linestyle=':')
        
    plt.legend()
    plt.xlim(k_min, k_max)
    plt.xlabel(r'$kv_A / \Omega_p$')
    plt.ylabel(r'$\omega/\Omega_p$')
    plt.ylim(0, 1.0)
    plt.show()
    
    
    
    