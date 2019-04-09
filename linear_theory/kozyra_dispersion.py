# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy             as np
from emperics import sheely_plasmasphere, geomagnetic_magnitude
from   scipy.special     import wofz
from   scipy.optimize    import fsolve
import matplotlib.pyplot as plt
import pdb


def dispersion_function(w, k):
    '''
    w is a vector: [wr, wi]
    
    Function used in scipy.fsolve minimizer to find roots of dispersion relation.
    Iterates over each k to find values of w that minimize 'solution'.
    
    type_out allows purely real or purely imaginary (coefficient only) for root
    finding. Set as anything else for complex output. 
    
    Can have 
    '''
    wc = w[0] + 1j*w[1]
    
    # Sum over energetic ion components
    hot_components = 0
    for ii in range(N):
        pdisp_arg  = (wc - ion_cyc[ii]) / (alpha[ii]*k)
        pdisp_func = 1j*np.sqrt(np.pi)*wofz(pdisp_arg) / (alpha[ii]*k)
        brackets   = (A[ii] + 1) * (ion_cyc[ii] - wc) - ion_cyc[ii]
        
        hot_components += w_pw2[ii] * (A[ii] - pdisp_func * brackets)
    
    # Sum over cold ion components
    cold_components = 0
    for ii in range(N):
        cold_components += w_pc2[ii] * wc / (wc - ion_cyc[ii])
    
    # Add cold electron component
    cold_components += w_pe2 * wc / (wc - elec_cyc)
    
    # This equation should equal zero
    solution = (SPLIGHT ** 2) * (k ** 2) - hot_components + cold_components - (wc ** 2)
    return np.array([solution.real, solution.imag])

    

def cold_plasma_dispersion_function(w, k):
    
    cold_sum = w_pe2 / (w * (w - elec_cyc))                 # Add electrons
    for ii in range(N):
        cold_sum += w_pi2[ii] / (w * (w - ion_cyc[ii]))     # Add each ion species
    return 1 - cold_sum - (SPLIGHT * k / w) ** 2
  

# =============================================================================
# def create_legend(fn_ax):
#     legend_elements = []
#     for label, style in zip(run_labels, run_styles):
#         legend_elements.append(Line2D([0], [0], color='k', lw=1, label=label, linestyle=style))
#         
#     new_legend = fn_ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.6))
#     return new_legend, fn_ax
# =============================================================================


if __name__ == '__main__':
    PMASS    = 1.673E-24         # g
    EMASS    = 9.109E-28         # g
    CHARGE   = 4.80326e-10       # StatC (Fr)
    SPLIGHT  = 3E10              # cm/s
    BMANN    = 8.617e-5          # eV/K
    NPTS     = 500
    N        = 3
    L_shell  = 4
    
    plot_kozya = True
    plot_CPDR  = True
    
    field_nT = geomagnetic_magnitude(L_shell) * 1e-9  # T
    n0       = sheely_plasmasphere(L_shell)     # /cm3
    FIELD    = field_nT * 1e4                   # G

    H_frac   = 0.70
    He_frac  = 0.20
    O_frac   = 0.10
    RCH_frac = 0.90

    if round(H_frac + He_frac + O_frac, 8) != 1.0:
        raise ValueError('Ion fractions must sum to unity.')
    
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
    ndensw[1] = 0.
    ndensw[2] = 0.

    
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
    
    w_pc2   = 4 * np.pi * ndensc * CHARGE ** 2 / mi             # Cold ion
    w_pw2   = 4 * np.pi * ndensw * CHARGE ** 2 / mi             # Hot ion
    w_pe2   = 4 * np.pi * ndense * CHARGE ** 2 / EMASS          # Cold electron
    w_pi2   = 4 * np.pi * ndense * CHARGE ** 2 / mi             # Cold ion incl. hot density (for CPDR)
    
    ion_cyc  = CHARGE * FIELD / (mi    * SPLIGHT)
    elec_cyc = CHARGE * FIELD / (EMASS * SPLIGHT)
    
    etac    = M * (w_pc2 / w_pw2[0])
    etaw    = M * (w_pw2 / w_pw2[0])
    delta   = w_pc2[0] / w_pw2[0]
    
    EVJOULE = 6.242E18
    tper    = temperp / EVJOULE                               # Bit of a cheat: Converting to SI, then CGS
    tpar    = tper / (1.0 + A)                                # Temps converted from eV to J
    alpha   = np.sqrt(2.0 * tpar / mi) * 100                  # in m/s, converted to cm/s
    
    
    ########################
    ### SOLVERS: KOZYRA ####
    ########################
    plt.ioff()
    fig, ax = plt.subplots()
    
    rho       = ((ndensc + ndensw) * mi).sum()
    alfven    = FIELD / np.sqrt(4 * np.pi * rho)
        
    # Initialize k space: Normalized by va/pcyc. 
    k_min  = 0.   * ion_cyc[0] / alfven
    k_max  = 1.   * ion_cyc[0] / alfven 
    Nk     = 1000
    k_vals = np.linspace(k_min, k_max, Nk, endpoint=False)
    
# =============================================================================
#     k_min_plot  = k_min  * alfven / ion_cyc[0]
#     k_max_plot  = k_max  * alfven / ion_cyc[0]
#     k_vals_plot = k_vals * alfven / ion_cyc[0]
# =============================================================================

    species_colors   = ['r', 'b', 'g']
    initial_wr_guess = np.array([ion_cyc[1], ion_cyc[2], 1e-10])*1.01
    initial_wi_guess = np.array([        0.,         0., 0.])
    
    if plot_kozya == True:
        solns          = np.zeros((Nk, initial_wr_guess.shape[0], 2)) 
        
        for jj in range(N):
            solns[0, jj, 0] = initial_wr_guess[jj]
            solns[0, jj, 1] = initial_wi_guess[jj]
    
        for ii in range(1, Nk):
            '''For this k-value'''
            for jj in range(N):
                '''For this branch'''
                solns[ii, jj] = fsolve(dispersion_function, x0=solns[ii - 1, jj], args=(k_vals[ii]))
                
        w_real = solns[:, :, 0]
        w_imag = solns[:, :, 1]
        
        norm = True
        if norm == True:
            for ii in range(N):
                ax.plot(k_vals[1:], w_real[1:, ii] / ion_cyc[0], c=species_colors[ii], label='Norm. Kozyra')
                ax.axhline(ion_cyc[ii] / ion_cyc[0], c='k', linestyle=':')
                ax.set_ylabel(r'$\omega/\Omega_p$')
        else:
            for ii in range(N):
                ax.plot(k_vals[1:], w_real[1:, ii]/ (2 * np.pi), c=species_colors[ii], label='Kozyra')
                ax.axhline(ion_cyc[ii] / (2 * np.pi), c='k', linestyle=':')
                ax.set_ylabel(r'f (Hz)')
            
        ax.set_title('Dispersion Relation: Kozyra vs. CPDR')
        ax.set_xlim(k_min, k_max)
        ax.set_xlabel(r'$kv_A / \Omega_p$')
        ax.set_ylim(0, 1.0)
        plt.show()
    
    if plot_CPDR == True:
        #######################
        ### COLD DISPERSION ###
        #######################
        normalize_frequency = True
        species_colors      = ['r', 'b', 'g']
        
        CPDR_solns          = np.zeros((Nk, N)) 
        CPDR_solns[0]       = initial_wr_guess   # Initial guess
        
        for ii in range(1, Nk):
            for jj in range(N):
                CPDR_solns[ii, jj] = fsolve(cold_plasma_dispersion_function, x0=CPDR_solns[ii - 1, jj], args=(k_vals[ii]))
        
        for ii in range(N):
            if normalize_frequency == True:
                ax.plot(k_vals[1:], CPDR_solns[1:, ii] / ion_cyc[0], c=species_colors[ii], linestyle='--', label='Norm. Cold')
                ax.axhline(ion_cyc[ii] / ion_cyc[0], c='k', linestyle=':')
                ax.set_ylabel(r'$\omega/\Omega_p$')
            else:
                ax.plot(k_vals[1:], CPDR_solns[1:, ii] / (2 * np.pi), c=species_colors[ii], linestyle='--', label='Cold')
                ax.axhline(ion_cyc[ii] / (2 * np.pi), c='k', linestyle=':')
                ax.set_ylabel(r'f (Hz)')
                
        ax.set_xlim(k_min, k_max)
        ax.set_xlabel(r'$kv_A / \Omega_p$')
        ax.set_ylim(0, 1.0)
    plt.show()
