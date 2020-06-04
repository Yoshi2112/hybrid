# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:29:15 2019

@author: Yoshi
"""

import numpy as np
from   scipy.optimize    import fsolve
from   scipy.special     import wofz
import extract_parameters_from_data as data
import os
'''
Equations from Wang et al. 2016. Is for cold species of warm dispersion
relation simplify for alpha = 0 under the asymptotic expansion of the plasma
dispersion function. Though Wang was fairly inconsistent with the placing of
his signs. Slight differences for Fig 3c (?).

Should double check this against the Wang code at some point. Also multi the wang code (verification is never bad)
'''
def get_k_CPDR(wr):
    cold_sum = w_pe2 / (wr * (wr - e_cyc))                      # Electrons
    for ii in range(w_cyc.shape[0]):
        if ndens[ii] != 0:
            if wr == w_cyc[ii]:
                return np.inf
            else:
                cold_sum += w_ps2[ii] / (wr * (wr - w_cyc[ii]))     # Add each ion species
    return np.sqrt((wr / c) ** 2 - cold_sum * (wr / c) ** 2)

def cold_plasma_dispersion_relation(w, k):
    cold_sum = w_pe2 / (w * (w - e_cyc))                  # Electrons
    for ii in range(w_ps2.shape[0]):
        cold_sum += w_ps2[ii] / (w * (w - w_cyc[ii]))     # Add each ion species
    return 1 - cold_sum - (c * k / w) ** 2

def Z(arg):
    '''Return Plasma Dispersion Function : Normalized Fadeeva function'''
    return 1j*np.sqrt(np.pi)*wofz(arg)

def warm_plasma_dispersion_relation(wt, k, A, A2):
    '''
    w is a vector: [wr, wi]
    
    Function used in scipy.fsolve minimizer to find roots of dispersion relation.
    Iterates over each k to find values of w that minimize 'solution'.
    
    type_out allows purely real or purely imaginary (coefficient only) for root
    finding. Set as anything else for complex output.
    
    Plasma dispersion function related to Fadeeva function (Summers & Thorne, 1993) by
    i*sqrt(pi) factor.
    
    test_output flag allows w and solution to be complex, rather than wrapped. Additional
    arguments may also be selected to be exported this way.
    '''
    wc = wt[0] + 1j*wt[1]
    
    if any(np.isnan(wt) == True):
        return np.array([np.nan, np.nan])

    components = (w_pe2) * wc / (e_cyc - wc)    # Electrons

    for ii in range(w_cyc.shape[0]):
        if w_pc2[ii] != 0:
            if (w_cyc[ii] - wc) == 0:
                Isc = np.inf
            else:
                Isc = wc / (w_cyc[ii] - wc)
            components += (w_pc2[ii]) * Isc

        if w_pw2[ii] != 0:
            pdisp_arg   = (wc - w_cyc[ii]) / (alpha_par[ii]*k)
            pdisp_func  = Z(pdisp_arg)*w_cyc[ii] / (alpha_par[ii]*k)
            brackets    = (A[ii] + 1) * (wc - w_cyc[ii])/w_cyc[ii] + 1
            Isw         = brackets * pdisp_func + A[ii]
            components += w_pw2[ii] * Isw
            
        if w_pw2b[ii] != 0:
            pdisp_arg2  = (wc - w_cyc[ii]) / (alpha_par2[ii]*k)
            pdisp_func2 = Z(pdisp_arg2)*w_cyc[ii] / (alpha_par2[ii]*k)
            brackets    = (A2[ii] + 1) * (wc - w_cyc[ii])/w_cyc[ii] + 1
            Isw2        = brackets * pdisp_func2 + A2[ii]
            components += w_pw2b[ii] * Isw2

    solution = (wc ** 2) - (c * k) ** 2 + components
    return np.array([solution.real, solution.imag])
    

def estimate_first_and_complexify(solutions):
    '''
    Sets dispersion solution for k = 0 (currently just nearest-point), and
    transforms the 2xN array of reals into a 1xN array of complex w = w_r + i*w_i
    '''
    outarray = np.zeros((solutions.shape[0], solutions.shape[1]), dtype=np.complex128)
    
    for jj in range(solutions.shape[1]):
        for ii in range(solutions.shape[0]):
            outarray[ii, jj] = solutions[ii, jj, 0] + 1j*solutions[ii, jj, 1]
    
    # Set value for k = 0
    outarray[0] = outarray[1]   
    return outarray


def set_frequencies_and_variables(field, ndensc, ndensw, t_perp, A, ndensw2, t_perp2, A2):
    global ndens, w_pc2, w_ps2, w_pe2, w_cyc, p_cyc, e_cyc, alfven
    global w_pw2,  alpha_par
    global w_pw2b, alpha_par2
    global c

    c     = 3E8                                 # m/s
    mp    = 1.673E-27                           # kg
    me    = 9.109E-31                           # kg
    q     = 1.602e-19                           # C
    qe    = -q
    e0    = 8.854e-12                           # F/m
    mu0   = (4e-7) * np.pi                      # units
    
    mi    = np.zeros(3)
    mi[0] = 1.  * mp
    mi[1] = 4.  * mp
    mi[2] = 16. * mp

    qi    = np.zeros(3)
    qi[0] = 1.0*q
    qi[1] = 1.0*q
    qi[2] = 1.0*q

    t_par   = t_perp  / (A + 1)
    t_par2  = t_perp2 / (A2 + 1)
    ndens   = ndensc + ndensw + ndensw2
    
    w_pc2   = ndensc      * qi ** 2 / (mi * e0)     # Cold      ion plasma frequencies (rad/s)
    w_pw2   = ndensw      * qi ** 2 / (mi * e0)     # Warm      ion plasma frequencies
    w_pw2b  = ndensw2     * qi ** 2 / (mi * e0)     # Warm2     ion plasma frequencies
    w_ps2   = ndens       * qi ** 2 / (mi * e0)     # Total     ion plasma frequencies
    w_pe2   = ndens.sum() * qe ** 2 / (me * e0)     # Electron  ion plasma frequencies
    
    w_cyc   =  q * field / mi                       # Ion      cyclotron frequencies (rad/s)
    p_cyc   =  q * field / mp                       # Proton   cyclotron frequency (used for normalization)
    e_cyc   =  qe* field / me                       # Electron cyclotron frequency
    
    rho       = (ndens * mi).sum()                  # Mass density (kg/m3)
    alfven    = field / np.sqrt(mu0 * rho)          # Alfven speed (m/s)
   
    alpha_par  = np.sqrt(2.0 * q * t_par  / mi)     # Thermal velocity in m/s (make relativistic?)  
    alpha_par2 = np.sqrt(2.0 * q * t_par2 / mi)     # Thermal velocity in m/s (make relativistic?)
    return


def get_dispersion_relation(field_, ndensc_, ndensw_, t_perp_, A_, ndensw2_, t_perp2_, A2_, Nk=5000, kmin=0.0, kmax=1.0):
    '''
    field  -- Background magnetic field in nT
    ndensc -- Cold plasma density (H, He, O) in /cc
    ndensw -- Warm plasma density (H, He, O) in /cc
    A      -- Warm plasma anisotropy
    t_perp -- Warm plasma perpendicular (to B) temperature, eV
    Nk     -- Number of points in k-space to solve for
    kmin   -- Minimum k-value, in units of p_cyc/vA (Default 0)
    kmax   -- Maximum k-value, in units of p_cyc/vA (Default 1)
    save   -- Flag: Save output to directory kwarg 'savepath'
    
    OUTPUT:
        k_vals     -- Wavenumbers solved for. In /m3 or normalized to p_cyc/v_A
        CPDR_solns -- Cold plasma dispersion relation: w(k) for each k in k_vals. In Hz or normalized to p_cyc
        warm_solns -- Warm plasma dispersion relation: w(k) for each k in k_vals. In Hz or normalized to p_cyc
    
    Note:
        warm_solns is np.complex128 array: Real component is the dispersion relation, 
        Imaginary component is the growth rate at that k.
    '''
    field   = field_   * 1e-9   # Convert from nT to T
    ndensc  = ndensc_  * 1e6    # Convert from cc to /m3
    ndensw  = ndensw_  * 1e6    # Convert from cc to /m3
    ndensw2 = ndensw2_ * 1e6    # Convert from cc to /m3
        
    set_frequencies_and_variables(field, ndensc, ndensw, t_perp_, A_, ndensw2, t_perp2_, A2_)

    # Initialize k space: Normalized by va/pcyc
    knorm_fac = p_cyc / alfven
    k_min     = kmin  * knorm_fac
    k_max     = kmax  * knorm_fac
    k_vals    = np.linspace(k_min, k_max, Nk, endpoint=False)
    
    eps    = 0.01       # 'Epsilon' value (Can't remember what this was for)
    tol    = 1e-15      # Solution tolerance
    fev    = 1000000    # Number of retries to get below tolerance
        
    CPDR_solns = np.ones((Nk, 3   )) * eps
    warm_solns = np.ones((Nk, 3, 2)) * eps

    # Initial guesses
    for ii in range(1, 3):
        CPDR_solns[0, ii - 1]  = w_cyc[ii] * 1.05
        warm_solns[0, ii - 1]  = np.array([[w_cyc[ii] * 1.05, 0.0]])

    # Numerical solutions: Use previous solution as starting point for new solution
    for jj in range(3):             # For each frequency band
        for ii in range(1, Nk):     # For each value of k
            CPDR_solns[ii, jj] = fsolve(cold_plasma_dispersion_relation, x0=CPDR_solns[ii - 1, jj], args=(k_vals[ii]),          xtol=tol, maxfev=fev)
            warm_solns[ii, jj] = fsolve(warm_plasma_dispersion_relation, x0=warm_solns[ii - 1, jj], args=(k_vals[ii], A_, A2_), xtol=tol, maxfev=fev)

    warm_solns  = estimate_first_and_complexify(warm_solns)
    CPDR_solns /= (2 * np.pi)   # Units of Herz
    warm_solns /= (2 * np.pi)

    return k_vals, CPDR_solns, warm_solns

def get_all_DRs(data_path, time_start, time_end, probe, pad, cmp, Nk=5000):
    param_dict   = data.load_and_interpolate_plasma_params(time_start, time_end, probe, pad, cold_composition=cmp)

    if os.path.exists(data_path) == True:
        print('Save file found: Loading...')
        data_pointer = np.load(data_path)
        all_CPDR     = data_pointer['all_CPDR']
        all_WPDR     = data_pointer['all_WPDR']
        all_k        = data_pointer['all_k']
    else:
        Nt         = param_dict['times'].shape[0]
        all_CPDR   = np.zeros((Nt, Nk, 3), dtype=np.float64)
        all_WPDR   = np.zeros((Nt, Nk, 3), dtype=np.complex128)
        all_k      = np.zeros((Nt, Nk)   , dtype=np.float64)
        for ii in range(Nt):
            print('Calculating dispersion/growth relation for {}'.format(param_dict['times'][ii]))
            
            try:
                k, CPDR, warm_solns = get_dispersion_relation(
                        param_dict['field'][ii],
                        param_dict['ndensc'][:, ii],
                        param_dict['ndensw'][:, ii],
                        param_dict['temp_perp'][:, ii],
                        param_dict['A'][:, ii],
                        param_dict['ndensw2'][:, ii],
                        param_dict['temp_perp2'][:, ii],
                        param_dict['A2'][:, ii],
                        w_isnormalized=False, k_isnormalized=False, Nk=Nk)    
                
                all_CPDR[ii, :, :] = CPDR 
                all_WPDR[ii, :, :] = warm_solns
                all_k[ii, :]       = k
            except:
                all_CPDR[ii, :, :] = np.ones((Nk, 3), dtype=np.float64   ) * np.nan 
                all_WPDR[ii, :, :] = np.ones((Nk, 3), dtype=np.complex128) * np.nan
                all_k[ii, :]       = np.ones(Nk     , dtype=np.float64   ) * np.nan
                
            if ii == Nt - 1:
               print('Saving dispersion history...')
               np.savez(data_path, all_CPDR=all_CPDR, all_WPDR=all_WPDR, all_k=all_k)
    return all_CPDR, all_WPDR, all_k, param_dict