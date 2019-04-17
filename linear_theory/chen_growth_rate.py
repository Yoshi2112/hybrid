# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:29:15 2019

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt
from   scipy.optimize    import fsolve
from   scipy.special     import wofz
from emperics            import geomagnetic_magnitude, sheely_plasmasphere
from matplotlib.ticker   import MultipleLocator
from matplotlib.lines    import Line2D
from convective_growth_rate import calculate_growth_rate
import os
import pdb
'''
Equations from Wang et al. 2016. Is for cold species of warm dispersion
relation simplify for alpha = 0 under the asymptotic expansion of the plasma
dispersion function. Though Wang was fairly inconsistent with the placing of
his signs. Slight differences for Fig 3c (?).
'''
def get_k_CPDR(wr):
    cold_sum = w_pe2 / (wr * (wr - e_cyc))                      # Electrons
    for ii in range(N):
        if ndens[ii] != 0:
            if wr == w_cyc[ii]:
                return np.inf
            else:
                cold_sum += w_ps2[ii] / (wr * (wr - w_cyc[ii]))     # Add each ion species
    return np.sqrt((wr / c) ** 2 - cold_sum * (wr / c) ** 2)

def create_band_legend(fn_ax, labels, colors):
    legend_elements = []
    for label, color in zip(labels, colors):
        legend_elements.append(Line2D([0], [0], color=color, lw=1, label=label))
        
    new_legend = fn_ax.legend(handles=legend_elements, loc='upper right')#, bbox_to_anchor=(1, 0.6))
    return new_legend

def create_type_legend(fn_ax, labels, linestyles):
    legend_elements = []
    for label, style in zip(labels, linestyles):
        legend_elements.append(Line2D([0], [0], color='k', lw=1, label=label, linestyle=style))
        
    new_legend = fn_ax.legend(handles=legend_elements, loc='upper left')#, bbox_to_anchor=(1, 0.6))
    return new_legend

def Z(arg):
    '''Return Plasma Dispersion Function : Normalized Fadeeva function'''
    return 1j*np.sqrt(np.pi)*wofz(arg)


def get_dr_dk(wr, k):
    '''No contribution from cold ions or electrons
    '''
    the_sum = 2 * (c * k) ** 2
    for ii in range(N):
        if ndensw[ii] != 0:
            arg     = (wr - w_cyc[ii]) / (k * alpha_par[ii])
            y_diff  = -2 * (1 + arg * Z(arg).real)
            
            outside = w_pw2[ii] * w_cyc[ii] / (k * alpha_par[ii])
            first   = (A[ii] + 1) * (wr / w_cyc[ii]) - A[ii]
            second  = Z(arg).real + arg * y_diff
            
            the_sum += outside * first * second

    the_sum *= - 1/k
    return the_sum


def get_dr_dw(wr, k):
    '''
    Summand varies for warm/cold components due to limit as T -> 0. Coded explicitly
    to avoid nasty errors and/or warnings.
    '''
    the_sum = 2 * (wr ** 2)
    for ii in range(N):
        # Sum warm components
        if ndensw[ii] != 0:
            arg     = (wr - w_cyc[ii]) / (k * alpha_par[ii])
            y_diff  = -2 * (1 + arg * Z(arg).real)
            
            outside = w_pw2[ii] * w_cyc[ii] / (k * alpha_par[ii])
            inside1 = (A[ii] + 1) * wr * Z(arg).real / w_cyc[ii]
            inside2 = (A[ii] + 1) * (wr / w_cyc[ii]) - A[ii]
            inside3 = wr * y_diff / (k * alpha_par[ii])
            
            the_sum += outside * (inside1 + inside2*inside3)
        
        # Sum cold components
        if ndensc[ii] != 0:
            w_pc2[ii] * w_cyc[ii] * wr / (wr - w_cyc[ii]) ** 2
            
    # Add electrons
    the_sum += w_pe2 * e_cyc * wr / (wr - e_cyc) ** 2
    the_sum *= 1/wr
    return the_sum


def get_D_imag(wr, k):
    ''' Cold components don't contribute to the imaginary component, i.e. only
    warm components are responsible for wave growth.
    '''
    the_sum = 0
    for ii in range(N):
        if ndensw[ii] != 0:
            arg      = (wr - w_cyc[ii]) / (k * alpha_par[ii])
            
            brackets = (alpha_perp[ii] / alpha_par[ii])  ** 2 * (wr - w_cyc[ii]) / w_cyc[ii] + 1
            end      = w_cyc[ii] / (k * alpha_par[ii]) * np.sqrt(np.pi) * np.exp(-arg**2)
            
            the_sum += w_pw2[ii] * brackets * end
    return the_sum


def get_dr_dk_cold(wr, k):
    return -2*k*(c**2)


def get_dr_dw_cold(wr, k):
    '''use 'c' or 's'? '''
    the_sum = 2 * wr
    for ii in range(N):
        if ndens[ii] != 0:
            the_sum += w_ps2[ii] * w_cyc[ii] / ((wr - w_cyc[ii]) ** 2)
    
    # Electrons
    the_sum += w_pe2 * e_cyc / ((wr - e_cyc) ** 2)
    return the_sum


def get_omega_variants():
    npts  = 1000
    min_f = 0.0
    max_f = 0.5
    X     = np.linspace(min_f, max_f, npts, endpoint=False)
    
    k     = np.zeros(npts)
    vg    = np.zeros(npts) 
    gr    = np.zeros(npts)
    st    = np.zeros(npts)
    Sw    = np.zeros(npts)
    
    for ii in range(1, npts):
        wr      = X[ii] * p_cyc
        k[ii]   = get_k_CPDR(wr)              # If k = np.nan or np.inf : Stop band
        
        if np.isnan(k[ii]) == False:
            Dr_dk = get_dr_dk(wr, k[ii])
            Dr_dw = get_dr_dw(wr, k[ii])
            Di    = get_D_imag(    wr, k[ii])
            
            vg[ii] = - Dr_dk / Dr_dw
            gr[ii] = - Di    / Dr_dw
        else:
            vg[ii] = 1e-50
            gr[ii] = 0
            st[ii] = 1

    Sw[1:]     = - gr[1:] / abs(vg[1:]) * 1e7       # Multiplicative factor either 1e9 (/m) or 1e7 (/cm)
    Sw[Sw < 0] = 0                                  # Stop bands
    
    xk, grk, stk = calculate_growth_rate(field*1e9, ndensc*1e-6, ndensw*1e-6, A, temperp=t_perp/q, norm_freq=1, maxfreq=max_f)

    plt.figure()
    plt.plot(X, Sw, label='mine')
    plt.plot(xk, grk, label='koz')
    plt.title('CGR')
    plt.legend()
    plt.show()
    
    plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    
    ax1.plot(X, gr, label='mu')
    ax2.plot(X, vg, label='vg')

    for ax in [ax1, ax2]:
        ax.legend()
        ax.set_xlim(min_f, max_f)
    return

if __name__ == '__main__':
    '''
    This version includes all species in one array, cold and warm components are
    counted as separate species
    '''
    log_dir    = 'C://Users//iarey//Documents//GitHub//hybrid//linear_theory//Logs//'
    log_output = False
    
    mp    = 1.673E-27                           # kg
    me    = 9.109E-31                           # kg
    q     = 1.602e-19                           # C
    qe    = -q
    c     = 3E8                                 # m/s
    e0    = 8.854e-12                           # F/m
    mu0   = (4e-7) * np.pi                      # units
    
    N        = 3                                # Number of species
    L_shell  = 4                                # L-shell at which magnetic field and density are calculated
    n0       = sheely_plasmasphere(L_shell)     # /m3
    field    = 300e-9                           #geomagnetic_magnitude(L_shell)   # T
    
    ndensc    = np.zeros(N)
    ndensc[0] = 196e6
    ndensc[1] = 22e6
    ndensc[2] = 2e6

    # Density of warm species (same order as cold) (number/cc)
    ndensw    = np.zeros(N)
    ndensw[0] = 5.1e6
    ndensw[1] = 0.05e6
    ndensw[2] = 0.13e6


    # Input the perpendicular temperature (ev)
    t_perp    = np.zeros(N)
    t_perp[0] = 30000.
    t_perp[1] = 10000.
    t_perp[2] = 10000.

    # Input the parallel temperature (ev)
    A = np.zeros(N)
    A[0] = 1.
    A[1] = 0.
    A[2] = 0.

    mi    = np.zeros(N)
    mi[0] = 1.  * mp
    mi[1] = 4.  * mp
    mi[2] = 16. * mp

    qi    = np.zeros(N)
    qi[0] = 1.0*q
    qi[1] = 1.0*q
    qi[2] = 1.0*q
    
    t_par   = t_perp / (A + 1)
    
    #################################################
    ### CALCULATE FREQUENCIES AND OTHER VARIABLES ###
    #################################################
    ndens   = ndensc + ndensw
    
    w_pc2   = ndensc      * qi ** 2 / (mi * e0)  # Cold      ion plasma frequencies (rad/s)
    w_pw2   = ndensw      * qi ** 2 / (mi * e0)  # Warm      ion plasma frequencies
    w_ps2   = ndens       * qi ** 2 / (mi * e0)  # Total     ion plasma frequencies
    w_pe2   = ndens.sum() * qe ** 2 / (me * e0)  # Electron  ion plasma frequencies
    
    w_cyc   =  q * field / mi                    # Ion      cyclotron frequencies (rad/s)
    p_cyc   =  q * field / mp                    # Proton   cyclotron frequency (used for normalization)
    e_cyc   =  qe* field / me                    # Electron cyclotron frequency
    
    rho       = (ndens * mi).sum()               # Mass density (kg/m3)
    alfven    = field / np.sqrt(mu0 * rho)       # Alfven speec (m/s)
    
    t_par    *= q                                # Convert temperatures in eV to Joules
    t_perp   *= q
    
    alpha_par  = np.sqrt(2.0 * t_par  / mi)      # Thermal velocity in m/s (make relativistic?)
    alpha_perp = np.sqrt(2.0 * t_perp / mi)      # Thermal velocity in m/s (make relativistic?)

    ######################
    ### SOLVE AND TEST ###
    ######################
    normalize_wavenumber= True
    normalize_frequency = True
    knorm_fac           = p_cyc / alfven
    species_colors      = ['r', 'b', 'g']
    
    # Initialize k space: Normalized by va/pcyc
    Nk     = 1000
    k_min  = 0.0  * knorm_fac
    k_max  = 2.0  * knorm_fac
    k_vals = np.linspace(k_min, k_max, Nk, endpoint=False)
    eps    = 0.01
    tol    = 1e-15
    fev    = 1000000
    
    get_omega_variants()
