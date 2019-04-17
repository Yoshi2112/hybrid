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
from matplotlib.lines    import Line2D
import os


'''
Equations from Wang et al. 2016. Is for cold species of warm dispersion
relation simplify for alpha = 0 under the asymptotic expansion of the plasma
dispersion function. Though Wang was fairly inconsistent with the placing of
his signs. Slight differences for Fig 3c (?).
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


def cold_plasma_dispersion_relation(w, k):
    cold_sum = w_pe2 / (w * (w - e_cyc))                  # Electrons
    for ii in range(w_ps2.shape[0]):
        cold_sum += w_ps2[ii] / (w * (w - w_cyc[ii]))     # Add each ion species
    return 1 - cold_sum - (c * k / w) ** 2


def warm_plasma_dispersion_relation(w, k, A):
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
    wc = w[0] + 1j*w[1]

    components = (w_pe2) * wc / (e_cyc - wc)    # Electrons
    for ii in range(w_cyc.shape[0]):
        if w_pc2[ii] != 0:
            Is          = wc / (w_cyc[ii] - wc)
            components += (w_pc2[ii]) * Is

        if w_pw2[ii] != 0:
            pdisp_arg   = (wc - w_cyc[ii]) / (alpha_par[ii]*k)
            pdisp_func  = Z(pdisp_arg)*w_cyc[ii] / (alpha_par[ii]*k)
            brackets    = (A[ii] + 1) * (wc - w_cyc[ii])/w_cyc[ii] + 1
            Is          = brackets * pdisp_func + A[ii]
            components += w_pw2[ii] * Is

    solution = (wc ** 2) - (c * k) ** 2 + components
    return np.array([solution.real, solution.imag])
    

def estimate_first_and_complexify(solutions):
    outarray = np.zeros((solutions.shape[0], solutions.shape[1]), dtype=np.complex128)
    
    for jj in range(solutions.shape[1]):
        for ii in range(solutions.shape[0]):
            outarray[ii, jj] = solutions[ii, jj, 0] + 1j*solutions[ii, jj, 1]
            
    outarray[0] = outarray[1]
    return outarray


def group_velocity_finite_difference(dc):
    '''
    This gives group velocity = dw/dk as a function of wavenumber by second order
    centered finite difference. The first term is zero.
    
    INPUT:
        dc -- Dispersion Curves in an NxM array for N points and M species.
    
    OUTPUT:
        vg -- Group velocity as NxM array
        
    vg[0] doesn't exist since k = 0
    vg[1] uses a forwards difference since dc[0] doesn't exist (k = 0)
    '''
    vg = np.zeros(dc.shape)
    dk = k_vals[1] - k_vals[0]
    
    vg[ 1] = -3*dc[ 1] + 4*dc[ 2] - dc[ 3]      # Forward difference first valid point (vg[0] doesn't exist)
    vg[-1] =  3*dc[-1] - 4*dc[-2] + dc[-3]
    
    for ii in range(2, dc.shape[0] - 1):
        vg[ii] = dc[ii + 1] - dc[ii - 1]
            
    return vg / (2*dk)


def get_dispersion_relation(field, ndensc, ndensw, A, t_perp, norm_k=False, norm_w=False, Nk=1000, kmin=0.0, kmax=1.0, plot=False, save=False, savepath=None):
    '''
    field  -- Background magnetic field in T
    ndensc -- Cold plasma density (H, He, O) in /m3
    ndensw -- Warm plasma density (H, He, O) in /m3
    A      -- Warm plasma anisotropy
    t_perp -- Warm plasma perpendicular (to B) temperature, eV
    norm_k -- Flag: Normalize wavenumber to units of p_cyc/vA
    norm_w -- Flag: Normalize frequency to units of p_cyc
    Nk     -- Number of points in k-space to solve for
    kmin   -- Minimum k-value, in units of p_cyc/vA
    kmax   -- Maximum k-value, in units of p_cyc/vA
    plot   -- Flag: Plot output
    save   -- Flag: Save output to directory kwarg 'savepath'
    '''
    global c, k_vals
    mp    = 1.673E-27                           # kg
    me    = 9.109E-31                           # kg
    q     = 1.602e-19                           # C
    qe    = -q
    c     = 3E8                                 # m/s
    e0    = 8.854e-12                           # F/m
    mu0   = (4e-7) * np.pi                      # units
    N     = A.shape[0]
    
    mi    = np.zeros(N)
    mi[0] = 1.  * mp
    mi[1] = 4.  * mp
    mi[2] = 16. * mp

    qi    = np.zeros(N)
    qi[0] = 1.0*q
    qi[1] = 1.0*q
    qi[2] = 1.0*q
    
    #################################################
    ### CALCULATE FREQUENCIES AND OTHER VARIABLES ###
    #################################################
    global t_par, ndens, w_pc2, w_pw2, w_ps2, w_pe2, w_cyc, p_cyc, e_cyc, alpha_par, alpha_perp
    
    t_par   = t_perp / (A + 1)
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
    knorm_fac           = p_cyc / alfven
    
    # Initialize k space: Normalized by va/pcyc
    k_min  = kmin  * knorm_fac
    k_max  = kmax  * knorm_fac
    k_vals = np.linspace(k_min, k_max, Nk, endpoint=False)
    
    eps    = 0.01
    tol    = 1e-15
    fev    = 1000000
        
    CPDR_solns           = np.ones((Nk, 3   )) * eps
    warm_solns           = np.ones((Nk, 3, 2)) * eps

    # Initial guesses
    for ii in range(1, N):
        CPDR_solns[0, ii - 1]  = w_cyc[ii] * 1.05
        warm_solns[0, ii - 1]  = np.array([[w_cyc[ii] * 1.05, 0.0]])

    # Numerical solutions
    for jj in range(3):
        for ii in range(1, Nk):
            CPDR_solns[ii, jj] = fsolve(cold_plasma_dispersion_relation, x0=CPDR_solns[ii - 1, jj], args=(k_vals[ii]),    xtol=tol, maxfev=fev)
            warm_solns[ii, jj] = fsolve(warm_plasma_dispersion_relation, x0=warm_solns[ii - 1, jj], args=(k_vals[ii], A), xtol=tol, maxfev=fev)

    warm_solns     = estimate_first_and_complexify(warm_solns)
    
    ###############
    ## NORMALIZE ##
    ###############
    if norm_w == True:
        CPDR_solns /= p_cyc
        warm_solns /= p_cyc
        w_cyc      /= p_cyc        
    else:
        CPDR_solns /= (2 * np.pi)
        warm_solns /= (2 * np.pi)
        w_cyc      /= (2 * np.pi)
         
    if norm_k == True:
        k_min  /= knorm_fac
        k_max  /= knorm_fac
        k_vals /= knorm_fac
        
    if plot==True or save==True:
        plot_dispersion(k_vals, CPDR_solns, warm_solns, norm_k=norm_k, norm_w=norm_w, save=save, savepath=savepath)
    
    return k_vals, CPDR_solns, warm_solns


def plot_dispersion(k_vals, CPDR_solns, warm_solns, norm_k=False, norm_w=False, save=False, savepath=None):
    '''
    docstring
    '''
    species_colors      = ['r', 'b', 'g']
    
    if norm_w == True:
        f_max       = 1.0
        ysuff       = '$/\Omega_p$'
    else:
        f_max       = p_cyc / (2 * np.pi)
        ysuff       = ' (Hz)'
    
    if norm_k == True:
        xlab        = r'$kv_A / \Omega_p$'
    else:
        xlab        = r'$k (m^{-1})$'
        
    plt.ioff()
    plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    
    for ii in range(3):
        ax1.plot(k_vals[1:], CPDR_solns[1:, ii],      c=species_colors[ii], linestyle='--', label='Cold')
        ax1.plot(k_vals[1:], warm_solns[1:, ii].real, c=species_colors[ii], linestyle='-',  label='Warm')
        ax1.axhline(w_cyc[ii], c='k', linestyle=':')

    ax1.set_title('Dispersion Relation')
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(r'$\omega${}'.format(ysuff))
    ax1.set_xlim(k_vals[0], k_vals[-1])
    
    ax1.set_ylim(0, f_max)
    ax1.minorticks_on()
    
    type_label = ['Cold Plasma Approx.', 'Hot Plasma Approx.', 'Cyclotron Frequencies']
    type_style = ['--', '-', ':']
    type_legend = create_type_legend(ax1, type_label, type_style)
    ax1.add_artist(type_legend)
    
    band_labels = [r'$H^+$', r'$He^+$', r'$O^+$']
    band_legend = create_band_legend(ax2, band_labels, species_colors)
    ax2.add_artist(band_legend)
    
    for ii in range(3):
        ax2.plot(k_vals[1:], warm_solns[1:, ii].imag, c=species_colors[ii], linestyle='-',  label='Growth')

    ax2.set_title('Temporal Growth Rate')
    ax2.set_xlabel(xlab)
    ax2.set_ylabel(r'$\gamma${}'.format(ysuff))
    ax2.set_xlim(k_vals[0], k_vals[-1])
    
    if norm_w == True:
        ax2.set_ylim(-0.05, 0.05)
        
    ax2.minorticks_on()
    
    if save == True:
        path     = os.getcwd() + '\\'
        
        vercount = 0
        name     = 'dispersion_relation{}.png'.format(vercount)
        while os.path.exists(path + name) == True:
            vercount += 1
            name     = 'dispersion_relation{}'.format(vercount)

        plt.savefig(path + name)
    else:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()    
    return


if __name__ == '__main__':
    Nn       = 3                                # Number of species
    L_shell  = 4                                # L-shell at which magnetic field and density are calculated
    n0       = sheely_plasmasphere(L_shell)     # /m3
    tfield   = geomagnetic_magnitude(L_shell)   # T
    
    tndensc    = np.zeros(Nn)
    tndensc[0] = 0.1*n0
    tndensc[1] = 0.2*n0
    tndensc[2] = 0.1*n0

    # Density of warm species (same order as cold) (number/cc)
    tndensw    = np.zeros(Nn)
    tndensw[0] = 0.6*n0
    tndensw[1] = 0.0*n0
    tndensw[2] = 0.0*n0

    # Input the perpendicular temperature (ev)
    tt_perp    = np.zeros(Nn)
    tt_perp[0] = 50000.
    tt_perp[1] = 00000.
    tt_perp[2] = 00000.

    # Input the parallel temperature (ev)
    tA = np.zeros(Nn)
    tA[0] = 1.
    tA[1] = 0.
    tA[2] = 0.

    get_dispersion_relation(tfield, tndensc, tndensw, tA, tt_perp, norm_k = True, norm_w=True, plot=True)    