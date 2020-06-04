# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:29:15 2019

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec       as gs
from   scipy.optimize    import fsolve
from   scipy.special     import wofz
from matplotlib.table import Table
from emperics import geomagnetic_magnitude, sheely_plasmasphere
from matplotlib.lines    import Line2D
import os
import pdb
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

    try:
        components = (w_pe2) * wc / (e_cyc - wc)    # Electrons
    except:
        pdb.set_trace()

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
    outarray = np.zeros((solutions.shape[0], solutions.shape[1]), dtype=np.complex128)
    
    for jj in range(solutions.shape[1]):
        for ii in range(solutions.shape[0]):
            outarray[ii, jj] = solutions[ii, jj, 0] + 1j*solutions[ii, jj, 1]
            
    outarray[0] = outarray[1]
    return outarray


def get_dispersion_relation(field_, ndensc_, ndensw_, A_, t_perp_, ndensw2_, A2_, t_perp2_, k_isnormalized=False, w_isnormalized=False, Nk=5000, kmin=0.0, kmax=1.0, plot=False, save=False, savepath=None):
    '''
    field  -- Background magnetic field in nT
    ndensc -- Cold plasma density (H, He, O) in /cc
    ndensw -- Warm plasma density (H, He, O) in /cc
    A      -- Warm plasma anisotropy
    t_perp -- Warm plasma perpendicular (to B) temperature, eV
    norm_k -- Flag: Normalize wavenumber to units of p_cyc/vA
    norm_w -- Flag: Normalize frequency to units of p_cyc
    Nk     -- Number of points in k-space to solve for
    kmin   -- Minimum k-value, in units of p_cyc/vA
    kmax   -- Maximum k-value, in units of p_cyc/vA
    plot   -- Flag: Plot output
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
    
    global c, k_vals
    mp    = 1.673E-27                           # kg
    me    = 9.109E-31                           # kg
    q     = 1.602e-19                           # C
    qe    = -q
    c     = 3E8                                 # m/s
    e0    = 8.854e-12                           # F/m
    mu0   = (4e-7) * np.pi                      # units
    N     = A_.shape[0]
    
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
    global t_par2, w_pw2b, alpha_par2, alpha_perp2
    
    t_par   = t_perp_  / (A_ + 1)
    t_par2  = t_perp2_ / (A2_ + 1)
    ndens   = ndensc + ndensw + ndensw2
    
    w_pc2   = ndensc      * qi ** 2 / (mi * e0)  # Cold      ion plasma frequencies (rad/s)
    w_pw2   = ndensw      * qi ** 2 / (mi * e0)  # Warm      ion plasma frequencies
    w_pw2b  = ndensw2     * qi ** 2 / (mi * e0)  # Warm2     ion plasma frequencies
    w_ps2   = ndens       * qi ** 2 / (mi * e0)  # Total     ion plasma frequencies
    w_pe2   = ndens.sum() * qe ** 2 / (me * e0)  # Electron  ion plasma frequencies
    
    w_cyc   =  q * field / mi                    # Ion      cyclotron frequencies (rad/s)
    p_cyc   =  q * field / mp                    # Proton   cyclotron frequency (used for normalization)
    e_cyc   =  qe* field / me                    # Electron cyclotron frequency
    
    rho       = (ndens * mi).sum()               # Mass density (kg/m3)
    alfven    = field / np.sqrt(mu0 * rho)       # Alfven speec (m/s)
    
    t_par    *= q                                # Convert temperatures in eV to Joules
    t_perp    = t_perp_ * q
    
    t_par2   *= q                                # Convert temperatures in eV to Joules
    t_perp2   = t_perp2_ * q
    
    alpha_par  = np.sqrt(2.0 * t_par  / mi)      # Thermal velocity in m/s (make relativistic?)
    alpha_perp = np.sqrt(2.0 * t_perp / mi)      # Thermal velocity in m/s (make relativistic?)
    
    alpha_par2  = np.sqrt(2.0 * t_par2  / mi)    # Thermal velocity in m/s (make relativistic?)
    alpha_perp2 = np.sqrt(2.0 * t_perp2 / mi)    # Thermal velocity in m/s (make relativistic?)

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
            CPDR_solns[ii, jj] = fsolve(cold_plasma_dispersion_relation, x0=CPDR_solns[ii - 1, jj], args=(k_vals[ii]),        xtol=tol, maxfev=fev)
            warm_solns[ii, jj] = fsolve(warm_plasma_dispersion_relation, x0=warm_solns[ii - 1, jj], args=(k_vals[ii], A_, A2_), xtol=tol, maxfev=fev)

    warm_solns     = estimate_first_and_complexify(warm_solns)
    
    ###############
    ## NORMALIZE ##
    ###############
    if w_isnormalized == True:
        CPDR_solns /= p_cyc
        warm_solns /= p_cyc
        w_cyc      /= p_cyc        
    else:
        CPDR_solns /= (2 * np.pi)
        warm_solns /= (2 * np.pi)
        w_cyc      /= (2 * np.pi)
         
    if k_isnormalized == True:
        k_min  /= knorm_fac
        k_max  /= knorm_fac
        k_vals /= knorm_fac
        
    if plot==True or save==True:
        plot_dispersion(k_vals, CPDR_solns, warm_solns, k_isnormalized=k_isnormalized, w_isnormalized=w_isnormalized, save=save, savepath=savepath)
    
    return k_vals, CPDR_solns, warm_solns


def plot_dispersion(k_vals, CPDR_solns, warm_solns, k_isnormalized=False, w_isnormalized=False, save=False, savepath=None):
    '''
    Plots the CPDR and WPDR nicely as per Wang et al 2016. 
    
    INPUT:
        k_vals     -- Wavenumber values in /m3 or normalized to p_cyc/v_A
        CPDR_solns -- Cold-plasma frequencies in Hz or normalized to p_cyc
        WPDR_solns -- Warm-plasma frequencies in Hz or normalized to p_cyc. 
                   -- .real is dispersion relation, .imag is growth rate vs. k
    '''
    species_colors      = ['r', 'b', 'g']
    
    if w_isnormalized == True:
        f_max       = 1.0
        ysuff       = '$/\Omega_p$'
    else:
        f_max       = p_cyc / (2 * np.pi)
        ysuff       = ' (Hz)'
    
    if k_isnormalized == True:
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
    
    if w_isnormalized == True:
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


def plot_dispersion_multiple(ax_disp, ax_growth, k_vals, CPDR_solns, warm_solns, k_isnormalized=False,
                             w_isnormalized=False, save=False, savepath=None, alpha=1.0):
    '''
    Plots the CPDR and WPDR nicely as per Wang et al 2016. Can plot multiple dispersion/growth curves for varying parameters.
    
    INPUT:
        k_vals     -- Wavenumber values in /m3 or normalized to p_cyc/v_A
        CPDR_solns -- Cold-plasma frequencies in Hz or normalized to p_cyc
        WPDR_solns -- Warm-plasma frequencies in Hz or normalized to p_cyc. 
                   -- .real is dispersion relation, .imag is growth rate vs. k
    '''
    species_colors      = ['r', 'b', 'g']
    
    # Plot dispersion #
    for ii in range(3):
        ax_disp.plot(k_vals[1:]*1e-6, CPDR_solns[1:, ii],      c=species_colors[ii], linestyle='--', label='Cold', alpha=alpha)
        ax_disp.plot(k_vals[1:]*1e-6, warm_solns[1:, ii].real, c=species_colors[ii], linestyle='-',  label='Warm', alpha=alpha)
        ax_disp.axhline(w_cyc[ii], c='k', linestyle=':')
    
    type_label = ['Cold Plasma Approx.', 'Hot Plasma Approx.', 'Cyclotron Frequencies']
    type_style = ['--', '-', ':']
    type_legend = create_type_legend(ax_disp, type_label, type_style)
    ax_disp.add_artist(type_legend)
    
    # Plot growth #
    band_labels = [r'$H^+$', r'$He^+$', r'$O^+$']
    band_legend = create_band_legend(ax_growth, band_labels, species_colors)
    ax_growth.add_artist(band_legend)
    
    for ii in range(3):
        ax_growth.plot(k_vals[1:]*1e-6, warm_solns[1:, ii].imag, c=species_colors[ii], linestyle='-',  label='Growth', alpha=alpha)
    return


def set_figure_text(ax):
    font    = 'monospace'
    fsize   = 10
    top     = 1.0               # Top of text block
    left    = 1.15               # Left boundary of text block
    
    TPER_kev  = _t_perp / 1e3
    TPER_kev2 = _t_perp2 / 1e3
    
    ax.text(left, top - 0.02, '$B_0 = ${}nT'.format(_field),           transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, top - 0.05, '$n_0 = $%.1f$cm^{-3}$' % 0, transform=ax.transAxes, fontsize=fsize, fontname=font)

    v_space = 0.03              # Vertical spacing between lines
    
    c_top   = top - 0.15         # 'Cold' top
    w_top   = top - 0.35         # 'Warm' top
    h_top   = top - 0.55         # 'Hot' (Warmer) top

    # Cold Table
    ax.text(left, c_top + 1*v_space, 'Cold Population', transform=ax.transAxes, fontsize=fsize+2, fontname=font, fontweight='bold')
    ax.text(left + 0.05, c_top, r'  $n_c (cm^{-3})$', transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, c_top - 1*v_space, ' H+:   {:>7.2f} '.format(round(_ndensc[0], 2), TPER_kev[0], _A[0]) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, c_top - 2*v_space, 'He+:   {:>7.2f} '.format(round(_ndensc[1], 2), TPER_kev[1], _A[1])    , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, c_top - 3*v_space, ' O+:   {:>7.2f} '.format(round(_ndensc[2], 2), TPER_kev[2], _A[2])    , transform=ax.transAxes, fontsize=fsize, fontname=font)

    
    # Warm Table
    ax.text(left, w_top + 1*v_space, 'Warm Population 1', transform=ax.transAxes, fontsize=fsize+2, fontname=font, fontweight='bold')
    ax.text(left + 0.05, w_top, r'  $n_i (cm^{-3})$  $T_{\perp} (keV)$   $A_i$ ', transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, w_top - 1*v_space, ' H+:   {:>5.2f}   {:>5}   {:>5}'.format(round(_ndensw[0], 2), TPER_kev[0], _A[0]) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, w_top - 2*v_space, 'He+:   {:>5.2f}   {:>5}   {:>5}'.format(round(_ndensw[1], 2), TPER_kev[1], _A[1])    , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, w_top - 3*v_space, ' O+:   {:>5.2f}   {:>5}   {:>5}'.format(round(_ndensw[2], 2), TPER_kev[2], _A[2])    , transform=ax.transAxes, fontsize=fsize, fontname=font)


    # Hot Table
    ax.text(left, h_top + 1*v_space, 'Warm Population 2', transform=ax.transAxes, fontsize=fsize+2, fontname=font, fontweight='bold')
    ax.text(left + 0.05, h_top, r'  $n_i (cm^{-3})$  $T_{\perp} (keV)$   $A_i$ ', transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, h_top - 1*v_space, ' H+:   {:>5.2f}   {:>5}   {:>5}'.format(round(_ndensw2[0], 2), TPER_kev2[0], _A2[0]) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, h_top - 2*v_space, 'He+:   {:>5.2f}   {:>5}   {:>5}'.format(round(_ndensw2[1], 2), TPER_kev2[1], _A2[1]) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    ax.text(left, h_top - 3*v_space, ' O+:   {:>5.2f}   {:>5}   {:>5}'.format(round(_ndensw2[2], 2), TPER_kev2[2], _A2[2]) , transform=ax.transAxes, fontsize=fsize, fontname=font)
    return


if __name__ == '__main__':
    save_path = 'G://TEMP//'
    output    = 'show'
    #np.seterr(all='raise')
    figtext   = True
    
    Nn       = 3                                    # Number of species
        
    variable = np.array([1.0])
    
    # Put this bit outside the plot if you want to plot all in same
    plt.ioff()
    fig     = plt.figure(figsize=(16, 10))
    grid    = gs.GridSpec(1, 2)
    
    _ax1    = fig.add_subplot(grid[0, 0])
    _ax2    = fig.add_subplot(grid[0, 1])

    _field = 157.3
    
    # Density of cold species (number/cc)
    _ndensc    = np.zeros(Nn)
    _ndensc[0] = 57.21     # Cold Hydrogen
    _ndensc[1] = 16.35     # Cold Helium
    _ndensc[2] = 8.17      # Cold Oxygen
    
    # Density of warm species (number/cc)
    _ndensw    = np.zeros(Nn)
    _ndensw[0] = 1.54      # Warm Hydrogen
    _ndensw[1] = 0.12      # Warm Helium
    _ndensw[2] = 0.44      # Warm Oxygen

    # Perpendicular temperature (ev)
    _t_perp    = np.zeros(Nn)
    _t_perp[0] = 9658
    _t_perp[1] = 7008
    _t_perp[2] = 3531

    # Ion species anisotropy : A = T_perp / T_par - 1
    _A    = np.zeros(Nn)
    _A[0] = 0.14
    _A[1] = 0.31
    _A[2] = 0.11
    
    # Density of second warm species (number/cc)
    _ndensw2    = np.zeros(Nn)
    _ndensw2[0] = 0.404      # Warm Hydrogen
    _ndensw2[1] = 0.0048     # Warm Helium
    _ndensw2[2] = 0.0022     # Warm Oxygen

    # Second perpendicular temperature (ev)
    _t_perp2    = np.zeros(Nn)
    _t_perp2[0] = 56189
    _t_perp2[1] = 96773
    _t_perp2[2] = 158218

    # second parallel temperature (ev)
    _A2    = np.zeros(Nn)
    _A2[0] = 0.2
    _A2[1] = 0.153
    _A2[2] = 0.356
   
    _k, _CPDR, _warm_solns = get_dispersion_relation( _field, _ndensc, _ndensw, _A, _t_perp, _ndensw2, _A2, _t_perp2, w_isnormalized=True)    
    plot_dispersion_multiple(_ax1, _ax2, _k, _CPDR, _warm_solns, save=False, savepath=None, w_isnormalized=True)    

    _ax1.set_title('Dispersion Relation')
    _ax1.set_xlabel(r'$k (\times 10^{-6} m^{-1})$')
    _ax1.set_ylabel(r'$\omega${}'.format(' (Hz)'))
    _ax1.set_xlim(k_vals[0]*1e-6, k_vals[-1]*1e-6)
    
    _ax2.set_title('Temporal Growth Rate')
    _ax2.set_xlabel(r'$k (\times 10^{-6}m^{-1})$')
    _ax2.set_ylabel(r'$\gamma$')
    _ax2.set_xlim(k_vals[0]*1e-6, k_vals[-1]*1e-6)
    _ax2.set_ylim(-0.05, 0.05)
    
    _ax1.set_ylim(0, p_cyc / (2 * np.pi))
    _ax1.minorticks_on()
    _ax2.minorticks_on() 
    
    _ax2.yaxis.set_label_position("right")
    _ax2.yaxis.tick_right()
    plt.setp(_ax2.get_yticklabels()[0], visible=False)
    #%%
    if figtext == True:
        set_figure_text(_ax2)
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0, right=0.75)
    
    if output == 'show':
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    elif output == 'save':
        figsave_path = save_path + 'test_.png'
        print('Saving {}'.format(figsave_path))
        fig.savefig(figsave_path)
        plt.close('all')
