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
import os
import pdb
'''
Equations from Wang et al. 2016. Is for cold species of warm dispersion
relation simplify for alpha = 0 under the asymptotic expansion of the plasma
dispersion function. Though Wang was fairly inconsistent with the placing of
his signs.
'''
def create_band_legend(fn_ax, labels, colors):
    legend_elements = []
    for label, color in zip(labels, colors):
        legend_elements.append(Line2D([0], [0], color=color, lw=1, label=label))
        
    new_legend = fn_ax.legend(handles=legend_elements, loc='upper left')#, bbox_to_anchor=(1, 0.6))
    return new_legend

def create_type_legend(fn_ax, labels, linestyles):
    legend_elements = []
    for label, style in zip(labels, linestyles):
        legend_elements.append(Line2D([0], [0], color='k', lw=1, label=label, linestyle=style))
        
    new_legend = fn_ax.legend(handles=legend_elements, loc='upper left')#, bbox_to_anchor=(1, 0.6))
    return new_legend

def wofz_vs_linear():
    '''
    Test function to see what the last term of equation (6) tends to as T -> 0
    
    It tends to ion_cyc / (ion_cyc - w)
    '''
    sp         = 2
    N_test     = 500
    test_w     = 0.75 * w_cyc[sp]
    test_k     = 0.001
    amin       = -25
    amax       = 5
    
    test_alpha = np.logspace(amin, amax, N_test)
    inverse    = np.zeros(N_test)
    pdisp      = np.zeros(N_test, dtype=np.complex128)
    
    for ii in range(N_test):
        inverse[ii]= w_cyc[sp] / (test_k * test_alpha[ii])
        pdisp_arg  = (test_w - w_cyc[sp]) / (test_alpha[ii]*test_k)
        pdisp[ii]  = 1j*np.sqrt(np.pi)*wofz(pdisp_arg)
    
    product = inverse*pdisp
    print(product[0])
    print('pcyc: {}'.format(w_cyc[sp]))
    print('tstw: {}'.format(test_w))
    print('arg: {}'.format(w_cyc[sp] / (w_cyc[sp] - test_w)))
    return


def Z(arg):
    '''Return Plasma Dispersion Function : Normalized Fadeeva function'''
    return 1j*np.sqrt(np.pi)*wofz(arg)


def test_Z():
    N_test       = 5
    x_test       = np.round(np.random.rand(N_test) * 10., decimals=1)
    y_test       = np.round(np.random.rand(N_test) * 10., decimals=1)
    test_vals    = x_test - 1j * y_test
    
    log_file = open(log_dir + 'disp_function_test.txt', 'a')
    
    for ii in range(N_test):
        test_result = Z(test_vals[ii])
        print('y: {} \t x: {} \t ReZ: {} \t ImZ: {}'.format(y_test[ii], x_test[ii], test_result.real, test_result.imag), file=log_file)
    log_file.close()
    return


def cold_plasma_dispersion_relation(w, k):
    cold_sum = w_pe2 / (w * (w - e_cyc))                  # Electrons
    for ii in range(N):
        cold_sum += w_ps2[ii] / (w * (w - w_cyc[ii]))     # Add each ion species
    return 1 - cold_sum - (c * k / w) ** 2


def warm_plasma_dispersion_relation(w, k, test_output=False):
    '''
    w is a vector: [wr, wi]
    
    Function used in scipy.fsolve minimizer to find roots of dispersion relation.
    Iterates over each k to find values of w that minimize 'solution'.
    
    type_out allows purely real or purely imaginary (coefficient only) for root
    finding. Set as anything else for complex output.
    
    Plasma dispersion function related to Fadeeva function (Summers & Thorne, 1993) by
    i*sqrt(pi) factor.
    '''
    wc = w[0] + 1j*w[1]

    components = (w_pe2) * wc / (e_cyc - wc)    # Electrons
    for ii in range(N):
        if ndensc[ii] != 0:
            Is          = wc / (w_cyc[ii] - wc)
            components += (w_pc2[ii]) * Is

        if ndensw[ii] != 0:
            pdisp_arg   = (wc - w_cyc[ii]) / (alpha_par[ii]*k)
            pdisp_func  = Z(pdisp_arg)*w_cyc[ii] / (alpha_par[ii]*k)
            brackets    = (A[ii] + 1) * (wc - w_cyc[ii])/w_cyc[ii] + 1
            Is          = brackets * pdisp_func + A[ii]
            components += w_pw2[ii] * Is

    solution = (wc ** 2) - (c * k) ** 2 + components
    
    if test_output == True:
        return np.array([solution.real, solution.imag]), pdisp_arg
    else:
        return np.array([solution.real, solution.imag])
    
    
def test_warm_solutions(filename='warm_solution_tests', plot=False):
    dwk_store = np.zeros((warm_solns.shape[0], warm_solns.shape[1]), dtype=np.complex128)
    
    if os.path.exists(log_dir + filename + '.txt') == True:
        os.remove(log_dir + filename + '.txt')
    
    text_file = open(log_dir + filename + '.txt', 'a')
    print('{:<10}{:>26}{:>26}{:>26}'.format('WAVENUMBER', 'COMPLEX_FREQUENCY', 'RESIDUAL_OUTPUT_D(w, k)', 'PDF_ARGUMENT'), file=text_file)
    for jj in range(warm_solns.shape[1]):
        print('', file=text_file)
        for ii in range(1, warm_solns.shape[0]):
            dwk, zeta = warm_plasma_dispersion_relation(warm_solns[ii, jj], k_vals[ii], test_output=True)
            wc = warm_solns[ii, jj, 0] + 1j*warm_solns[ii, jj, 1]
            dwk_out = dwk[0]+1j*dwk[1]
            dwk_store[ii, jj] = dwk_out
            print('{:<10.3e}{:>26.3e}{:>26.3e}{:>26.3e}'.format(k_vals[ii], wc, dwk_out, zeta), file=text_file)
    text_file.close()

    if plot == True:
        figpath = log_dir + filename + '_Error_plot.png'
        for jj in range(N):
            fig = plt.figure(figsize=(15,10))
            ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
            
            ax1.set_title('Dispersion Relation Absolute Error')
            ax1.plot(k_vals, dwk_store[:, jj].real, c=species_colors[jj])
            ax1.set_xlabel('$k (m^{-1})$')
            ax1.set_ylabel('Error')
            
            ax2.set_title('Growth Rate Absolute Error')
            ax2.plot(k_vals, dwk_store[:, jj].imag, c=species_colors[jj])
            ax2.set_xlabel('$k (m^{-1})$')
            ax2.set_ylabel('Error')
            fig.savefig(figpath)
            plt.close('all')
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
    c     = 3E8                                 # m/s
    e0    = 8.854e-12                           # F/m
    mu0   = (4e-7) * np.pi                      # units
    
    N        = 3                                # Number of species
    L_shell  = 4                                # L-shell at which magnetic field and density are calculated
    n0       = sheely_plasmasphere(L_shell)     # /m3
    field    = geomagnetic_magnitude(L_shell)   # T
    
    ndensc   = np.zeros(N)
    ndensc[0] = 0.10 * n0                       # Cold H+
    ndensc[1] = 0.30 * n0                       # Cold He+
    ndensc[2] = 0.50 * n0                       # Cold O+

    ndensw    = np.zeros(N)
    ndensw[0] = 0.10 * n0                       # Hot H+
    ndensw[1] = 0.00 * n0                       # Hot He+
    ndensw[2] = 0.00 * n0                       # Hot O+

    # Input the perpendicular temperature (ev)
    t_perp    = np.zeros(N)
    t_perp[0] = 50000.
    t_perp[1] = 0.
    t_perp[2] = 0.

    # Input the parallel temperature (ev)
    t_par    = np.zeros(N)
    t_par[0] = 25000.
    t_par[1] = 0.
    t_par[2] = 0.

    mi    = np.zeros(N)
    mi[0] = 1.  * mp
    mi[1] = 4.  * mp
    mi[2] = 16. * mp
    
    A = np.zeros(N)
    for ii in range(N):
        if t_par[ii] != 0:
            A[ii] = t_perp[ii] / t_par[ii] - 1

    ndens   = ndensc + ndensw
    if ndens.sum() != n0:
        raise ValueError('Density issue')

    w_pc2   = ndensc      * q ** 2 / (mi * e0)   # Cold      ion plasma frequencies
    w_pw2   = ndensw      * q ** 2 / (mi * e0)   # Warm      ion plasma frequencies
    w_ps2   = ndens       * q ** 2 / (mi * e0)   # Total     ion plasma frequencies
    w_pe2   = ndens.sum() * q ** 2 / (me * e0)   # Electron  ion plasma frequencies
    
    w_cyc   =  q * field / mi
    p_cyc   =  q * field / mp
    e_cyc   = -q * field / me
    
    rho       = (ndens * mi).sum()
    alfven    = field / np.sqrt(mu0 * rho)
    
    t_par    *= q                                            # Convert temperatures in eV to Joules
    alpha_par = np.sqrt(2.0 * t_par / mi)                    # in m/s

    test_w = [18.92, -0.3696]
    test_k = 7.338e-6
    warm_plasma_dispersion_relation(test_w, test_k)

    ######################
    ### SOLVE AND TEST ###
    ######################
    plt.ioff()
    normalize_wavenumber= True
    normalize_frequency = True
    knorm_fac           = p_cyc / alfven
    species_colors      = ['r', 'b', 'g']
    
    # Initialize k space: Normalized by va/pcyc
    Nk     = 5000
    k_min  = 0.0  * knorm_fac
    k_max  = 1.0  * knorm_fac
    k_vals = np.linspace(k_min, k_max, Nk, endpoint=False)
    eps    = 0.01
    
    CPDR_solns           = np.ones((Nk, 3   )) * eps
    warm_solns           = np.ones((Nk, 3, 2)) * eps

    for ii in range(1, N):
        CPDR_solns[0, ii - 1]  = w_cyc[ii] * 1.05
        warm_solns[0, ii - 1]  = np.array([[w_cyc[ii] * 1.05, 0.0]])                # Initial guess
    
    tol = 1e-15
    fev = 1000000

    for ii in range(1, Nk):
        for jj in range(3):
            CPDR_solns[ii, jj] = fsolve(cold_plasma_dispersion_relation, x0=CPDR_solns[ii - 1, jj], args=(k_vals[ii]), xtol=tol, maxfev=fev)
            warm_solns[ii, jj] = fsolve(warm_plasma_dispersion_relation, x0=warm_solns[ii - 1, jj], args=(k_vals[ii]), xtol=tol, maxfev=fev)
    
    if log_output == True:
        test_warm_solutions(plot=True)
        
        
    ###############
    ## NORMALIZE ##
    ###############
    if normalize_frequency == True:
        CPDR_solns /= p_cyc
        warm_solns /= p_cyc
        w_cyc      /= p_cyc
    else:
        CPDR_solns /= (2 * np.pi)
        warm_solns /= (2 * np.pi)
        w_cyc      /= (2 * np.pi)
        
    if normalize_wavenumber == True:
        k_min  /= knorm_fac
        k_max  /= knorm_fac
        k_vals /= knorm_fac
        
    wave_solns = warm_solns[:, :, 0]
    grth_solns = warm_solns[:, :, 1]
    
    ##########
    ## PLOT ##
    ##########
    plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    
    for ii in range(3):
        ax1.plot(k_vals[1:], CPDR_solns[1:, ii], c=species_colors[ii], linestyle='--', label='Cold')
        ax1.plot(k_vals[1:], wave_solns[1:, ii], c=species_colors[ii], linestyle='-',  label='Warm')
        ax1.axhline(w_cyc[ii], c='k', linestyle=':')

    ax1.set_title('Dispersion Relation')
    ax1.set_xlabel(r'$kv_A / \Omega_p$')
    ax1.set_ylabel(r'$\omega/\Omega_p$')
    ax1.set_xlim(k_min, k_max)
    ax1.set_ylim(0, 1.0)
    ax1.minorticks_on()
    ax1.yaxis.set_minor_locator(MultipleLocator(0.04))
    
    type_label = ['Cold Plasma Approx.', 'Hot Plasma Approx.', 'Cyclotron Frequencies']
    type_style = ['--', '-', ':']
    type_legend = create_type_legend(ax1, type_label, type_style)
    ax1.add_artist(type_legend)
    
    band_labels = [r'$H^+$', r'$He^+$', r'$O^+$']
    band_legend = create_band_legend(ax2, band_labels, species_colors)
    ax2.add_artist(band_legend)
    
    for ii in range(3):
        ax2.plot(k_vals[1:], grth_solns[1:, ii], c=species_colors[ii], linestyle='-',  label='Growth')

    ax2.set_title('Temporal Growth Rate')
    ax2.set_xlabel(r'$kv_A / \Omega_p$')
    ax2.set_ylabel(r'$\gamma/\Omega_p$')
    ax2.set_xlim(k_min, k_max)
    ax2.set_ylim(-0.05, 0.05)
    ax2.minorticks_on()
    ax2.yaxis.set_minor_locator(MultipleLocator(0.005))
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()




