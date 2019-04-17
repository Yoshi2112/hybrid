# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:29:15 2019

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt
from   scipy.optimize    import fsolve
from   scipy.special     import wofz
#from emperics            import geomagnetic_magnitude, sheely_plasmasphere
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


def Z(arg):
    '''Return Plasma Dispersion Function : Normalized Fadeeva function'''
    return 1j*np.sqrt(np.pi)*wofz(arg)


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
    
    test_output flag allows w and solution to be complex, rather than wrapped. Additional
    arguments may also be selected to be exported this way.
    '''
    if test_output == False:
        wc = w[0] + 1j*w[1]
    else:
        wc = w

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
        return solution, pdisp_arg
    else:
        return np.array([solution.real, solution.imag])
    

def calculate_group_velocity(dc):
    '''
    This gives group velocity as a function of wavenumber by second order
    centered finite difference. The first term is zero.
    How do I get it as a function of frequency like Kozyra did?
    '''
    vg = np.zeros(dc.shape)
    dk = k_vals[1] - k_vals[0]
    
    vg[-1] =  3*dc[-1] - 4*dc[-2] + dc[-3]
    
    for ii in range(1, dc.shape[0] - 1):
        vg[ii] = dc[ii + 1] - dc[ii - 1]
            
    return vg / (2*dk)


def estimate_first_and_complexify(solutions):
    outarray = np.zeros((solutions.shape[0], solutions.shape[1]), dtype=np.complex128)
    
    for jj in range(solutions.shape[1]):
        for ii in range(solutions.shape[0]):
            outarray[ii, jj] = solutions[ii, jj, 0] + 1j*solutions[ii, jj, 1]
            
    outarray[0] = outarray[1]
    return outarray


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
    n0       = 10e6   #sheely_plasmasphere(L_shell)     # /m3
    field    = 200e-9 #geomagnetic_magnitude(L_shell)   # T
    
    ndensc   = np.zeros(N)
    ndensc[0] = 0.90 * n0                       # Cold H+
    ndensc[1] = 0.00 * n0                       # Cold He+
    ndensc[2] = 0.00 * n0                       # Cold O+

    ndensw    = np.zeros(N)
    ndensw[0] = 0.10 * n0                       # Hot H+
    ndensw[1] = 0.00 * n0                       # Hot He+
    ndensw[2] = 0.00 * n0                       # Hot O+

    # Input the perpendicular temperature (ev)
    t_perp    = np.zeros(N)
    t_perp[0] = 200000.
    t_perp[1] = 0.
    t_perp[2] = 0.

    # Input the parallel temperature (ev)
    t_par    = np.zeros(N)
    t_par[0] = 100000.
    t_par[1] = 0.
    t_par[2] = 0.
    
# =============================================================================
#     ndensc   = np.zeros(N)
#     ndensc[0] = 0.10 * n0                       # Cold H+
#     ndensc[1] = 0.20 * n0                       # Cold He+
#     ndensc[2] = 0.10 * n0                       # Cold O+
# 
#     ndensw    = np.zeros(N)
#     ndensw[0] = 0.60 * n0                       # Hot H+
#     ndensw[1] = 0.00 * n0                       # Hot He+
#     ndensw[2] = 0.00 * n0                       # Hot O+
# 
#     # Input the perpendicular temperature (ev)
#     t_perp    = np.zeros(N)
#     t_perp[0] = 50000.
#     t_perp[1] = 0.
#     t_perp[2] = 0.
# 
#     # Input the parallel temperature (ev)
#     t_par    = np.zeros(N)
#     t_par[0] = 25000.
#     t_par[1] = 0.
#     t_par[2] = 0.
# =============================================================================

    mi    = np.zeros(N)
    mi[0] = 1.  * mp
    mi[1] = 4.  * mp
    mi[2] = 16. * mp

    qi    = np.zeros(N)
    qi[0] = 1.0*q
    qi[1] = 1.0*q
    qi[2] = 1.0*q
    
    A = np.zeros(N)
    for ii in range(N):
        if t_par[ii] != 0:
            A[ii] = t_perp[ii] / t_par[ii] - 1

    ndens   = ndensc + ndensw
    if round(ndens.sum(), 2) != round(n0, 2):
        raise ValueError('Density issue')


    #################################################
    ### CALCULATE FREQUENCIES AND OTHER VARIABLES ###
    #################################################
    w_pc2   = ndensc      * q ** 2 / (mi * e0)   # Cold      ion plasma frequencies (rad/s)
    w_pw2   = ndensw      * q ** 2 / (mi * e0)   # Warm      ion plasma frequencies
    w_ps2   = ndens       * q ** 2 / (mi * e0)   # Total     ion plasma frequencies
    w_pe2   = ndens.sum() * q ** 2 / (me * e0)   # Electron  ion plasma frequencies
    
    w_cyc   =  q * field / mi                    # Ion      cyclotron frequencies (rad/s)
    p_cyc   =  q * field / mp                    # Proton   cyclotron frequency (used for normalization)
    e_cyc   = -q * field / me                    # Electron cyclotron frequency
    
    rho       = (ndens * mi).sum()               # Mass density (kg/m3)
    alfven    = field / np.sqrt(mu0 * rho)       # Alfven speec (m/s)
    
    t_par    *= q                                # Convert temperatures in eV to Joules
    alpha_par = np.sqrt(2.0 * t_par / mi)        # Thermal velocity in m/s (make relativistic?)


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

    for jj in range(3):
        for ii in range(1, Nk):
            CPDR_solns[ii, jj] = fsolve(cold_plasma_dispersion_relation, x0=CPDR_solns[ii - 1, jj], args=(k_vals[ii]), xtol=tol, maxfev=fev)
            warm_solns[ii, jj] = fsolve(warm_plasma_dispersion_relation, x0=warm_solns[ii - 1, jj], args=(k_vals[ii]), xtol=tol, maxfev=fev)

    warm_solns     = estimate_first_and_complexify(warm_solns)
    group_velocity = calculate_group_velocity(warm_solns) 
    
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

    
    ##########
    ## PLOT ##
    ##########
    plt.ioff()
    plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    
    for ii in range(3):
        ax1.plot(k_vals[1:], CPDR_solns[1:, ii],      c=species_colors[ii], linestyle='--', label='Cold')
        ax1.plot(k_vals[1:], warm_solns[1:, ii].real, c=species_colors[ii], linestyle='-',  label='Warm')
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
        ax2.plot(k_vals[1:], warm_solns[1:, ii].imag, c=species_colors[ii], linestyle='-',  label='Growth')

    ax2.set_title('Temporal Growth Rate')
    ax2.set_xlabel(r'$kv_A / \Omega_p$')
    ax2.set_ylabel(r'$\gamma/\Omega_p$')
    ax2.set_xlim(k_min, k_max)
    ax2.set_ylim(-0.05, 0.05)
    ax2.minorticks_on()
    ax2.yaxis.set_minor_locator(MultipleLocator(0.005))
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
# =============================================================================
#     
#     plt.figure()
#     for ii in range(3):
#         ax1.plot(k_vals[1:], group_velocity[1:, ii],      c=species_colors[ii], linestyle='--', label='Cold')
# =============================================================================
    
    plt.show()
