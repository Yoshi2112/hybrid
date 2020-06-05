# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:29:15 2019

@author: Yoshi
"""
import sys
import os
import pdb
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from   scipy.optimize    import fsolve
from   scipy.special     import wofz
from   emperics          import geomagnetic_magnitude, sheely_plasmasphere
from matplotlib.lines    import Line2D
c     = 3E8                                 # m/s
'''
Simplified dispersion solver based on Wang (2016), using structs instead of 
predefined warm/cold arrays. Eventual tests:
     - Normal compared to wang (VERIFIED 21/05/2020)
     - Split one population in two (half density in each). Should return identical result (VERIFIED 21/05/2020)
     - Test with H/O plasma (two species = two bands to solve)
     
NOTE: Plotting subroutine contains hard-coded legends, etc. Would need to be modified
    to be truly general.
    
ISSUES: Completely different H-band solution found for Wang's Figure 1c at 60% RC
        Also different growth rate for Wang's Figure 3c at 2.0 warm hydrogen anisotropy    
        
How?
 -- (DONE: SEEMS FINE)Incorrect fsolve solution? Check by back substitution to examine tolerance/absolute error
 -- (DONE) Also look at getting np.nan for bad fsolve solution
 -- Wang is wrong? Possible
 -- Something weird at high energy/density? One changes alpha_par, the other changes plasma frequency
'''

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


def cold_plasma_dispersion_relation(w, k, Species):
    '''
    w_ps :: Plasma frequency of whole species? Or just cold component? (Based on density)
            Assumes hot components are also cold, adds their density to it.
    '''
    cold_sum = 0.0
    for ii in range(Species.shape[0]):
        cold_sum += Species[ii]['plasma_freq_sq'] / (w * (w - Species[ii]['gyrofreq']))
    return 1 - cold_sum - (c * k / w) ** 2


def warm_plasma_dispersion_relation(w, k, Species):
    '''
    w is a vector: [wr, wi]
    
    Function used in scipy.fsolve minimizer to find roots of dispersion relation.
    Iterates over each k to find values of w that minimize 'solution'.
    
    type_out allows purely real or purely imaginary (coefficient only) for root
    finding. Set as anything else for complex output.
    
    Plasma dispersion function related to Fadeeva function (Summers & Thorne, 1993) by
    i*sqrt(pi) factor.
    
    FSOLVE OPTIONS :: If bad solution, return np.nan?
    '''
    wc = w[0] + 1j*w[1]

    components = 0.0
    for ii in range(Species.shape[0]):
        sp = Species[ii]
        if sp['tper'] == 0:
            components += sp['plasma_freq_sq'] * wc / (sp['gyrofreq'] - wc)
        else:
            pdisp_arg   = (wc - sp['gyrofreq']) / (sp['vth_par']*k)
            pdisp_func  = Z(pdisp_arg)*sp['gyrofreq'] / (sp['vth_par']*k)
            brackets    = (sp['anisotropy'] + 1) * (wc - sp['gyrofreq'])/sp['gyrofreq'] + 1
            Is          = brackets * pdisp_func + sp['anisotropy']
            components += sp['plasma_freq_sq'] * Is

    solution = (wc ** 2) - (c * k) ** 2 + components
    return np.array([solution.real, solution.imag])
    

def estimate_first_and_complexify(solutions):
    outarray = np.zeros((solutions.shape[0], solutions.shape[1]), dtype=np.complex128)
    
    for jj in range(solutions.shape[1]):
        for ii in range(solutions.shape[0]):
            outarray[ii, jj] = solutions[ii, jj, 0] + 1j*solutions[ii, jj, 1]
            
    outarray[0] = outarray[1]
    return outarray


def dispersion_relation_solver(Species, PlasmaParams, norm_k_in=True, norm_k_out=False, norm_w=False, Nk=1000,
                             kmin=0.0, kmax=1.0, plot=False, save=False, savepath=None,
                             return_all=False):
    '''
    Function that solves dispersion relation for a given range of k in a plasma consisting of Species and 
    with given PlasmaParams
    
    Species      -- Structured array of each ion species (including entry for cold electrons)
    PlasmaParams -- Dict containing information about the macroscopic plasma (B0, n0, alven speed, etc.)
    
    *kwargs ::
    norm_k_in -- Flag: kmin/kmax are normalized to units of p_cyc/vA
    norm_k_in -- Flag: For plot routine, k values are (re)normalized to units of p_cyc/vA
    norm_w    -- Flag: Normalize frequency to units of p_cyc
    Nk        -- Number of points in k-space to solve for
    kmin      -- Minimum k-value, in units of p_cyc/vA
    kmax      -- Maximum k-value, in units of p_cyc/vA
    plot      -- Flag: Plot output, show on screen
    save      -- Flag: Save output to directory kwarg 'savepath', suppresses on-screen plotting
    
    OUTPUT:
        k_vals     -- Wavenumber array for solved k values 
        CPDR_solns -- Cold plasma dispersion relation: w(k) for each k in k_vals
        warm_solns -- Warm plasma dispersion relation: w(k) for each k in k_vals
    '''
    gyfreqs, counts = np.unique(Species['gyrofreq'], return_counts=True)
    
    # Remove electron count, 
    gyfreqs = gyfreqs[1:]
    N_solns = counts.shape[0] - 1
    
    # Initialize k space: (un)Normalized by va/pcyc to get into SI units
    if norm_k_in == True:
        knorm_fac = PlasmaParams['p_cyc'] / PlasmaParams['va']
        k_min     = kmin  * knorm_fac
        k_max     = kmax  * knorm_fac
        k_vals    = np.linspace(k_min, k_max, Nk, endpoint=False)
    else:
        k_vals    = np.linspace(kmin, kmax, Nk, endpoint=False)
    
    # fsolve arguments
    eps    = 0.01       # Offset used to supply initial guess (since right on w_cyc returns an error)
    tol    = 1e-10      # Absolute solution convergence tolerance in rad/s
    fev    = 1000000    # Maximum number of iterations
    
    # Solution arrays
    CPDR_solns = np.ones((Nk, N_solns   )) * eps
    warm_solns = np.ones((Nk, N_solns, 2)) * eps
    
    # Error Arrays (ignoring infodict because that's not useful information right now)
    cold_ier = np.zeros((Nk, N_solns), dtype=int)
    cold_msg = np.zeros((Nk, N_solns), dtype='<U256')
    warm_ier = np.zeros((Nk, N_solns), dtype=int)
    warm_msg = np.zeros((Nk, N_solns), dtype='<U256')

    # Initial guesses
    for ii in range(1, N_solns):
        CPDR_solns[0, ii - 1]  = gyfreqs[-ii - 1] * 1.05
        warm_solns[0, ii - 1]  = np.array([[gyfreqs[-ii - 1] * 1.05, 0.0]])

    # Numerical solutions
    for jj in range(N_solns):
        for ii in range(1, Nk):
            CPDR_solns[ii, jj], infodict, cold_ier[ii, jj], cold_msg[ii, jj] =\
                fsolve(cold_plasma_dispersion_relation, x0=CPDR_solns[ii - 1, jj], args=(k_vals[ii], Species), xtol=tol, maxfev=fev, full_output=True)
            
            warm_solns[ii, jj], infodict, warm_ier[ii, jj], warm_msg[ii, jj] =\
                fsolve(warm_plasma_dispersion_relation, x0=warm_solns[ii - 1, jj], args=(k_vals[ii], Species), xtol=tol, maxfev=fev, full_output=True)
            
    warm_solns = estimate_first_and_complexify(warm_solns)

    # Filter out bad solutions
    if True:
        for jj in range(N_solns):
            for ii in range(1, Nk):
                if cold_ier[ii, jj] == 5:
                    CPDR_solns[ii, jj] = np.nan
                
                if warm_ier[ii, jj] == 5:
                    warm_solns[ii, jj] = np.nan
        
    if plot==True or save==True:
        plot_dispersion(k_vals, CPDR_solns, warm_solns, PlasmaParams, norm_k=norm_k_out, norm_w=norm_w, save=save, savepath=savepath)

    if return_all == False:
        return k_vals, CPDR_solns, warm_solns
    else:
        return k_vals, CPDR_solns, cold_ier, cold_msg,\
                       warm_solns, warm_ier, warm_msg


def plot_dispersion(k_vals, CPDR_solns, warm_solns, PlasmaParams,
                    norm_k=False, norm_w=False, save=False, savename=None,
                    title='', growth_only=False, glims=0.05):
    ###############
    ## NORMALIZE ##
    ###############
    if norm_w == True:
        y_cold      = CPDR_solns.copy() / PlasmaParams['p_cyc']
        y_warm      = warm_solns.copy() / PlasmaParams['p_cyc']   
        f_max       = 1.0
        ysuff       = '$/\Omega_p$'
    else:
        y_cold      = CPDR_solns.copy() / (2 * np.pi)
        y_warm      = warm_solns.copy() /(2 * np.pi)
        f_max       = PlasmaParams['p_cyc'] / (2 * np.pi)
        ysuff       = ' (Hz)'
         
    if norm_k == True:
        x_vals  = k_vals.copy() / PlasmaParams['p_cyc'] * PlasmaParams['va']
        xlab    = r'$kv_A / \Omega_p$'
    else:
        x_vals  = k_vals
        xlab    = r'$k (m^{-1})$'
        
    species_colors = ['r', 'b', 'g']
    
    plt.ioff()
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    plt.suptitle(title)
    
    for ii in range(CPDR_solns.shape[1]):
        ax1.plot(x_vals[1:], y_cold[1:, ii],      c=species_colors[ii], linestyle='--', label='Cold')
        ax1.plot(x_vals[1:], y_warm[1:, ii].real, c=species_colors[ii], linestyle='-',  label='Warm')
        #ax1.axhline(w_cyc[ii], c='k', linestyle=':')

    ax1.set_title('Dispersion Relation')
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(r'$\omega${}'.format(ysuff))
    ax1.set_xlim(x_vals[0], x_vals[-1])
    
    ax1.set_ylim(0, f_max)
    ax1.minorticks_on()
    
    type_label = ['Cold Plasma Approx.', 'Hot Plasma Approx.']
    type_style = ['--', '-']
    type_legend = create_type_legend(ax1, type_label, type_style)
    ax1.add_artist(type_legend)
    
    band_labels = [r'$H^+$', r'$He^+$', r'$O^+$']
    band_legend = create_band_legend(ax2, band_labels, species_colors)
    ax2.add_artist(band_legend)
    
    for ii in range(CPDR_solns.shape[1]):
        ax2.plot(x_vals[1:], y_warm[1:, ii].imag, c=species_colors[ii], linestyle='-',  label='Growth')

    ax2.set_title('Temporal Growth Rate')
    ax2.set_xlabel(xlab)
    ax2.set_ylabel(r'$\gamma${}'.format(ysuff))
    ax2.set_xlim(x_vals[0], x_vals[-1])
    
    if norm_w == True:
        ax2.set_ylim(-glims, glims)
    
    ax2.minorticks_on()
    
    if growth_only == True:
        ax2.set_ylim(0, glims)
    
    if save == True:
        plt.savefig(savename)
        plt.close('all')
    else:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()    
    return


def create_species_array(B0, name, mass, charge, density, tper, ani):
    '''
    For each ion species, total density is collated and an entry for 'electrons' added (treated as cold)
    Also output a PlasmaParameters dict containing things like alfven speed, density, hydrogen gyrofrequency, etc.
    
    Inputs must be in SI units: nT, kg, C, /m3, eV, etc.
    '''
    e0        = 8.854e-12
    mu0       = 4e-7*np.pi
    q         = 1.602e-19
    me        = 9.101e-31
    mp        = 1.673e-27
    t_par     = q*tper / (ani + 1)            # Convert Perp temp in eV to Par evergy in Joules  
    alpha_par = np.sqrt(2.0 * t_par  / mass)  # Par Thermal velocity in m/s (make relativistic?)
    ne        = density.sum()
    nsp       = name.shape[0]
    
    # Create initial fields
    Species = np.array([], dtype=[('name', 'U20'),          # Species name
                                  ('mass', 'f8'),           # Mass in kg
                                  ('density', 'f8'),        # Species density in /m3
                                  ('tper', 'f8'),           # Perpendicular temperature in eV
                                  ('anisotropy', 'f8'),     # Anisotropy: T_perp/T_par - 1
                                  ('plasma_freq_sq', 'f8'), # Square of the plasma frequency
                                  ('gyrofreq', 'f8'),       # Cyclotron frequency
                                  ('vth_par', 'f8')])       # Parallel Thermal velocity

    # Insert species values into each
    for ii in range(nsp):
        new_species = np.array([(name[ii], mass[ii], density[ii], tper[ii], ani[ii],
                                                density[ii] * charge[ii] ** 2 / (mass[ii] * e0),
                                                charge[ii]  * B0 / mass[ii],
                                                alpha_par[ii])], dtype=Species.dtype)
        Species = np.append(Species, new_species)
    
    # Add cold electrons
    Species = np.append(Species, np.array([('Electrons', me, ne, 0, 0,
                                            ne * q ** 2 / (me * e0),
                                            -q  * B0 / me,
                                            0.)], dtype=Species.dtype))
    
    PlasParams = {}
    PlasParams['va']    = B0 / np.sqrt(mu0*(density * mass).sum())  # Alfven speed
    PlasParams['n0']    = ne                                        # Electron number density
    PlasParams['p_cyc'] = q*B0 / mp                                 # Proton cyclotron frequency
    return Species, PlasParams


def get_dispersion_relations(B0, name, mass, charge, density, tper, ani, kmin=0.0, kmax=1.0, Nk=1000,
                         norm_k_in=True, norm_k_out=False, norm_w=False, plot=False):
    '''
    Wrapper around dispersion solver, taking in numpy arrays of species and field quantities.
    
    INPUT:
        B0      -- Magnetic field, in T
        name    -- Species name, as a string
        mass    -- Species mass, in kg
        charge  -- Species charge, in C
        density -- Species number density, in /m3
        tper    -- Species perpendicular (to B0) temperature, in eV
        ani     -- Species anisotropy, T_perp / T_par - 1
    
    OUTPUT:
        k_vals      -- Angular wavenumbers for which the dispersion relations were solved (rad/m)
        CPDR_solns  -- Cold Plasma Dispersion Relation solutions, in angular frequency (rad/s)
        warm_solns  -- Warm Plasma Dispersion Relation solutions, in angular frequency (rad/s)
    
    kwarg:
        kmin=0.0            -- Minimum wavenumber to solve for
        kmax=1.0            -- Maximum wavenumber to solve for
        norm_k_in=True      -- Flag to signify if kmin/kmax kwargs are to be normalized by va/pcyc
        norm_k_out=False    -- Flag to signify if output wavenumbers are     normalized by va/pcyc
        norm_w=False        -- Flag to signify if output frequencies are     normalized by pcyc
        plot=False          -- Plotting flag. Should only be used if function is called a single time.
    
    Note: 
        The 'Warm Plasma Dispersion Relation' is its own thing, if you believe Stix, and is probably
        not this simple. These equations take after the Chen/Wang papers of 2016, and are remarkably
        similar to those used by Kozyra in 1984 (i.e. only valid for small growth rates, gam << w)
        
    To Do:
        - Put in a check to make sure all arrays are the same size
    '''
    Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
    
    if False:
        # Diagnostic to check residuals
        k_vals, CPDR_solns, cold_ier, cold_msg, warm_solns, warm_ier, warm_msg = \
            dispersion_relation_solver(Species, PP, norm_k_in=norm_k_in, norm_k_out=norm_k_out, \
                                     norm_w=norm_w, plot=plot, \
                                     Nk=Nk, kmin=kmin, kmax=kmax, return_all=True)

        back_substitute(k_vals, CPDR_solns, warm_solns, Species)
    else:
        k_vals, CPDR_solns, warm_solns = dispersion_relation_solver(Species, PP, norm_k_in=norm_k_in, norm_k_out=norm_k_out, \
                                     norm_w=norm_w, plot=plot, kmin=kmin, kmax=kmax, Nk=Nk)
    return k_vals, CPDR_solns, warm_solns


def back_substitute(k_vals, CPDR_solns, warm_solns, Species):
    '''
    DIAGNOSTIC PLOT 
    For each k_value, substitute in the calculated w_value to collect residuals.
    
    Plot residuals as a function of k. Note that the relative error in growth rate isn't
    particularly useful, since it is zero for many k values (i.e. massive 'relatve' error)
    '''
    print('Doing back-substitution...')
    Nk      = CPDR_solns.shape[0]
    N_solns = CPDR_solns.shape[1]

    WPDR           = np.zeros((warm_solns.shape[0], warm_solns.shape[1], 2), dtype=float)
    WPDR[:, :, 0]  = warm_solns.real
    WPDR[:, :, 1]  = warm_solns.imag
    warm_residuals = np.zeros((warm_solns.shape[0], warm_solns.shape[1], 2), dtype=float)
    cold_residuals = np.zeros(CPDR_solns.shape)
    
    for jj in range(N_solns):
        for ii in range(1, Nk):
            cold_residuals[ii, jj] = cold_plasma_dispersion_relation(CPDR_solns[ii, jj], k_vals[ii], Species)
            warm_residuals[ii, jj] = warm_plasma_dispersion_relation(      WPDR[ii, jj], k_vals[ii], Species)
    
    # Plot residuals as a percentage of solution
    species_colors      = ['r', 'b', 'g']
            
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    for ii in range(3):
        axes[0].plot(k_vals, cold_residuals[:, ii]   , c=species_colors[ii], ls=':')
        axes[0].plot(k_vals, warm_residuals[:, ii, 0], c=species_colors[ii])
        axes[1].plot(k_vals, warm_residuals[:, ii, 1], c=species_colors[ii])
    fig.suptitle('Absolute Error')
    axes[0].set_xlabel('k')
    axes[1].set_xlabel('k')
    axes[0].set_ylabel('$\Delta \omega_r$')
    axes[0].set_ylabel('$\Delta \omega_i$')
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    for jj in range(3):
        axes[0].plot(k_vals, cold_residuals[:, jj]    / CPDR_solns[:, jj]   , c=species_colors[jj], ls=':')
        axes[0].plot(k_vals, warm_residuals[:, jj, 0] /       WPDR[:, jj, 0], c=species_colors[jj])
        axes[1].plot(k_vals, warm_residuals[:, jj, 1] /       WPDR[:, jj, 1], c=species_colors[jj], marker='o')
    fig.suptitle('Relative Error')
    axes[0].set_xlabel('k')
    axes[1].set_xlabel('k')
    axes[0].set_ylabel('$\Delta \omega_r$ / $\omega_r$')
    axes[0].set_ylabel('$\Delta \omega_i$ / $\omega_i$')
    plt.show()
    return


if __name__ == '__main__':
    '''
    Test quantities/direct interface
    '''                               # Number of species
    L_shell  = 4                                # L-shell at which magnetic field and density are calculated
    n0       = sheely_plasmasphere(L_shell)     # Plasma density, /m3
    _B0      = geomagnetic_magnitude(L_shell)   # Background magnetic field, T
    mp       = 1.673e-27                        # Proton mass (kg)
    qi       = 1.602e-19                        # Elementary charge (C)
    
    if True:
        '''
        Standard Wang (2016) test values DON'T CHANGE
        '''
        H_ab = 0.7
        He_ab= 0.2
        O_ab = 0.1
        RC_ab= 0.1
        
        _name    = np.array(['Warm H'  , 'Cold H'        , 'Cold He', 'Cold O'])
        _mass    = np.array([1.0       , 1.0             , 4.0      , 16.0    ]) * mp
        _charge  = np.array([1.0       , 1.0             , 1.0      ,  1.0    ]) * qi
        _density = np.array([RC_ab*H_ab, (1 - RC_ab)*H_ab, He_ab    ,  O_ab,  ]) * n0
        _tpar    = np.array([25e3      , 0.0             , 0.0      ,  0.0    ])
        _ani     = np.array([1.0       , 0.0             , 0.0      ,  0.0    ])
        _tper    = (_ani + 1) * _tpar
    else:
        '''
        Change these ones as much as you want
        '''
        H_ab = 0.9
        O_ab = 0.1
        RC_ab= 0.1
        
        _name    = np.array(['Warm H'  , 'Cold H'        , 'Cold O'])
        _mass    = np.array([1.0       , 1.0             , 16.0    ]) * mp
        _charge  = np.array([1.0       , 1.0             , 1.0     ]) * qi
        _density = np.array([RC_ab*H_ab, (1 - RC_ab)*H_ab, O_ab,   ]) * n0
        _tpar    = np.array([25e3      , 0.0             , 0.0     ])
        _ani     = np.array([1.0       , 0.0             , 0.0     ])
        _tper    = (_ani + 1) * _tpar
        
    _k, CPDR, WPDR = get_dispersion_relations(_B0, _name, _mass, _charge, _density, _tper, _ani,
                         kmin=0.0, kmax=1.0, Nk=1000, norm_k_in=True, norm_k_out=True, 
                         norm_w=True, plot=True)