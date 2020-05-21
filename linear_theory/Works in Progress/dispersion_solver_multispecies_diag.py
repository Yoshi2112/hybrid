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
from plasma_lib.emperics import geomagnetic_magnitude, sheely_plasmasphere
from matplotlib.lines    import Line2D
c     = 3E8                                 # m/s
'''
Simplified dispersion solver based on Wang (2016), using structs instead of 
predefined warm/cold arrays. Eventual tests:
     - Normal compared to wang (VERIFIED 21/05/2020)
     - Split one population in two (half density in each). Should return identical result.
     - Test with H/O plasma (two species = two bands to solve)
     
NOTE: Plotting subroutine contains hard-coded legends, etc. Would need to be modified
    to be truly general.
    
ISSUES: Completely different H-band solution found for Wang's Figure 1c at 60% RC
        Also different growth rate for Wang's Figure 3c at 2.0 warm hydrogen anisotropy    
        
How?
 -- Incorrect fsolve solution? Check by back substitution to examine tolerance/absolute error
 -- Also look at getting np.nan for bad fsolve solution
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


def get_dispersion_relations(Species, PlasmaParams, norm_k=False, norm_w=False, Nk=1000, kmin=0.0, kmax=1.0, plot=False, save=False, savepath=None):
    '''
    Species      -- Structured array of each ion species (including entry for cold electrons)
    PlasmaParams -- Dict containing information about the macroscopic plasma (B0, n0, alven speed, etc.)
    
    *kwargs ::
    norm_k -- Flag: Normalize wavenumber to units of p_cyc/vA
    norm_w -- Flag: Normalize frequency to units of p_cyc
    Nk     -- Number of points in k-space to solve for
    kmin   -- Minimum k-value, in units of p_cyc/vA
    kmax   -- Maximum k-value, in units of p_cyc/vA
    plot   -- Flag: Plot output, show on screen
    save   -- Flag: Save output to directory kwarg 'savepath', suppresses on-screen plotting
    
    OUTPUT:
        k_vals     -- Wavenumber array for solved k values 
        CPDR_solns -- Cold plasma dispersion relation: w(k) for each k in k_vals
        warm_solns -- Warm plasma dispersion relation: w(k) for each k in k_vals
    '''
    gyfreqs, counts = np.unique(Species['gyrofreq'], return_counts=True)
    
    # Remove electron count, 
    gyfreqs = gyfreqs[1:]
    N_solns = counts.shape[0] - 1
    
    # Initialize k space: Normalized by va/pcyc
    knorm_fac = PlasmaParams['p_cyc'] / PlasmaParams['va']
    k_min     = kmin  * knorm_fac
    k_max     = kmax  * knorm_fac
    k_vals    = np.linspace(k_min, k_max, Nk, endpoint=False)
    
    # fsolve arguments
    eps    = 0.01
    tol    = 1e-15
    fev    = 1000000
    
    # Solution arrays
    CPDR_solns = np.ones((Nk, N_solns   )) * eps
    warm_solns = np.ones((Nk, N_solns, 2)) * eps

    # Error Arrays (ignoring infodict because that's not useful information right now)
    cold_ier = np.zeros(Nk)
    cold_msg = np.zeros(Nk, dtype='<U256')
    warm_ier = np.zeros(Nk)
    warm_msg = np.zeros(Nk, dtype='<U256')
    
    # Initial guesses
    for ii in range(1, N_solns):
        CPDR_solns[0, ii - 1]  = gyfreqs[-ii - 1] * 1.05
        warm_solns[0, ii - 1]  = np.array([[gyfreqs[-ii - 1] * 1.05, 0.0]])

    # Numerical solutions
    for jj in range(N_solns):
        for ii in range(1, Nk):
            CPDR_solns[ii, jj] = fsolve(cold_plasma_dispersion_relation, x0=CPDR_solns[ii - 1, jj], args=(k_vals[ii], Species), xtol=tol, maxfev=fev)
            warm_solns[ii, jj] = fsolve(warm_plasma_dispersion_relation, x0=warm_solns[ii - 1, jj], args=(k_vals[ii], Species), xtol=tol, maxfev=fev)

    warm_solns = estimate_first_and_complexify(warm_solns)
    
    ###############
    ## NORMALIZE ##
    ###############
    if norm_w == True:
        CPDR_solns /= PlasmaParams['p_cyc']
        warm_solns /= PlasmaParams['p_cyc']       
    else:
        CPDR_solns /= (2 * np.pi)
        warm_solns /= (2 * np.pi)
         
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
        #ax1.axhline(w_cyc[ii], c='k', linestyle=':')

    ax1.set_title('Dispersion Relation')
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(r'$\omega${}'.format(ysuff))
    ax1.set_xlim(k_vals[0], k_vals[-1])
    
    ax1.set_ylim(0, f_max)
    ax1.minorticks_on()
    
    type_label = ['Cold Plasma Approx.', 'Hot Plasma Approx.']
    type_style = ['--', '-']
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


def create_species_array(B0, name, mass, charge, density, tper, ani):
    '''
    For each ion species, total density is collated and an entry for 'electrons' added (treated as cold)
    Also output a PlasmaParameters dict containing things like alfven speed, density, hydrogen gyrofrequency, etc.
    '''
    e0        = 8.854e-12
    mu0       = 4e-7*np.pi
    q         = 1.602e-19
    me        = 9.101e-31
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


def dispersion_relations(B0, name, mass, charge, density, tper, ani, norm_k=False, norm_w=False, plot=False):
    '''
    Wrapper around dispersion solver, taking in general numpy arrays of plasma params
    
    Every input must be in standard SI units, and be of the same shape (Put check in?)
    '''
    Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
    k_vals, CPDR_solns, warm_solns = get_dispersion_relations(Species, PP, norm_k=norm_k, norm_w=norm_w, plot=plot)
    return k_vals, CPDR_solns, warm_solns


if __name__ == '__main__':
    Nn       = 3                                # Number of species
    L_shell  = 4                                # L-shell at which magnetic field and density are calculated
    n0       = sheely_plasmasphere(L_shell)     # Plasma density, /m3
    _B0      = geomagnetic_magnitude(L_shell)   # Background magnetic field, T
    HPA      = 0.1                              # Hot proton abundance
    mp       = 1.673e-27                        # Proton mass (in kg)
    qi       = 1.602e-19
    
    H_ab = 0.7
    He_ab= 0.2
    O_ab = 0.1
    
    RC_ab= 0.1
    
    # These are the sorts of arguments I'd pass
    _name    = np.array(['Warm H', 'Cold H', 'Cold He', 'Cold O'])
    _mass    = np.array([1.0, 1.0, 4.0, 16.0]) * mp
    _charge  = np.array([1.0, 1.0, 1.0,  1.0]) * qi
    _density = np.array([RC_ab*H_ab, (1 - RC_ab)*H_ab, He_ab, O_ab]) * n0
    _tpar    = np.array([25e3, 0.0, 0.0, 0.0])
    _ani     = np.array([2.0, 0.0, 0.0, 0.0])
    _tper    = (_ani + 1) * _tpar
    
    dispersion_relations(_B0, _name, _mass, _charge, _density, _tper, _ani,
                         norm_k=True, norm_w=True, plot=True)
