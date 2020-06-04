# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:29:15 2019

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt
from   scipy.optimize    import fsolve
from   scipy.special     import wofz
import extract_parameters_from_data as data
from matplotlib.lines    import Line2D
import os
import pdb
'''
Still technically hard-coded for 3 species (H+, He+, O+)
This doesn't need to be the case, but I'd need a root-finder that doesn't rely on 
initial guesses as heavily, and that finds an arbitrary number of (unique) roots.
'''
def plot_dispersion(k_vals, CPDR_solns, warm_solns, k_isnormalized=False, w_isnormalized=False, save=False, savepath=None):
    '''
    Plots the CPDR and WPDR nicely as per Wang et al 2016. 
    
    INPUT:
        k_vals     -- Wavenumber values in /m3 or normalized to p_cyc/v_A
        CPDR_solns -- Cold-plasma frequencies in Hz or normalized to p_cyc
        WPDR_solns -- Warm-plasma frequencies in Hz or normalized to p_cyc. 
                   -- .real is dispersion relation, .imag is growth rate vs. k
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

    species_colors      = ['r', 'b', 'g']
        
    plt.ioff()
    plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    
    for ii in range(3):
        ax1.plot(k_vals[1:], CPDR_solns[1:, ii],      c=species_colors[ii], linestyle='--', label='Cold')
        ax1.plot(k_vals[1:], warm_solns[1:, ii].real, c=species_colors[ii], linestyle='-',  label='Warm')

    ax1.set_ylim(0, None)
    ax1.set_xlim(k_vals[0], k_vals[-1])
    ax2.set_xlim(k_vals[0], k_vals[-1])
    
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


class ParticlePopulation:
    '''
    Constructor for the ParticlePopulation (PP) class. PP objects take as inputs
    their population parameters:
        Magnetic field (in T)
        mass (in proton units/amu)
        charge (in elementary units)
        density (in /cc)
        Perpendicular temperature (in eV)
        Anisotropy
    and automatically calculates:
        Parallel temperature (in eV)
        Plasma frequency (in rad/s)
        thermal velocity (in m/s)
        cyclotron frequency (in rad/s)
    '''
    def __init__(self, _B, _mass, _charge, _dens, _Tperp=0, _A=0, _name='', _index=None):
        PMASS    = 1.673e-27
        PCHARGE  = 1.602e-19
        EPNAUGHT = 8.854e-12
        self.mass    = _mass*PMASS
        self.charge  = _charge*PCHARGE
        self.density = _dens*1e6
        self.A       = _A
        self.Tperp   = _Tperp
        self.Tpar    = _Tperp / (_A + 1)
        self.wp2     = self.density * self.charge ** 2 / (self.mass * EPNAUGHT)
        self.vth     = np.sqrt(2.0 * PCHARGE * self.Tpar  / self.mass)
        self.w_cyc   = self.charge * _B / self.mass

        self.name    = _name
        self.index   = _index


def Z(arg):
    '''Return Plasma Dispersion Function : Normalized Fadeeva function'''
    return 1j*np.sqrt(np.pi)*wofz(arg)


def warm_plasma_dispersion_relation(wt, k, sp, all_cold):
    '''
    w is a vector: [wr, wi]
    sp is the species list of ParticlePopulation objects. The first one is always electrons.
    Function used in scipy.fsolve minimizer to find roots of dispersion relation.
    Iterates over each k to find values of w that minimize 'solution'.
    
    type_out allows purely real or purely imaginary (coefficient only) for root
    finding. Set as anything else for complex output.
    
    Plasma dispersion function related to Fadeeva function (Summers & Thorne, 1993) by
    i*sqrt(pi) factor.
    
    test_output flag allows w and solution to be complex, rather than wrapped. Additional
    arguments may also be selected to be exported this way.
    
    SWAP BETWEEN WARM/COLD DISPERSION RELATIONS BY DISABLING TEMPERATURE
    '''
    wc = wt[0] + 1j*wt[1]                       # Complex frequency
    
    N  = len(sp)                                # Number of species
    c  = 3E8                                    # Speed of light (m/s)
    
    if any(np.isnan(wt) == True):
        return np.array([np.nan, np.nan])       # Immediately return invalid value for wt

    components = 0
    for ii in range(N):
        if sp[ii].Tperp == 0 or all_cold == True:
            if (sp[ii].w_cyc - wc) == 0:
                Isc = np.inf
            else:
                Isc = wc / (sp[ii].w_cyc - wc)
            components += sp[ii].wp2 * Isc
        else:
            pdisp_arg   = (wc - sp[ii].w_cyc) / (sp[ii].vth*k)
            pdisp_func  = Z(pdisp_arg)*sp[ii].w_cyc / (sp[ii].vth*k)
            brackets    = (sp[ii].A + 1) * (wc - sp[ii].w_cyc)/sp[ii].w_cyc + 1
            Isw         = brackets * pdisp_func + sp[ii].A
            components += sp[ii].wp2 * Isw

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


def num_bands(sp, field, eps=1.05):
    '''
    This is really the only thing that hard-codes this script to 3 species.
    If there was a better, globally convergent and multi-solution solver, 
    then we wouldn't need this bit.
    '''
    has_solns  = np.array([0,  0,  0 ])
    
    for ii in range(len(sp)):
        if sp[ii].index == 0:
            has_solns[0]  = 1   
        elif sp[ii].index == 1:
            has_solns[1]  = 1  
        elif sp[ii].index == 2:
            has_solns[2]  = 1
            
    mp         = 1.673E-27; qp  = 1.602E-19 
    init_guess = np.array([0., 0., 0.])
    cyclotron  = qp * field / (mp * np.array([1.0, 4.0, 16.0]))
    
    init_guess[0] = cyclotron[1] * eps      # Slightly above He+ gyrofrequency (H  band)
    init_guess[1] = cyclotron[2] * eps      # Slightly above O+  gyrofrequency (He band)
    init_guess[2] = cyclotron[2] * 0.01     # 1% of          O+ gyrofrequency  (O  band)
    return has_solns, init_guess


def get_dispersion_relation(field, spec, Nk=5000, kmin=0.0, kmax=1.0):
    '''
    field  -- Background magnetic field in T
    spec   -- Plasma species, as ParticlePopulation objects (contains densities, temperatures, etc.)
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
    N   = len(spec)
    
    qp  = 1.602E-19
    mp  = 1.673E-27
    mu0 = (4E-7) * np.pi

    p_cyc  = qp * field / mp
    rho    = sum([spec[ii].density * spec[ii].mass for ii in range(N)])     # Mass density (kg/m3)
    alfven = field / np.sqrt(mu0 * rho)                                     # Alfven speed (m/s)

    band_exists, init_guess = num_bands(spec, field)
    
    # Initialize k space: Normalized by va/pcyc
    knorm_fac = p_cyc / alfven
    k_min     = kmin  * knorm_fac
    k_max     = kmax  * knorm_fac
    k_vals    = np.linspace(k_min, k_max, Nk, endpoint=False)

    eps    = 0.01       # 'Epsilon' value (keeps initial guesses off zero)
    tol    = 1e-15      # Solution tolerance
    fev    = 1000000    # Number of retries to get below tolerance
        
    CPDR_solns = np.ones((Nk, 3, 2))
    warm_solns = np.ones((Nk, 3, 2))

    # Initial guesses (lower band cyclotron frequency)
    for ii in range(3):
        CPDR_solns[0, ii - 1]  = np.array([[init_guess[ii], 0. ]])
        warm_solns[0, ii - 1]  = np.array([[init_guess[ii], eps]])

    # Numerical solutions: Use previous solution as starting point for new solution
    for jj in range(3):                 # For each frequency band
        if band_exists[jj] == 1:
            for ii in range(1, Nk):     # For each value of k
                CPDR_solns[ii, jj] = fsolve(warm_plasma_dispersion_relation, x0=CPDR_solns[ii - 1, jj], args=(k_vals[ii], spec, True),  xtol=tol, maxfev=fev)
                warm_solns[ii, jj] = fsolve(warm_plasma_dispersion_relation, x0=warm_solns[ii - 1, jj], args=(k_vals[ii], spec, False), xtol=tol, maxfev=fev)

    warm_solns  = estimate_first_and_complexify(warm_solns)
        
    CPDR_solns /= (2 * np.pi)   # Units of Herz
    warm_solns /= (2 * np.pi)
    return k_vals, CPDR_solns, warm_solns


def get_DRs_from_data(data_path, time_start, time_end, probe, pad, cmp, Nk=5000):
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



if __name__ == '__main__':
    _n0 = 346.4
    _B0 = 487.5e-9
    
    population_list = []
    
    # Electrons
    population_list.append(ParticlePopulation(_B0, 1/1836 , -1.0, _n0, _name='e-', _index=-1))
    
    # Cold populations
    population_list.append(ParticlePopulation(_B0, 1.0 , 1.0, 0.6*_n0, _name='Cold H+' , _index=0))
    population_list.append(ParticlePopulation(_B0, 4.0 , 1.0, 0.2*_n0, _name='Cold He+', _index=1))
    population_list.append(ParticlePopulation(_B0, 16.0, 1.0, 0.1*_n0, _name='Cold O+' , _index=2))
    
    # Warm populations
    population_list.append(ParticlePopulation(_B0, 1.0,  1.0, 0.1*_n0, _Tperp=50000, _A=1, _name='Warm H+', _index=0))
    
    _k, _CPDR, _WPDR = get_dispersion_relation(_B0, population_list)
    
    plot_dispersion(_k, _CPDR, _WPDR)
    