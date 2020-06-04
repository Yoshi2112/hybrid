# -*- coding: utf-8 -*-
"""
Created on Tues Aug 19 20:26:09 2019

@author: Yoshi

Script Function:
"""
import warnings
import pdb
import numpy as np
import matplotlib.pyplot as plt

from plasma_lib.emperics    import geomagnetic_magnitude, sheely_plasmasphere
from plasma_lib.convective_growth_rate import calculate_growth_rate
from plasma_lib.warm_dispersion_chen   import get_dispersion_relation, plot_dispersion, create_type_legend, create_band_legend
'''
Kozyra calculate_growth_rate function: Returns freqs, CGR, stop band. Takes /cm3 and nT
Chen get_dispersion_relation function: Returns k, CPDR, WPDR (complex). Takes /m3 and T

Expected range of k values for EMIC waves: up to 1 or 2 p_cyc/v_A (normalized units)
Up to around 2e-4 m^-1
Frequencies up to proton cyclotron : Up to around 5Hz for high field values (but generally lower)
'''

def get_and_plot_dispersion(save=False):
    '''
    2x2 grid, A4 size
    
    Put dispersion relation in top half, 1x2
    Both growth rates underneath, CGR then Chen/Wang WPDR
    
    To Do:
        - Check growth rate units for Kozyra/Chen
        - Output to PDF/png
        - Label cyclotron frequencies (or remove with flag for varying B0)
        - Work out how to display it for multiple values
        - Do some investigating : Adjust ion composition parameter space with constant field (for marginal instability)
        
    -- Two figures
      - First figure: Chen modified code - HOPE + RBSPICE + COLD densities. Set for arbitrary number of species.
      - Second figure: Kozyra CGR - Only does one hot population, try one with HOPE (top), one with RBSPICE (bottom)
      
    Time axis: Maybe use alpha? Do for all points or if >10, pick 5 roughly equally spaced?
    '''
    ###################################
    ### CALL FUNCTIONS, INIT VALUES ###
    ###################################
    freqs,  cgr, stop  = calculate_growth_rate(  field, cold_dens, warm_dens, anisotropy, temperp=t_perp, maxfreq=1.0)
    k_vals, cpdr, wpdr = get_dispersion_relation(field, cold_dens, warm_dens, anisotropy, t_perp, kmin=0.0, kmax=1.0)
    
    k_vals *= 1e6
    species_colors     = ['r', 'b', 'g']
    species_cyc        = np.array([1.0, 1/4, 1/16]) * q * field * 1e-9 / (mp * 2 * np.pi)
    print(species_cyc)

    plt.ioff()
    fig = plt.figure(figsize=(8.27, 11.69))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    
    #####################################
    ### TOP PLOT: DISPERSION RELATION ###
    #####################################
    for ii in range(3):
        ax1.plot(k_vals[1:], cpdr[1:, ii],      c=species_colors[ii], linestyle='--', label='Cold')
        ax1.plot(k_vals[1:], wpdr[1:, ii].real, c=species_colors[ii], linestyle='-',  label='Warm')
        ax1.axhline(species_cyc[ii], c='k', linestyle=':')
        
    ax1.set_title('Dispersion Relation')
    ax1.set_xlabel(r'k ($\times 10^{-6} m^{-1}$)')
    ax1.set_ylabel(r'$\omega$ (Hz)')
    ax1.set_xlim(k_vals[0], k_vals[-1])
    
    ax1.set_ylim(0, f_max)
    ax1.minorticks_on()
    
    type_label = ['Cold Plasma Approx.', 'Hot Plasma Approx.', 'Cyclotron Frequencies']
    type_style = ['--', '-', ':']
    type_legend = create_type_legend(ax1, type_label, type_style)
    ax1.add_artist(type_legend)
    
    band_labels = [r'$H^+$', r'$He^+$', r'$O^+$']
    band_legend = create_band_legend(ax3, band_labels, species_colors)
    ax3.add_artist(band_legend)
    
    #################################
    ### SECOND PLOTS: GROWTH RATE ###
    #################################
    # Convective
    ax2.plot(freqs, cgr)
    for ii in range(stop.shape[0] - 1):
        if stop[ii] == 1:
            ax2.axvspan(freqs[ii], freqs[ii + 1], color='k')
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Convective Growth Rate')
    
    ax2.set_xlim(0, f_max)
    
    # Warm-Plasma (Temporal? Still convective in Wang?)
    for ii in range(3):
        ax3.plot(k_vals[1:], wpdr[1:, ii].imag, c=species_colors[ii])

    ax3.set_xlabel(r'Wavenumber ($\times 10^{-6} m^{-1}$)')
    
    ax3.set_xlim(k_vals[0], k_vals[-1])
    ax3.set_ylim(None, None)
        
    ax3.tick_params("y", right=True, labelright=True, labelleft=False, left=False)
    ax3.yaxis.set_label_position('right')
    ax3.set_ylabel('Temporal Growth Rate')
    #ax3.yaxis.tick_right()
    
    fig.subplots_adjust(wspace=0)
    
    ####################
    ### SAVE OR SHOW ###
    ####################
    if save == True:
        name = None
        fig.savefig(save_path + name)
    else:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()  
    return


if __name__ == '__main__':
    #warnings.filterwarnings('error')
    save_path = None
    
    Nn       = 3                                    # Number of species
    L_shell  = 4                                    # L-shell at which magnetic field and density are calculated
    n0       = sheely_plasmasphere(L_shell)*1e-6    # /cc
    field    = geomagnetic_magnitude(L_shell)*1e9   # nT
    
    f_max    = 7.0
    q        = 1.602e-19
    mp       = 1.673e-27
    
    cold_dens = np.zeros(Nn)
    cold_dens[0] = 0.6*n0     # Cold Hydrogen
    cold_dens[1] = 0.2*n0     # Cold Helium
    cold_dens[2] = 0.1*n0     # Cold Oxygen

    # Density of warm species (same order as cold) (number/cc)
    warm_dens    = np.zeros(Nn)
    warm_dens[0] = 0.1*n0     # Warm Hydrogen
    warm_dens[1] = 0.0*n0     # Warm Helium
    warm_dens[2] = 0.0*n0     # Warm Oxygen
    
    # Density of warmer species (same order as cold) (number/cc)
    warm_dens    = np.zeros(Nn)
    warm_dens[0] = 0.1*n0     # Warm Hydrogen
    warm_dens[1] = 0.0*n0     # Warm Helium
    warm_dens[2] = 0.0*n0     # Warm Oxygen

    # Input the perpendicular temperature (ev)
    t_perp    = np.zeros(Nn)
    t_perp[0] = 50000.
    t_perp[1] = 00000.
    t_perp[2] = 00000.

    # Anisotropy value (A = T_perp/T_par - 1, A = 0 when isotropic)
    anisotropy    = np.zeros(Nn)
    anisotropy[0] = 1.
    anisotropy[1] = 0.
    anisotropy[2] = 0.
    
    get_and_plot_dispersion()