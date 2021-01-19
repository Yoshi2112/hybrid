# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:50:49 2020

@author: Yoshi
"""
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines               import Line2D
from dispersion_solver_multispecies import get_dispersion_relations

e0     = 8.854e-12
mu0    = 4e-7*np.pi
B_surf = 3.12e-5
qi     = 1.602177e-19                       # Elementary charge (C)
c      = 2.998925e+08                       # Speed of light (m/s)
mp     = 1.672622e-27                       # Mass of proton (kg)
me     = 9.109384e-31                       # Mass of electron (kg)
kB     = 1.380649e-23                       # Boltzmann's Constant (J/K)

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


def get_params_from_sim_input(filepath):
    '''
    Read particle file, distill down into something that the dispersion reader
    can parse. What do we need?
    -- Magnetic field (equatorial)
    (, norm_k_in=False, norm_w=False,
        kmin=kmin, kmax=kmax)
        
    Maybe have optional to read from run_inputs too? Or just specify a priori?
    
    Need to join together every pair in species_lbl (because it separates temp from ion)
    '''
    ### PARTICLE/PLASMA PARAMETERS ###
    with open(filepath, 'r') as f:
        species_lbl = np.array(f.readline().split()[1:])
        
        temp_color = np.array(f.readline().split()[1:])
        temp_type  = np.array(f.readline().split()[1:], dtype=int)
        dist_type  = np.array(f.readline().split()[1:], dtype=int)
        nsp_ppc    = np.array(f.readline().split()[1:], dtype=int)
        
        mass       = np.array(f.readline().split()[1:], dtype=float)
        charge     = np.array(f.readline().split()[1:], dtype=float)
        drift_v    = np.array(f.readline().split()[1:], dtype=float)
        density    = np.array(f.readline().split()[1:], dtype=float)*1e6
        anisotropy = np.array(f.readline().split()[1:], dtype=float)
        
        # Particle energy: If beta == 1, energies are in beta. If not, they are in eV                                    
        E_per      = np.array(f.readline().split()[1:], dtype=float)
        E_e        = float(f.readline().split()[1])
        beta_flag  = int(f.readline().split()[1])
    
        L         = float(f.readline().split()[1])         # Field line L shell
        B_eq      = f.readline().split()[1]                # Initial magnetic field at equator: None for L-determined value (in T) :: 'Exact' value in node ND + NX//2

    if B_eq == '-':
        B_eq = (B_surf / (L ** 3))
    else:
        B_eq = float(B_eq)
    
    ne         = density.sum()                 # Electron number density
    va         = B_eq / np.sqrt(mu0*ne*mp)     # Alfven speed at equator: Assuming pure proton plasma
    charge    *= qi                            # Cast species charge to Coulomb
    mass      *= mp                            # Cast species mass to kg
    drift_v   *= va                            # Cast species velocity to m/s

    # E_per in eV or in terms of beta :: Convert to eV
    if beta_flag == 0:
        Tperp      = E_per
    else:    
        Tperp      = (E_per * B_eq ** 2 / (2 * mu0 * ne * kB)) // 11603.

    full_lbls = np.empty(species_lbl.shape[0] // 2, dtype='<U15')
    for ii in range(species_lbl.shape[0] // 2):
        bit1 = species_lbl[2*ii]
        bit2 = species_lbl[2*ii + 1]
        full_lbls[ii] = bit1 + ' ' + bit2
    
    return B_eq, temp_type, full_lbls, mass, charge, density, Tperp, anisotropy


def get_linear_dispersion_from_sim(filepath, zero_cold=True):
    '''
    This should control all the inputs and units stuff for a single param file to export
    '''
    # Need to set filepath
    B_eq, temp_type, species_lbl, mass, charge, density, Tperp, anisotropy = get_params_from_sim_input(filepath)
            
    if zero_cold == True:
        for ii in range(Tperp.shape[0]):
            if temp_type[ii] == 0:
                Tperp[ii] = 0.0
    
    p_cyc = qi * B_eq / mp 
    wpi   = np.sqrt(density.sum() * qi ** 2 / (mp * e0))
    dx    = c / wpi
    
    # define k value range to solve within :: What would be the maximum k solved for in the simulation?
    # Just set normalized inputs for now
    k_min = 0.0
    
    # Convert from linear units to angular units for k range to solve over :: Maximum resolvable by simulation
    k_max = np.pi / dx

    k_vals, CPDR_solns, WPDR_solns = get_dispersion_relations(B_eq, species_lbl, mass, charge, \
                                           density, Tperp, anisotropy, norm_k_in=False, norm_k_out=False,
                                           norm_w=False, kmin=k_min, kmax=k_max, Nk=1000)
    
    # Convert back from angular units to linear units
    k_vals     /= 2*np.pi
    CPDR_solns /= 2*np.pi
    WPDR_solns /= 2*np.pi
    return k_vals, CPDR_solns, WPDR_solns, p_cyc


def plot_series(save=False):
    '''
    To do:
        Easy way to put legend/title in for each different series
        Easy way to distinguish between solutions. Maybe just put the cyclotron frequencies in and use all black?
    '''
    gitdir    = 'C://Users//iarey//Documents//GitHub'
    folder    = '//hybrid//simulation_codes//run_inputs//from_data//event_25Jul_quiet//Changing_Cold_He//'
    file_list = os.listdir(gitdir + folder)
    num_files = len(file_list)
    
    alpha     = np.linspace(0.1, 1.0, num_files)
    
    # Generate plot space
    plt.ioff()
    plt.figure(figsize=(15, 10))
    plt.suptitle('Changing Helium Concentration')
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    
    # Load and plot each file
    for ff in range(num_files):
        # Cycle through files in folder :: Use that for alpha as well
        filename = file_list[ff]
        filepath = gitdir + folder + filename
        
        k_vals, CPDR_solns, WPDR_solns, p_cyc = get_linear_dispersion_from_sim(filepath, zero_cold=True)
            
        species_colors = ['r', 'b', 'g']

        for ii in range(CPDR_solns.shape[1]):
            ax1.plot(k_vals[1:], CPDR_solns[1:, ii],      c=species_colors[ii], linestyle='--', label='Cold', alpha=alpha[ff])
            ax1.plot(k_vals[1:], WPDR_solns[1:, ii].real, c=species_colors[ii], linestyle='-',  label='Warm', alpha=alpha[ff])
            #ax1.axhline(w_cyc[ii], c='k', linestyle=':')
    
        ax1.set_title('Dispersion Relation')
        ax1.set_xlabel(r'$k (m^{-1})$')
        ax1.set_ylabel(r'$\omega$ (Hz)', rotation=0)
        ax1.set_xlim(k_vals[0], k_vals[-1])
        
        ax1.set_ylim(0, None)
        ax1.minorticks_on()
        
        type_label = ['Cold Plasma Approx.', 'Hot Plasma Approx.']
        type_style = ['--', '-']
        type_legend = create_type_legend(ax1, type_label, type_style)
        ax1.add_artist(type_legend)
        
        band_labels = [r'$H^+$', r'$He^+$', r'$O^+$']
        band_legend = create_band_legend(ax2, band_labels, species_colors)
        ax2.add_artist(band_legend)
        
        for ii in range(CPDR_solns.shape[1]):
            ax2.plot(k_vals[1:], WPDR_solns[1:, ii].imag, c=species_colors[ii], linestyle='-',  label='Growth', alpha=alpha[ff])
    
        ax2.set_title('Temporal Growth Rate')
        ax2.set_xlabel(r'$k (m^{-1})$')
        ax2.set_ylabel(r'$\gamma (s^{-1})$', rotation=0)
        ax2.set_xlim(k_vals[0], k_vals[-1])
            
        ax2.minorticks_on()
        
        if save == True:
            plt.savefig('')
            plt.close('all')
        else:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()    

    return


if __name__ == '__main__':
    plot_series(save=False)
    
    # Load parameters (for species) from hybrid files
    # Plot it similar to existing plotting routines, maybe send axes to get new line each time
