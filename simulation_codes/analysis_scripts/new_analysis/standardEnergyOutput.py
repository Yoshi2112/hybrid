# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:51:39 2023

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb


def plotEnergies(Sim, normalize=True, save=True):
    '''
    Parameters
    ----------
    normalize : Bool, optional
        Normalize each energy to its initial value. Default True.
    save : Bool, optional
        Save as an image file, else plot in figure. Default False.
        
    Magnetic, electron energies are on field timebase
    Particle and total energies on particle timebase
    
    TODO: Option to specify maximum time (in order to check conservation/exchange
          before things like finite grid effects take over)
    
    WRITE THIS NEATER!!! WHAT DO YOU WANT FROM THIS??
    Maybe text should be 'share of energy at start and end', so  B will be 0% at
        start and then some percentage at end.
    '''
    print('Plotting energies...')
    mag_energy, electron_energy, particle_energy, total_energy = Sim.get_energies()

    fig     = plt.figure(figsize=(15, 7))
    ax      = plt.subplot2grid((7, 7), (0, 0), colspan=6, rowspan=7)
    
    if normalize == True:
        ax.plot(Sim.field_sim_time, mag_energy      / total_energy[0],      label = r'$U_B$', c='green')
        ax.plot(Sim.field_sim_time, electron_energy / total_energy[0], label = r'$U_e$', c='orange')
        ax.plot(Sim.particle_sim_time, total_energy / total_energy[0],    label = r'$Total$', c='k')
        
        for jj in range(Sim.Nj):
            ax.plot(Sim.particle_sim_time, particle_energy[:, jj, 0] / total_energy[0],
                     label=r'$K_{E\parallel}$ %s' % Sim.species_lbl[jj], c=Sim.temp_color[jj], linestyle=':')
            
            ax.plot(Sim.particle_sim_time, particle_energy[:, jj, 1] / total_energy[0],
                     label=r'$K_{E\perp}$ %s' % Sim.species_lbl[jj], c=Sim.temp_color[jj], linestyle='-')
    else:
        ax.plot(Sim.field_sim_time, mag_energy,      label = r'$U_B$', c='green')
        ax.plot(Sim.field_sim_time, electron_energy, label = r'$U_e$', c='orange')
        ax.plot(Sim.particle_sim_time, total_energy,    label = r'$Total$', c='k')
        
        for jj in range(Sim.Nj):
            ax.plot(Sim.particle_sim_time, particle_energy[:, jj, 0],
                     label=r'$K_{E\parallel}$ %s' % Sim.species_lbl[jj], c=Sim.temp_color[jj], linestyle=':')
            
            ax.plot(Sim.particle_sim_time, particle_energy[:, jj, 1],
                     label=r'$K_{E\perp}$ %s' % Sim.species_lbl[jj], c=Sim.temp_color[jj], linestyle='-')
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))
    fig.tight_layout()

    # Calculate energy partition at start and end of simulation
    # Calculated as a percentage of the total energy at t = 0, so the percentages might not add up to 100% at the end
    total_ion_energy = particle_energy.sum(axis=2)
    percent_ion = np.zeros((Sim.Nj, 2))
    for jj in range(Sim.Nj):
        percent_ion[jj, 0] = round(100.*(total_ion_energy[0, jj]) / total_energy[0], 2)
        percent_ion[jj, 1] = round(100.*(total_ion_energy[-1, jj]) / total_energy[0], 2)
        
    percent_elec  = np.zeros(2)
    percent_mag   = np.zeros(2)
    percent_total = np.zeros(2)

    percent_elec  = round(100.*(electron_energy[-1] - electron_energy[0]) / electron_energy[0], 2)
    percent_mag   = round(100.*(mag_energy[-1]      - mag_energy[0])      / mag_energy[0], 2)
    percent_total = round(100.*(total_energy[-1]    - total_energy[0])    / total_energy[0], 2)

    fsize = 14; fname='monospace'
    plt.figtext(0.85, 0.92, r'$\Delta E$ OVER RUNTIME',            fontsize=fsize+2, fontname=fname)
    plt.figtext(0.85, 0.92, '________________________',            fontsize=fsize+2, fontname=fname)
    plt.figtext(0.85, 0.88, 'TOTAL   : {:>7}%'.format(percent_total),  fontsize=fsize,  fontname=fname)
    plt.figtext(0.85, 0.84, 'MAGNETIC: {:>7}%'.format(percent_mag),    fontsize=fsize,  fontname=fname)
    plt.figtext(0.85, 0.80, 'ELECTRON: {:>7}%'.format(percent_elec),   fontsize=fsize,  fontname=fname)

    for jj in range(Sim.Nj):
        plt.figtext(0.85, 0.76-jj*0.04, 'ION{}    : {:>7}%'.format(jj, percent_ion[jj, 1]), fontsize=fsize,  fontname=fname)

    ax.set_xlabel('Time (seconds)')
    ax.set_xlim(0, Sim.particle_sim_time[-1])

    if normalize == True:
        ax.set_title('Normalized Energy Distribution in Simulation Space')
        ax.set_ylabel('Normalized Energy', rotation=90)
        fullpath = Sim.anal_dir + 'norm_energy_plot'
        fig.subplots_adjust(bottom=0.07, top=0.96, left=0.04)
    else:
        ax.set_title('Energy Distribution in Simulation Space')
        ax.set_ylabel('Energy (Joules)', rotation=90)
        fullpath = Sim.anal_dir + 'energy_plot'
        fig.subplots_adjust(bottom=0.07, top=0.96, left=0.055)

    if save == True:
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
    
    plt.close('all')
    print('Energy plot saved')
    return


def plot_ion_energy_components(Sim, normalize=True, save=True, tmax=600):
    mag_energy, electron_energy, particle_energy, total_energy = Sim.get_energies()
    time_radperiods_particle = Sim.particle_sim_time * Sim.gyfreq_eq
    time_radperiods_field    = Sim.field_sim_time    * Sim.gyfreq_eq
    
    if normalize == True:
        for jj in range(Sim.Nj):
            particle_energy[:, jj] /= particle_energy[0, jj]
    
    lpad = 20
    plt.ioff()
    
    for jj in range(Sim.Nj):
        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(figsize=(18, 10), nrows=2, ncols=2)
        fig.subplots_adjust(hspace=0)
        
        ax1.plot(time_radperiods_particle, particle_energy[:, jj, 1])
        ax3.plot(time_radperiods_particle, particle_energy[:, jj, 0])
        
        ax2.plot(time_radperiods_particle, particle_energy[:, jj, 1])
        ax4.plot(time_radperiods_particle, particle_energy[:, jj, 0])
        
        ax1.set_ylabel(r'Perpendicular Energy', rotation=90, labelpad=lpad)
        ax3.set_ylabel(r'Parallel Energy', rotation=90, labelpad=lpad)
        
        for ax in [ax1, ax2]:
            ax.set_xticklabels([])
                    
        for ax in [ax1, ax3]:
            ax.set_xlim(0, tmax)
            
        for ax in [ax2, ax4]:
            ax.set_xlim(0, time_radperiods_field[-1])
                
        for ax in [ax3, ax4]:
            ax.set_xlabel(r'Time $(\Omega^{-1})$')
                
        plt.suptitle('{} ions'.format(Sim.species_lbl[jj]), fontsize=20, x=0.5, y=.93)
        plt.figtext(0.125, 0.05, 'Total time: {:.{p}g}s'.format(Sim.field_sim_time[-1], p=6), fontweight='bold')
        fig.savefig(Sim.anal_dir + 'ion_energy_species_{}.png'.format(jj), facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')
    return