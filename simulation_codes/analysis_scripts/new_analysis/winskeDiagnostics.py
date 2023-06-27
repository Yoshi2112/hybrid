# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:54:06 2023

@author: Yoshi

Diagnostic analysis postprocessing functions designed to output plots in the
style of Winske et al., 1993 from Matsumoto & Omura 1993. These are mainly
just basic diagnostics and sanity checks, with no quantitative checks of
accuracy (but maybe I can add some later)

Some of these functions may only work effectively for two-species runs
(cold, hot protons).
-- 5 panel particle/field plots for each time
-- Wavenumber spectrum for select times (4 panels, 20, 40, 60, 100)
    -- Generate axes in function and use master function to apply them to a figure
-- 4 panel B energy density, beam velocity, para/perp temperatures
-- Space-time profiles of By
-- Time histories of fourier modes for positive helicity part |B_yk^+|**2

CHECK: No functions acting on attributes
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mtick
import os, sys, pdb

import doSpectra as ds
from _constants import UNIT_CHARGE, PROTON_MASS, ELECTRON_MASS, ELEC_PERMITTIVITY, MAGN_PERMEABILITY, BOLTZMANN_CONSTANT, LIGHT_SPEED


def summaryPlot5Panel(Sim, save=True, skip=1):
    '''
    Diagnostic summary of main particle and field values in the style of Winske et al. (1993)
    5 plots:
        -- Hot proton beam, vx vs. x scatter
        -- Hot proton beam, vy vs. x scatter
        -- Hot proton beam number density vs x
        -- Magnetic field, By vs. x
        -- Magnetic field, Phase angle vs x
    '''  
    if Sim.num_particle_steps == 0:
        print('ABORT: No particle data present to create summary plots.')
        return
    print('Creating Winske summary outputs...')
    np.set_printoptions(suppress=True)

    save_dir = Sim.anal_dir + '/winske_plots/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    plt.ioff()
    pbx, pby, pbz, pex, pey, pez, pvex, pvey,\
    pvez, pte, pjx, pjy, pjz, pqdens = Sim.interpolateFields2ParticleTimes()

    radperiods = Sim.particle_sim_time * Sim.gyfreq_eq
    
    # Normalize units
    qdens_norm = pqdens / (Sim.density*Sim.charge).sum()     
    BY_norm    = pby / Sim.B_eq
    BZ_norm    = pbz / Sim.B_eq
    PHI        = np.arctan2(BZ_norm, BY_norm)
    
    for ii in range(Sim.num_particle_steps):
        if ii%skip == 0:
            filename = 'winske_summary%05d.png' % ii
            fullpath = save_dir + filename
            
            if os.path.exists(fullpath) and False:
                sys.stdout.write('\rSummary plot already present for timestep [{}]{}'.format(Sim.run_num, ii))
                sys.stdout.flush()
                continue
            sys.stdout.write('\rCreating summary plot for particle timestep [{}]{}'.format(Sim.run_num, ii))
            sys.stdout.flush()
    
            fig, axes = plt.subplots(5, figsize=(8.27,11.69), sharex=True)                  # Initialize Figure Space
            fig.suptitle('{}[{}] :: IT={:04d} :: T={:5.2f}'.format(Sim.series_name, Sim.run_num, ii, radperiods[ii]), family='monospace')
            
            xp, vp, idx, psim_time, idx_start, idx_end = Sim.load_particles(ii)
            
            pos       = xp / Sim.dx
            vel       = vp / LIGHT_SPEED
            cell_B    = Sim.B_nodes/Sim.dx
            cell_E    = Sim.E_nodes/Sim.dx
            st, en    = idx_start[1], idx_end[1]
            xmin      = Sim.xmin / Sim.dx
            xmax      = Sim.xmax / Sim.dx
    
            axes[0].scatter(pos[st: en], 1e3*vel[0, st: en], s=1, c='k')
            axes[1].scatter(pos[st: en], 1e3*vel[1, st: en], s=1, c='k')
            axes[2].plot(cell_E, qdens_norm[ii], color='k', lw=1.0)
            axes[3].plot(cell_B, 10.*BY_norm[ii], color='k')
            axes[4].plot(cell_B, PHI[ii],     color='k')
            
            axes[0].set_ylim(-2, 2)
            axes[1].set_ylim(-2, 2)
            axes[2].set_ylim(0.5, 1.5)
            axes[3].set_ylim(-2.5, 2.5)
            axes[4].set_ylim(-np.pi, np.pi)

            axes[0].set_title('BEAM')
            axes[1].set_title('BEAM')
            
            axes[0].set_ylabel('VX\n($\\times 10^-3$)', labelpad=20, rotation=0)
            axes[1].set_ylabel('VY\n($\\times 10^-3$)', labelpad=20, rotation=0)
            axes[2].set_ylabel('DN', labelpad=20, rotation=0)
            axes[3].set_ylabel('BY\n($\\times 10^-1$)', labelpad=20, rotation=0)
            axes[4].set_ylabel('PHI', labelpad=20, rotation=0)
            axes[4].set_xlabel('X')
            
            for ax in axes:
                ax.set_xlim(xmin, xmax)
            
            plt.tight_layout()
            fig.align_ylabels()
            if save == True:
                plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=100)
            plt.close('all')

    print('\n')
    return


def energyPlot4Panel(Sim, save=True):
    '''
    Diagnostic summary of particle-wave energy transfer in the style of Winske et al. (1993)
    4 plots:
        -- Normalized magnetic field energy density vs. time
        -- Normalized beam velocity vs. time
        -- Perpendicular beam temperature vs. time
        -- Parallel beam temperature vs. time
    '''  
    np.set_printoptions(suppress=True)

    by    = getattr(Sim, 'by')
    bz    = getattr(Sim, 'bz')
    b_squared = (by ** 2 + bz ** 2).mean(axis=1) / Sim.B_eq ** 2

    f_radperiods = Sim.field_sim_time    * Sim.gyfreq_eq
    p_radperiods = Sim.particle_sim_time * Sim.gyfreq_eq
    
    V_beam  = np.zeros(Sim.num_particle_steps)  # Average beam velocity
    TX_beam = np.zeros(Sim.num_particle_steps)  # Average beam perp. temp.
    TP_beam = np.zeros(Sim.num_particle_steps)  # Average beam para. temp.
    for ii in range(Sim.num_particle_steps):
        xp, vp, idx, psim_time, idx_start, idx_end = Sim.load_particles(ii)
        
        st, en = idx_start[1], idx_end[1]
        V_beam[ii] = vp[0, st:en].mean()
    V_beam /= V_beam[0]
    
    fig, axes = plt.subplots(4, figsize=(8.27,11.69), sharex=True)                  # Initialize Figure Space

    axes[0].plot(f_radperiods, b_squared, color='k')
    axes[1].plot(p_radperiods, V_beam,    color='k')
    axes[2].plot(p_radperiods, TX_beam,   color='k')
    axes[3].plot(p_radperiods, TP_beam,   color='k')
    
    axes[0].set_ylabel('B**2', labelpad=20, rotation=0)
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 0.48)
    
    axes[1].set_ylabel('VX', labelpad=20, rotation=0)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0.3, 1.00)
    
    axes[2].set_ylabel('TX', labelpad=20, rotation=0)
    axes[2].set_xlim(0, 100)
    axes[2].set_ylim(1.0, 30.)
    
    axes[3].set_ylabel('TP', labelpad=20, rotation=0)
    axes[3].set_xlim(0, 100)
    axes[3].set_ylim(1.0, 50.)
    axes[3].set_xlabel('T')
    
    plt.tight_layout()
    fig.align_ylabels()
    
    if save == True:
        fullpath = Sim.anal_dir + 'winske_magnetic_timeseries' + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=100)
        print('winskeEnergy Plot saved')
        plt.close('all')
    else:
        plt.show()
    return


def plot_kt_winske(Sim, component='by'):
    qi     = 1.602e-19       # Elementary charge (C)
    c      = 3e8             # Speed of light (m/s)
    mp     = 1.67e-27        # Mass of proton (kg)
    e0     = 8.854e-12       # Epsilon naught - permittivity of free space
    
    ftime = getattr(Sim, 'field_sim_time')
    arr = getattr(Sim, component)
    
    radperiods = ftime * Sim.gyfreq
    gperiods   = ftime / Sim.gyperiod
    
    ts_folder = Sim.anal_dir + '//winske_fourier_modes//'
    if os.path.exists(ts_folder) == False:
        os.makedirs(ts_folder)
    
    # Get first/last indices for FFT range and k-space array
    if component[0].upper() == 'B':
        st = Sim.x0B; en = Sim.x1B
        k  = np.fft.fftfreq(Sim.NX, Sim.dx)
    else:
        st = Sim.x0E; en = Sim.x1E
        k  = np.fft.fftfreq(Sim.NX, Sim.dx)
    
    # Normalize to c/wpi
    cwpi = c/np.sqrt(Sim.ne * qi ** 2 / (mp * e0))
    
    k   *= cwpi
    k    = k[k>=0]
    kmax = k.shape[0]
    
    fft_matrix  = np.zeros((arr.shape[0], en-st), dtype='complex128')
    for ii in range(arr.shape[0]): # Take spatial FFT at each time, ii
        fft_matrix[ii, :] = np.fft.fft(arr[ii, st:en] - arr[ii, st:en].mean())

    kt = (fft_matrix[:, :k.shape[0]] * np.conj(fft_matrix[:, :k.shape[0]])).real
    
    plt.ioff()

    for ii in range(ftime.shape[0]):
        fig, ax = plt.subplots()
        ax.semilogy(k[1:kmax], kt[ii, 1:kmax], ds='steps-mid')
        ax.set_title('IT={:04d} :: T={:5.2f} :: GP={:5.2f}'.format(ii, radperiods[ii], gperiods[ii]), family='monospace')
        ax.set_xlabel('K')
        ax.set_ylabel('BYK**2')
        ax.set_xlim(k[1], k[kmax-1])
        fig.savefig(ts_folder + 'winske_fourier_{}_{}.png'.format(component, ii), edgecolor='none')
        plt.close('all') 
        
        sys.stdout.write('\rCreating fourier mode plot for timestep {}'.format(ii))
        sys.stdout.flush()

    print('\n')
    return