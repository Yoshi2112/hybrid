# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:56:34 2016

@author: c3134027
"""
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import os

import analysis_backend as bk
import analysis_config  as cf

qi  = 1.602e-19               # Elementary charge (C)
c   = 3e8                     # Speed of light (m/s)
me  = 9.11e-31                # Mass of electron (kg)
mp  = 1.67e-27                # Mass of proton (kg)
e   = -qi                     # Electron charge (C)
mu0 = (4e-7) * np.pi          # Magnetic Permeability of Free Space (SI units)
kB  = 1.38065e-23             # Boltzmann's Constant (J/K)
e0  = 8.854e-12               # Epsilon naught - permittivity of free space


def plot_energies(normalize=True, save=False):
    mag_energy, electron_energy, particle_energy, total_energy = bk.get_energies()

    ftime_sec  = cf.dt_field    * np.arange(mag_energy.shape[0])
    ptime_sec  = cf.dt_particle * np.arange(particle_energy.shape[0])

    fig     = plt.figure(figsize=(15, 7))
    ax      = plt.subplot2grid((7, 7), (0, 0), colspan=6, rowspan=7)

    if normalize == True:
        ax.plot(ftime_sec, mag_energy      / mag_energy[0],      label = r'$U_B$', c='green')
        ax.plot(ftime_sec, electron_energy / electron_energy[0], label = r'$U_e$', c='orange')
        ax.plot(ptime_sec, total_energy    / total_energy[0],    label = r'$Total$', c='k')
        
        for jj in range(cf.Nj):
            ax.plot(ptime_sec, particle_energy[:, jj, 0] / particle_energy[0, jj, 0],
                     label=r'$K_{E\parallel}$ %s' % cf.species_lbl[jj], c=cf.temp_color[jj], linestyle=':')
            
            ax.plot(ptime_sec, particle_energy[:, jj, 1] / particle_energy[0, jj, 1],
                     label=r'$K_{E\perp}$ %s' % cf.species_lbl[jj], c=cf.temp_color[jj], linestyle='-')
    else:
        ax.plot(ftime_sec, mag_energy,      label = r'$U_B$', c='green')
        ax.plot(ftime_sec, electron_energy, label = r'$U_e$', c='orange')
        ax.plot(ptime_sec, total_energy,    label = r'$Total$', c='k')
        
        for jj in range(cf.Nj):
            ax.plot(ptime_sec, particle_energy[:, jj, 0],
                     label=r'$K_{E\parallel}$ %s' % cf.species_lbl[jj], c=cf.temp_color[jj], linestyle=':')
            
            ax.plot(ptime_sec, particle_energy[:, jj, 1],
                     label=r'$K_{E\perp}$ %s' % cf.species_lbl[jj], c=cf.temp_color[jj], linestyle='-')
    
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))
    fig.tight_layout()

    percent_ion = np.zeros(cf.Nj)
    for jj in range(cf.Nj):
        tot_ion         = particle_energy[:, jj, 0] + particle_energy[:, jj, 1]
        percent_ion[jj] = round(100.*(tot_ion[-1] - tot_ion[0]) / tot_ion[0], 2)

    percent_elec  = round(100.*(electron_energy[-1] - electron_energy[0]) / electron_energy[0], 2)
    percent_mag   = round(100.*(mag_energy[-1]      - mag_energy[0])      / mag_energy[0], 2)
    percent_total = round(100.*(total_energy[-1]    - total_energy[0])    / total_energy[0], 2)

    fsize = 14; fname='monospace'
    plt.figtext(0.85, 0.92, r'$\Delta E$ OVER RUNTIME',            fontsize=fsize+2, fontname=fname)
    plt.figtext(0.85, 0.92, '________________________',            fontsize=fsize+2, fontname=fname)
    plt.figtext(0.85, 0.88, 'TOTAL   : {:>7}%'.format(percent_total),  fontsize=fsize,  fontname=fname)
    plt.figtext(0.85, 0.84, 'MAGNETIC: {:>7}%'.format(percent_mag),    fontsize=fsize,  fontname=fname)
    plt.figtext(0.85, 0.80, 'ELECTRON: {:>7}%'.format(percent_elec),   fontsize=fsize,  fontname=fname)

    for jj in range(cf.Nj):
        plt.figtext(0.85, 0.76-jj*0.04, 'ION{}    : {:>7}%'.format(jj, percent_ion[jj]), fontsize=fsize,  fontname=fname)

    ax.set_xlabel('Time (seconds)')
    ax.set_xlim(0, ptime_sec[-1])

    if normalize == True:
        ax.set_title('Normalized Energy Distribution in Simulation Space')
        ax.set_ylabel('Normalized Energy', rotation=90)
        fullpath = cf.anal_dir + 'norm_energy_plot'
        fig.subplots_adjust(bottom=0.07, top=0.96, left=0.04)
    else:
        ax.set_title('Energy Distribution in Simulation Space')
        ax.set_ylabel('Energy (Joules)', rotation=90)
        fullpath = cf.anal_dir + 'energy_plot'
        fig.subplots_adjust(bottom=0.07, top=0.96, left=0.055)

    if save == True:
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
    
    plt.close('all')
    print('Energy plot saved')
    return


def plot_ion_energy_components(normalize=True, save=True, tmax=600):
    mag_energy, electron_energy, particle_energy, total_energy = bk.get_energies()
    
    if normalize == True:
        for jj in range(cf.Nj):
            particle_energy[:, jj] /= particle_energy[0, jj]
    
    lpad = 20
    plt.ioff()
    
    for jj in range(cf.Nj):
        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(figsize=(18, 10), nrows=2, ncols=2)
        fig.subplots_adjust(hspace=0)
        
        ax1.plot(cf.time_radperiods_particle, particle_energy[:, jj, 1])
        ax3.plot(cf.time_radperiods_particle, particle_energy[:, jj, 0])
        
        ax2.plot(cf.time_radperiods_particle, particle_energy[:, jj, 1])
        ax4.plot(cf.time_radperiods_particle, particle_energy[:, jj, 0])
        
        ax1.set_ylabel(r'Perpendicular Energy', rotation=90, labelpad=lpad)
        ax3.set_ylabel(r'Parallel Energy', rotation=90, labelpad=lpad)
        
        for ax in [ax1, ax2]:
            ax.set_xticklabels([])
                    
        for ax in [ax1, ax3]:
            ax.set_xlim(0, tmax)
            
        for ax in [ax2, ax4]:
            ax.set_xlim(0, cf.time_radperiods_field[-1])
                
        for ax in [ax3, ax4]:
            ax.set_xlabel(r'Time $(\Omega^{-1})$')
                
        plt.suptitle('{} ions'.format(cf.species_lbl[jj]), fontsize=20, x=0.5, y=.93)
        plt.figtext(0.125, 0.05, 'Total time: {:.{p}g}s'.format(cf.time_seconds_field[-1], p=6), fontweight='bold')
        fig.savefig(cf.anal_dir + 'ion_energy_species_{}.png'.format(jj), facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')
    return


def analyse_particle_motion(it_max=None):
    '''
    Mainly looking at a few particles at a time to get a sense of the motion
    of these particles in a bottle/with waves
    '''
    # To Do:
    #   - Track bounce period of some hot/cold particles (maybe a handful each?)
    #   - Look at their magnetic moments with time

    if it_max is None:
        num_particle_steps = len(os.listdir(cf.particle_dir))
    
    ptime = np.zeros(num_particle_steps)
    np.random.seed(cf.seed)
    
    # CREATE SAMPLE ARRAY :: Either equal number from each, or just from the one
    N_samples = 5
    
    if False:
        # Collect a sample from each species
        sloc = np.zeros((cf.Nj * N_samples), dtype=int)  # Sample location (to not confuse with particle index)
        for ii in range(cf.Nj):
            sloc[ii*N_samples: (ii + 1)*N_samples] = np.random.randint(cf.idx_start[ii], cf.idx_end[ii], N_samples, dtype=int)
    elif True:
        # Collect a sample from just one species
        jj   = 1
        sloc = np.random.randint(cf.idx_start[jj], cf.idx_end[jj], N_samples, dtype=int)
    
    ## COLLECT DATA ON THESE PARTICLES
    sidx      = np.zeros((num_particle_steps, sloc.shape[0]), dtype=int)    # Sample particle index
    spos      = np.zeros((num_particle_steps, sloc.shape[0], 3))            # Sample particle position
    svel      = np.zeros((num_particle_steps, sloc.shape[0], 3))            # Sample particle velocity
    
    # Load up species index and particle position, velocity for samples
    for ii in range(num_particle_steps):
        pos, vel, idx, ptime[ii] = cf.load_particles(ii)
        print('Loading sample particle data for particle file {}'.format(ii))
        for jj in range(sloc.shape[0]):
            sidx[ii, jj]    = idx[sloc[jj]]
            spos[ii, jj, :] = pos[:, sloc[jj]]
            svel[ii, jj, :] = vel[:, sloc[jj]]

    if False:
        # Plot position/velocity (will probably have to put a catch in here for absorbed particles: ylim?)
        fig, axes = plt.subplots(2, sharex=True)
        for ii in range(sloc.shape[0]):
            axes[0].plot(ptime, spos[:, ii, 0], c=cf.temp_color[sidx[0, ii]], marker='o')
            
            axes[1].plot(ptime, svel[:, ii, 0], c=cf.temp_color[sidx[0, ii]], marker='o')
            
            axes[0].set_title('Sample Positions/Velocities of Particles :: Indices {}'.format(sloc))
            axes[1].set_xlabel('Time (s)')
            axes[0].set_ylabel('Position (m)')
            axes[1].set_ylabel('Velocity (m/s)') 
    return


def plot_particle_loss_with_time(it_max=None, save=True):
    #   - What is the initial pitch angle of particles that have been lost?
    #   - Sum all magnetic moments to look for conservation of total mu
    #
    # 1) Plot of N particles lost vs. time (per species in color)
    # 2) Some sort of 2D plot to look at the (initial equatorial?) pitch angle of the particles lost?
    # 3) Do the mu thing, also by species?
    savedir = cf.anal_dir + '/Particle_Loss_Analysis/'
    
    if os.path.exists(savedir) == False:
        os.makedirs(savedir)
    
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
           
    ptime        = np.zeros(it_max)
    N_lost       = np.zeros((it_max, cf.Nj), dtype=int)
    
    last_pos, last_vel, last_idx, last_time = cf.load_particles(len(os.listdir(cf.particle_dir)) - 1)
    all_lost_idx, N_lost_total = locate_lost_ions(last_idx)

    ## Load up species index and particle position, velocity for calculations
    for ii in range(it_max):
        print('Loading data for particle file {}'.format(ii))
        pos, vel, idx, ptime[ii] = cf.load_particles(ii)
        lost_idx, N_lost[ii, :] = locate_lost_ions(idx)
    
    plt.ioff()
    # N_lost per species with time
    if True:
        fig, axes = plt.subplots()
        for ii in range(cf.Nj):
            axes.plot(ptime, N_lost[:, ii], c=cf.temp_color[ii], marker='o', label=cf.species_lbl[ii])
            
        axes.set_title('Number of particles lost from simulation with time')
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('N', rotation=0)
        
    if save == True:
        fpath = savedir + 'particle_loss_vs_time.png'
        fig.savefig(fpath)
        plt.close('all')
        print('Particle loss graph saved as {}'.format(fpath))
    else:
        plt.show()
    return


@nb.njit()
def locate_lost_ions(idx):
    '''
    Checked this. Works great. Returns a 1/0 array indicating if a particular
    particle has been lost (1: Lost). Indices of these particles can be called
    via lost_indices.nonzero().
    N_lost is just a counter per species of how many lost particles there are.
    '''
    lost_indices = np.zeros(cf.N,  dtype=nb.int64)
    N_lost       = np.zeros(cf.Nj, dtype=nb.int64)
    for ii in range(idx.shape[0]):
        if idx[ii] < 0:
            lost_indices[ii] = 1        # Locate in index list
            N_lost[idx[ii]+128] += 1    # Count in lost array
    return lost_indices, N_lost


def plot_initial_configurations(it_max=None, save=True, plot_lost=True):
    ## Count those that have been lost by the end of the simulation
    ## and plot that against initial distro phase spaces
    #
    ## Notes:
    ##  -- Why are lost particles only in the negative side of the simulation space?
    ##  -- Why is there seemingly no connection between lost particles and loss cone?
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
    
    savedir = cf.anal_dir + '/Particle_Loss_Analysis/Initial Particle Configuration/'

    if os.path.exists(savedir) == False:                                   # Create directories
        os.makedirs(savedir)
    
    if plot_lost == True:
        final_pos, final_vel, final_idx, ptime2 = cf.load_particles(it_max-1)
        lost_indices, N_lost     = locate_lost_ions(final_idx)

    init_pos , init_vel , init_idx , ptime1 = cf.load_particles(0)
    v_mag  = np.sqrt(init_vel[0] ** 2 + init_vel[1] ** 2 + init_vel[2] ** 2)
    v_perp = np.sign(init_vel[2]) * np.sqrt(init_vel[1] ** 2 + init_vel[2] ** 2)
    v_para = init_vel[0]
    
    plt.ioff()
    cf.temp_color[0] = 'c'
    
    plt.ioff()
    for jj in range(cf.Nj):
        print('Plotting phase spaces for species {}'.format(jj))
        fig1, ax1 = plt.subplots(figsize=(15, 10))
        fig2, ax2 = plt.subplots(figsize=(15, 10))
        fig3, ax3 = plt.subplots(3, sharex=True, figsize=(15, 10))
    
        lost_vals = lost_indices[cf.idx_start[jj]: cf.idx_end[jj]].nonzero()[0] + cf.idx_start[jj]

        # Loss cone diagram
        ax1.scatter(v_perp[cf.idx_start[jj]: cf.idx_end[jj]], v_para[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        
        if plot_lost == True:
            ax1.scatter(v_perp[lost_vals], v_para[lost_vals], c='k', marker='x', s=20, label='Lost particles')
        
        ax1.set_title('Initial Loss Cone Distribution :: {}'.format(cf.species_lbl[jj]))
        ax1.set_ylabel('$v_\parallel$ (m/s)')
        ax1.set_xlabel('$v_\perp$ (m/s)')
        ax1.legend()
        
        # v_mag vs. x
        ax2.scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], v_mag[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        
        if plot_lost == True:
            ax2.scatter(init_pos[0, lost_vals], v_mag[lost_vals], c='k', marker='x', s=20, label='Lost particles')
        ax2.set_title('Initial Velocity vs. Position :: {}'.format(cf.species_lbl[jj]))
        ax2.set_xlabel('Position (m)')
        ax2.set_ylabel('Velocity |v| (m/s)')
        ax2.legend()
            
        # v components vs. x (3 plots)
        ax3[0].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[0, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        ax3[1].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[1, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        ax3[2].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[2, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
   
        if plot_lost == True:
            ax3[0].scatter(init_pos[0, lost_vals], init_vel[0, lost_vals], c='k', marker='x', s=20, label='Lost particles')
            ax3[1].scatter(init_pos[0, lost_vals], init_vel[1, lost_vals], c='k', marker='x', s=20, label='Lost particles')
            ax3[2].scatter(init_pos[0, lost_vals], init_vel[2, lost_vals], c='k', marker='x', s=20, label='Lost particles')
        
        ax3[0].set_ylabel('$v_x$ (m/s)')
        ax3[1].set_ylabel('$v_y$ (m/s)')
        ax3[2].set_ylabel('$v_z$ (m/s)')
        
        ax3[0].set_title('Initial Velocity Components vs. Position :: {}'.format(cf.species_lbl[jj]))
        ax3[2].set_xlabel('Position (m)')
        
        for ax in ax3:
            ax.legend()
            
        if save == True:
            fig1.savefig(savedir + 'loss_velocity_space_species_{}'.format(jj))
            fig2.savefig(savedir + 'loss_position_velocity_magnitude_species_{}'.format(jj))
            fig3.savefig(savedir + 'loss_position_velocity_components_species_{}'.format(jj))
            print('Plots saved for species {}'.format(jj))
            plt.close('all')
        else:
            plt.show()
    return


def plot_initial_configurations_loss_with_time(it_max=None, save=True, skip=1):
    ## Count those that have been lost by the end of the simulation
    ## and plot that against initial distro phase spaces
    #
    ## Notes:
    ##  -- Why are lost particles only in the negative side of the simulation space?
    ##  -- Why is there seemingly no connection between lost particles and loss cone?
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
    
    savedir = cf.anal_dir + '/Particle_Loss_Analysis/Phase Space Loss with Time/'
    
    dir1 = savedir + '/velocity_space/'
    dir2 = savedir + '/v_mag_vs_x/'
    dir3 = savedir + '/v_components_vs_x/'
    
    for this_dir in[dir1, dir2, dir3]:
        for ii in range(cf.Nj):
            this_path = this_dir + '/species_{}/'.format(ii)
            if os.path.exists(this_path) == False:                                   # Create directories
                os.makedirs(this_path)
    
    init_pos , init_vel , init_idx , ptime1 = cf.load_particles(0)
    
    v_mag  = np.sqrt(init_vel[0] ** 2 + init_vel[1] ** 2 + init_vel[2] ** 2) / cf.va
    v_perp = np.sign(init_vel[2]) * np.sqrt(init_vel[1] ** 2 + init_vel[2] ** 2)
    v_para = init_vel[0]
    
    plt.ioff()
    cf.temp_color[0] = 'c'
    
    plt.ioff()
    for ii in range(0, it_max, skip):
        final_pos, final_vel, final_idx, ptime2 = cf.load_particles(ii)
        lost_indices, N_lost = locate_lost_ions(final_idx)

        for jj in range(cf.Nj):
            lost_vals = lost_indices[cf.idx_start[jj]: cf.idx_end[jj]].nonzero()[0] + cf.idx_start[jj]

            print('Plotting phase spaces for species {}'.format(jj))
            fig1, ax1 = plt.subplots(figsize=(15, 10))
            fig2, ax2 = plt.subplots(figsize=(15, 10))
            fig3, ax3 = plt.subplots(3, sharex=True, figsize=(15, 10))
        
            # Loss cone diagram
            ax1.scatter(v_perp[cf.idx_start[jj]: cf.idx_end[jj]], v_para[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
            ax1.scatter(v_perp[lost_vals], v_para[lost_vals], c='k', marker='x', s=20, label='Lost particles')
            
            ax1.set_title('Initial Velocity Distribution :: {} :: Lost Particles at t={:.2f}s'.format(cf.species_lbl[jj], ptime2))
            ax1.set_ylabel('$v_\parallel$ (m/s)')
            ax1.set_xlabel('$v_\perp$ (m/s)')
            ax1.legend()
            
            # v_mag vs. x
            ax2.scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], v_mag[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
            ax2.scatter(init_pos[0, lost_vals], v_mag[lost_vals], c='k', marker='x', s=20, label='Lost particles')
            
            ax2.set_title('Initial Velocity vs. Position :: {} :: Lost Particles at t={:.2f}s'.format(cf.species_lbl[jj], ptime2))
            ax2.set_xlabel('Position (m)')
            ax2.set_ylabel('Velocity |v| (/vA)')
            ax2.legend()
                
            # v components vs. x (3 plots)
            ax3[0].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[0, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
            ax3[1].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[1, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
            ax3[2].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[2, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
       
            ax3[0].scatter(init_pos[0, lost_vals], init_vel[0, lost_vals], c='k', marker='x', s=20, label='Lost particles')
            ax3[1].scatter(init_pos[0, lost_vals], init_vel[1, lost_vals], c='k', marker='x', s=20, label='Lost particles')
            ax3[2].scatter(init_pos[0, lost_vals], init_vel[2, lost_vals], c='k', marker='x', s=20, label='Lost particles')
            
            ax3[0].set_ylabel('$v_x$ (m/s)')
            ax3[1].set_ylabel('$v_y$ (m/s)')
            ax3[2].set_ylabel('$v_z$ (m/s)')
            
            ax3[0].set_title('Initial Velocity Components vs. Position :: {} :: Lost Particles at t={:.2f}s'.format(cf.species_lbl[jj], ptime2))
            ax3[2].set_xlabel('Position (m)')
            
            for ax in ax3:
                ax.legend()
                
            if save == True:
                savedir1 = dir1 + '/species_{}/'.format(jj)
                savedir2 = dir2 + '/species_{}/'.format(jj)
                savedir3 = dir3 + '/species_{}/'.format(jj)
                
                fig1.savefig(savedir1 + 'loss_velocity_space_species_{}_t{:05}'.format(jj, ii))
                fig2.savefig(savedir2 + 'loss_position_velocity_magnitude_species_{}_t{:05}'.format(jj, ii))
                fig3.savefig(savedir3 + 'loss_position_velocity_components_species_{}_t{:05}'.format(jj, ii))
                print('Plots saved for species {}'.format(jj))
                plt.close('all')
            else:
                plt.show()
    return


def plot_phase_space_with_time(it_max=None, plot_all=True, lost_black=True, skip=1):
    ## Same plotting routines as above, just for all times, and saving output
    ## to a file
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
        
    path_cone = cf.anal_dir + '/Particle_Loss_Analysis/phase_spaces/velocity_phase_space/'
    path_mag  = cf.anal_dir + '/Particle_Loss_Analysis/phase_spaces/velocity_mag_vs_x/'
    path_comp = cf.anal_dir + '/Particle_Loss_Analysis/phase_spaces/velocity_components_vs_x/'
    
    for path in [path_cone, path_mag, path_comp]:
        if os.path.exists(path) == False:                                   # Create directories
            os.makedirs(path)
        
    final_pos, final_vel, final_idx, ptime2 = cf.load_particles(it_max-1)
    lost_indices, N_lost                    = locate_lost_ions(final_idx)
    
    v_max = 16
    
    for ii in range(0, it_max, skip):
        print('Plotting phase space diagrams for particle output {}'.format(ii))
        pos, vel, idx, ptime = cf.load_particles(ii)
    
        vel   /= cf.va 
        v_mag  = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        v_perp = np.sign(vel[2]) * np.sqrt(vel[1] ** 2 + vel[2] ** 2)
        v_para = vel[0]
        
        plt.ioff()
        fig1, ax1 = plt.subplots(figsize=(15, 10))
        fig2, ax2 = plt.subplots(figsize=(15, 10))
        fig3, ax3 = plt.subplots(3, sharex=True, figsize=(15, 10))
        
        for jj in [1]:#range(cf.Nj):
            if lost_black == True:
                lc = 'k'
            else:
                lc = cf.temp_color[jj]
        
            lost_vals = lost_indices[cf.idx_start[jj]: cf.idx_end[jj]].nonzero()[0] + cf.idx_start[jj]
    
            if True:
                # Loss cone diagram
                ax1.scatter(v_perp[cf.idx_start[jj]: cf.idx_end[jj]], v_para[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
                ax1.scatter(v_perp[lost_vals], v_para[lost_vals], c=lc, marker='x', s=20)
                ax1.set_title('Initial Loss Cone Distribution :: t = {:5.4f}'.format(ptime))
                ax1.set_ylabel('$v_\parallel$ (m/s)')
                ax1.set_xlabel('$v_\perp$ (m/s)')
                ax1.set_xlim(-v_max, v_max)
                ax1.set_ylim(-v_max, v_max)
            
            if True:
                # v_mag vs. x
                ax2.scatter(pos[0, cf.idx_start[jj]: cf.idx_end[jj]], v_mag[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])       
                ax2.scatter(pos[0, lost_vals], v_mag[lost_vals], c=lc, marker='x', s=20)
                ax2.set_title('Initial Velocity vs. Position :: t = {:5.4f}'.format(ptime))
                ax2.set_xlabel('Position (m)')
                ax2.set_ylabel('Velocity |v| (m/s)')
                
                ax2.set_xlim(cf.xmin, cf.xmax)
                ax2.set_ylim(0, v_max)
            
            if True:
                # v components vs. x (3 plots)
                ax3[0].scatter(pos[0, cf.idx_start[jj]: cf.idx_end[jj]], vel[0, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
                ax3[1].scatter(pos[0, cf.idx_start[jj]: cf.idx_end[jj]], vel[1, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
                ax3[2].scatter(pos[0, cf.idx_start[jj]: cf.idx_end[jj]], vel[2, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
       
                ax3[0].scatter(pos[0, lost_vals], vel[0, lost_vals], c=lc, marker='x', s=20)
                ax3[1].scatter(pos[0, lost_vals], vel[1, lost_vals], c=lc, marker='x', s=20)
                ax3[2].scatter(pos[0, lost_vals], vel[2, lost_vals], c=lc, marker='x', s=20)
                
                ax3[0].set_ylabel('$v_x$ (m/s)')
                ax3[1].set_ylabel('$v_y$ (m/s)')
                ax3[2].set_ylabel('$v_z$ (m/s)')
                
                for ax in ax3:
                    ax.set_xlim(cf.xmin, cf.xmax)
                    ax.set_ylim(-v_max, v_max)
                
                ax3[0].set_title('Initial Velocity Components vs. Position :: t = {:5.4f}'.format(ptime))
                ax3[2].set_xlabel('Position (m)')
                       
        fig1.savefig(path_cone + 'cone%06d.png' % ii)
        fig2.savefig(path_mag  +  'mag%06d.png' % ii)
        fig3.savefig(path_comp + 'comp%06d.png' % ii)
        
        plt.close('all')
    return


def plot_loss_paths(it_max=None, save_to_file=True):
    savedir = cf.anal_dir + '/particle_loss_paths/'
    
    if os.path.exists(savedir) == False:                                   # Create directories
        os.makedirs(savedir)
            
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))

    # Find lost particles        
    final_pos, final_vel, final_idx, ptime2 = cf.load_particles(it_max-1)
    lost_indices, N_lost                    = locate_lost_ions(final_idx)
    
    ptime    = np.zeros(it_max) 
    lost_pos = np.zeros((it_max, 3, N_lost.sum())) 
    lost_vel = np.zeros((it_max, 3, N_lost.sum())) 
    lost_idx = np.zeros((it_max, N_lost.sum()), dtype=int) 
    
    for ii in range(it_max):
        print('Getting particle loss data from dump file {}'.format(ii))
        lval = lost_indices.nonzero()[0]
        pos, vel, idx, ptime[ii] = cf.load_particles(ii)

        lost_pos[ii, :, :] = pos[:, lval]
        lost_vel[ii, :, :] = vel[:, lval]
        lost_idx[ii, :]    = idx[   lval]
    
    if save_to_file == True:  
        print('Saving lost particle information to file.')
        np.savez(cf.temp_dir + 'lost_particle_info', lval=lval, lost_pos=lost_pos,
                 lost_vel=lost_vel, lost_idx=lost_idx)
    
    lost_vel /= cf.va
    
    v_mag  = np.sqrt(lost_vel[:, 0] ** 2 + lost_vel[:, 1] ** 2 + lost_vel[:, 2] ** 2)
    v_perp = np.sign(lost_vel[:, 2]) * np.sqrt(lost_vel[:, 1] ** 2 + lost_vel[:, 2] ** 2)
    rL     =                           np.sqrt(lost_pos[:, 1] ** 2 + lost_pos[:, 2] ** 2)
    v_para = lost_vel[:, 0]

    # Lost particle : idx 12968
    # lval indx     : 142
    # Initial pos   : [-1020214.38977955,  -100874.673573  ,        0.        ]
    # Initial vel   : [ -170840.94864185, -8695629.67092295,  3474619.54765129]

    plt.ioff()
    
    for ii in range(N_lost.sum()):
        print('Plotting diagnostic outputs for particle {}'.format(lval[ii]))
        particle_path = savedir + 'pidx_%05d//' % lval[ii]
        
        if os.path.exists(particle_path) == False:                                   # Create directories
            os.makedirs(particle_path)
        
        fig1, ax1 = plt.subplots(figsize=(15, 10))
        fig2, ax2 = plt.subplots(figsize=(15, 10))
        fig3, ax3 = plt.subplots(3, sharex=True, figsize=(15, 10))
        fig4, ax4 = plt.subplots(5, sharex=True, figsize=(15, 10))
    
        # Phase space plots as above
        # But for a single (or small group of) particle/s with time
        
        # Fig 1 : Loss Cone
        if True:
            # Loss cone diagram
            ax1.plot(v_perp[:, ii], v_para[:, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax1.set_title('Loss Cone Path')
            ax1.set_ylabel('$v_\parallel$ (m/s)')
            ax1.set_xlabel('$v_\perp$ (m/s)')
            
            filename1 = 'velocity_space_idx_%07d.png' % lval[ii]
            fig1.savefig(particle_path + filename1)
            plt.close('all')  
        
        # Fig 2 : v_mag vs. x
        if True:
            # v_mag vs. x
            ax2.plot(lost_pos[:, 0, ii], v_mag[:, ii], c=cf.temp_color[lost_idx[0, ii]])       
            ax2.set_title('Velocity Magnitude vs. Position')
            ax2.set_xlabel('Position (m)')
            ax2.set_ylabel('Velocity |v| (m/s)')
            
            ax2.set_xlim(cf.xmin, cf.xmax)
            
            filename2 = 'vmag_vs_x_idx_%07d.png' % lval[ii]
            fig2.savefig(particle_path + filename2)
            plt.close('all') 

        
        # Fig 3 : v_components vs. x
        if True:
            ax3[0].plot(lost_pos[:, 0, ii], lost_vel[:, 0, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax3[1].plot(lost_pos[:, 0, ii], lost_vel[:, 1, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax3[2].plot(lost_pos[:, 0, ii], lost_vel[:, 2, ii], c=cf.temp_color[lost_idx[0, ii]])
               
            ax3[0].set_ylabel('$v_x$ (m/s)')
            ax3[1].set_ylabel('$v_y$ (m/s)')
            ax3[2].set_ylabel('$v_z$ (m/s)')
            
            ax3[0].set_title('Velocity Components vs. Position')
            ax3[2].set_xlabel('Position (m)')
            ax3[2].set_xlim(cf.xmin, cf.xmax)
            
            filename3 = 'vcomp_vs_x_idx_%07d.png' % lval[ii]
            fig3.savefig(particle_path + filename3)
            plt.close('all') 
        
        # Fig 4 : x, v vs. t (4-5 plot)
        if True:
            fn_idx = np.argmax(lost_idx[:, ii] < 0)
            
            ax4[0].plot(ptime, lost_pos[:, 0, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax4[1].plot(ptime, lost_vel[:, 0, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax4[2].plot(ptime, lost_vel[:, 1, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax4[3].plot(ptime, lost_vel[:, 2, ii], c=cf.temp_color[lost_idx[0, ii]])
            ax4[4].plot(ptime,       rL[:, ii],    c=cf.temp_color[lost_idx[0, ii]])

            ax4[0].set_ylabel('$x$ (m)')
            ax4[1].set_ylabel('$v_x$ (/va)')
            ax4[2].set_ylabel('$v_y$ (/va)')
            ax4[3].set_ylabel('$v_z$ (/va)')
            ax4[4].set_ylabel('$r_L$ (m)')
            
            ax4[0].set_title('Position and Velocity Components vs. Time')
            ax4[4].set_xlabel('Time (s)')
            ax4[4].set_xlim(0, ptime[-1])
            
            for ax in ax4:
                ax.axvline(ptime[fn_idx], c='k', ls='--', alpha=0.5)
            
            filename4 = 'components_vs_time_idx_%07d.png' % lval[ii]
            fig4.savefig(particle_path + filename4)
            plt.close('all') 

    return

@nb.njit(parallel=True, fastmath=True)
def interrogate_B0(Px, Py, Pz, B0_out):
    print('Interrogating B0 function')
    for ii in nb.prange(Px.shape[0]):
        for jj in range(Py.shape[0]):
            for kk in range(Pz.shape[0]):
                pos = np.array([Px[ii], Py[jj], Pz[kk]])
                bk.eval_B0_particle(pos, B0_out[ii, jj, kk, :])
    return


def plot_B0():
    savedir = cf.anal_dir + '/B0_quiver_plots_2D_slices/'
    yz_dir  = savedir + 'yz_plane//'
    xz_dir  = savedir + 'xz_plane//'
    xy_dir  = savedir + 'xy_plane//'
    
    for dpath in [yz_dir, xz_dir, xy_dir]:
        if os.path.exists(dpath) == False:                                   # Create directories
            os.makedirs(dpath)
            
    q    = 1.602177e-19                  # Elementary charge (C)
    mp   = 1.672622e-27                  # Mass of proton (kg)

    vmax = 20                                    # Maximum expected particle velocity 
    rmax = mp * vmax * cf.va / (q * cf.B_eq)     # Maximum expected Larmor radii
    
    # Sample number
    Nx = 80
    Ny = 40
    Nz = 40
    
    Px = np.linspace(cf.xmin, cf.xmax, Nx, dtype=np.float64)
    Py = np.linspace(  -rmax,    rmax, Ny, dtype=np.float64)
    Pz = np.linspace(  -rmax,    rmax, Nz, dtype=np.float64)
    
    B0_out = np.zeros((Px.shape[0], Py.shape[0], Pz.shape[0], 3), dtype=np.float64)

    interrogate_B0(Px, Py, Pz, B0_out)
    
    # Convert distances to km
    Px *= 1e-3; Py *= 1e-3; Pz *= 1e-3
    
    # Normalize vector lengths (for constant arrow shape)
    Uyz = B0_out[:, :, :, 1] / np.sqrt(B0_out[:, :, :, 1] ** 2 + B0_out[:, :, :, 2] ** 2)
    Vyz = B0_out[:, :, :, 2] / np.sqrt(B0_out[:, :, :, 1] ** 2 + B0_out[:, :, :, 2] ** 2)
    
    Uxz = B0_out[:, :, :, 0] / np.sqrt(B0_out[:, :, :, 0] ** 2 + B0_out[:, :, :, 2] ** 2)
    Vxz = B0_out[:, :, :, 2] / np.sqrt(B0_out[:, :, :, 0] ** 2 + B0_out[:, :, :, 2] ** 2)
    
    Uxy = B0_out[:, :, :, 0] / np.sqrt(B0_out[:, :, :, 0] ** 2 + B0_out[:, :, :, 1] ** 2)
    Vxy = B0_out[:, :, :, 1] / np.sqrt(B0_out[:, :, :, 0] ** 2 + B0_out[:, :, :, 1] ** 2)

    B0r   = np.sqrt(B0_out[:, :, :, 1] ** 2 + B0_out[:, :, :, 2] ** 2)*1e9
    B0xy  = np.sqrt(B0_out[:, :, :, 0] ** 2 + B0_out[:, :, :, 1] ** 2)*1e9
    B0xz  = np.sqrt(B0_out[:, :, :, 0] ** 2 + B0_out[:, :, :, 2] ** 2)*1e9
    
    plt.ioff()
    # yz slice at some x
    for ii in range(Nx):
        # Magnitude (for quiver colour)
        fig, ax = plt.subplots(figsize=(16,10))
        im1     = ax.quiver(Py, Pz, Uyz[ii, :, :].T, Vyz[ii, :, :].T, B0r[ii, :, :], clim=(B0r.min(), B0r.max()))
        
        ax.set_xlabel('y (km)')
        ax.set_ylabel('z (km)')
        ax.set_title('YZ slice at X = {:5.2f} km :: B0r vectors :: Vmax = {}vA'.format(Px[ii], vmax))
        fig.colorbar(im1).set_label('B0r (nT)')
        
        filename = 'yz_plane_%05d.png' % ii
        savepath = yz_dir + filename
        plt.savefig(savepath)
        print('yz plot {} saved'.format(ii))
        plt.close('all')
    
    # xy slice at some z
    for jj in range(Ny):
        fig, ax = plt.subplots(figsize=(16,10))
        im2     = ax.quiver(Px, Py, Uxy[:, :, jj].T, Vxy[:, :, jj].T, B0xy[:, :, jj], clim=(B0xy.min(), B0xy.max()))
        
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_title('XY slice at Z = {:5.2f} km :: B0xy vectors :: Vmax = {}vA'.format(Pz[jj], vmax))
        fig.colorbar(im2).set_label('B0xy (nT)')   
        
        filename = 'xy_plane_%05d.png' % jj
        savepath = xy_dir + filename
        plt.savefig(savepath)
        print('xy plot {} saved'.format(jj))
        plt.close('all')
    
    # xz slice at some y
    for kk in range(Nz):        
        fig, ax = plt.subplots(figsize=(16,10))
        im3 = ax.quiver(Px, Pz, Uxz[:, kk, :].T, Vxz[:, kk, :].T, B0xz[:, kk, :], clim=(B0xz.min(), B0xz.max()))
        
        ax.set_xlabel('x (km)')
        ax.set_ylabel('z (km)')
        ax.set_title('XZ slice at Y = {:5.2f} km :: B0xz vectors :: Vmax = {}vA'.format(Py[kk], vmax))
        fig.colorbar(im3).set_label('B0xz (nT)')  
        
        filename = 'xz_plane_%05d.png' % kk
        savepath = savedir + 'xz_plane//' + filename
        plt.savefig(savepath)
        print('xz plot {} saved'.format(kk))
        plt.close('all')    
    return


def plot_adiabatic_parameter():
    '''
    Change later to plot for each species charge/mass ratio, but for now its just protons
    
    What are the units for this? Does it require some sort of normalization? No, because its
    just larmor radius (mv/qB) divided by spatial length
    '''
    max_v  = 20 * cf.va
    N_plot = 1000
    B_av   = 0.5 * (cf.B_xmax + cf.B_eq)
    z0     = cf.xmax
    
    v_perp = np.linspace(0, max_v, N_plot)
    
    epsilon = mp * v_perp / (qi * B_av * z0)
    
    plt.title(r'Adiabatic Parameter $\epsilon$ vs. Expected v_perp range :: {}[{}]'.format(series, run_num))
    plt.ylabel(r'$\epsilon$', rotation=0)
    plt.xlabel(r'$v_\perp (/v_A)$')
    plt.xlim(0, max_v/cf.va)
    plt.plot(v_perp/cf.va, epsilon)
    plt.show()
    return


#%% MAIN
if __name__ == '__main__':
    drive       = 'F:'
    series      = 'pusher_only_test'
    series_dir  = '{}/runs//{}//'.format(drive, series)
    num_runs    = len([name for name in os.listdir(series_dir) if 'run_' in name])
    
    for run_num in [0]:#range(num_runs):
        print('Run {}'.format(run_num))
        cf.load_run(drive, series, run_num, extract_arrays=True)
        
        plot_initial_configurations_loss_with_time()
        plot_adiabatic_parameter()
        # Particle Loss Analysis :: For Every Time (really time consuming)
        #analyse_particle_motion()
        
        #plot_loss_paths()
        #plot_B0()
        #analyse_particle_motion_manual()

        try:
            plot_initial_configurations()
            plot_particle_loss_with_time()
            plot_phase_space_with_time()
        except:
            pass
