# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:44:46 2017

@author: iarey
"""
import numpy as np
import pickle

import os
import sys
import matplotlib.pyplot as plt
from shutil import rmtree
import simulation_parameters_1D as const
from   simulation_parameters_1D import drive, save_path, NX, ne, density, save_particles, save_fields
from   simulation_parameters_1D import idx_bounds, Nj, species_lbl, temp_type, dist_type, mass, charge,\
                                       drift_v, sim_repr, Tpar, Tper, temp_color, HM_amplitude, HM_frequency


def manage_directories():
    print('Checking directories...')
    if (save_particles == 1 or save_fields == 1) == True:
        if os.path.exists('%s/%s' % (drive, save_path)) == False:
            os.makedirs('%s/%s' % (drive, save_path))                        # Create master test series directory
            print('Master directory created')

        path = ('%s/%s/run_%d' % (drive, save_path, const.run_num))          # Set root run path (for images)

        if os.path.exists(path) == False:
            os.makedirs(path)
            print('Run directory created')
        else:
            print('Run directory already exists')
            overwrite_flag = input('Overwrite? (Y/N) \n')
            if overwrite_flag.lower() == 'y':
                rmtree(path)
                os.makedirs(path)
            elif overwrite_flag.lower() == 'n':
                sys.exit('Program Terminated: Change run_num in simulation_parameters_1D')
            else:
                sys.exit('Unfamiliar input: Run terminated for safety')
    return


def store_run_parameters(dt, part_save_iter, field_save_iter):
    d_path = ('%s/%s/run_%d/data/' % (drive, save_path, const.run_num))    # Set path for data
    f_path = d_path + '/fields/'
    p_path = d_path + '/particles/'
    
    manage_directories()

    for folder in [d_path, f_path, p_path]:
        if os.path.exists(folder) == False:                               # Create data directories
            os.makedirs(folder)

    # Single parameters
    params = dict([('seed', const.seed),
                   ('Nj', Nj),
                   ('dt', dt),
                   ('NX', NX),
                   ('dxm', const.dxm),
                   ('dx', const.dx),
                   ('cellpart', const.cellpart),
                   ('B0', const.B0),
                   ('HM_amplitude', HM_amplitude),
                   ('HM_frequency', HM_frequency),
                   ('ne', ne),
                   ('Te0', const.Te0),
                   ('ie', const.ie),
                   ('theta', const.theta),
                   ('part_save_iter', part_save_iter),
                   ('field_save_iter', field_save_iter),
                   ('max_rev', const.max_rev),
                   ('orbit_res', const.orbit_res),
                   ('freq_res', const.freq_res),
                   ('run_desc', const.run_description),
                   ('method_type', 'PREDCORR_HM'),
                   ('particle_shape', 'TSC')
                   ])

    with open(d_path + 'simulation_parameters.pckl', 'wb') as f:
        pickle.dump(params, f)
        f.close()
        print('Simulation parameters saved')
        
    # Particle values: Array parameters
    p_file = d_path + 'particle_parameters'
    np.savez(p_file, idx_bounds  = idx_bounds,
                     species_lbl = species_lbl,
                     temp_color  = temp_color,
                     temp_type   = temp_type,
                     dist_type   = dist_type,
                     mass        = mass,
                     charge      = charge,
                     drift_v     = drift_v,
                     density     = density,
                     sim_repr    = sim_repr,
                     Tpar        = Tpar,
                     Tper        = Tper)
    print('Particle data saved')
    return


def save_field_data(dt, field_save_iter, qq, Ji, E, B, Ve, Te, dns):
    sim_time = np.array([qq*dt])    # Timestamp: Useful for debugging
    d_path   = '%s/%s/run_%d/data/fields/' % (drive, save_path, const.run_num)
    r        = qq / field_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, E = E[1:NX+1, 0:3], B = B[1:NX+2, 0:3],   J = Ji[1:NX+1, 0:3],
                       dns = dns[1:NX+1],   Ve = Ve[1:NX+1, 0:3], Te = Te[1:NX+1], sim_time = sim_time)
    print('Field data saved')
    
def save_particle_data(dt, part_save_iter, qq, pos, vel):
    sim_time = np.array([qq*dt])    # Timestamp: Useful for debugging
    d_path   = '%s/%s/run_%d/data/particles/' % (drive, save_path, const.run_num)
    r        = qq / part_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, pos = pos, vel = vel, sim_time = sim_time)
    print('Particle data saved')
    
    
def save_diagnostic_plot(pos, vel, E, B, q_dens, qq, dt):
    plt.ioff()

    fig_size = 4, 7                                                             # Set figure grid dimensions
    fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
    fig.patch.set_facecolor('w')                                                # Set figure face color

    x_pos       = pos.copy() / 1000        # Particle x-positions (km) (For looking at particle characteristics)
    x_cell_num  = np.arange(q_dens.shape[0])            # Numerical cell numbering: x-axis

#----- Velocity (vy) Plots: Hot and Cold Species
    ax_vx   = plt.subplot2grid(fig_size, (0, 0), rowspan=2, colspan=3)
    ax_vy   = plt.subplot2grid(fig_size, (2, 0), rowspan=2, colspan=3)

    norm_vel   = vel.copy() / const.va
    
    for jj in range(const.Nj):
        ax_vx.scatter(x_pos[idx_bounds[jj, 0]: idx_bounds[jj, 1]], norm_vel[0, idx_bounds[jj, 0]: idx_bounds[jj, 1]], s=1, c=const.temp_color[jj], lw=0)
        ax_vy.scatter(x_pos[idx_bounds[jj, 0]: idx_bounds[jj, 1]], norm_vel[1, idx_bounds[jj, 0]: idx_bounds[jj, 1]], s=1, c=const.temp_color[jj], lw=0)

    ax_vx.set_title(r'Beam velocities ($va^{-1}$) vs. Position (x)')
    ax_vy.set_xlabel(r'Position (km)', labelpad=10)

    ax_vx.set_ylabel(r'$v_{b, x}$', rotation=90)
    ax_vy.set_ylabel(r'$v_{b, y}$', rotation=90)

    plt.setp(ax_vx.get_xticklabels(), visible=False)
    #ax_vx.set_yticks(ax_vx.get_yticks()[1:])

    for ax in [ax_vy, ax_vx]:
        ax.set_xlim(0, const.xmax/1000)
        ax.set_ylim(-20, 20)

#----- Density Plot
    ax_den = plt.subplot2grid((fig_size), (0, 3), colspan=3)                            # Initialize axes

    for ii in range(Nj):
        ax_den.plot(x_cell_num, q_dens / (const.ne * const.q))

    ax_den.set_title('Normalized Ion Densities and Magnetic Fields (y, mag) vs. Cell')  # Axes title (For all, since density plot is on top
    ax_den.set_ylabel('Normalized Density', fontsize=14, rotation=90, labelpad=5)       # Axis (y) label for this specific axes
    ax_den.set_ylim(0, 1.5)
    
#----- Electric Field (Ez) Plot
    ax_Ez = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_den)

    Ez = E[:, 0] *1e6

    ax_Ez.plot(x_cell_num, Ez, color='magenta')

    ax_Ez.set_xlim(0, q_dens.shape[0])
    
# =============================================================================
#     Ez_lim = 0.5
#     if abs(Ez).max() > 500e-5:
#         Ez_lim = 1e-2
#     elif abs(Ez).max() > 100e-5:
#         Ez_lim = 500e-5 
#         
#     if abs(Ez).max() > 0.5:
#         Ez_lim = 20e-5
#     elif abs(Ez).max() > 20e-5:
#         Ez_lim = 100e-5
# =============================================================================
    
    #ax_Ez.set_ylim(-Ez_lim, Ez_lim)
    ax_Ez.set_ylabel(r'$E_x$ ($\mu V m^{-1}$)', labelpad=25, rotation=0, fontsize=14)

#----- Magnetic Field (By) and Magnitude (|B|) Plots
    ax_By = plt.subplot2grid((fig_size), (2, 3), colspan=3, sharex=ax_den)              # Initialize Axes
    ax_B  = plt.subplot2grid((fig_size), (3, 3), colspan=3, sharex=ax_den)

    mag_B = (np.sqrt(B[:, 0] ** 2 + B[:, 1] ** 2 + B[:, 2] ** 2)) / const.B0
    B_y   = B[:, 1] / const.B0                                                          # Normalize grid values

    ax_B.plot(x_cell_num, mag_B, color='g')                                             # Create axes plots
    ax_By.plot(x_cell_num, B_y, color='g')

    ax_B.set_xlim(0,  q_dens.shape[0])                                                               # Set x limit
    ax_By.set_xlim(0, q_dens.shape[0])

    ax_B.set_ylim(0, 1.5)                                                               # Set y limit
    ax_By.set_ylim(-1, 1)

    ax_B.set_ylabel( r'$|B|$', rotation=0, labelpad=20, fontsize=14)                    # Set labels
    ax_By.set_ylabel(r'$\frac{B_y}{B_0}$', rotation=0, labelpad=10, fontsize=14)
    ax_B.set_xlabel('Cell Number')                                                      # Set x-axis label for group (since |B| is on bottom)

    for ax in [ax_den, ax_Ez, ax_By]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_yticks(ax.get_yticks()[1:])

    for ax in [ax_den, ax_Ez, ax_By, ax_B]:
        ax.grid()

#----- Figure Text
    font = 18; spacing = 0.04
        
    fig.text(0.87, 0.94 - 0*spacing, 'NX = {}'.format(NX), fontsize = font)
    fig.text(0.87, 0.94 - 1*spacing, 'N  = {}'.format(const.N), fontsize = font)
    fig.text(0.87, 0.94 - 2*spacing, '$B_0$ = %.2fnT' % (const.B0*1e9), fontsize = font)
    fig.text(0.87, 0.94 - 3*spacing, '$n_0$ = %.2f$cm^{-3}$' % (ne*1e-6), fontsize = font)
    
    if const.smooth_sources == 0:
        fig.text(0.87, 0.25, 'Smoothing OFF', fontsize = font, color='r')
    elif const.smooth_sources == 1:
        fig.text(0.87, 0.25, 'Smoothing ON', fontsize = font, color='g')
        
    time_font = 18; time_spacing = 0.04; time_top = 0.19
    
    fig.text(0.87, time_top - 0*time_spacing, 'it  = %d' % qq, fontsize = time_font)
    fig.text(0.87, time_top - 2*time_spacing, 't    = %.3fs' % (qq*dt), fontsize = time_font)
    fig.text(0.87, time_top - 3*time_spacing, 'dt  = %.4fs' % (dt), fontsize = time_font)

#----- Plot Adjustments
    plt.tight_layout(pad=1.0, w_pad=1.8)
    fig.subplots_adjust(hspace=0)

    filename = 'plot%05d.png' % qq
    path     = drive + save_path + '/run_{}/diag_plots/'.format(const.run_num)
    
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
        
    fullpath = path + filename
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close('all')
    return