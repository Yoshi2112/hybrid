# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:44:46 2017

@author: iarey
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import sys
from shutil import rmtree
import simulation_parameters_1D as const
from   simulation_parameters_1D import generate_data, generate_plots, drive, save_path, N, NX, dx, xmax, ne, B0, density, va
from   simulation_parameters_1D import idx_bounds, Nj, species, temp_type, dist_type, mass, charge, velocity, sim_repr, Tpar, Tper
from   matplotlib  import rcParams

def manage_directories():
    print 'Checking directories...'
    if (generate_data == 1 or generate_plots == 1) == True:
        if os.path.exists('%s/%s' % (drive, save_path)) == False:
            os.makedirs('%s/%s' % (drive, save_path))                        # Create master test series directory
            print 'Master directory created'

        path = ('%s/%s/run_%d' % (drive, save_path, const.run_num))          # Set root run path (for images)

        if os.path.exists(path) == False:
            os.makedirs(path)
            print 'Run directory created'
        else:
            print 'Run directory already exists'
            overwrite_flag = raw_input('Overwrite? (Y/N) \n')
            if overwrite_flag.lower() == 'y':
                rmtree(path)
                os.makedirs(path)
            elif overwrite_flag.lower() == 'n':
                sys.exit('Program Terminated: Change run_num in simulation_parameters_1D')
            else:
                sys.exit('Unfamiliar input: Run terminated for safety')
    return


def create_figure_and_save(part, J, B, qn, qq, dt, plot_dump_iter):
    plt.ioff()

    r = qq / plot_dump_iter                                                     # Capture number

    fig_size = 4, 7                                                             # Set figure grid dimensions
    fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
    fig.patch.set_facecolor('w')                                                # Set figure face color

    x_pos       = part[0, :] / dx                                               # Particle x-positions (km) (For looking at particle characteristics)
    x_cell_num  = np.arange(NX)                                                 # Numerical cell numbering: x-axis

#----- Velocity (x, y) Plots: Hot Species
    ax_vx   = plt.subplot2grid(fig_size, (0, 0), rowspan=2, colspan=3)
    ax_vy   = plt.subplot2grid(fig_size, (2, 0), rowspan=2, colspan=3)

    norm_xvel   = part[3, :] / const.c
    norm_yvel   = part[4, :] / const.c       # y-velocities (for normalization)

    ax_vx.scatter(x_pos[idx_bounds[1, 0]: idx_bounds[1, 1]], norm_xvel[idx_bounds[1, 0]: idx_bounds[1, 1]], s=1, c='r', lw=0)        # Hot population
    ax_vy.scatter(x_pos[idx_bounds[1, 0]: idx_bounds[1, 1]], norm_yvel[idx_bounds[1, 0]: idx_bounds[1, 1]], s=1, c='r', lw=0)        # 'Other' population

    ax_vx.set_title(r'Beam velocities ($v_A^{-1}$) vs. Position (x)')
    ax_vy.set_xlabel(r'Position (cell)', labelpad=10)

    ax_vx.set_ylabel(r'$\frac{v_x}{v_A}$', rotation=90)
    ax_vy.set_ylabel(r'$\frac{v_y}{v_A}$', rotation=90)

    plt.setp(ax_vx.get_xticklabels(), visible=False)
    ax_vx.set_yticks(ax_vx.get_yticks()[1:])

    for ax in [ax_vy, ax_vx]:
        ax.set_xlim(0, xmax/dx)
        ax.set_ylim(-2e-3, 2e-3)

#----- Density Plot
    ax_qn = plt.subplot2grid((fig_size), (0, 3), colspan=3)                            # Initialize axes
    
    qn_norm   = qn[1: NX + 1] / ((density*charge).sum())                                # Normalize density for each species to initial values
    ax_qn.plot(qn_norm, color='green')                                     # Create overlayed plots for densities of each species

    ax_qn.set_title('Normalized Charge/Current Density and B-Fields (y, mag)')  # Axes title (For all, since density plot is on top
    ax_qn.set_ylabel(r'$\rho_c$', fontsize=14, rotation=0, labelpad=5)       # Axis (y) label for this specific axes
    ax_qn.set_ylim(0, 3)
    
#----- Current (Jx) Plot
    ax_Jz = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_qn)

    Jz = J[1: NX + 1, 0]

    ax_Jz.plot(Jz, color='magenta')

    ax_Jz.set_xlim(0, NX)
    #ax_Jz.set_ylim(-200e-5, 200e-5)

    #ax_Jz.set_yticks(np.arange(-200e-5, 201e-5, 50e-5))
    #ax_Jz.set_yticklabels(np.arange(-150, 201, 50))
    ax_Jz.set_ylabel(r'$J_x$ ($\mu$A)', labelpad=25, rotation=0, fontsize=14)

#----- Magnetic Field (By) and Magnitude (|B|) Plots
    ax_By = plt.subplot2grid((fig_size), (2, 3), colspan=3, sharex=ax_qn)              # Initialize Axes
    ax_B  = plt.subplot2grid((fig_size), (3, 3), colspan=3, sharex=ax_qn)

    mag_B = (np.sqrt(B[1:NX, 0] ** 2 + B[1:NX, 1] ** 2 + B[1:NX, 2] ** 2)) / B0
    B_y   = B[1:NX , 1] / B0                                                         # Normalize grid values

    ax_B.plot(mag_B, color='g')                                             # Create axes plots
    ax_By.plot(B_y, color='g')

    ax_B.set_xlim(0,  NX)                                                               # Set x limit
    ax_By.set_xlim(0, NX)

    ax_B.set_ylim(0, 2)                                                                 # Set y limit
    ax_By.set_ylim(-1, 1)

    ax_B.set_ylabel( r'$|B|$', rotation=0, labelpad=20, fontsize=14)                                 # Set labels
    ax_By.set_ylabel(r'$\frac{B_y}{B_0}$', rotation=0, labelpad=10, fontsize=14)
    ax_B.set_xlabel('Cell Number')                                                      # Set x-axis label for group (since |B| is on bottom)

    for ax in [ax_qn, ax_Jz, ax_By]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_yticks(ax.get_yticks()[1:])

    for ax in [ax_qn, ax_Jz, ax_By, ax_B]:
        qrt = NX / (4.)
        ax.set_xticks(np.arange(0, NX + qrt, qrt))
        ax.grid()

#----- Plot Adjustments
    plt.tight_layout(pad=1.0, w_pad=1.8)
    fig.subplots_adjust(hspace=0)

    filename = 'anim%05d.png' % r
    path     = drive + save_path + '/run_{}/anim/'.format(const.run_num)
    
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
        
    fullpath = path + filename
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
    print 'Plot saved'.format(r)
    plt.close('all')
    return


def store_run_parameters(dt, data_dump_iter):
    d_path = ('%s/%s/run_%d/data' % (drive, save_path, const.run_num))    # Set path for data

    manage_directories()

    if os.path.exists(d_path) == False:                                   # Create data directory
        os.makedirs(d_path)

    # Single parameters
    params = dict([('seed', const.seed),
                   ('Nj', Nj),
                   ('dt', dt),
                   ('NX', NX),
                   ('dxm', const.dxm),
                   ('dx', const.dx),
                   ('cellpart', const.cellpart),
                   ('subcycles', const.subcycles),
                   ('B0', const.B0),
                   ('ne', ne),
                   ('Te0', const.Te0),
                   ('ie', const.ie),
                   ('theta', const.theta),
                   ('data_dump_iter', data_dump_iter),
                   ('max_sec', const.max_rev),
                   ('run_desc', const.run_description)])

    h_name = os.path.join(d_path, 'Header.pckl')            # Data file containing dictionary of variables used in run

    with open(h_name, 'wb') as f:
        pickle.dump(params, f)
        f.close()
        print 'Header file saved'

    # Particle values: Array parameters
    p_file = os.path.join(d_path, 'p_data')
    np.savez(p_file, idx_bounds=idx_bounds, species=species, temp_type=temp_type, dist_type=dist_type, mass=mass,
             charge=charge, velocity=velocity, density=density, sim_repr=sim_repr, Tpar=Tpar, Tper=Tper)
    print 'Particle data saved'
    return

def save_data(dt, data_dump_iter, qq, part, Ji, E, B, dns):
    d_path = ('%s/%s/run_%d/data' % (drive, save_path, const.run_num))    # Set path for data
    r      = qq / data_dump_iter                                          # Capture number

    d_filename = 'data%05d' % r
    d_fullpath = os.path.join(d_path, d_filename)
    np.savez(d_fullpath, part=part, E = E[:, 0:3], B = B[:, 0:3], J = Ji, dns = dns)   # Data file for each iteration
    print 'Data saved'.format(qq)
