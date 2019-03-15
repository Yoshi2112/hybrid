# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:44:46 2017

@author: iarey
"""
import numpy as np
import matplotlib.pyplot as plt

import os
import simulation_parameters_1D as const
from   simulation_parameters_1D import drive, save_path, NX, dx, xmax, B0, density
from   simulation_parameters_1D import idx_bounds, Nj, species_lbl, charge, temp_color


def create_figure_and_save(pos, vel, E, B, qn, qq, DT, plot_iter):
    plt.ioff()

    r = qq / plot_iter                                                     # Capture number

    fig_size = 4, 7                                                             # Set figure grid dimensions
    fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
    fig.patch.set_facecolor('w')                                                # Set figure face color

    x_pos       = pos / dx                                                      # Particle x-positions (km) (For looking at particle characteristics)
    #pdb.set_trace()
#----- Velocity (x, y) Plots: Hot Species
    ax_vx   = plt.subplot2grid(fig_size, (0, 0), rowspan=2, colspan=3)
    ax_vy   = plt.subplot2grid(fig_size, (2, 0), rowspan=2, colspan=3)

    norm_xvel   = vel[0, :] / const.c
    norm_yvel   = vel[1, :] / const.c

    for jj in range(Nj):
        ax_vx.scatter(x_pos[idx_bounds[jj, 0]: idx_bounds[jj, 1]], norm_xvel[idx_bounds[jj, 0]: idx_bounds[jj, 1]], s=3, c=temp_color[jj], lw=0, label=species_lbl[jj])
        ax_vy.scatter(x_pos[idx_bounds[jj, 0]: idx_bounds[jj, 1]], norm_yvel[idx_bounds[jj, 0]: idx_bounds[jj, 1]], s=3, c=temp_color[jj], lw=0)

    ax_vx.legend()
    ax_vx.set_title(r'Particle velocities vs. Position (x)')
    ax_vy.set_xlabel(r'Cell', labelpad=10)

    ax_vx.set_ylabel(r'$\frac{v_x}{c}$', rotation=90)
    ax_vy.set_ylabel(r'$\frac{v_y}{c}$', rotation=90)

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
    ax_qn.set_ylim(0, 2)
    
#----- Current (Jx) Plot
    ax_Ex = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_qn)

    Ex = E[1: NX + 1, 0]

    ax_Ex.plot(Ex, color='magenta')

    ax_Ex.set_xlim(0, NX)
    #ax_Jx.set_ylim(-200e-5, 200e-5)

    #ax_Jx.set_yticks(np.arange(-200e-5, 201e-5, 50e-5))
    #ax_Jx.set_yticklabels(np.arange(-150, 201, 50))
    ax_Ex.set_ylabel(r'$E_x$', labelpad=25, rotation=0, fontsize=14)

#----- Magnetic Field (By) and Magnitude (|B|) Plots
    ax_By = plt.subplot2grid((fig_size), (2, 3), colspan=3, sharex=ax_qn)              # Initialize Axes
    ax_B  = plt.subplot2grid((fig_size), (3, 3), colspan=3, sharex=ax_qn)

    mag_B = (np.sqrt(B[1:NX+2, 0] ** 2 + B[1:NX+2, 1] ** 2 + B[1:NX+2, 2] ** 2)) / B0
    B_y   = B[1:NX+2 , 1] / B0                                                         # Normalize grid values

    ax_B.plot(mag_B, color='g')                                                        # Create axes plots
    ax_By.plot(B_y, color='g') 

    ax_B.set_xlim(0,  NX)                                                               # Set x limit
    ax_By.set_xlim(0, NX)

    ax_B.set_ylim(0, 2)                                                                 # Set y limit
    ax_By.set_ylim(-1, 1)

    ax_B.set_ylabel( r'$|B|$', rotation=0, labelpad=20, fontsize=14)                    # Set labels
    ax_By.set_ylabel(r'$\frac{B_y}{B_0}$', rotation=0, labelpad=10, fontsize=14)
    ax_B.set_xlabel('Cell Number')                                                      # Set x-axis label for group (since |B| is on bottom)

    for ax in [ax_qn, ax_Ex, ax_By]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_yticks(ax.get_yticks()[1:])

    for ax in [ax_qn, ax_Ex, ax_By, ax_B]:
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