# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:44:46 2017

@author: iarey
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.gridspec as gs

import os
from matplotlib  import rcParams
from mpl_toolkits.mplot3d import Axes3D
import const
from const import generate_data, generate_plots, drive, save_path, N, NX, NY, xmax, ymax, ne, B0, k, \
                  kB, mu0, va, Bc, dx, dy, cellpart, Te0, theta, run_desc
from part_params import idx_bounds, Nj, species, temp_type, dist_type, mass, \
                        charge, velocity, proportion, sim_repr, Tpar, Tper, seed, rand_sample
import pdb
def manage_directories(qq, framegrab):
    print 'Checking directories...'
    if (generate_data == 1 or generate_plots == 1) == True:
        if os.path.exists('%s/%s' % (drive, save_path)) == False:
            os.makedirs('%s/%s' % (drive, save_path))                        # Create master test series directory
            print 'Master directory created'
            
        const.run_num = len(os.listdir('%s/%s' % (drive, save_path)))        # Count number of existing runs. Set to run number manually for static save
        path = ('%s/%s/run_%d' % (drive, save_path, const.run_num))          # Set root run path (for images)    
        
        if os.path.exists(path) == False:
            os.makedirs(path)
            print 'Run directory created' 
    return


def create_figure_and_save(part, E, B, dns, qq, dt, framegrab):
    if qq == 0:
        manage_directories(qq, framegrab)

    # Set Figure constants
    plt.ioff()
    fig_size = 4, 6  
    
    # Set font things
    rcParams.update({'text.color'   : 'k',
                'axes.labelcolor'   : 'k',
                'axes.edgecolor'    : 'k',
                'axes.facecolor'    : 'w',
                'mathtext.default'  : 'regular',
                'xtick.color'       : 'k',
                'ytick.color'       : 'k',
                'axes.labelsize'    : 16,
                })
    
    fig   = plt.figure(figsize=(20,10))                 # Initialize figure
    fig.patch.set_facecolor('w')                        # Set figure background to white
    
    grids = gs.GridSpec(2, 3)                           # Set figure grid specifications/layout (GridSpec)
    fig.subplots_adjust(wspace=0, hspace=0)             # Set subplot grid spacing
    
    # Slice some things for simplicity
    sim_time    = qq*dt
    pos         = part[0:2, :]              # Particle x-positions in Earth-Radii 
    vel         = part[3:6, :] / va         # Velocities as multiples of the alfven speed
    B_norm      = B[:, :, :]   / B0
    B_norm[:, :, 0] -= 1                    # Remove background field from x component
    E_norm      = E[:, :, :]   * 1e6        # Transform from V/m to mV/m
    x_cell_num  = np.arange(NX + 2)         # Numerical cell numbering: x-axis
    y_cell_num  = np.arange(NY + 2)
    X, Y        = np.meshgrid(x_cell_num, y_cell_num)

    for (ii, comp) in zip(range(3), ['x', 'y', 'z']):   # Do magnetic field
        ax = fig.add_subplot(grids[0, ii], projection='3d')
        ax.plot_wireframe(X, Y, B_norm[:, :, ii])
        
        ax.set_xlim(0, NX + 2)
        ax.set_ylim(0, NY + 2)
        ax.set_zlim(-1, 1)
        ax.view_init(elev=25., azim=300.)
    
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        
        ax.zaxis.set_rotate_label(False)                        # Disable automatic rotation
        ax.set_zlabel(r'$B_{%s}$' % comp, rotation=0)
        
    for (ii, comp) in zip(range(3), ['x', 'y', 'z']):   # Do electric field
        ax = fig.add_subplot(grids[1, ii], projection='3d')
        ax.plot_wireframe(X, Y, E_norm[:, :, ii])
        
        ax.set_xlim(0, NX + 2)
        ax.set_ylim(0, NY + 2)
        ax.set_zlim(-50, 50)
        ax.view_init(elev=25., azim=300.)
    
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        
        ax.zaxis.set_rotate_label(False)                        # Disable automatic rotation
        ax.set_zlabel(r'$E_{%s}$' % comp, rotation=0)
        
    plt.tight_layout(pad=1.0, w_pad=1.5)
    #fig.subplots_adjust(hspace=0, top=0.95)
        
# =============================================================================
#     Ns = len(rand_sample)
#     samps = np.zeros((2, Ns))
#     
#     for ii, idx in zip(range(Ns), rand_sample):
#         samps[:, ii] = pos[:, idx]
# =============================================================================
    
# =============================================================================
#     # PLOT: Density
#     ax_main = plt.subplot2grid(fig_size, (0, 0), rowspan=4, colspan=4)
#     ax_main.scatter(pos[0,:], pos[1, :], s=1, lw=0)
#     ax_main.scatter(samps[0], samps[1], s=20, c='r', marker='x')
#     
#     #ax_main.pcolormesh((dns[:, :, 0] / (proportion[0]*ne)), cmap='seismic', vmin=0.25, vmax=1.75)
# 
#     #ax_main.set_title('Normalized Beam Density')
#     ax_main.set_xlabel('x (m)')
#     ax_main.set_ylabel('y (m)')
#     ax_main.set_xlim(0, xmax)
#     ax_main.set_ylim(0, ymax)
# =============================================================================

    

# =============================================================================
#     # PLOT: Spatial values of By
#     ax_main2 = plt.subplot2grid(fig_size, (0, 4), projection='3d', rowspan=4, colspan=4)
#     ax_main2.set_title(r'$B_y$')
#     X, Y = np.meshgrid(x_cell_num, y_cell_num)
# 
#     ax_main2.plot_wireframe(X, Y, B_norm[:, :, 2])
#     ax_main2.set_xlim(0, NX + 2)
#     ax_main2.set_ylim(0, NY + 2)
#     ax_main2.set_zlim(-1, 1)
#     ax_main2.view_init(elev=25., azim=300.)
# 
#     ax_main2.set_xlabel('x (m)')
#     ax_main2.set_ylabel('y (m)')
#     ax_main2.set_zlabel(r'$B_y$')
# =============================================================================

# =============================================================================
#     # Diagnostic and Informative figures
#     beta_par = (2*mu0*(proportion*ne)*kB*Tpar) / (B0 ** 2)
#     beta_per = (2*mu0*(proportion*ne)*kB*Tper) / (B0 ** 2)
#     
#     plt.figtext(0.85, 0.90, 'N  = %d' % N, fontsize=24)
#     plt.figtext(0.85, 0.85, r'$NX$ = %d' % NX, fontsize=24)
#     plt.figtext(0.85, 0.80, r'$NY$ = %d' % NY, fontsize=24)
#     
#     plt.figtext(0.85, 0.70, r'$n_e = %.2f cm^{-1}$' % (ne / 1e6), fontsize=24)
#     plt.figtext(0.85, 0.65, r'$n_b$ = %.1f%%' % (proportion[1]*100), fontsize=24)
# 
#     plt.figtext(0.85, 0.55, r'$\beta_{\parallel} = %.1f$' % beta_par[1], fontsize=24)
#     plt.figtext(0.85, 0.50, r'$\beta_{\perp} = %.1f$' % beta_per[1], fontsize=24)
# 
#     plt.figtext(0.85, 0.40, r'$V_b = %.2f v_A$' % (velocity[1] / va), fontsize=24)
#     plt.figtext(0.85, 0.35, r'$V_c = %.2f v_A$' % (velocity[0] / va), fontsize=24)
# 
#     plt.figtext(0.82, 0.05, r'$B_0$ = [%.1f, %.1f, %.1f] nT' % (Bc[0]*1e9, Bc[1]*1e9, Bc[2]*1e9), fontsize=20)
#     plt.figtext(0.813, 0.10, r'$t_{real} = %.2fs$' % sim_time, fontsize=20)
# =============================================================================
            
    if qq%framegrab == 0:       # Dump data at specified interval   
        r = qq / framegrab          # Capture number

        filename = 'anim%05d.png' % r
        path     = drive + save_path + '/run_{}'.format(const.run_num)
        fullpath = os.path.join(path, filename)
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        print 'Plot %d produced' % r
        plt.close('all')     
    return


def save_data(dt, framegrab, qq, part, Ji, E, B, dns):
    d_path = ('%s/%s/run_%d/data' % (drive, save_path, const.run_num))    # Set path for data                
    r      = qq / framegrab                                               # Capture number

    if qq ==0:
        if os.path.exists(d_path) == False:                               # Create data directory
            os.makedirs(d_path)
            
        # Save Header File: Important variables for Data Analysis
        params = dict([('Nj', Nj),
                                   ('DT', dt),
                                   ('NX', NX),
                                   ('NY', NY),
                                   ('dx', dx),
                                   ('dy', dy),
                                   ('xmax', xmax),
                                   ('ymax', ymax),
                                   ('k' , k ),
                                   ('ne', ne),
                                   ('cellpart', cellpart),
                                   ('B0', B0),
                                   ('Te0', Te0),
                                   ('seed', seed),
                                   ('theta', theta),
                                   ('framegrab', framegrab),
                                   ('run_desc', run_desc)])
                       
        h_name = os.path.join(d_path, 'Header.pckl')            # Data file containing variables used in run
        
        with open(h_name, 'wb') as f:
            pickle.dump(params, f)
            f.close() 
            print 'Header file saved'
        
        p_file = os.path.join(d_path, 'p_data')
        np.savez(p_file, idx_bounds=idx_bounds, species=species, temp_type=temp_type, dist_type=dist_type, mass=mass,
                 charge=charge, velocity=velocity, proportion=proportion, sim_repr=sim_repr)               # Data file containing particle information
        print 'Particle data saved'

    d_filename = 'data%05d' % r
    d_fullpath = os.path.join(d_path, d_filename)
    np.savez(d_fullpath, part=part, Ji=Ji, dns=dns, E = E[:, 0:3], B = B[:, 0:3])   # Data file for each iteration
    print 'Data saved'
