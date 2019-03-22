# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:44:46 2017

@author: iarey
"""
import numpy as np
import pickle

import os
import sys
from shutil import rmtree
import simulation_parameters_1D as const
from   simulation_parameters_1D import generate_data, generate_plots, drive, save_path, NX, ne, density
from   simulation_parameters_1D import idx_bounds, Nj, species_lbl, temp_type, dist_type, mass, charge, velocity, sim_repr, Tpar, Tper, temp_color


def manage_directories():
    print('Checking directories...')
    if (generate_data == 1 or generate_plots == 1) == True:
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
                   ('B0', const.B0),
                   ('ne', ne),
                   ('Te0', const.Te0),
                   ('ie', const.ie),
                   ('theta', const.theta),
                   ('data_dump_iter', data_dump_iter),
                   ('max_rev', const.max_rev),
                   ('orbit_res', const.orbit_res),
                   ('freq_res', const.freq_res),
                   ('run_desc', const.run_description),
                   ('method_type', 'PREDCORR'),
                   ('particle_shape', 'TSC')
                   ])

    h_name = os.path.join(d_path, 'Header.pckl')            # Data file containing dictionary of variables used in run

    with open(h_name, 'wb') as f:
        pickle.dump(params, f)
        f.close()
        print('Header file saved')
        
    # Particle values: Array parameters
    p_file = os.path.join(d_path, 'p_data')
    np.savez(p_file, idx_bounds  = idx_bounds,
                     species_lbl = species_lbl,
                     temp_color  = temp_color,
                     temp_type   = temp_type,
                     dist_type   = dist_type,
                     mass        = mass,
                     charge      = charge,
                     velocity    = velocity,
                     density     = density,
                     sim_repr    = sim_repr,
                     Tpar        = Tpar,
                     Tper        = Tper)
    print('Particle data saved')
    return


def save_data(real_time, dt, data_iter, qq, pos, vel, Ji, E, B, Ve, Te, dns):
    d_path = ('%s/%s/run_%d/data' % (drive, save_path, const.run_num))  # Set path for data
    r      = qq / data_iter                                             # Capture number

    d_filename = 'data%05d' % r
    d_fullpath = os.path.join(d_path, d_filename)
    np.savez(d_fullpath, pos = pos, vel = vel, E = E[1:NX+1, 0:3], B = B[1:NX+2, 0:3], J = Ji[1:NX+1],
                         dns = dns[1:NX+1], Ve = Ve[1:NX+1], Te = Te[1:NX+1], real_time = np.array(real_time))   # Data file for each iteration
    print('Data saved'.format(qq))
