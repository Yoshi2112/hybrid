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
from   simulation_parameters_1D import drive, save_path, NX, ND, NC, r_damp, ne, density, save_particles, save_fields
from   simulation_parameters_1D import idx_start, idx_end, Nj, species_lbl, temp_type, dist_type, mass, charge,\
                                       drift_v, Tpar, Tper, temp_color, nsp_ppc, Bc, N_species


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
                   ('ND', ND),
                   ('NC', NC),
                   ('N' , const.N),
                   ('dxm', const.dxm),
                   ('dx', const.dx),
                   ('r_damp', r_damp),
                   ('L', const.L), 
                   ('B_eq', const.B_eq),
                   ('xmax', const.xmax),
                   ('B_xmax', const.B_xmax),
                   ('a', const.a),
                   ('theta_xmax', const.theta_xmax),
                   ('rc_hwidth', const.rc_hwidth),
                   ('ne', ne),
                   ('Te0', const.Te0),
                   ('ie', const.ie),
                   ('part_save_iter', part_save_iter),
                   ('field_save_iter', field_save_iter),
                   ('max_rev', const.max_rev),
                   ('orbit_res', const.orbit_res),
                   ('freq_res', const.freq_res),
                   ('run_desc', const.run_description),
                   ('method_type', 'PREDCORR_HM'),
                   ('particle_shape', 'TSC'),
                   ('boundary_type', 'damped'),
                   ('particle_boundary', 'reflective'),
                   ])

    with open(d_path + 'simulation_parameters.pckl', 'wb') as f:
        pickle.dump(params, f)
        f.close()
        print('Simulation parameters saved')
    
    # Particle values: Array parameters
    p_file = d_path + 'particle_parameters'
    np.savez(p_file, idx_start   = idx_start,
                     idx_end     = idx_end,
                     species_lbl = species_lbl,
                     temp_color  = temp_color,
                     temp_type   = temp_type,
                     dist_type   = dist_type,
                     mass        = mass,
                     charge      = charge,
                     drift_v     = drift_v,
                     nsp_ppc     = nsp_ppc,
                     density     = density,
                     N_species   = N_species,
                     Tpar        = Tpar,
                     Tper        = Tper,
                     Bc          = Bc)
    print('Particle data saved')
    return


def save_field_data(sim_time, dt, field_save_iter, qq, Ji, E, B, Ve, Te, dns):
    d_path   = '%s/%s/run_%d/data/fields/' % (drive, save_path, const.run_num)
    r        = qq / field_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, E = E[:, 0:3], B = B[:, 0:3],   J = Ji[:, 0:3],
                       dns = dns,      Ve = Ve[:, 0:3], Te = Te, sim_time = sim_time)
    print('Field data saved')
    
    
def save_particle_data(sim_time, dt, part_save_iter, qq, pos, vel):
    d_path   = '%s/%s/run_%d/data/particles/' % (drive, save_path, const.run_num)
    r        = qq / part_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, pos = pos, vel = vel, sim_time = sim_time)
    print('Particle data saved')