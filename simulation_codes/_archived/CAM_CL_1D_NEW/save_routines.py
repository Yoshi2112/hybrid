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


def manage_directories():
    print('Checking directories...')
    if (const.save_particles == 1 or const.save_fields == 1) == True:
        if os.path.exists('%s/%s' % (const.drive, const.save_path)) == False:
            os.makedirs('%s/%s' % (const.drive, const.save_path))                        # Create master test series directory
            print('Master directory created')

        path = ('%s/%s/run_%d' % (const.drive, const.save_path, const.run_num))          

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
    d_path = '%s/%s/run_%d/data/' % (const.drive, const.save_path, const.run_num)     # Set main dir for data
    f_path = d_path + '/fields/'
    p_path = d_path + '/particles/'
    manage_directories()

    for folder in [d_path, f_path, p_path]:
        if os.path.exists(folder) == False:                               # Create data directories
            os.makedirs(folder)

    # Save simulation parameters to file
    params = dict([('seed', const.seed),
                   ('Nj', const.Nj),
                   ('dt', dt),
                   ('NX', const.NX),
                   ('dxm', const.dxm),
                   ('dx', const.dx),
                   ('subcycles', const.subcycles),
                   ('B0', const.B0),
                   ('ne', const.ne),
                   ('Te0', const.Te0),
                   ('ie', const.ie),
                   ('theta', const.theta),
                   ('part_save_iter', part_save_iter),
                   ('field_save_iter', field_save_iter),
                   ('max_rev', const.max_rev),
                   ('LH_frac', const.LH_frac),
                   ('orbit_res', const.orbit_res),
                   ('freq_res', const.freq_res),
                   ('run_desc', const.run_description),
                   ('method_type', 'CAM_CL_NEW'),
                   ('particle_shape', 'TSC')
                   ])

    with open(d_path + 'simulation_parameters.pckl', 'wb') as f:
        pickle.dump(params, f)
        f.close()
        
    print('Simulation parameters saved')
    
    # Save particle parameters to file
    p_file = os.path.join(d_path, 'particle_parameters')
    np.savez(p_file, idx_start   = const.idx_start,
                     idx_end     = const.idx_end,
                     species_lbl = const.species_lbl,
                     temp_color  = const.temp_color,
                     temp_type   = const.temp_type,
                     dist_type   = const.dist_type,
                     mass        = const.mass,
                     charge      = const.charge,
                     drift_v     = const.drift_v,
                     nsp_ppc     = const.nsp_ppc,
                     N_species   = const.N_species,
                     density     = const.density,
                     Tpar        = const.Tpar,
                     Tper        = const.Tper,
                     Bc          = const.Bc)
    
    print('Particle parameters saved')
    return


def save_field_data(dt, field_save_iter, qq, Ji, E, B, Ve, Te, dns):
    NX = const.NX
    d_path = '%s/%s/run_%d/data/fields/' % (const.drive, const.save_path, const.run_num)
    r      = qq / field_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, E = E[1:NX+1, 0:3], B = B[1:NX+2, 0:3], J = Ji[1:NX+1],
                         dns = dns[1:NX+1], Ve = Ve[1:NX+1], Te = Te[1:NX+1])
    
    
def save_particle_data(dt, part_save_iter, qq, pos, vel):
    d_path = '%s/%s/run_%d/data/particles/' % (const.drive, const.save_path, const.run_num)
    r      = qq / part_save_iter

    d_filename = 'data%05d' % r
    d_fullpath = os.path.join(d_path, d_filename)
    np.savez(d_fullpath, pos = pos, vel = vel)
