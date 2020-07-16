# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:44:46 2017

@author: iarey
"""
import numpy as np
import pickle

import os
import sys
import simulation_parameters_2D as const

def manage_directories():
    # Create master test series directory, if needed
    print('Checking directories...')
    if (const.save_particles == 1 or const.save_fields == 1) == True:
        if os.path.exists('%s/%s' % (const.drive, const.save_path)) == False:
            os.makedirs('%s/%s' % (const.drive, const.save_path))                        
            print('Master directory created')

        path = ('%s/%s/run_%d' % (const.drive, const.save_path, const.run_num))          

        if os.path.exists(path) == False:
            os.makedirs(path)
            print('Run directory created')
        else:
            print('Run directory already exists')
            sys.exit('Program Terminated: Change run_num in simulation_parameters_1D')

    return


def store_run_parameters(dt, part_save_iter, field_save_iter):
    # Set paths for data
    d_path = ('%s/%s/run_%d/data/' % (const.drive, const.save_path, const.run_num))    
    f_path = d_path + '/fields/'
    p_path = d_path + '/particles/'
    
    manage_directories()

    for folder in [d_path, f_path, p_path]:
        if os.path.exists(folder) == False:                               # Create data directories
            os.makedirs(folder)

    # Single parameters
    params = dict([('seed', const.seed),
                   ('Nj', const.Nj),
                   ('dt', dt),
                   ('NX', const.NX),
                   ('dxm', const.dxm),
                   ('dx', const.dx),
                   ('nsp_ppc', const.nsp_ppc),
                   ('B0', const.B0),
                   ('HM_amplitude', const.HM_amplitude),
                   ('HM_frequency', const.HM_frequency),
                   ('ne', const.ne),
                   ('Te0', const.Te0),
                   ('ie', const.ie),
                   ('theta', const.theta),
                   ('part_save_iter', part_save_iter),
                   ('field_save_iter', field_save_iter),
                   ('max_rev', const.max_rev),
                   ('orbit_res', const.orbit_res),
                   ('freq_res', const.freq_res),
                   ('run_desc', const.run_description),
                   ('method_type', 'PREDCORR_2D_TIMEVAR'),
                   ('particle_shape', 'TSC')
                   ])

    with open(d_path + 'simulation_parameters.pckl', 'wb') as f:
        pickle.dump(params, f)
        f.close()
        print('Simulation parameters saved')
        
    # Particle values: Array parameters
    p_file = d_path + 'particle_parameters'
    np.savez(p_file, idx_start   = const.idx_start,
                     idx_end     = const.idx_end,
                     species_lbl = const.species_lbl,
                     temp_color  = const.temp_color,
                     temp_type   = const.temp_type,
                     dist_type   = const.dist_type,
                     mass        = const.mass,
                     charge      = const.charge,
                     drift_v     = const.drift_v,
                     density     = const.density,
                     n_contr     = const.n_contr,
                     Tpar        = const.Tpar,
                     Tper        = const.Tper)
    print('Particle data saved')
    return


def save_field_data(dt, field_save_iter, qq, Ji, E, B, Ve, Te, dns):
    '''Saves ghost cells as well (for checks later on)'''
    sim_time = np.array([qq*dt])    # Timestamp: Useful for debugging
    d_path   = '%s/%s/run_%d/data/fields/' % (const.drive, const.save_path, const.run_num)
    r        = qq / field_save_iter

    d_fullpath = d_path + 'data%05d' % r
        
    np.savez(d_fullpath, E = E, B = B, J = Ji, dns = dns, Ve = Ve, Te = Te, sim_time = sim_time)
    print('Field data saved')
    
def save_particle_data(dt, part_save_iter, qq, pos, vel):
    sim_time = np.array([qq*dt])    # Timestamp: Useful for debugging
    d_path   = '%s/%s/run_%d/data/particles/' % (const.drive, const.save_path, const.run_num)
    r        = qq / part_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, pos = pos, vel = vel, sim_time = sim_time)
    print('Particle data saved')