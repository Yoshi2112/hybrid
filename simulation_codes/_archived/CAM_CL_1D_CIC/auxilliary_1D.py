# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:15:59 2017

@author: iarey
"""
import numba as nb
import numpy as np

import particles_1D as particles
from simulation_parameters_1D import data_res, plot_res, max_rev, dx, gyfreq, orbit_res, charge, mass, NX


@nb.njit(cache=True)
def cross_product_single(A, B):
    '''Vector (cross) product between 3-vectors, A and B of same dimensions.

    INPUT:
        A, B -- 3-vectors (single values)

    OUTPUT:
        output -- The resultant cross product as a 3-vector
    '''
    output = np.zeros(A.shape)

    output[0] =    A[1] * B[2] - A[2] * B[1]
    output[1] = - (A[0] * B[2] - A[2] * B[0])
    output[2] =    A[0] * B[1] - A[1] * B[0]

    return output


@nb.njit(cache=True)
def cross_product(A, B):
    '''Vector (cross) product between two vectors, A and B of same dimensions.

    INPUT:
        A, B -- 3D vectors (ndarrays)

    OUTPUT:
        output -- The resultant cross product with same dimensions as input vectors
    '''
    output = np.zeros(A.shape)

    output[:, 0] =    A[:, 1] * B[:, 2] - A[:, 2] * B[:, 1]
    output[:, 1] = - (A[:, 0] * B[:, 2] - A[:, 2] * B[:, 0])
    output[:, 2] =    A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]

    return output


def set_timestep(part):
    gyperiod = (2*np.pi) / gyfreq               # Seconds (gyfreq in rad/s)
    ion_ts   = orbit_res * gyperiod             # Timestep to resolve gyromotion
    vel_ts   = dx / (2 * np.max(part[3, :]))    # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step

    DT       = min(ion_ts, vel_ts)              # Smallest of the two
    max_time = max_rev * gyperiod               # Total number of seconds required
    max_inc  = int(max_time / DT) + 1           # Total number of increments

    if plot_res == 0:                           # Decide plot and data increments (if enabled)
        plot_dump_iter = 1
    else:
        plot_dump_iter = int(plot_res*gyperiod / DT)

    if data_res == 0:
        data_dump_iter = 1
    else:
        data_dump_iter = int(data_res*gyperiod / DT)

    print('Timestep: %.4fs, %d iterations total' % (DT, max_inc))
    return DT, max_inc, data_dump_iter, plot_dump_iter


def check_timestep(qq, DT, part, B, E, dns, maxtime, data_dump_iter, plot_dump_iter):
    max_V           = np.max(part[3, :])
    #k_wave          = np.pi / dx
    
    B_cent = np.zeros((NX + 2, 3))
    
    for ii in range(3):
        B_cent[1:-1, ii] = 0.5*(B[:-1, ii] + B[1:, ii])
        
    B_tot           = np.sqrt(B_cent[:, 0] ** 2 + B_cent[:, 1] ** 2 + B_cent[:, 2] ** 2)
    high_rat        = np.max(charge/mass)
    
    gyfreq          = high_rat*max(abs(B_tot)) / (2 * np.pi)         # Gyrofrequency
    #elecfreq        = high_rat*max(abs(E[:, 0] / max_V))            # Electron acceleration "frequency"
    #dispfreq        = (k_wave ** 2) * np.max(B_tot / (mu0 * dns))   # Dispersion frequency
    
    ion_ts          = orbit_res / gyfreq                 # Timestep to resolve gyromotion
    #acc_ts          = orbit_res / elecfreq              # Timestep to resolve electric field acceleration
    #dis_ts          = orbit_res / dispfreq              # Timestep to resolve magnetic field dispersion
    vel_ts          = 0.60*dx / max_V                    # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than 'half' a cell in one time step
                                                         # Slightly larger than half to stop automatically halving DT at start
    DT_new          = min(ion_ts, vel_ts)                # Smallest of the two (four, later)
    
    change_flag = 0
    if DT_new < DT:
        part = particles.position_update(part, -0.5*DT)     # Roll back particle position before halving timestep
        
        change_flag = 1
        DT         *= 0.5
        maxtime    *= 2
        qq         *= 2
        
        if plot_dump_iter != None:
            plot_dump_iter *= 2
            
        if data_dump_iter != None:
            data_dump_iter *= 2
        
    return part, qq, DT, maxtime, data_dump_iter, plot_dump_iter, change_flag

