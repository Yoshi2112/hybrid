# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:15:59 2017

@author: iarey
"""
import numba as nb
import numpy as np

import particles_1D as particles
from simulation_parameters_1D import data_res, plot_res, max_rev, dx, gyfreq, orbit_res, charge, mass, NX, do_parallel


@nb.njit(fastmath=True, parallel=do_parallel)
def cross_product_single(A, B):
    '''
    Vector (cross) product between 3-vectors, A and B of same dimensions.

    INPUT:
        A, B -- 3-vectors (single values)

    OUTPUT:
        output -- The resultant cross product as a 3-vector
    '''
    output = np.zeros(A.shape)

    output[0] = A[1] * B[2] - A[2] * B[1]
    output[1] = A[2] * B[0] - A[0] * B[2]
    output[2] = A[0] * B[1] - A[1] * B[0]
    return output


@nb.njit(fastmath=True, parallel=do_parallel)
def cross_product(A, B):
    '''
    Vector (cross) product between two vectors, A and B of same dimensions.

    INPUT:
        A, B -- 3D vectors (ndarrays)

    OUTPUT:
        output -- The resultant cross product with same dimensions as input vectors
    '''
    output = np.zeros(A.shape)

    for ii in nb.prange(A.shape[0]):
        output[ii, 0] = A[ii, 1] * B[ii, 2] - A[ii, 2] * B[ii, 1]
        output[ii, 1] = A[ii, 2] * B[ii, 0] - A[ii, 0] * B[ii, 2]
        output[ii, 2] = A[ii, 0] * B[ii, 1] - A[ii, 1] * B[ii, 0]
    return output


@nb.njit()
def interpolate_to_center_cspline(arr, DX=dx):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    1D array
    '''
    interp = np.zeros(arr.shape[0], dtype=nb.float64)	
    y2     = np.zeros(arr.shape[0], dtype=nb.float64)

    # Calculate second derivative for interior points
    for ii in range(1, arr.shape[0] - 1):                       
        y2[ii] = (arr[ii - 1] - 2*arr[ii] + arr[ii + 1]) 
    
    y2[0]      = y2[NX + 1]                                     # Update edge cells
    y2[NX + 2] = y2[1]                                          

    # Interpolate midpoints using cubic spline
    for ii in range(NX + 2):
        interp[ii] = 0.5 * (arr[ii] + arr[ii + 1]) - 0.0625*(y2[ii] + y2[ii + 1])

    interp[0]      = interp[NX + 1]
    interp[NX + 2] = interp[1]
    return interp


@nb.njit()
def interpolate_to_center_linear(val):
    ''' 
    Interpolates vector cell edge values (i.e. B-grid quantities) to cell centers (i.e. E-grid quantities)
    Note: First and last (two) array values return zero due to ghost cell values
    '''
    center = np.zeros((NX + 3, 3))
    
    for ii in range(3):
        center[1:NX+1, :] = 0.5*(val[1: NX+1, :] + val[2:NX+2, :])
        
    return center


def set_timestep(vel):
    gyperiod = (2*np.pi) / gyfreq               
    ion_ts   = orbit_res * gyperiod             # Timestep to resolve gyromotion
    vel_ts   = dx / (2 * np.max(vel[0, :]))     # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step

    DT       = min(ion_ts, vel_ts)              # Smallest of the two
    max_time = max_rev * gyperiod               # Total number of seconds required
    max_inc  = int(max_time / DT) + 1           # Total number of increments

    if plot_res == 0:                           # Decide plot and data increments (if enabled)
        plot_iter = 1
    else:
        plot_iter = int(plot_res*gyperiod / DT)

    if data_res == 0:
        data_iter = 1
    else:
        data_iter = int(data_res*gyperiod / DT)

    print 'Timestep: %.4fs, %d iterations total' % (DT, max_inc)
    return DT, max_inc, data_iter, plot_iter


def check_timestep(qq, DT, pos, vel, B, E, dns, maxtime, data_iter, plot_iter):
    max_V           = np.max(vel[0, :])
    #k_wave          = np.pi / dx

    B_cent          = interpolate_to_center_linear(B)
    B_tot           = np.sqrt(B_cent[:, 0] ** 2 + B_cent[:, 1] ** 2 + B_cent[:, 2] ** 2)
    high_rat        = np.max(charge/mass)

    gyfreq          = high_rat*max(abs(B_tot)) / (2 * np.pi)         # Gyrofrequency
    #elecfreq        = high_rat*max(abs(E[:, 0] / max_V))            # Electron acceleration "frequency"
    #dispfreq        = (k_wave ** 2) * np.max(B_tot / (mu0 * dns))   # Dispersion frequency

    ion_ts          = orbit_res / gyfreq                 # Timestep to resolve gyromotion
    #acc_ts          = orbit_res / elecfreq              # Timestep to resolve electric field acceleration
    #dis_ts          = orbit_res / dispfreq              # Timestep to resolve magnetic field dispersion
    vel_ts          = 0.75*dx / max_V                    # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than 'half' a cell in one time step
                                                         # Slightly larger than half to stop automatically halving DT at start
    DT_new          = min(ion_ts, vel_ts)                # Smallest of the two (four, later)
    
    change_flag = 0
    if DT_new < DT:
        # Roll back particle position before halving timestep
        pos = particles.position_update(pos, vel, -0.5*DT)
        
        change_flag = 1
        DT         *= 0.5
        maxtime    *= 2
        qq         *= 2
        
        if plot_iter != None:
            plot_iter *= 2
            
        if data_iter != None:
            data_iter *= 2
    
    return pos, qq, DT, maxtime, data_iter, plot_iter, change_flag

