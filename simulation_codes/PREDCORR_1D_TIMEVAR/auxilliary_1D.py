# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:15:59 2017

@author: iarey
"""
import numba as nb
import numpy as np

import particles_1D as particles
import simulation_parameters_1D as const
import save_routines as save


@nb.njit()
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


@nb.njit(parallel=const.do_parallel)
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



@nb.njit(parallel=const.do_parallel)
def interpolate_to_center_cspline1D(arr, DX=const.dx):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    1D array
    '''
    interp = np.zeros(arr.shape[0], dtype=nb.float64)	
    
    for ii in nb.prange(1, arr.shape[0] - 2):                       
        interp[ii] = 0.5 * (arr[ii] + arr[ii + 1]) \
                 - 1./16 * (arr[ii + 2] - arr[ii + 1] - arr[ii] + arr[ii - 1])
         
    interp[0]                = interp[arr.shape[0] - 3]
    interp[arr.shape[0] - 2] = interp[1]
    interp[arr.shape[0] - 1] = interp[2]
    return interp


@nb.njit(parallel=const.do_parallel)
def interpolate_to_center_cspline3D(arr, DX=const.dx):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    1D array
    '''
    dim    = arr.shape[1]
    interp = np.zeros((arr.shape[0], dim), dtype=nb.float64)	

    # Calculate second derivative for interior points
    for jj in range(dim):
        interp[:, jj] = interpolate_to_center_cspline1D(arr[:, jj])
    return interp


@nb.njit(parallel=const.do_parallel)
def interpolate_to_center_linear_1D(val):
    ''' 
    Interpolates vector cell edge values (i.e. B-grid quantities) to cell centers (i.e. E-grid quantities)
    Note: First and last (two) array values return zero due to ghost cell values
    '''
    center = np.zeros(val.shape)
    
    center[1:const.NX + 1] = 0.5*(val[1: const.NX + 1] + val[2:const.NX + 2])
        
    return center


def set_timestep(vel):
    gyperiod = (2*np.pi) / const.gyfreq               # Gyroperiod within uniform field (s)         
    k_max    = np.pi / const.dx
    ion_ts   = const.orbit_res * gyperiod             # Timestep to resolve gyromotion
    vel_ts   = 0.5 * const.dx / np.max(vel[0, :])     # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step 
   
    if const.account_for_dispersion == True:
        dispfreq = (k_max ** 2) * (const.B0 / (const.mu0 * const.ne))           # Dispersion frequency
        disp_ts  = const.dispersion_allowance * const.freq_res / dispfreq
    else:
        disp_ts  = ion_ts

    DT       = min(ion_ts, vel_ts, disp_ts)
    max_time = const.max_rev * gyperiod               # Total runtime in seconds
    max_inc  = int(max_time / DT) + 1                 # Total number of time steps

    if const.part_res == 0:
        part_save_iter = 1
    else:
        part_save_iter = int(const.part_res*gyperiod / DT)

    if const.field_res == 0:
        field_save_iter = 1
    else:
        field_save_iter = int(const.field_res*gyperiod / DT)

    if const.save_fields == 1 or const.save_particles == 1:
        save.store_run_parameters(DT, part_save_iter, field_save_iter)
        
    return DT, max_inc, part_save_iter, field_save_iter


@nb.njit()
def check_timestep(qq, DT, pos, vel, B, E, dns, Ie, W_elec, max_inc, part_save_iter, field_save_iter, idx):
    max_Vx          = np.max(vel[0, :])
    max_V           = np.max(vel)
    k_max           = np.pi / const.dx

    B_cent          = interpolate_to_center_cspline3D(B)
    B_tot           = np.sqrt(B_cent[:, 0] ** 2 + B_cent[:, 1] ** 2 + B_cent[:, 2] ** 2)

    dispfreq        = (k_max ** 2) * (B_tot / (const.mu0 * dns)).max()           # Dispersion frequency
    gyfreq          = const.high_rat  * np.abs(B_tot).max() / (2 * np.pi)      
    ion_ts          = const.orbit_res * 1./gyfreq
    
    if E.max() != 0:
        elecfreq        = const.high_rat*(np.abs(E[:, 0] / max_V).max())               # Electron acceleration "frequency"
        Eacc_ts         = const.freq_res / elecfreq                            
    else:
        Eacc_ts = ion_ts
    
    if const.account_for_dispersion == True:
        disp_ts     = const.dispersion_allowance * const.freq_res / dispfreq     # Making this a little bigger so it doesn't wreck everything
    else:
        disp_ts     = ion_ts
    
    vel_ts          = 0.80 * const.dx / max_Vx                                   # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than 'half' a cell in one time step
    DT_part         = min(Eacc_ts, vel_ts, ion_ts, disp_ts)                      # Smallest of the allowable timesteps

    if DT_part < 0.9*DT:
        vel         = particles.velocity_update(pos, vel, Ie, W_elec, idx, B, E, 0.5*DT)    # Re-sync vel/pos       
        DT         *= 0.5
        max_inc    *= 2
        qq         *= 2
        vel         = particles.velocity_update(pos, vel, Ie, W_elec, idx, B, E, -0.5*DT)   # De-sync vel/pos 

        part_save_iter *= 2
        field_save_iter *= 2
            
        #print('Timestep halved. Syncing particle velocity with DT = {}'.format(DT))

            
    elif DT_part >= 4.0*DT and qq%2 == 0 and part_save_iter%2 == 0 and field_save_iter%2 == 0 and max_inc%2 == 0:
        vel         = particles.velocity_update(pos, vel, Ie, W_elec, idx, B, E, 0.5*DT)    # Re-sync vel/pos          
        DT         *= 2.0
        max_inc    /= 2
        qq         /= 2
        vel         = particles.velocity_update(pos, vel, Ie, W_elec, idx, B, E, -0.5*DT)   # De-sync vel/pos 

        part_save_iter  /= 2
        field_save_iter /= 2
        
        #print('Timestep Doubled. Syncing particle velocity with DT = {}'.format(DT))

    return vel, qq, DT, max_inc, part_save_iter, field_save_iter
