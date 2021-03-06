# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:15:59 2017

@author: iarey
"""
import numba as nb
import numpy as np
import pdb

import save_routines as save
import particles_1D as particles
import simulation_parameters_1D as const


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


@nb.njit()
def cross_product(A, B):
    '''
    Vector (cross) product between two vectors, A and B of same dimensions.

    INPUT:
        A, B -- 3D vectors (ndarrays)

    OUTPUT:
        output -- The resultant cross product with same dimensions as input vectors
    '''
    output = np.zeros(A.shape)

    for ii in np.arange(A.shape[0]):
        output[ii, 0] = A[ii, 1] * B[ii, 2] - A[ii, 2] * B[ii, 1]
        output[ii, 1] = A[ii, 2] * B[ii, 0] - A[ii, 0] * B[ii, 2]
        output[ii, 2] = A[ii, 0] * B[ii, 1] - A[ii, 1] * B[ii, 0]
    return output



@nb.njit()
def interpolate_to_center_cspline1D(arr):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    1D array
    '''
    interp = np.zeros(arr.shape[0], dtype=np.float64)	
    
    for ii in range(1, arr.shape[0] - 2):                       
        interp[ii] = 0.5 * (arr[ii] + arr[ii + 1]) \
                 - 1./16 * (arr[ii + 2] - arr[ii + 1] - arr[ii] + arr[ii - 1])
         
    interp[0]                = interp[arr.shape[0] - 3]
    interp[arr.shape[0] - 2] = interp[1]
    interp[arr.shape[0] - 1] = interp[2]
    return interp


@nb.njit()
def interpolate_to_center_cspline3D(arr):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    1D array
    '''
    dim    = arr.shape[1]
    interp = np.zeros((arr.shape[0], dim), dtype=np.float64)	

    for jj in range(dim):
        interp[:, jj] = interpolate_to_center_cspline1D(arr[:, jj])
    return interp


@nb.njit()
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
    ion_ts   = const.orbit_res * gyperiod             # Timestep to resolve gyromotion
    vel_ts   = const.dx / (2 * np.max(vel[0, :]))     # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step

    DT       = min(ion_ts, vel_ts)
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
    
    print('Timestep: %.4fs, %d iterations total' % (DT, max_inc))
    
    if const.adaptive_subcycling == True:
        k_max           = np.pi / const.dx
        dispfreq        = (k_max ** 2) * const.B0 / (const.mu0 * const.ne * const.q)            # Dispersion frequency
        dt_sc           = const.freq_res * 1./dispfreq
        subcycles       = int(DT / dt_sc + 1)
        print('Number of subcycles required: {}'.format(subcycles))
    else:
        subcycles = const.subcycles
        print('Number of subcycles set at default: {}'.format(subcycles))
    
    if const.save_fields == 1 or const.save_particles == 1:
        save.store_run_parameters(DT, part_save_iter, field_save_iter)
        
    return DT, max_inc, part_save_iter, field_save_iter, subcycles


#@nb.njit()
def check_timestep(qq, DT, pos, vel, Ie, W_elec, B, E, dns, max_inc, part_save_iter, field_save_iter, subcycles):
    max_Vx          = vel[0, :].max()
    max_V           = vel.max()
    
    B_cent          = interpolate_to_center_cspline3D(B)
    B_tot           = np.sqrt(B_cent[:, 0] ** 2 + B_cent[:, 1] ** 2 + B_cent[:, 2] ** 2)
    high_rat        = const.qm_ratios.max()
    
    gyfreq          = high_rat  * np.abs(B_tot).max() / (2 * np.pi)      
    ion_ts          = const.orbit_res * 1./gyfreq
    
    if E.max() != 0:
        elecfreq    = high_rat * (np.abs(E[:, 0] / max_V)).max()
        freq_ts     = const.freq_res / elecfreq                            
    else:
        freq_ts     = ion_ts
    
    vel_ts          = 0.75*const.dx / max_Vx
    DT_part         = min(freq_ts, vel_ts, ion_ts)
    
    # Reduce timestep
    change_flag       = 0
    if DT_part < 0.9*DT:
        particles.position_update(pos, vel, Ie, W_elec, -0.5*DT)
        
        change_flag      = 1
        DT              *= 0.5
        max_inc         *= 2
        qq              *= 2
        part_save_iter  *= 2
        field_save_iter *= 2
        print('Timestep halved. Syncing particle velocity/position with DT =', DT)
    
    # Increase timestep (only if previously decreased, or everything's even - saves wonky cuts)
    elif DT_part >= 4.0*DT and qq%2 == 0 and part_save_iter%2 == 0 and field_save_iter%2 == 0 and max_inc%2 == 0:
        particles.position_update(pos, vel, Ie, W_elec, -0.5*DT)
        
        change_flag       = 1
        DT               *= 2.0
        max_inc         //= 2
        qq              //= 2
        part_save_iter  //= 2
        field_save_iter //= 2
        
        print('Timestep doubled. Syncing particle velocity/position with DT =', DT)

    if const.adaptive_subcycling == 1:
        k_max           = np.pi / const.dx
        dispfreq        = (k_max ** 2) * (B_tot / (const.mu0 * dns)).max()             # Dispersion frequency
        dt_sc           = const.freq_res / dispfreq
        new_subcycles   = int(DT / dt_sc + 1)
        
        if subcycles < 0.75*new_subcycles:                                       
            subcycles *= 2
            print('Number of subcycles per timestep doubled to', subcycles)
            
        if subcycles > 3.0*new_subcycles and subcycles%2 == 0:                                      
            subcycles //= 2
            print('Number of subcycles per timestep halved to', subcycles)
            
        if subcycles >= 1000:
            subcycles = 1000
            print('Maxmimum number of subcycles reached - she gon\' blow')

    return qq, DT, max_inc, part_save_iter, field_save_iter, change_flag, subcycles


def dump_to_file(pos, vel, E, Ve, Te, B, J, J_minus, J_plus, rho_int, rho_half, qq, suff='', print_particles=False):
    import os, sys
    np.set_printoptions(threshold=sys.maxsize)
    
    dirpath = const.drive + const.save_path + '/run_{}/ts_{:05}/'.format(const.run_num, qq, suff) 
    if os.path.exists(dirpath) == False:
        os.makedirs(dirpath)
        
    print('Dumping arrays to file')
    if print_particles == True:
        with open(dirpath + 'pos{}.txt'.format(suff), 'w') as f:
            print(pos, file=f)
        with open(dirpath + 'vel{}.txt'.format(suff), 'w') as f:
            print(vel, file=f)
    with open(dirpath + 'E{}.txt'.format(suff), 'w') as f:
        print(E, file=f)
    with open(dirpath + 'Ve{}.txt'.format(suff), 'w') as f:
        print(Ve, file=f)
    with open(dirpath + 'Te{}.txt'.format(suff), 'w') as f:
        print(Te, file=f)
    with open(dirpath + 'B{}.txt'.format(suff), 'w') as f:
        print(B, file=f)
    with open(dirpath + 'J{}.txt'.format(suff), 'w') as f:
        print(J, file=f)
    with open(dirpath + 'J_minus{}.txt'.format(suff), 'w') as f:
        print(J_minus, file=f)
    with open(dirpath + 'J_plus{}.txt'.format(suff), 'w') as f:
        print(J_plus, file=f)
    with open(dirpath + 'rho_int{}.txt'.format(suff), 'w') as f:
        print(rho_int, file=f)
    with open(dirpath + 'rho_half{}.txt'.format(suff), 'w') as f:
        print(rho_half, file=f)
    np.set_printoptions(threshold=1000)
    return

#%%
#%% DEPRECATED OR UNTESTED FUNCTIONS
#%%
#@nb.njit()
def old_interpolate_to_center_cspline1D(arr, DX=const.dx):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    1D array
    '''

    interp = np.zeros(arr.shape[0], dtype=np.float64)	
    y2     = np.zeros(arr.shape[0], dtype=np.float64)

    # Calculate second derivative for interior points
    for ii in range(1, arr.shape[0] - 1):                       
        y2[ii] = (arr[ii - 1] - 2*arr[ii] + arr[ii + 1]) 
    
    # Average y2 at boundaries
    end_bit                = 0.5*(y2[1] + y2[arr.shape[0] - 2])
    y2[1]                  = end_bit
    y2[arr.shape[0] - 2]   = end_bit
    
    # Assign ghost cell values
    y2[0]                = y2[arr.shape[0] - 3]                      # Update edge cells
    y2[arr.shape[0] - 1] = y2[2]                                          

    # Interpolate midpoints using cubic spline
    for ii in range(arr.shape[0] - 1):
        interp[ii] = 0.5 * (arr[ii] + arr[ii + 1]) - 0.0625*(y2[ii] + y2[ii + 1])

    interp[0]                = interp[arr.shape[0] - 3]
    interp[arr.shape[0] - 2] = interp[1]
    interp[arr.shape[0] - 1] = interp[2]
    return interp


#@nb.njit()
def old_interpolate_to_center_cspline3D(arr, DX=const.dx):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    1D array
    '''
    dim    = arr.shape[1]
    interp = np.zeros((arr.shape[0], dim), dtype=np.float64)	

    # Calculate second derivative for interior points
    for jj in range(dim):
        interp[:, jj] = old_interpolate_to_center_cspline1D(arr[:, jj])
    return interp