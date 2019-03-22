# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:15:59 2017

@author: iarey
"""
import numba as nb
import numpy as np

import plot_and_save as pas
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

    for ii in np.arange(A.shape[0]):
        output[ii, 0] = A[ii, 1] * B[ii, 2] - A[ii, 2] * B[ii, 1]
        output[ii, 1] = A[ii, 2] * B[ii, 0] - A[ii, 0] * B[ii, 2]
        output[ii, 2] = A[ii, 0] * B[ii, 1] - A[ii, 1] * B[ii, 0]
    return output



@nb.njit()
def interpolate_to_center_cspline1D(arr, DX=const.dx):
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
def interpolate_to_center_cspline3D(arr, DX=const.dx):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    1D array
    '''
    dim    = arr.shape[1]
    interp = np.zeros((arr.shape[0], dim), dtype=np.float64)	

    # Calculate second derivative for interior points
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

    if const.plot_res == 0:                           # Decide plot and data increments (if enabled)
        plot_iter = 1
    else:
        plot_iter = int(const.plot_res*gyperiod / DT)

    if const.data_res == 0:
        data_iter = 1
    else:
        data_iter = int(const.data_res*gyperiod / DT)
    
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
    
    if const.generate_data == 1:
        pas.store_run_parameters(DT, data_iter)
        
    return DT, max_inc, data_iter, plot_iter, subcycles


def check_timestep(qq, DT, pos, vel, B, E, dns, max_inc, data_iter, plot_iter, subcycles):
    max_Vx          = np.max(vel[0, :])
    max_V           = np.max(vel)
    k_max           = np.pi / const.dx

    B_cent          = interpolate_to_center_cspline3D(B)
    B_tot           = np.sqrt(B_cent[:, 0] ** 2 + B_cent[:, 1] ** 2 + B_cent[:, 2] ** 2)
    high_rat        = (const.charge/const.mass).max()
    
    gyfreq          = high_rat  * np.abs(B_tot).max() / (2 * np.pi)      
    ion_ts          = const.orbit_res * 1./gyfreq
    
    if E.max() != 0:
        elecfreq        = high_rat*max(abs(E[:, 0] / max_V))             # Electron acceleration "frequency"
        freq_ts         = const.freq_res / elecfreq                            
    else:
        freq_ts = ion_ts
    
    vel_ts          = 0.75*const.dx / max_Vx                                   # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than 'half' a cell in one time step
    DT_part         = min(freq_ts, vel_ts, ion_ts)                       # Smallest of the particle conditions

    change_flag = 0
    if DT_part < 0.9*DT:
        pos, Ie, W_elec = particles.position_update(pos, vel, -0.5*DT)           # Roll back particle position before halving timestep
        
        change_flag = 1
        DT         *= 0.5
        max_inc    *= 2
        qq         *= 2
        
        if plot_iter != None:
            plot_iter *= 2
            
        if data_iter != None:
            data_iter *= 2
            
    elif DT_part >= 4.0*DT and qq%2 == 0 and data_iter%2 == 0 and max_inc%2 == 0:
        pos, Ie, W_elec = particles.position_update(pos, vel, -0.5*DT)          # Roll back particle position before halving timestep
        
        change_flag  = 2
        DT          *= 2.0
        max_inc      = max_inc // 2
        qq         //= 2
        
        if plot_iter == None or plot_iter == 1:
            plot_iter   = 1
        else:
            plot_iter //= 2
        
        if data_iter == None or data_iter == 1:
            data_iter   = 1
        else:
            data_iter //= 2

    if const.adaptive_subcycling == True:
        dispfreq        = (k_max ** 2) * (B_tot / (const.mu0 * dns)).max()             # Dispersion frequency
        dt_sc           = const.freq_res * 1./dispfreq
        new_subcycles   = int(DT / dt_sc + 1)
        
        if subcycles < 0.75*new_subcycles:                                       
            subcycles *= 2
            print('Number of subcycles per timestep doubled to {}'.format(subcycles))
            
        if subcycles > 3.0*new_subcycles and subcycles%2 == 0:                                      
            subcycles //= 2
            print('Number of subcycles per timestep halved to {}'.format(subcycles))
            
        if subcycles > 1000:
            const.adaptive_subcycling = False
            subcycles = 1000
    else:
        pass

    return pos, qq, DT, max_inc, data_iter, plot_iter, change_flag, subcycles



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