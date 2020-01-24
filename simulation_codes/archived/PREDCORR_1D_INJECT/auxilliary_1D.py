# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:15:59 2017

@author: iarey
"""
import numba as nb
import numpy as np

import particles_1D as particles
import fields_1D as fields
import simulation_parameters_1D as const


@nb.njit()
def cross_product(A, B, C):
    '''
    Vector (cross) product between two vectors, A and B of same dimensions.

    INPUT:
        A, B -- 3D vectors (ndarrays)

    OUTPUT:
        C -- The resultant cross product with same dimensions as input vectors
        
    Could be more memory efficient to "accumulate" operation, but would involve rewriting
    for each specific instance.
    '''
    for ii in nb.prange(A.shape[0]):
        C[ii, 0] = A[ii, 1] * B[ii, 2] - A[ii, 2] * B[ii, 1]
        C[ii, 1] = A[ii, 2] * B[ii, 0] - A[ii, 0] * B[ii, 2]
        C[ii, 2] = A[ii, 0] * B[ii, 1] - A[ii, 1] * B[ii, 0]
    return



@nb.njit()
def interpolate_to_center_cspline1D(arr, interp):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    1D array (e.g. grad_P)
    '''
    for ii in nb.prange(1, arr.shape[0] - 2):                       
        interp[ii] = 0.5 * (arr[ii] + arr[ii + 1]) \
                 - 1./16 * (arr[ii + 2] - arr[ii + 1] - arr[ii] + arr[ii - 1])
         
    interp[0]                = interp[arr.shape[0] - 3]
    interp[arr.shape[0] - 2] = interp[1]
    interp[arr.shape[0] - 1] = interp[2]
    return


@nb.njit()
def interpolate_to_center_cspline3D(arr, interp):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    with a 3D array (e.g. B)
    '''
    for jj in range(arr.shape[1]):
        for ii in range(1, arr.shape[0] - 2):                       
            interp[ii, jj] = 0.5 * (arr[ii, jj] + arr[ii + 1, jj]) \
                     - 1./16 * (arr[ii + 2, jj] - arr[ii + 1, jj] - arr[ii, jj] + arr[ii - 1, jj])
             
        interp[0, jj]                = interp[arr.shape[0] - 3, jj]
        interp[arr.shape[0] - 2, jj] = interp[1, jj]
        interp[arr.shape[0] - 1, jj] = interp[2, jj]
    return


@nb.njit()
def interpolate_to_center_linear_1D(val, center):
    ''' 
    Interpolates vector cell edge values (i.e. B-grid quantities) to cell centers (i.e. E-grid quantities)
    Note: First and last (two) array values return zero due to ghost cell values
    '''
    center[1:const.NX + 1] = 0.5*(val[1: const.NX + 1] + val[2:const.NX + 2])
    return


@nb.njit()
def check_timestep(pos, vel, B, E, q_dens, Ie, W_elec, Ib, W_mag, B_cent, \
                     qq, DT, max_inc, part_save_iter, field_save_iter, idx):
    
    interpolate_to_center_cspline3D(B, B_cent)
    B_tot           = np.sqrt(B_cent[:, 0] ** 2 + B_cent[:, 1] ** 2 + B_cent[:, 2] ** 2)

    dispfreq        = ((np.pi / const.dx) ** 2) * (B_tot / (const.mu0 * q_dens)).max()           # Dispersion frequency
    local_gyfreq    = const.high_rat  * np.abs(B_tot).max() / (2 * np.pi)      
    ion_ts          = const.orbit_res * 1./local_gyfreq
    
    if E[:, 0].max() != 0:
        if vel.max() != 0:
            elecfreq        = const.high_rat*(np.abs(E[:, 0] / vel.max()).max())               # Electron acceleration "frequency"
            Eacc_ts         = const.freq_res / elecfreq                            
        else:
            Eacc_ts = ion_ts
    else:
        Eacc_ts = ion_ts
    
    if const.account_for_dispersion == True:
        disp_ts     = const.dispersion_allowance * const.freq_res / dispfreq     # Making this a little bigger so it doesn't wreck everything
    else:
        disp_ts     = ion_ts

    vel_ts          = 0.80 * const.dx / vel[0, :].max()                          # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than 'half' a cell in one time step
    DT_part         = min(Eacc_ts, vel_ts, ion_ts, disp_ts)                      # Smallest of the allowable timesteps
    
    if DT_part < 0.9*DT:

        particles.velocity_update(vel, Ie, W_elec, Ib, W_mag, idx, B, E, 0.5*DT)    # Re-sync vel/pos       

        DT         *= 0.5
        max_inc    *= 2
        qq         *= 2
        
        field_save_iter *= 2
        part_save_iter *= 2

        particles.velocity_update(vel, Ie, W_elec, Ib, W_mag, idx, B, E, -0.5*DT)   # De-sync vel/pos 
        print('Timestep halved. Syncing particle velocity...')
        
            
    elif DT_part >= 4.0*DT and qq%2 == 0 and part_save_iter%2 == 0 and field_save_iter%2 == 0 and max_inc%2 == 0:
        particles.velocity_update(vel, Ie, W_elec, Ib, W_mag, idx, B, E, 0.5*DT)    # Re-sync vel/pos          
        DT         *= 2.0
        max_inc   //= 2
        qq        //= 2

        field_save_iter //= 2
        part_save_iter //= 2
            
        particles.velocity_update(vel, Ie, W_elec, Ib, W_mag, idx, B, E, -0.5*DT)   # De-sync vel/pos 
        print('Timestep Doubled. Syncing particle velocity...')

    
    return qq, DT, max_inc, part_save_iter, field_save_iter


#@nb.njit()
def main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag,                      \
              B, E_int, E_half, q_dens, q_dens_adv, Ji, ni, nu,          \
              Ve, Te, temp3D, temp3D2, temp1D, old_particles, old_fields,\
              qq, DT, max_inc, part_save_iter, field_save_iter):
    '''
    Main loop separated from __main__ function, since this is the actual computation bit.
    Could probably be optimized further, but I wanted to njit() it.
    The only reason everything's not njit() is because of the output functions.
    
    Future: Come up with some way to loop until next save point
    
    Thoughts: declare a variable steps_to_go. Output all time variables at return
    to resync everything, and calculate steps to next stop.
    If no saves, steps_to_go = max_inc
    '''
    # Check timestep
    qq, DT, max_inc, part_save_iter, field_save_iter \
    = check_timestep(pos, vel, B, E_int, q_dens, Ie, W_elec, Ib, W_mag, temp3D, \
                     qq, DT, max_inc, part_save_iter, field_save_iter, idx)
    
    # Move particles, collect moments
    particles.advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                            B, E_int, DT, q_dens_adv, Ji, ni, nu, temp1D)
    
    # Average N, N + 1 densities (q_dens at N + 1/2)
    q_dens *= 0.5
    q_dens += 0.5 * q_dens_adv
    
    # Push B from N to N + 1/2
    fields.push_B(B, E_int, temp3D, DT, qq, half_flag=1)
    
    # Calculate E at N + 1/2
    fields.calculate_E(B, Ji, q_dens, E_half, Ve, Te, temp3D, temp3D2, temp1D)
    
    
    ###################################
    ### PREDICTOR CORRECTOR SECTION ###
    ###################################

    # Store old values
    old_particles[0  , :] = pos
    old_particles[1:4, :] = vel
    old_particles[4  , :] = Ie
    old_particles[5:8, :] = W_elec
    
    old_fields[:, 0:3]    = B
    old_fields[:, 3:6]    = Ji
    old_fields[:, 6:9]    = Ve
    old_fields[:,   9]    = Te
    
    # Predict fields
    E_int *= -1.0
    E_int +=  2.0 * E_half
    
    fields.push_B(B, E_int, temp3D, DT, qq, half_flag=0)

    # Advance particles to obtain source terms at N + 3/2
    particles.advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                            B, E_int, DT, q_dens, Ji, ni, nu, temp1D, pc=1)
    
    q_dens *= 0.5;    q_dens += 0.5 * q_dens_adv
    
    # Compute predicted fields at N + 3/2
    fields.push_B(B, E_int, temp3D, DT, qq + 1, half_flag=1)
    fields.calculate_E(B, Ji, q_dens, E_int, Ve, Te, temp3D, temp3D2, temp1D)
    
    # Determine corrected fields at N + 1 
    E_int *= 0.5;    E_int += 0.5 * E_half

    # Restore old values: [:] allows reference to same memory (instead of creating new, local instance)
    pos[:]    = old_particles[0  , :]
    vel[:]    = old_particles[1:4, :]
    Ie[:]     = old_particles[4  , :]
    W_elec[:] = old_particles[5:8, :]
    B[:]      = old_fields[:, 0:3]
    Ji[:]     = old_fields[:, 3:6]
    Ve[:]     = old_fields[:, 6:9]
    Te[:]     = old_fields[:,   9]
    
    fields.push_B(B, E_int, temp3D, DT, qq, half_flag=0)                           # Advance the original B

    q_dens[:] = q_dens_adv

    return qq, DT, max_inc, part_save_iter, field_save_iter
