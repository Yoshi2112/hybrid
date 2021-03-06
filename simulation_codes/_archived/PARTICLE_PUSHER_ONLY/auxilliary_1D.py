# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:15:59 2017

@author: iarey
"""
import numba as nb
import numpy as np

import particles_1D as particles
import fields_1D    as fields
import init_1D      as init

from simulation_parameters_1D import dx, NC, NX, ND, qm_ratios, freq_res, orbit_res, E_nodes, disable_waves

@nb.njit()
def cross_product(A, B, C):
    '''
    Vector (cross) product between two vectors, A and B of same dimensions.

    INPUT:
        A, B -- 3D vectors (ndarrays)

    OUTPUT:
        C -- The resultant cross product with same dimensions as input vectors
    '''
    C[:, 0] += A[:, 1] * B[:, 2]
    C[:, 1] += A[:, 2] * B[:, 0]
    C[:, 2] += A[:, 0] * B[:, 1]
    
    C[:, 0] -= A[:, 2] * B[:, 1]
    C[:, 1] -= A[:, 0] * B[:, 2]
    C[:, 2] -= A[:, 1] * B[:, 0]
    return


@nb.njit()
def interpolate_edges_to_center(B, interp, zero_boundaries=False):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    with a 3D array (e.g. B). Second derivative y2 is calculated on the B-grid, with
    forwards/backwards difference used for endpoints.
    
    interp has one more gridpoint than required just because of the array used. interp[-1]
    should remain zero.
    
    This might be able to be done without the intermediate y2 array since the interpolated
    points don't require previous point values.
    '''
    y2      = np.zeros(B.shape, dtype=nb.float64)
    interp *= 0.
    
    # Calculate second derivative
    for jj in range(1, B.shape[1]):
        
        # Interior B-nodes, Centered difference
        for ii in range(1, NC):
            y2[ii, jj] = B[ii + 1, jj] - 2*B[ii, jj] + B[ii - 1, jj]
                
        # Edge B-nodes, Forwards/Backwards difference
        if zero_boundaries == True:
            y2[0 , jj] = 0.
            y2[NC, jj] = 0.
        else:
            y2[0,  jj] = 2*B[0 ,    jj] - 5*B[1     , jj] + 4*B[2     , jj] - B[3     , jj]
            y2[NC, jj] = 2*B[NC,    jj] - 5*B[NC - 1, jj] + 4*B[NC - 2, jj] - B[NC - 3, jj]
        
    # Do spline interpolation: E[ii] is bracketed by B[ii], B[ii + 1]
    for jj in range(1, B.shape[1]):
        for ii in range(NC):
            interp[ii, jj] = 0.5 * (B[ii, jj] + B[ii + 1, jj] + (1/6) * (y2[ii, jj] + y2[ii + 1, jj]))
    
    # Add B0x to interpolated array
    for ii in range(NC):
        interp[ii, 0] = fields.eval_B0x(E_nodes[ii])
    return


@nb.njit()
def check_timestep(pos, vel, B, E, q_dens, Ie, W_elec, Ib, W_mag, B_center, Ep, Bp, v_prime, S, T, temp_N,\
                     qq, DT, max_inc, part_save_iter, field_save_iter, idx, B_damping_array, E_damping_array):
    '''
    Evaluates all the things that could cause a violation of the timestep:
        - Magnetic field dispersion (switchable in param file since this can be tiny)
        - Gyromotion resolution
        - Ion velocity (Don't cross more than half a cell in a timestep)
        - Electric field acceleration
        
    When a violating condition found, velocity is advanced by 0.5DT (since this happens
    at the top of a loop anyway). The assumption is that the timestep isn't violated by
    enough to cause instant instability (each criteria should have a little give), which 
    should be valid except under extreme instability. The timestep is then halved and all
    time-dependent counters and quantities are doubled. Velocity is then retarded back
    half a timestep to de-sync back into a leapfrog scheme.
    '''
    interpolate_edges_to_center(B, B_center)
    B_magnitude     = np.sqrt(B_center[ND:ND+NX+1, 0] ** 2 +
                              B_center[ND:ND+NX+1, 1] ** 2 +
                              B_center[ND:ND+NX+1, 2] ** 2)
    gyfreq          = qm_ratios.max() * B_magnitude.max()     
    ion_ts          = orbit_res / gyfreq
    
    if E[:, 0].max() != 0:
        elecfreq        = qm_ratios.max()*(np.abs(E[:, 0] / np.abs(vel).max()).max())               # Electron acceleration "frequency"
        Eacc_ts         = freq_res / elecfreq                            
    else:
        Eacc_ts = ion_ts

    vel_ts          = 0.60 * dx / np.abs(vel[0, :]).max()                        # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than 'half' a cell in one time step
    DT_part         = min(Eacc_ts, vel_ts, ion_ts)                      # Smallest of the allowable timesteps
    
    if DT_part < 0.9*DT:

        particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E, v_prime, S, T,temp_N,0.5*DT)    # Re-sync vel/pos       

        DT         *= 0.5
        max_inc    *= 2
        qq         *= 2
        
        field_save_iter *= 2
        part_save_iter *= 2

        particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E, v_prime, S, T,temp_N,-0.5*DT)   # De-sync vel/pos 
        print('Timestep halved. Syncing particle velocity...')
        init.set_damping_array(B_damping_array, E_damping_array, DT)

    return qq, DT, max_inc, part_save_iter, field_save_iter, B_damping_array, E_damping_array


#@nb.njit()
def main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag, Ep, Bp, v_prime, S, T,temp_N,                      \
              B, E_int, E_half, q_dens, q_dens_adv, Ji, ni, nu,          \
              Ve, Te, Te0, temp3De, temp3Db, temp1D, old_particles, old_fields, flux, \
              B_damping_array, E_damping_array, qq, DT, max_inc, part_save_iter, field_save_iter):
    '''
    Main loop separated from __main__ function, since this is the actual computation bit.
    Could probably be optimized further, but I wanted to njit() it.
    The only reason everything's not njit() is because of the output functions.
    
    Future: Come up with some way to loop until next save point
    
    Thoughts: declare a variable steps_to_go. Output all time variables at return
    to resync everything, and calculate steps to next stop.
    If no saves, steps_to_go = max_inc
    '''
    

    return qq, DT, max_inc, part_save_iter, field_save_iter
