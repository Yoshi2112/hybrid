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

from simulation_parameters_1D import dx, mu0, NC, NX, ND, qm_ratios, freq_res, orbit_res,\
                                     account_for_dispersion, dispersion_allowance, E_nodes
from fields_1D import eval_B0x

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
def interpolate_edges_to_center(B, interp, zero_boundaries=False):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    with a 3D array (e.g. B). Second derivative y2 is calculated on the B-grid, with
    forwards/backwards difference used for endpoints.
    
    interp has one more gridpoint than required just because of the array used. interp[-1]
    should remain zero.
    
    This might be able to be done without the intermediate y2 array since the interpolated
    points don't require previous point values.
    
    ADDS B0 TO X-AXIS ON TOP OF INTERPOLATION
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
    
    for ii in range(NC):
        interp[ii, 0] = eval_B0x(E_nodes[ii])
    return


@nb.njit()
def check_timestep(pos, vel, B, E, q_dens, Ie, W_elec, Ib, W_mag, B_center, \
                     qq, DT, max_inc, part_save_iter, field_save_iter, idx, damping_array):
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
    
    Also evaluates if a timestep is unnneccessarily too small, which can sometimes happen
    after wave-particle interactions are complete and energetic particles are slower. This
    criteria is higher in order to provide a little hysteresis and prevent constantly switching
    timesteps.
    '''
    interpolate_edges_to_center(B, B_center)
    B_magnitude     = np.sqrt(B_center[ND:ND+NX+1, 0] ** 2 +
                              B_center[ND:ND+NX+1, 1] ** 2 +
                              B_center[ND:ND+NX+1, 2] ** 2)
    gyfreq          = qm_ratios.max() * B_magnitude.max()     
    ion_ts          = orbit_res / gyfreq
    
    if E[:, 0].max() != 0:
        elecfreq        = qm_ratios.max()*(np.abs(E[:, 0] / vel.max()).max())               # Electron acceleration "frequency"
        Eacc_ts         = freq_res / elecfreq                            
    else:
        Eacc_ts = ion_ts
    
    if account_for_dispersion == True:
        B_tot           = np.sqrt(B_center[:, 0] ** 2 + B_center[:, 1] ** 2 + B_center[:, 2] ** 2)
    
        dispfreq        = ((np.pi / dx) ** 2) * (B_tot / (mu0 * q_dens)).max()           # Dispersion frequency

        disp_ts     = dispersion_allowance * freq_res / dispfreq     # Making this a little bigger so it doesn't wreck everything
    else:
        disp_ts     = ion_ts

    vel_ts          = 0.60 * dx / np.abs(vel[0, :]).max()                        # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than 'half' a cell in one time step
    DT_part         = min(Eacc_ts, vel_ts, ion_ts, disp_ts)                      # Smallest of the allowable timesteps
    
    if DT_part < 0.9*DT:

        particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, 0.5*DT)    # Re-sync vel/pos       

        DT         *= 0.5
        max_inc    *= 2
        qq         *= 2
        
        field_save_iter *= 2
        part_save_iter *= 2

        particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, -0.5*DT)   # De-sync vel/pos 
        print('Timestep halved. Syncing particle velocity...')
        init.set_damping_array(damping_array, DT)
            
    elif DT_part >= 4.0*DT and qq%2 == 0 and part_save_iter%2 == 0 and field_save_iter%2 == 0 and max_inc%2 == 0:
        particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, 0.5*DT)    # Re-sync vel/pos          
        DT         *= 2.0
        max_inc   //= 2
        qq        //= 2

        field_save_iter //= 2
        part_save_iter //= 2
            
        particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, -0.5*DT)   # De-sync vel/pos 
        print('Timestep Doubled. Syncing particle velocity...')
        init.set_damping_array(damping_array, DT)

    return qq, DT, max_inc, part_save_iter, field_save_iter, damping_array


@nb.njit()
def main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag,                      \
              B, E_int, E_half, q_dens, q_dens_adv, Ji, ni, nu,          \
              Ve, Te, temp3De, temp3Db, temp1D, old_particles, old_fields,\
              damping_array, qq, DT, max_inc, part_save_iter, field_save_iter):
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
    qq, DT, max_inc, part_save_iter, field_save_iter, damping_array \
    = check_timestep(pos, vel, B, E_int, q_dens, Ie, W_elec, Ib, W_mag, temp3De, \
                     qq, DT, max_inc, part_save_iter, field_save_iter, idx, damping_array)
    
    # Move particles, collect moments
    particles.advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                            B, E_int, DT, q_dens_adv, Ji, ni, nu, temp1D)
    
    # Average N, N + 1 densities (q_dens at N + 1/2)
    q_dens *= 0.5
    q_dens += 0.5 * q_dens_adv
    
    # Push B from N to N + 1/2
    fields.push_B(B, E_int, temp3Db, DT, qq, damping_array, half_flag=1)
    
    # Calculate E at N + 1/2
    fields.calculate_E(B, Ji, q_dens, E_half, Ve, Te, temp3De, temp3Db, temp1D)
    
    
    ###################################
    ### PREDICTOR CORRECTOR SECTION ###
    ###################################

    # Store old values
    old_particles[0:3 , :] = pos
    old_particles[3:6 , :] = vel
    old_particles[6   , :] = Ie
    old_particles[7:10, :] = W_elec
    
    old_fields[:,   0:3]  = B
    old_fields[:NC, 3:6]  = Ji
    old_fields[:NC, 6:9]  = Ve
    old_fields[:NC,   9]  = Te
    
    # Predict fields
    E_int *= -1.0
    E_int +=  2.0 * E_half
    
    fields.push_B(B, E_int, temp3Db, DT, qq, damping_array, half_flag=0)

    # Advance particles to obtain source terms at N + 3/2
    particles.advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                            B, E_int, DT, q_dens, Ji, ni, nu, temp1D, pc=1)
    
    q_dens *= 0.5;    q_dens += 0.5 * q_dens_adv
    
    # Compute predicted fields at N + 3/2
    fields.push_B(B, E_int, temp3Db, DT, qq + 1, damping_array, half_flag=1)
    fields.calculate_E(B, Ji, q_dens, E_int, Ve, Te, temp3De, temp3Db, temp1D)
    
    # Determine corrected fields at N + 1 
    E_int *= 0.5;    E_int += 0.5 * E_half

    # Restore old values: [:] allows reference to same memory (instead of creating new, local instance)
    pos[:]    = old_particles[0:3 , :]
    vel[:]    = old_particles[3:6 , :]
    Ie[:]     = old_particles[6   , :]
    W_elec[:] = old_particles[7:10, :]
    B[:]      = old_fields[:,   0:3]
    Ji[:]     = old_fields[:NC, 3:6]
    Ve[:]     = old_fields[:NC, 6:9]
    Te[:]     = old_fields[:NC,   9]
    
    fields.push_B(B, E_int, temp3Db, DT, qq, damping_array, half_flag=0)   # Advance the original B

    q_dens[:] = q_dens_adv

    return qq, DT, max_inc, part_save_iter, field_save_iter
