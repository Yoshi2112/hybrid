# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:15:59 2017

@author: iarey
"""
import numba as nb
import numpy as np

from sources_2D import manage_E_grid_ghost_cells
import particles_2D as particles
import fields_2D as fields
import simulation_parameters_2D as const


@nb.njit()
def cross_product(A, B, C):
    '''
    Vector (cross) product between two vectors, A and B of same dimensions.

    INPUT:
        A, B -- 3D vectors (ndarrays)

    OUTPUT:
        C -- The resultant cross product with same dimensions as input vectors
    '''
    for ii in nb.prange(A.shape[0]):
        for jj in nb.prange(A.shape[1]):
            C[ii, jj, 0] = A[ii, jj, 1] * B[ii, jj, 2] - A[ii, jj, 2] * B[ii, jj, 1]
            C[ii, jj, 1] = A[ii, jj, 2] * B[ii, jj, 0] - A[ii, jj, 0] * B[ii, jj, 2]
            C[ii, jj, 2] = A[ii, jj, 0] * B[ii, jj, 1] - A[ii, jj, 1] * B[ii, jj, 0]
    return


@nb.njit()
def linear_Bgrid_to_Egrid_scalar(val, center):
    ''' 
    Interpolates vector cell edge values (i.e. B-grid quantities) to cell centers
    (i.e. E-grid quantities)
    SCALAR FORM FOR ARRAYS [ii, jj] FOR NODE (ii, jj)
    '''
    sx = val.shape[0]; sy = val.shape[1]
    
    for ii in range(1, sx-2):
        for jj in range(1, sy-2):
            center[ii, jj] = 0.25 * (val[ii, jj  ] + val[ii+1, jj  ] +
                                     val[ii, jj+1] + val[ii+1, jj+1])
    
    manage_E_grid_ghost_cells(center)
    return


@nb.njit()
def linear_Bgrid_to_Egrid_vector(val, center):
    ''' 
    Interpolates vector cell edge values (i.e. B-grid quantities) to cell centers 
    (i.e. E-grid nodes)
    VECTOR FORM FOR ARRAYS [ii, jj, kk] FOR NODE (ii, jj) AND COMPONENT kk
    '''
    sx = val.shape[0]; sy = val.shape[1]
    
    for kk in range(3):
        for ii in range(1, sx-2):
            for jj in range(1, sy-2):
                center[ii, jj, kk] = 0.25 * (val[ii, jj  , kk] + val[ii+1, jj  , kk] + 
                                             val[ii, jj+1, kk] + val[ii+1, jj+1, kk])
    
    manage_E_grid_ghost_cells(center)
    return


@nb.njit()
def check_timestep(pos, vel, B, E, q_dens, Ie, W_elec, Ib, W_mag, B_cent, \
                     qq, DT, max_inc, part_save_iter, field_save_iter, idx):
    
    linear_Bgrid_to_Egrid_vector(B, B_cent)
    B_tot           = np.sqrt(B_cent[:, :, 0] ** 2 + B_cent[:, :, 1] ** 2 + B_cent[:, :, 2] ** 2)

    dispfreq        = ((np.pi / const.dx) ** 2) * (B_tot / (const.mu0 * q_dens)).max()           # Dispersion frequency
    gyfreq          = const.high_rat  * np.abs(B_tot).max() / (2 * np.pi)      
    ion_ts          = const.orbit_res * 1./gyfreq
    
    if E[:, 0].max() != 0:
        elecfreq        = const.high_rat*(np.abs(E[:, 0] / vel.max()).max())               # Electron acceleration "frequency"
        Eacc_ts         = const.freq_res / elecfreq                            
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


@nb.njit()
def main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag,                      \
              B, E_int, E_half, q_dens, q_dens_adv, Ji, ni, nu,          \
              Ve, Te, temp3Da, temp3Db, temp3Dc, old_particles, old_fields,\
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
    = check_timestep(pos, vel, B, E_int, q_dens, Ie, W_elec, Ib, W_mag, temp3Da, \
                     qq, DT, max_inc, part_save_iter, field_save_iter, idx)
    
    # Move particles, collect moments
    particles.advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                            B, E_int, DT, q_dens_adv, Ji, ni, nu)
    
    # Average N, N + 1 densities (q_dens at N + 1/2)
    q_dens *= 0.5
    q_dens += 0.5 * q_dens_adv
    
    # Push B from N to N + 1/2
    fields.push_B(B, E_int, temp3Da, DT, qq, half_flag=1)
    
    # Calculate E at N + 1/2
    fields.calculate_E(B, Ji, q_dens, E_half, Ve, Te, temp3Da, temp3Db, temp3Dc)
    
    
    ###################################
    ### PREDICTOR CORRECTOR SECTION ###
    ###################################

    # Store old values:
    # ALSO HAVE THE OPTION TO RECALCULATE IE AND W_ELEC RATHER THAN STORING
    # CPU SPEED VS MEMORY STORAGE
    old_particles[0 : 2 , :] = pos
    old_particles[2 : 5 , :] = vel
    old_particles[5 : 7 , :] = Ie
    old_particles[7 : 10, :] = W_elec[0, :, :].T
    old_particles[10: 13, :] = W_elec[1, :, :].T
    
    old_fields[:, :, 0:3] = B
    old_fields[:, :, 3:6] = Ji
    old_fields[:, :, 6:9] = Ve
    old_fields[:, :,   9] = Te
    
    # Predict fields
    E_int *= -1.0
    E_int +=  2.0 * E_half
    
    fields.push_B(B, E_int, temp3Da, DT, qq, half_flag=0)

    # Advance particles to obtain source terms at N + 3/2
    particles.advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                            B, E_int, DT, q_dens, Ji, ni, nu)
    
    q_dens *= 0.5;    q_dens += 0.5 * q_dens_adv
    
    # Compute predicted fields at N + 3/2
    fields.push_B(B, E_int, temp3Da, DT, qq + 1, half_flag=1)
    fields.calculate_E(B, Ji, q_dens, E_int, Ve, Te, temp3Da, temp3Db, temp3Dc)
    
    # Determine corrected fields at N + 1 
    E_int *= 0.5;    E_int += 0.5 * E_half

    # Restore old values: [:] allows reference to same memory (instead of creating new, local instance)
    # THIS ALSO NEEDS TO BE FIXED: LOTS OF PARTICLE INFORMATION TO STORE
    pos[:]          = old_particles[0 : 2 , :]
    vel[:]          = old_particles[2 : 5 , :]
    Ie[:   ]        = old_particles[5 : 7 , :]
    W_elec[0, :, :] = old_particles[7 : 10, :].T
    W_elec[1, :, :] = old_particles[10: 13, :].T
    
    B[:, :]   = old_fields[:, :, 0:3]
    Ji[:, :]  = old_fields[:, :, 3:6]
    Ve[:, :]  = old_fields[:, :, 6:9]
    Te[:, :]  = old_fields[:, :,   9]
    
    fields.push_B(B, E_int, temp3Da, DT, qq, half_flag=0)                           # Advance the original B

    q_dens[:, :] = q_dens_adv

    return qq, DT, max_inc, part_save_iter, field_save_iter