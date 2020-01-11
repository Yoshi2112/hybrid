# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb
import numpy as np

from   simulation_parameters_1D  import ND, dx, xmax, xmin, charge, mass, kB, Tpar, Tper, drift_v, renew_particles
from   sources_1D                import collect_moments

@nb.njit()
def advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                  B, E, DT, q_dens_adv, Ji, ni, nu, temp1D, pc=0):
    '''
    Helper function to group the particle advance and moment collection functions
    ''' 
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)
    velocity_update(vel, Ie, W_elec, Ib, W_mag, idx, B, E, DT)
    position_update(pos, vel, idx, DT, Ie, W_elec)  
    collect_moments(vel, Ie, W_elec, idx, q_dens_adv, Ji, ni, nu, temp1D)
    return


@nb.njit()
def assign_weighting_TSC(pos, I, W, E_nodes=True):
    '''Triangular-Shaped Cloud (TSC) weighting scheme used to distribute particle densities to
    nodes and interpolate field values to particle positions. Ref. Lipatov? Or Birdsall & Langdon?

    INPUT:
        pos     -- particle positions (x)
        I       -- Leftmost (to nearest) nodes. Output array
        W       -- TSC weights, 3xN array starting at respective I
        E_nodes -- True/False flag for calculating values at electric field
                   nodes (grid centres) or not (magnetic field, edges)
    
    The maths effectively converts a particle position into multiples of dx (i.e. nodes),
    rounded (to get nearest node) and then offset to account for E/B grid staggering and 
    to get the leftmost node. This is then offset by the damping number of nodes, ND. The
    calculation for weighting (dependent on delta_left).
    
    NOTE: The addition of `epsilon' prevents banker's rounding due to precision limits. This
          is the easiest way to get around it.
    '''
    Np         = pos.shape[0]
    epsilon    = 1e-15
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 0.0
    
    for ii in np.arange(Np):
        I[ii]       = int(round(pos[ii] / dx - grid_offset + epsilon) - 1.0) + ND
        delta_left  = I[ii] - (pos[ii] + epsilon) / dx + grid_offset - ND
    
        W[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))
        W[1, ii] = 0.75 - np.square(delta_left + 1.)
        W[2, ii] = 1.0  - W[0, ii] - W[1, ii]
    return


@nb.njit()
def velocity_update(vel, Ie, W_elec, Ib, W_mag, idx, B, E, dt):
    '''
    Interpolates the fields to the particle positions using TSC weighting, then
    updates velocities using a Boris particle pusher.
    Based on Birdsall & Langdon (1985), pp. 59-63.

    INPUT:
        part -- Particle array containing velocities to be updated
        B    -- Magnetic field on simulation grid
        E    -- Electric field on simulation grid
        dt   -- Simulation time cadence
        W    -- Weighting factor of particles to rightmost node

    OUTPUT:
        None -- vel array is mutable (I/O array)
        
    Notes: Still not sure how to parallelise this: There are a lot of array operations
    Probably need to do it more algebraically? Find some way to not create those temp arrays.
    Removed the "cross product" and "field interpolation" functions because I'm
    not convinced they helped.
    '''
    for ii in nb.prange(vel.shape[1]):
        qmi = 0.5 * dt * charge[idx[ii]] / mass[idx[ii]]                    # Charge-to-mass ration for ion of species idx[ii]
        
        Ep = E[Ie[ii]    , 0:3] * W_elec[0, ii]                             \
           + E[Ie[ii] + 1, 0:3] * W_elec[1, ii]                             \
           + E[Ie[ii] + 2, 0:3] * W_elec[2, ii]                             # Vector E-field at particle location
        
        Bp = B[Ib[ii]    , 0:3] * W_mag[0, ii]                              \
           + B[Ib[ii] + 1, 0:3] * W_mag[1, ii]                              \
           + B[Ib[ii] + 2, 0:3] * W_mag[2, ii]                              # Vector B-field at particle location
        
        T = qmi * Bp                                                        # Vector Boris variable
        S = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)                 # Vector Boris variable

        v_minus    = vel[:, ii] + qmi * Ep
        
        v_prime    = np.zeros(3)
        v_prime[0] = v_minus[0] + v_minus[1] * T[2] - v_minus[2] * T[1]
        v_prime[1] = v_minus[1] + v_minus[2] * T[0] - v_minus[0] * T[2]
        v_prime[2] = v_minus[2] + v_minus[0] * T[1] - v_minus[1] * T[0]
                
        v_plus     = np.zeros(3)
        v_plus[0]  = v_minus[0] + v_prime[1] * S[2] - v_prime[2] * S[1]
        v_plus[1]  = v_minus[1] + v_prime[2] * S[0] - v_prime[0] * S[2]
        v_plus[2]  = v_minus[2] + v_prime[0] * S[1] - v_prime[1] * S[0]

        vel[:, ii] = v_plus +  qmi * Ep
    return


@nb.njit()
def position_update(pos, vel, idx, dt, Ie, W_elec):
    '''Updates the position of the particles using x = x0 + vt. 
    Also updates particle nearest node and weighting.

    INPUT:
        part   -- Particle array with positions to be updated
        dt     -- Time cadence of simulation

    OUTPUT:
        pos    -- Particle updated positions
        W_elec -- (0) Updated nearest E-field node value and (1-2) left/centre weights
        
    Need to work out where I want these new particles to enter the simulation space
    ATM they're just swapping ends, but I probably want them to go back into the middle
    Screw it, let's just do that change.
    '''
    for ii in nb.prange(pos.shape[0]):
        pos[ii] += vel[0, ii] * dt
        
        if pos[ii] < xmin:
            #pos[ii] += xmax
            new_flag = 1
            
        if pos[ii] > xmax:
            #pos[ii] -= xmax
            new_flag = 1

        if new_flag == 1 and renew_particles == True:
            # Re-initialize temperature. "New" particle. 
            # Should be able to disable this functionality by replacing
            # if statement with "False"
            pos[ii]    = xmax / 2.          # This might give me crazy density errors at some point
            
            sp         = idx[ii]
            vel[0, ii] = np.random.normal(0, np.sqrt(kB *  Tpar[sp] /  mass[sp]) +  drift_v[sp])
            vel[1, ii] = np.random.normal(0, np.sqrt(kB *  Tper[sp] /  mass[sp]))
            vel[2, ii] = np.random.normal(0, np.sqrt(kB *  Tper[sp] /  mass[sp]))
        elif new_flag == 1 and renew_particles == False:
            # Terribly inefficient. Maybe I can use the idx flag with a -1 to indicate particles
            # that have been "turned off"? Their species probably doesn't matter too much after
            # they've left the simulation space. Also, might this screw with node calc. equations?
            pos[ii]    = - xmax
            vel[0, ii] = 0.
            vel[1, ii] = 0.
            vel[2, ii] = 0.
            
    assign_weighting_TSC(pos, Ie, W_elec)
    return
