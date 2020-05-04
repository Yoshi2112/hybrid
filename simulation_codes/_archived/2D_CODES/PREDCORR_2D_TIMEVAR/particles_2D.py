# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb
import numpy as np

# Package simulation dimensions into array? [[xmin, xmax, dx], [ymin, ymax, dy]]
from   simulation_parameters_2D  import dx, dy, xmax, xmin, ymax, ymin, charge, mass
from   sources_2D                import collect_moments

import pdb

@nb.njit()
def advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                  B, E, DT, q_dens_adv, Ji, ni, nu):
    '''
    Helper function to group the particle advance and moment collection functions
    ''' 
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)
    velocity_update(vel, Ie, W_elec, Ib, W_mag, idx, B, E, DT)
    position_update(pos, vel, DT, Ie, W_elec)  
    collect_moments(vel, Ie, W_elec, idx, q_dens_adv, Ji, ni, nu)
    return


@nb.njit()
def assign_weighting_TSC(pos, I, W, E_nodes=True):
    '''Triangular-Shaped Cloud (TSC) weighting scheme used to distribute particle densities to
    nodes and interpolate field values to particle positions.

    INPUT:
        pos  -- particle positions (x, y)
        BE   -- Flag: Weighting factor for Magnetic (0) or Electric (1) field node
        
    OUTPUT:
        I -- 2xN   : leftmost (to the nearest) node in x, y
        W -- 2xNx3 : Weights  array consisting of leftmost (to the nearest) node, 
        and weights for -1, 0 TSC nodes in x (0) and y (1) directions.
        Field interpolation and source term distribution for that node/particle combination
        is calculated as S = W(x) * W(y), where sum(W(x)*W(y)) = 1
        
    NOTE: The addition of `epsilon' in left_node prevents banker's rounding in left_node due to precision limits.
    '''
    Np         = pos.shape[1]
    epsilon    = 1e-15
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 1.0
    
    for jj in range(2):
        if jj == 0:
            dcell = dx
        elif jj == 1:
            dcell = dy
            
        for ii in np.arange(Np):
            I[jj, ii]   = int(round(pos[jj, ii] / dcell + grid_offset + epsilon) - 1.0)
            delta_left  = I[jj, ii] - (pos[jj, ii] + epsilon) / dcell - grid_offset
        
            W[jj, ii, 0] = 0.5  * np.square(1.5 - abs(delta_left))
            W[jj, ii, 1] = 0.75 - np.square(delta_left + 1.)
            W[jj, ii, 2] = 1.0  - W[jj, ii, 0] - W[jj, ii, 1] # This could be omitted and calculated on the spot: CPU vs. Memory importance
    return


@nb.njit()
def interpolate_fields_to_particle(xx, Ie, W_elec, Ib, W_mag, E, B):
    '''
    Interpolates fields at 9 nodes to each particle (Could be really slow: Go back to linear?)
    
    Check this though
    '''
    Ep = np.zeros(3); Bp = np.zeros(3)
    for ii in range(3):             # Nodes in x
        for jj in range(3):         # Nodes in y
            ndx = Ie[0, xx] + ii
            ndy = Ie[1, xx] + jj
            Ep += E[ndx, ndy, 0:3] * W_elec[0, xx, ii] * W_elec[1, xx, jj]     # Vector E-field at particle location
            
            ndx = Ib[0, xx] + ii
            ndy = Ib[1, xx] + jj
            Bp += B[ndx, ndy, 0:3] * W_mag[0, xx, ii] * W_mag[1, xx, jj]       # Vector B-field at particle location
    return Ep, Bp


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
        
        Ep, Bp = interpolate_fields_to_particle(ii, Ie, W_elec, Ib, W_mag, E, B)
        
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
def position_update(pos, vel, dt, Ie, W_elec):
    '''Updates the position of the particles using x = x0 + vt. 
    Also updates particle nearest node and weighting.

    INPUT:
        part   -- Particle array with positions to be updated
        dt     -- Time cadence of simulation

    OUTPUT:
        pos    -- Particle updated positions
        W_elec -- (0) Updated nearest E-field node value and (1-2) left/centre weights
    '''
    for ii in nb.prange(pos.shape[0]):
        pos[0, ii] += vel[0, ii] * dt
        pos[1, ii] += vel[1, ii] * dt
        
        if pos[0, ii] < xmin:
            pos[0, ii] += xmax

        if pos[0, ii] > xmax:
            pos[0, ii] -= xmax
            
        if pos[1, ii] < ymin:
            pos[1, ii] += ymax
            
        if pos[1, ii] > ymax:
            pos[1, ii] -= ymax
            
    assign_weighting_TSC(pos, Ie, W_elec)
    return
