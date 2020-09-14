# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb
import numpy as np

from simulation_parameters_1D  import N, dx, xmax, xmin, charge, mass, e_resis, q, qm_ratios
import auxilliary_1D as aux


@nb.njit()
def two_step_algorithm(v0, Bp, Ep, dt, idx):
    fac        = 0.5*dt*charge[idx]/mass[idx]
    v_half     = v0 + fac*(Ep + aux.cross_product_single(v0, Bp))
    v0        += 2*fac*(Ep + aux.cross_product_single(v_half, Bp))
    return v0


@nb.njit()
def assign_weighting_TSC(pos, I, W, E_nodes=True):
    '''Triangular-Shaped Cloud (TSC) weighting scheme used to distribute particle densities to
    nodes and interpolate field values to particle positions.

    INPUT:
        pos  -- particle positions (x)
        BE   -- Flag: Weighting factor for Magnetic (0) or Electric (1) field node
        
    OUTPUT:
        weights -- 3xN array consisting of leftmost (to the nearest) node, and weights for -1, 0 TSC nodes
        
    NOTE: The addition of `epsilon' in left_node prevents banker's rounding in left_node due to precision limits.
    '''
    Np         = pos.shape[0]
    epsilon    = 1e-15
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 1.0
    
    for ii in np.arange(Np):
        I[ii]       = int(round(pos[ii] / dx + grid_offset + epsilon) - 1.0)
        delta_left  = I[ii] - (pos[ii] + epsilon) / dx - grid_offset
    
        W[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))
        W[1, ii] = 0.75 - np.square(delta_left + 1.)
        W[2, ii] = 1.0  - W[0, ii] - W[1, ii]
    return


@nb.njit()
def boris_algorithm(v0, Bp, Ep, dt, idx):
    '''Updates the velocities of the particles in the simulation using a Boris particle pusher, as detailed
    in Birdsall & Langdon (1985),  59-63.

    INPUT:
        v0   -- Original particle velocity
        B    -- Magnetic field value at particle  position
        E    -- Electric field value at particle position
        dt   -- Simulation time cadence
        idx  -- Particle species identifier

    OUTPUT:
        v0   -- Updated particle velocity (overwrites input value on return)
        
    Note: Designed for single particle call (doesn't use array-specific operations)
    '''
    T = (charge[idx] * Bp / mass[idx]) * dt / 2.                        # Boris variable
    S = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)                 # Boris variable

    v_minus    = v0 + charge[idx] * Ep * dt / (2. * mass[idx])
    v_prime    = v_minus + aux.cross_product_single(v_minus, T)
    v_plus     = v_minus + aux.cross_product_single(v_prime, S)
 
    v0         = v_plus + charge[idx] * Ep * dt / (2. * mass[idx])
    return v0


@nb.njit()
def interpolate_forces_to_particle(E, B, J, Ie, W_elec, Ib, W_mag, idx):
    '''
    Same as previous function, but also interpolates current to particle position to return
    an electric field modified by electron resistance
    '''
    Ep = E[Ie    , 0:3] * W_elec[0]                 \
       + E[Ie + 1, 0:3] * W_elec[1]                 \
       + E[Ie + 2, 0:3] * W_elec[2]                 # E-field at particle location
    
    Bp = B[Ib    , 0:3] * W_mag[0]                  \
       + B[Ib + 1, 0:3] * W_mag[1]                  \
       + B[Ib + 2, 0:3] * W_mag[2]                  # B-field at particle location
   
    Jp = J[Ie    , 0:3] * W_elec[0]                 \
       + J[Ie + 1, 0:3] * W_elec[1]                 \
       + J[Ie + 2, 0:3] * W_elec[2]                 # Current at particle location
       
    Ep -= (charge[idx] / q) * e_resis * Jp          # "Effective" E-field accounting for electron resistance
    return Ep, Bp


@nb.njit()
def velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, J, dt):
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
        vel  -- Returns particle array with updated velocities
    '''
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)     # Magnetic field weighting
    
    for ii in np.arange(N):
        Ep, Bp     = interpolate_forces_to_particle(E, B, J, Ie[ii], W_elec[:, ii], Ib[ii], W_mag[:, ii], idx[ii])
        vel[:, ii] = boris_algorithm(   vel[:, ii], Bp, Ep, dt, idx[ii])
    return


@nb.njit()
def velocity_update_vectorised(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E, v_prime, S, T, qmi, DT):
    '''
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
    # Add B0 to these arrays, except that B0 doesn't do anything except in the JxB sense.
    Bp *= 0
    Ep *= 0
    
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)                       # Calculate magnetic node weights
    
    for ii in range(vel.shape[1]):
        qmi[ii] = 0.5 * DT * qm_ratios[idx[ii]]                               # q/m for ion of species idx[ii]
        for jj in range(3):                                                   # Nodes
            for kk in range(3):                                               # Components
                Ep[kk, ii] += E[Ie[ii] + jj, kk] * W_elec[jj, ii]             # Vector E-field  at particle location
                Bp[kk, ii] += B[Ib[ii] + jj, kk] * W_mag[ jj, ii]             # Vector b1-field at particle location

    vel[:, :] += qmi[:] * Ep[:, :]                                            # First E-field half-push IS NOW V_MINUS

    T[:, :] = qmi[:] * Bp[:, :]                                               # Vector Boris variable
    S[:, :] = 2.*T[:, :] / (1. + T[0, :] ** 2 + T[1, :] ** 2 + T[2, :] ** 2)  # Vector Boris variable
    
    v_prime[0, :] = vel[0, :] + vel[1, :] * T[2, :] - vel[2, :] * T[1, :]     # Magnetic field rotation
    v_prime[1, :] = vel[1, :] + vel[2, :] * T[0, :] - vel[0, :] * T[2, :]
    v_prime[2, :] = vel[2, :] + vel[0, :] * T[1, :] - vel[1, :] * T[0, :]
            
    vel[0, :] += v_prime[1, :] * S[2, :] - v_prime[2, :] * S[1, :]
    vel[1, :] += v_prime[2, :] * S[0, :] - v_prime[0, :] * S[2, :]
    vel[2, :] += v_prime[0, :] * S[1, :] - v_prime[1, :] * S[0, :]
    
    vel[:, :] += qmi[:] * Ep[:, :]                                           # Second E-field half-push
    return


@nb.njit()
def position_update(pos, vel, dt):
    '''Updates the position of the particles using x = x0 + vt. 
    Also updates particle nearest node and weighting.

    INPUT:
        part   -- Particle array with positions to be updated
        dt     -- Time cadence of simulation

    OUTPUT:
        pos    -- Particle updated positions
        W_elec -- (0) Updated nearest E-field node value and (1-2) left/centre weights
    '''
    pos += vel[0, :] * dt
    
    for ii in nb.prange(pos.shape[0]):        
        if pos[ii] < xmin:
            pos[ii] += xmax
        elif pos[ii] > xmax:
            pos[ii] -= xmax
            
    Ie, W_elec = assign_weighting_TSC(pos)
    return pos, Ie, W_elec