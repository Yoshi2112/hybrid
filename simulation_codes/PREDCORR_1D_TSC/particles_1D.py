# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb
import numpy as np

from simulation_parameters_1D  import N, dx, xmax, xmin, charge, mass, do_parallel
import auxilliary_1D as aux
from sources_1D import collect_moments


@nb.njit()
def advance_particles_and_moments(pos, vel, Ie, W_elec, idx, B, E, DT):
    '''
    Helper function to group the particle advance and moment collection functions
    '''
    vel             = velocity_update(pos, vel, Ie, W_elec, idx, B, E, DT)
    pos, Ie, W_elec = position_update(pos, vel, DT)    
    q_dens, Ji      = collect_moments(vel, Ie, W_elec, idx)
    return q_dens, Ji


@nb.njit(parallel=do_parallel)
def assign_weighting_TSC(pos, E_nodes=True):
    '''Triangular-Shaped Cloud (TSC) weighting scheme used to distribute particle densities to
    nodes and interpolate field values to particle positions.

    INPUT:
        pos  -- particle positions (x)
        BE   -- Flag: Weighting factor for Magnetic (0) or Electric (1) field node
        
    OUTPUT:
        weights -- 3xN array consisting of leftmost (to the nearest) node, and weights for -1, 0 TSC nodes
    '''
    Np         = pos.shape[0]
    
    left_node  = np.zeros(Np,      dtype=np.uint16)
    weights    = np.zeros((3, Np), dtype=np.float64)
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 1.0
    
    for ii in nb.prange(Np):
        left_node[ii]  = int(round(pos[ii] / dx + grid_offset) - 1.0)
        delta_left     = left_node[ii] - pos[ii] / dx - grid_offset
    
        weights[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))
        weights[1, ii] = 0.75 - np.square(delta_left + 1.)
        weights[2, ii] = 1.0  - weights[0, ii] - weights[1, ii]
    return left_node, weights


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
def two_step_algorithm(v0, Bp, Ep, dt, idx):
    fac        = 0.5*dt*charge[idx]/mass[idx]
    v_half     = v0 + fac*(Ep + aux.cross_product_single(v0, Bp))
    v0        += 2*fac*(Ep + aux.cross_product_single(v_half, Bp))
    return v0


@nb.njit()
def interpolate_forces_to_particle(E, B, Ie, W_elec, Ib, W_mag, idx):
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
    return Ep, Bp


@nb.njit(parallel=do_parallel)
def velocity_update(pos, vel, Ie, W_elec, idx, B, E, dt):
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
    '''
    Ib, W_mag = assign_weighting_TSC(pos, E_nodes=False)     # Magnetic field weighting
    
    for ii in nb.prange(N):
        Ep, Bp     = interpolate_forces_to_particle(E, B, Ie[ii], W_elec[:, ii], Ib[ii], W_mag[:, ii], idx[ii])
        vel[:, ii] = boris_algorithm(vel[:, ii], Bp, Ep, dt, idx[ii])
    return


@nb.njit(parallel=do_parallel)
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
    for ii in nb.prange(pos.shape[0]):
        pos[ii] += vel[0, ii] * dt
        
        if pos[ii] < xmin:
            pos[ii] += xmax

        if pos[ii] > xmax:
            pos[ii] -= xmax
            
    Ie, W_elec = assign_weighting_TSC(pos)
    return pos, Ie, W_elec
