# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb

from simulation_parameters_1D  import N, dx, xmax, xmin, charge, mass, Nj, idx_bounds
from auxilliary_1D             import cross_product_single

@nb.njit()
def calc_left_node(pos):
    node = pos / dx + 0.5                           # Leftmost (E-field) node, I
    return node.astype(nb.int32)


@nb.njit()
def assign_weighting(xpos, I, BE):
    '''Linear weighting scheme used to interpolate particle source term contributions to
    nodes and field contributions to particle positions.

    INPUT:
        xpos -- Particle positions
        I    -- Particle rightmost nodes
        BE   -- Flag: Weighting factor for Magnetic (0) or Electric (1) field node

    Notes: Last term displaces weighting factor by half a cell by adding 0.5 for a retarded
    electric field grid (i.e. all weightings are 0.5 larger due to further distance from
    left node, and this weighting applies to the I + 1 node.
    '''
    W_o = (xpos / dx) - I + (BE / 2.)
    return W_o


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
        
    DOES SINGLE PARTICLE (NEW FUNCTION)
    '''
    T = (charge[idx] * Bp / mass[idx]) * dt / 2.                        # Boris variable
    S = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)                 # Boris variable

    v_minus    = v0 + charge[idx] * Ep * dt / (2. * mass[idx])
    v_prime    = v_minus + cross_product_single(v_minus, T)
    v_plus     = v_minus + cross_product_single(v_prime, S)
 
    v0         = v_plus + charge[idx] * Ep * dt / (2. * mass[idx])
    return v0


@nb.njit()
def velocity_update(part, B, E, dt):
    '''Updates the velocities of the particles in the simulation using a Boris particle pusher, as detailed
    in Birdsall & Langdon (1985),  59-63.

    INPUT:
        part -- Particle array containing velocities to be updated
        B    -- Magnetic field on simulation grid
        E    -- Electric field on simulation grid
        dt   -- Simulation time cadence
        W    -- Weighting factor of particles to rightmost node

    OUTPUT:
        part -- Returns particle array with updated velocities
    '''
    W_magnetic = assign_weighting(part[0, :], part[1, :], 0)    # Magnetic field weighting

    for jj in range(Nj):
        for nn in range(idx_bounds[jj, 0], idx_bounds[jj, 1]):
            I   = int(part[1, nn])                              # Nearest (leftmost) node, I
            Ib  = int(part[0, nn] / dx)                         # Nearest (leftmost) magnetic node
            We  = part[2, nn]                                   # E-field weighting
            Wb  = W_magnetic[nn]                                # B-field weighting
    
            Ep = E[I,  0:3] * (1 - We) + E[I  + 1, 0:3] * We                    # E-field at particle location
            Bp = B[Ib, 0:3] * (1 - Wb) + B[Ib + 1, 0:3] * Wb                    # B-field at particle location
 
            part[3:6, nn] = boris_algorithm(part[3:6, nn], Bp, Ep, dt, jj)      # Advance particle velocity
    return part


@nb.njit()
def position_update(part, dt):
    '''Updates the position of the particles using x = x0 + vt. Also updates particle leftmost node and weighting.

    INPUT:
        part -- Particle array with positions to be updated
        dt   -- Time cadence of simulation

    OUTPUT:
        part -- Particle array with updated positions
    '''
    part[0, :] += part[3, :] * dt                       # Update position: x = x0 + vt

    for ii in range(N):                                 # Check particle boundary conditions
        if part[0, ii] < xmin:
            part[0, ii] = xmax + part[0,ii]

        if part[0, ii] > xmax:
            part[0, ii] = part[0,ii] - xmax
            
    part[1, :] = calc_left_node(part[0, :])
    part[2, :] = assign_weighting(part[0, :], part[1, :], 1)
    return part


def two_part_velocity_update(part, B, E, dt):
    ''' Backup velocity update from Matthews (1994), just in case Boris isn't compatible with it.
    
    Advances velocity full timestep by first approximating half timestep.
    '''
    W_magnetic = assign_weighting(part[0, :], part[1, :], 0)    # Magnetic field weighting

    for jj in range(Nj):
        for nn in range(idx_bounds[jj, 0], idx_bounds[jj, 1]):
            I   = int(part[1, nn])                              # Nearest (leftmost) node, I
            Ib  = int(part[0, nn] / dx)                         # Nearest (leftmost) magnetic node
            We  = part[2, nn]                                   # E-field weighting
            Wb  = W_magnetic[nn]                                # B-field weighting
            idx = jj                                            # Particle species identifier
    
            Ep = E[I,  0:3] * (1 - We) + E[I  + 1, 0:3] * We                    # E-field at particle location
            Bp = B[Ib, 0:3] * (1 - Wb) + B[Ib + 1, 0:3] * Wb                    # B-field at particle location
            
            fac        = 0.5*dt*charge[idx]/mass[idx]
            v_half     = part[3:6] + fac*(Ep + cross_product_single(part[3:6], Bp))
            
            part[3:6] += 2*fac*(Ep + cross_product_single(v_half, Bp))
    
    return part