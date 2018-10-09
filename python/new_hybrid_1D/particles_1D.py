# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numpy as np
import numba as nb

from simulation_parameters_1D  import N, dx, xmax, xmin, charge, mass
from sources_1D                import assign_weighting, calc_left_node


@nb.njit(cache=True)
def velocity_update(part, B, E, dt, W_in):  # Based on Appendix A of Ch5 : Hybrid Codes by Winske & Omidi.
    '''Updates the velocity of the particles using the explicit solution given in Chapter 5, Appendix A of Hybrid Codes
    by Winske & Omidi.

    INPUT:
        part -- Particle array containing velocities to be updated
        B    -- Magnetic field on simulation grid
        E    -- Electric field on simulation grid
        dt   -- Simulation time cadence
        W    -- Weighting factor of particles to rightmost node

    OUTPUT:
        part -- Returns particle array with updated velocities
    '''
    Wb = assign_weighting(part[0, :], part[1, :], 0)        # Magnetic field weighting

    for n in range(N):
        vn = part[3:6, n]                # Existing particle velocity

        # Weighted E & B fields at particle location (node) - Weighted average of two nodes on either side of particle
        I   = int(part[1,n])             # Nearest (leftmost) node, I
        Ib  = int(part[0, n] / dx)       # Nearest (leftmost) magnetic node
        We  = W_in[n]                    # E-field weighting
        idx = int(part[2, n])            # Particle species index

        E_p = E[I,  0:3] * (1 - We   ) + E[I  + 1, 0:3] * We
        B_p = B[Ib, 0:3] * (1 - Wb[n]) + B[Ib + 1, 0:3] * Wb[n]

        # Intermediate calculations
        h = (charge[idx] * dt) / mass[idx]
        f = 1. - (h**2) / 2. * (B_p[0]**2 + B_p[1]**2 + B_p[2]**2 )
        g = h / 2. * (B_p[0]*vn[0] + B_p[1]*vn[1] + B_p[2]*vn[2])
        v0 = vn + (h/2.)*E_p

        # Velocity push
        part[3,n] = f * vn[0] + h * ( E_p[0] + g * B_p[0] + (v0[1]*B_p[2] - v0[2]*B_p[1]) )
        part[4,n] = f * vn[1] + h * ( E_p[1] + g * B_p[1] - (v0[0]*B_p[2] - v0[2]*B_p[0]) )
        part[5,n] = f * vn[2] + h * ( E_p[2] + g * B_p[2] + (v0[0]*B_p[1] - v0[1]*B_p[0]) )
    return part


@nb.njit(cache=True)
def boris_velocity_update(part, B, E, dt, W):
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
    Wb = assign_weighting(part[0, :], part[1, :], 0)        # Magnetic field weighting

    for n in range(N):
        v_minus = np.zeros(3)                               # First velocity
        v_prime = np.zeros(3)                               # Rotation velocity
        v_plus  = np.zeros(3)                               # Second velocity

        I   = int(part[1,n])                                # Nearest (leftmost) node, I
        Ib  = int(part[0, n] / dx)                          # Nearest (leftmost) magnetic node
        We  = W[n]                                          # E-field weighting
        idx = int(part[2, n])                               # Particle species identifier

        Ep = E[I,  0:3] * (1 - We   ) + E[I  + 1, 0:3] * We                 # E-field at particle location
        Bp = B[Ib, 0:3] * (1 - Wb[n]) + B[Ib + 1, 0:3] * Wb[n]              # B-field at particle location

        T = (charge[idx] * Bp / mass[idx]) * dt / 2.                        # Boris variable
        S = 2.*T / (1. + np.sqrt(T[0] ** 2 + T[1] ** 2 + T[2] ** 2))        # Boris variable

        # Actual Boris Method
        v_minus    = part[3:6, n] + charge[idx] * Ep * dt / (2 * mass[idx])

        v_prime[0] = v_minus[0] + (v_minus[1] * T[2] - v_minus[2] * T[1])  #removed multiplicative from second term: (charge[idx] * dt / (2 * mass[idx]))
        v_prime[1] = v_minus[1] - (v_minus[0] * T[2] - v_minus[2] * T[0])
        v_prime[2] = v_minus[2] + (v_minus[0] * T[1] - v_minus[1] * T[0])

        v_plus[0]  = v_minus[0] + (v_prime[1] * S[2] - v_prime[2] * S[1])
        v_plus[1]  = v_minus[1] - (v_prime[0] * S[2] - v_prime[2] * S[0])
        v_plus[2]  = v_minus[2] + (v_prime[0] * S[1] - v_prime[1] * S[0])

        part[3:6, n] = v_plus + charge[idx] * Ep * dt / (2 * mass[idx])
    return part


@nb.njit(cache=True)
def position_update(part, dt):
    '''Updates the position of the particles using x = x0 + vt. Also updates particle leftmost node and weighting.

    INPUT:
        part -- Particle array with positions to be updated
        dt   -- Time cadence of simulation

    OUTPUT:
        part -- Particle array with updated positions
        W    -- Particle (E-field) node weightings
    '''
    part[0, :] += part[3, :] * dt                       # Update position: x = x0 + vt

    for ii in range(N):                                 # Check particle boundary conditions
        if part[0, ii] < xmin:
            part[0, ii] = xmax + part[0,ii]

        if part[0, ii] > xmax:
            part[0, ii] = part[0,ii] - xmax

    part[1, :] = calc_left_node(part[0, :])
    W          = assign_weighting(part[0, :], part[1, :], 1)
    return part, W
