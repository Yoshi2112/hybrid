# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb
import numpy as np

from   simulation_parameters_1D  import ND, dx, xmin, xmax, qm_ratios, B_eq, a
from   sources_1D                import collect_moments

from fields_1D import eval_B0x


@nb.njit()
def advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                  B, E, DT, q_dens_adv, Ji, ni, nu, temp1D, pc=0):
    '''
    Helper function to group the particle advance and moment collection functions
    ''' 
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)
    velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, DT)
    position_update(pos, vel, DT, Ie, W_elec)  
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
    epsil      = 1e-15
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 0.0
    
    particle_transform = xmax + (ND - grid_offset)*dx  + epsil  # Offset to account for E/B grid and damping nodes
    
    for ii in np.arange(Np):
        xp          = (pos[ii] + particle_transform) / dx       # Shift particle position >= 0
        I[ii]       = int(round(xp) - 1.0)                      # Get leftmost to nearest node
        delta_left  = I[ii] - xp                                # Distance from left node in grid units
    
        W[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))      # Get weighting factors
        W[1, ii] = 0.75 - np.square(delta_left + 1.)
        W[2, ii] = 1.0  - W[0, ii] - W[1, ii]
    return


@nb.njit()
def eval_B0_particle(x, v, qmi, b1):
    '''
    Calculates the B0 magnetic field at the position of a particle. Neglects B0_r
    and thus local cyclotron depends only on B0_x. Includes b1 in cyclotron, but
    since b1 < B0_r, maybe don't?
    
    Also, how accurate is this near the equator?
    '''
    B0_xp    = np.zeros(3)
    B0_xp[0] = eval_B0x(x)    
    
    b1t      = np.sqrt(b1[0] ** 2 + b1[1] ** 2 + b1[2] ** 2)
    l_cyc    = qmi * (B0_xp[0] + b1t)
    fac      = a * B_eq * x / l_cyc
    
    B0_xp[1] =  v[2] * fac
    B0_xp[2] = -v[1] * fac
    return B0_xp

@nb.njit()
def solve_quadratic(AA, BB, CC):
    '''
    DIAGNOSTIC FUNCTION
    Solves quadratic equation: Gives positive and negative solution for
    - b +/- sqrt(b ** 2 - 4ac) / 2a
    
    Only returns negative (of the +/- in quadratic) solution since the positive
    sign will always give a negative value (due to the signs of a, b, c and the
    squaring in the coefficients)
    '''
    if AA != 0:
        negative_soln = (- BB - np.sqrt(BB ** 2 - 4*AA*CC)) / (2*AA)
    else:
        negative_soln = 0.
    return negative_soln


@nb.njit()
def eval_B0_exact(x, v, qmi, approx):
    '''
    DIAGNOSTIC FUNCTION
    Solves (maybe?) the exact solution for By,Bz radial (coupled quadratic
    equations of their squares)
    
    Need to code this manually to separate out the two real solutions.
    
    The +det term of the quadratic always seems to be nan? Probably some 
    mathematical reason for this (b < 4ac? Check later)
    
    Just return positive results for each for now, but don't assume this will work
    generally - just for this particle.
    
    How to work out if it should be the positive or negative solution 
    (of the sqrt) taken? Use v_perp cross B?
    
    Currently using approximation as guide - not the best solution, but it works
    '''
    B0_particle = np.zeros(3)
    B0_x        = eval_B0x(x)
    K2          = (a * B_eq * x / qmi) ** 2     # Constant for calculation
    
    # Quadratic coefficients of R = By ** 2 :: 4 solutions
    ay = (v[1] ** 2 + v[2] ** 2) * (-K2)
    by = - v[2] ** 2 * K2 * B0_x ** 2
    cy = (v[2] ** 2 * K2) ** 2
    
    B0_y = np.sqrt(solve_quadratic(ay, by, cy))
    
    # Quadratic coefficients of Q = Bz ** 2 :: 4 solutions
    az = (v[1] ** 2 + v[2] ** 2) * (-K2)
    bz = - v[1] ** 2 * K2 * B0_x ** 2
    cz = (v[1] ** 2 * K2) ** 2
    
    B0_z = np.sqrt(solve_quadratic(az, bz, cz))
    
    # All three of these will always be positive
    # Need a way to work out how to set +/- status
    B0_particle[0] = B0_x
    B0_particle[1] = B0_y
    B0_particle[2] = B0_z
    
    if approx[1] < 0:
        B0_particle[1] *= -1.0
        
    if approx[2] < 0:
        B0_particle[2] *= -1.0
    
    return B0_particle


@nb.njit()
def velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, dt):
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
    for ii in nb.prange(vel.shape[1]):  
        qmi = 0.5 * dt * qm_ratios[idx[ii]]                                 # Charge-to-mass ration for ion of species idx[ii]

        Ep = E[Ie[ii]    , 0:3] * W_elec[0, ii]                             \
           + E[Ie[ii] + 1, 0:3] * W_elec[1, ii]                             \
           + E[Ie[ii] + 2, 0:3] * W_elec[2, ii]                             # Vector E-field at particle location

        Bp = B[Ib[ii]    , 0:3] * W_mag[0, ii]                              \
           + B[Ib[ii] + 1, 0:3] * W_mag[1, ii]                              \
           + B[Ib[ii] + 2, 0:3] * W_mag[2, ii]                              # b1 at particle location
        
        v_minus    = vel[:, ii] + qmi * Ep                                  # First E-field half-push
        
        Bp[0]  = 0                                                          # No wave b1 exists, removes B due to B-nodes (since they're analytic)
        B0_xp  = eval_B0_particle(pos[ii], v_minus, qm_ratios[idx[ii]], Bp) # B0 at particle location
                                                              
       
        # DIAGNOSTIC: DELETE (as well as return signature)
        B0_exact = eval_B0_exact(pos[ii], v_minus, qm_ratios[idx[ii]], B0_xp)
        ###
        
        # DIAGNOSTIC: Change this back to B0_xp
        Bp    += B0_exact                                                      # B  at particle location (total)
        
        T = qmi * Bp                                                        # Vector Boris variable
        S = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)                 # Vector Boris variable
        
        v_prime    = np.zeros(3)
        v_prime[0] = v_minus[0] + v_minus[1] * T[2] - v_minus[2] * T[1]     # Magnetic field rotation
        v_prime[1] = v_minus[1] + v_minus[2] * T[0] - v_minus[0] * T[2]
        v_prime[2] = v_minus[2] + v_minus[0] * T[1] - v_minus[1] * T[0]
                
        v_plus     = np.zeros(3)
        v_plus[0]  = v_minus[0] + v_prime[1] * S[2] - v_prime[2] * S[1]
        v_plus[1]  = v_minus[1] + v_prime[2] * S[0] - v_prime[0] * S[2]
        v_plus[2]  = v_minus[2] + v_prime[0] * S[1] - v_prime[1] * S[0]
        
        vel[:, ii] = v_plus +  qmi * Ep                                     # Second E-field half-push
    return Bp, B0_exact


@nb.njit()
def position_update(pos, vel, dt, Ie, W_elec, diag=False):
    '''Updates the position of the particles using x = x0 + vt. 
    Also updates particle nearest node and weighting.

    INPUT:
        part   -- Particle array with positions to be updated
        dt     -- Time cadence of simulation

    OUTPUT:
        pos    -- Particle updated positions
        W_elec -- (0) Updated nearest E-field node value and (1-2) left/centre weights
        
    Reflective boundaries to simulate the "open ends" that would have flux coming in from the ionosphere side.
    '''
    for ii in nb.prange(pos.shape[0]):
        pos[ii] += vel[0, ii] * dt

        if (pos[ii] <= xmin or pos[ii] >= xmax):
            vel[0, ii] *= -1.                   # Reflect velocity
            pos[ii]    += vel[0, ii] * dt       # Get particle back in simulation space
                
    assign_weighting_TSC(pos, Ie, W_elec)
    return
