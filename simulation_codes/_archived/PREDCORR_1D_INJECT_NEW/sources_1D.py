# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""
import numba as nb
from simulation_parameters_1D import ND, NX, Nj, n_contr, charge, q, ne, min_dens, particle_boundary


@nb.njit()
def deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu):
    '''Collect number and velocity moments in each cell, weighted by their distance
    from cell nodes.

    INPUT:
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier

    OUTPUT:
        ni     -- Species number moment array(size, Nj)
        nui    -- Species velocity moment array (size, Nj)
        
    13/03/2020 :: Modified to ignore contributions from particles with negative
                    indices (i.e. "deactivated" particles)
                    
    07/05/2020 :: Modified to allow contributions from negatively indexed particles
                    if reflection is enabled - these *were* hot particles, now
                    counted as cold.
    '''
    for ii in nb.prange(vel.shape[1]):
        I   = Ie[ii]
        sp  = idx[ii]
        
        if sp >= 0 or particle_boundary == 1:
            for kk in range(3):
                nu[I,     sp, kk] += W_elec[0, ii] * vel[kk, ii]
                nu[I + 1, sp, kk] += W_elec[1, ii] * vel[kk, ii]
                nu[I + 2, sp, kk] += W_elec[2, ii] * vel[kk, ii]
            
            ni[I,     sp] += W_elec[0, ii]
            ni[I + 1, sp] += W_elec[1, ii]
            ni[I + 2, sp] += W_elec[2, ii]
    return


@nb.njit()
def collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu, temp1D):
    '''
    Moment (charge/current) collection function.

    INPUT:
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier
        
    OUTPUT:
        q_dens -- Charge  density
        Ji     -- Current density
        
    TERTIARY:
        ni
        nu
        temp1D
        
    Source terms in damping region set to be equal to last valid cell value. 
    Smoothing routines deleted (can be found in earlier versions) since TSC 
    weighting effectively also performs smoothing.
    
    07/05/2020 :: Changed damped source terms to give zero gradient at ND-NX
                    boundary. Remaining gradient probably a particle
                    initialization error.
    '''
    q_dens *= 0.
    Ji     *= 0.
    ni     *= 0.
    nu     *= 0.

    deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu)
    
    # Sum contributions across species
    for jj in range(Nj):
        q_dens  += ni[:, jj] * n_contr[jj] * charge[jj]

        for kk in range(3):
            Ji[:, kk] += nu[:, jj, kk] * n_contr[jj] * charge[jj]

    # Mirror source term contributions at edge back into domain: Simulates having
    # some sort of source on the outside of the physical space boundary.
    q_dens[ND]          += q_dens[ND - 1]
    q_dens[ND + NX - 1] += q_dens[ND + NX]
    
    for ii in range(3):
        # Mirror source term contributions
        Ji[ND, ii]          += Ji[ND - 1, ii]
        Ji[ND + NX - 1, ii] += Ji[ND + NX, ii]

        # Set damping cell source values (zero gradient)
        Ji[:ND, ii] = Ji[ND + 1, ii]
        Ji[ND+NX:, ii]  = Ji[ND+NX - 2, ii]
        
    # Set damping cell source values
    q_dens[:ND]    = q_dens[ND + 1]
    q_dens[ND+NX:] = q_dens[ND+NX - 2]
        
    # Set density minimum
    for ii in range(q_dens.shape[0]):
        if q_dens[ii] < min_dens * ne * q:
            q_dens[ii] = min_dens * ne * q
    return













# OLD CODE

# =============================================================================
#         # Set damping cell source values (zero second derivative)
#         Ji[:ND, ii]    = 2*Ji[ND,      ii] - Ji[ND + 1     , ii]
#         Ji[ND+NX:, ii] = 2*Ji[ND+NX-1, ii] - Ji[ND + NX - 2, ii]
#         
#     # Set damping cell source values
#     q_dens[:ND]    = 2*q_dens[ND]      - q_dens[ND + 1]
#     q_dens[ND+NX:] = 2*q_dens[ND+NX-1] - q_dens[ND+NX-2]
# =============================================================================
# =============================================================================
#         # Set damping cell source values (copy last)
#         Ji[:ND, ii] = Ji[ND, ii]
#         Ji[ND+NX:]  = Ji[ND+NX-1]
#         
#     # Set damping cell source values
#     q_dens[:ND]    = q_dens[ND]
#     q_dens[ND+NX:] = q_dens[ND+NX-1]
# =============================================================================