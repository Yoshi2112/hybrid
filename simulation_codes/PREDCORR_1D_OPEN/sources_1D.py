# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""
import numba as nb

from simulation_parameters_1D import ND, NX, Nj, n_contr, charge, smooth_sources, q, ne, min_dens

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
    '''
    for ii in nb.prange(vel.shape[1]):
        I   = Ie[ii]
        sp  = idx[ii]
    
        for kk in range(3):
            nu[I,     sp, kk] += W_elec[0, ii] * vel[kk, ii]
            nu[I + 1, sp, kk] += W_elec[1, ii] * vel[kk, ii]
            nu[I + 2, sp, kk] += W_elec[2, ii] * vel[kk, ii]
        
        ni[I,     sp] += W_elec[0, ii]
        ni[I + 1, sp] += W_elec[1, ii]
        ni[I + 2, sp] += W_elec[2, ii]
    return

import pdb
@nb.njit()
def collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu, temp1D, mirror=True):
    '''
    Moment (charge/current) collection function.

    INPUT:
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier
        
    OUTPUT:
        rho_c  -- Charge  density
        Ji     -- Current density
        
    Source terms in damping region set to be equal to last valid cell value
    '''
    q_dens *= 0.
    Ji     *= 0.
    ni     *= 0.
    nu     *= 0.

    deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu)

    if smooth_sources == 1:
        for jj in range(Nj):
            smooth(ni[:, jj], temp1D)
            
            for kk in range(3):
                smooth(nu[:,  jj, kk], temp1D)

    for jj in range(Nj):
        q_dens  += ni[:, jj] * n_contr[jj] * charge[jj]

        for kk in range(3):
            Ji[:, kk] += nu[:, jj, kk] * n_contr[jj] * charge[jj]

    if mirror == True:
        # Mirror source term contributions at edge back into domain: Simulates having
        # some sort of source on the outside of the physical space boundary.
        # Is this going to cause "rippling" when particles disappear from simulation domain?
        q_dens[ND]          += q_dens[ND - 1]
        q_dens[ND + NX - 1] += q_dens[ND + NX]
    
        # Set damping cell source values
        q_dens[:ND]    = q_dens[ND]
        q_dens[ND+NX:] = q_dens[ND+NX-1]
        
        for ii in range(3):
            Ji[ND, ii]          += Ji[ND - 1, ii]
            Ji[ND + NX - 1, ii] += Ji[ND + NX, ii]
        
            Ji[:ND, ii] = Ji[ND, ii]
            Ji[ND+NX:]  = Ji[ND+NX-1]
        
# =============================================================================
#     # Set density minimum
#     for ii in range(q_dens.shape[0]):
#         if q_dens[ii] < min_dens * ne * q:
#             q_dens[ii] = min_dens * ne * q
# =============================================================================
    return


@nb.njit()
def smooth(arr, temp1D):
    '''Smoothing function: Applies Gaussian smoothing routine across adjacent cells. 
    Assummes no contribution from ghost cells. Designed for source terms.
    
    HOW TO DEAL WITH OPEN BOUNDARIES? If smoothed before damping regions filled, 
    then they'll be overwritten anyway (and there will be a loss of source from
    the boundaries). Leaving the boundaries unchanged should be fine.
    
    Some weird stuff going on with memory management: Does it create a new numpy array
    or not? Or new numpy instance pointing to same memory locations? Passing
    slices as function arguments seems weird. But the function works, it just might not be efficient.
    '''
    nc      = arr.shape[0]
    temp1D *= 0
             
    for ii in nb.prange(1, nc - 1):
        temp1D[ii - 1] += 0.25*arr[ii]
        temp1D[ii]     += 0.50*arr[ii]
        temp1D[ii + 1] += 0.25*arr[ii]
    
    # Output smoothed array
    arr[:] = temp1D[:]
    return