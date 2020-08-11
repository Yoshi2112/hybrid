# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""
import numba as nb
from simulation_parameters_1D import ND, NX, Nj, n_contr, charge, q, ne, min_dens,\
                                     xmin, xmax, dx, source_smoothing


@nb.njit()
def collect_velocity_moments(pos, vel, Ie, W_elec, idx, nu, Ji):
    '''
    Collect first and second moments of macroparticle velocity distributions.
    Calculate average current density at each grid point. 

    INPUT:
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier

    OUTPUT:
        ni     -- Species number moment array(size, Nj)
        nui    -- Species velocity moment array (size, Nj)
        
    Might need to put in a check for ghost cell values?
    Or just not distribute those values to the ghost cells, since they
    get overwritten anyway. For now just leave it because I +/- 1 is simpler
    to visualise, check, and code
    '''
    nu     *= 0.
    # Deposit average velocity across all cells :: First moment
    for ii in nb.prange(vel.shape[1]):
        I   = Ie[ii]
        sp  = idx[ii]
        
        if sp >= 0:
            for kk in range(3):
                nu[I,     sp, kk] += W_elec[0, ii] * vel[kk, ii]
                nu[I + 1, sp, kk] += W_elec[1, ii] * vel[kk, ii]
                nu[I + 2, sp, kk] += W_elec[2, ii] * vel[kk, ii]
                
            # Simulate virtual particles in boundary ghost cells
            # Same position but dx further forward or behind
            if abs(pos[0, ii] - xmin) < dx and pos[0, ii] != xmin:
                for kk in range(3):
                    nu[I - 1, sp, kk] += W_elec[0, ii] * vel[kk, ii]
                    nu[I    , sp, kk] += W_elec[1, ii] * vel[kk, ii]
                    nu[I + 1, sp, kk] += W_elec[2, ii] * vel[kk, ii]
                    
            elif abs(pos[0, ii] - xmax) < dx and pos[0, ii] != xmax:
                for kk in range(3):
                    nu[I + 1, sp, kk] += W_elec[0, ii] * vel[kk, ii]
                    nu[I + 2, sp, kk] += W_elec[1, ii] * vel[kk, ii]
                    nu[I + 3, sp, kk] += W_elec[2, ii] * vel[kk, ii]

    if source_smoothing == True:
        for jj in range(Nj):
            for ii in range(3):
                three_point_smoothing(nu[:, jj, ii], Ji[:,  0])

    Ji     *= 0.
    # Convert to real moment, and accumulate charge density
    for jj in range(Nj):
        for kk in range(3):
            nu[:, jj, kk] *= n_contr[jj]
            Ji[:,     kk] += nu[:, jj, kk] * charge[jj]
            
# =============================================================================
#     # Set damping cell source values (last value)
#     for ii in range(3):
#         Ji[:ND, ii]    = Ji[ND, ii]
#         Ji[ND+NX:, ii] = Ji[ND+NX - 1, ii]
# =============================================================================
    return


@nb.njit()
def collect_position_moment(pos, Ie, W_elec, idx, q_dens, ni):
    '''Collect number density in each cell, weighted by their distance
    from cell nodes.

    INPUT:
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier

    OUTPUT:
        q_dens -- Total charge density in each cell
        ni     -- Species number moment array(size, Nj)
        
    Again, check the ghost cell access or remove it altogether. Also, surely
    there's a more efficient way to do this.
    '''
    epsilon = 1e-6
    ni     *= 0.
    # Deposit macroparticle moment on grid
    for ii in nb.prange(Ie.shape[0]):
        I   = Ie[ii]
        sp  = idx[ii]
        
        if sp >= 0:
            ni[I,     sp] += W_elec[0, ii]
            ni[I + 1, sp] += W_elec[1, ii]
            ni[I + 2, sp] += W_elec[2, ii]
            
            # Simulate virtual particles in boundary ghost cells
            if pos[0, ii] - xmin < dx and pos[0, ii] != xmin:
                print('LHS Ghost for particle ', ii)
                ni[I - 1, sp] += W_elec[0, ii]
                ni[I    , sp] += W_elec[1, ii]
                ni[I + 1, sp] += W_elec[2, ii]
            elif xmax - pos[0, ii] < dx  and pos[0, ii] != xmax:
                print('RHS Ghost for particle ', ii)
                ni[I + 1, sp] += W_elec[0, ii]
                ni[I + 2, sp] += W_elec[1, ii]
                ni[I + 3, sp] += W_elec[2, ii]
    
    if source_smoothing == 1:
        for ii in range(Nj):
            three_point_smoothing(ni[:, ii], q_dens)
            
    q_dens *= 0.
    # Sum charge density contributions across species
    for jj in range(Nj):
        ni[:, jj] *= n_contr[jj]
        q_dens    += ni[:, jj] * charge[jj]
        
# =============================================================================
#     # Set damping cell source values
#     q_dens[:ND]    = q_dens[ND]
#     q_dens[ND+NX:] = q_dens[ND+NX - 1]
#         
#     # Set density minimum
#     for ii in range(q_dens.shape[0]):
#         if q_dens[ii] < min_dens * ne * q:
#             q_dens[ii] = min_dens * ne * q
# =============================================================================
    return


@nb.njit()
def three_point_smoothing(arr, temp):
    '''
    Three point Gaussian (1/4-1/2-1/4) smoothing function. arr, temp are both
    1D arrays of size NC = NX + 2*ND (i.e. on the E-grid)
    '''
    NC = arr.shape[0]
    
    temp *= 0.0
    for ii in range(1, NC - 1):
        temp[ii] = 0.25*arr[ii - 1] + 0.5*arr[ii] + 0.25*arr[ii + 1]
        
    temp[0]      = temp[1]
    temp[NC - 1] = temp[NC - 2]
    
    arr[:]       = temp
    return