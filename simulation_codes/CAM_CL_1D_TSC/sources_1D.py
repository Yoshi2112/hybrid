# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""

import numpy as np
import numba as nb

import particles_1D as particles

from simulation_parameters_1D import N, NX, Nj, n_contr, charge, mass, smooth_sources, do_parallel
from fields_1D                import interpolate_to_center_cspline3D
from auxilliary_1D            import cross_product

@nb.njit(parallel=do_parallel)
def push_current(J, E, B, L, G, dt):
    '''Uses an MHD-like equation to advance the current with a moment method as 
    per Matthews (1994) CAM-CL method. Fills in ghost cells at edges (excluding very last one)
    
    INPUT:
        J  -- Ionic current
        E  -- Electric field
        B  -- Magnetic field (offset from E by 0.5dx)
        L  -- "Lambda" MHD variable
        G  -- "Gamma"  MHD variable
        dt -- Timestep
        
    OUTPUT:
        J_out -- Advanced current
    '''
    J_out        = np.zeros(J.shape)
    B_center     = interpolate_to_center_cspline3D(B)
    G_cross_B    = cross_product(G, B_center)
    
    for ii in range(3):
        J_out[:, ii] = J[:, ii] + 0.5*dt * (L * E[:, ii] + G_cross_B[:, ii]) 

    J_out[0]                = J_out[J.shape[0] - 3]
    J_out[J.shape[0] - 2]   = J_out[1]
    J_out[J.shape[0] - 1]   = J_out[2]
    return J_out


@nb.njit(parallel=do_parallel)
def deposit_both_moments(pos, vel, Ie, W_elec, idx):
    '''Collect number and velocity moments in each cell, weighted by their distance
    from cell nodes.

    INPUT:
        pos    -- Particle positions (x)
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier
        
    OUTPUT:
        n_i    -- Species number moment array(size, Nj)
        nu_i   -- Species velocity moment array (size, Nj)
    '''
    size      = NX + 3
    n_i       = np.zeros((size, Nj))
    nu_i      = np.zeros((size, Nj, 3))
    
    for ii in nb.prange(pos.shape[0]):
        I   = Ie[ ii]
        sp  = idx[ii]
    
        for kk in range(3):
            nu_i[I,     sp, kk] += W_elec[0, ii] * vel[kk, ii]
            nu_i[I + 1, sp, kk] += W_elec[1, ii] * vel[kk, ii]
            nu_i[I + 2, sp, kk] += W_elec[2, ii] * vel[kk, ii]
        
        n_i[I,     sp] += W_elec[0, ii]
        n_i[I + 1, sp] += W_elec[1, ii]
        n_i[I + 2, sp] += W_elec[2, ii]

    n_i   = manage_ghost_cells(n_i)
    nu_i  = manage_ghost_cells(nu_i)
    return n_i, nu_i


@nb.njit(parallel=do_parallel)
def deposit_velocity_moments(vel, Ie, W_elec, idx):
    '''Collect velocity moment in each cell, weighted by their distance
    from cell nodes.

    INPUT:
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier
        
    OUTPUT:
        nu_i   -- Species velocity moment array (size, Nj)
    '''
    size      = NX + 3
    nu_i      = np.zeros((size, Nj, 3))

    for ii in range(N):
        I   = Ie[ ii]
        sp  = idx[ii]
        
        for kk in range(3):
            nu_i[I,     sp, kk] += W_elec[0, ii] * vel[kk, ii]
            nu_i[I + 1, sp, kk] += W_elec[1, ii] * vel[kk, ii]
            nu_i[I + 2, sp, kk] += W_elec[2, ii] * vel[kk, ii]
                      
    nu_i  = manage_ghost_cells(nu_i)
    return nu_i


@nb.njit(parallel=do_parallel)
def init_collect_moments(pos, vel, Ie, W_elec, idx, DT):
    '''Moment collection and position advance function. Specifically used at initialization or
    after timestep synchronization.

    INPUT:
        pos    -- Particle positions (x)
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier
        DT     -- Timestep for position advance
        
    OUTPUT:
        pos     -- Advanced particle positions
        Ie      -- Updated leftmost to nearest E-nodes
        W_elec  -- Updated TSC weighting coefficients
        rho_0   -- Charge  density at initial time (p0)
        rho     -- Charge  density at +0.5 timestep
        J_init  -- Current density at initial time (J0)
        J_plus  -- Current density at +0.5 timestep
        G       -- "Gamma"  MHD variable for current advance : Current-like
        L       -- "Lambda" MHD variable for current advance :  Charge-like
    '''
    size    = NX + 3
    
    rho_0   = np.zeros( size)
    rho     = np.zeros( size)    
    J_plus  = np.zeros((size, 3))
    J_init  = np.zeros((size, 3))
    L       = np.zeros( size)
    G       = np.zeros((size, 3))

    ni_init, nu_init     = deposit_both_moments(pos, vel, Ie, W_elec, idx)
    pos, Ie, W_elec      = particles.position_update(pos, vel, DT)
    ni, nu_plus          = deposit_both_moments(pos, vel, Ie, W_elec, idx)

    if smooth_sources == 1:
        for jj in range(Nj):
            ni[:, jj]  = smooth(ni[:, jj])
        
            for kk in range(3):
                nu_plus[:, jj, kk] = smooth(nu_plus[:,  jj, kk])
                nu_init[:, jj, kk] = smooth(nu_init[:, jj, kk])
    
    for jj in range(Nj):
        rho_0   += ni_init[:, jj]   * n_contr[jj] * charge[jj]
        rho     += ni[:, jj]        * n_contr[jj] * charge[jj]
        L       += ni[:, jj]        * n_contr[jj] * charge[jj] ** 2 / mass[jj]
        
        for kk in range(3):
            J_init[:, kk]  += nu_init[:, jj, kk] * n_contr[jj] * charge[jj]
            J_plus[ :, kk] += nu_plus[:, jj, kk] * n_contr[jj] * charge[jj]
            G[      :, kk] += nu_plus[:, jj, kk] * n_contr[jj] * charge[jj] ** 2 / mass[jj]

    return pos, Ie, W_elec, rho_0, rho, J_plus, J_init, G, L


@nb.njit(parallel=do_parallel)
def collect_moments(pos, vel, Ie, W_elec, idx, DT):
    '''
    Moment collection and position advance function.

    INPUT:
        pos    -- Particle positions (x)
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier
        DT     -- Timestep for position advance
        
    OUTPUT:
        pos     -- Advanced particle positions
        Ie      -- Updated leftmost to nearest E-nodes
        W_elec  -- Updated TSC weighting coefficients
        rho     -- Charge  density at +0.5 timestep
        J_plus  -- Current density at +0.5 timestep
        J_minus -- Current density at initial time (J0)
        G       -- "Gamma"  MHD variable for current advance
        L       -- "Lambda" MHD variable for current advance    
    '''
    size    = NX + 3
    
    rho     = np.zeros(size)    
    J_plus  = np.zeros((size, 3))
    J_minus = np.zeros((size, 3))
    L       = np.zeros(size)
    G       = np.zeros((size, 3))
    
    nu_minus        = deposit_velocity_moments(vel, Ie, W_elec, idx)
    pos, Ie, W_elec = particles.position_update(pos, vel, DT)
    ni, nu_plus     = deposit_both_moments(pos, vel, Ie, W_elec, idx)
    
    if smooth_sources == 1:
        for jj in range(Nj):
            ni[:, jj]  = smooth(ni[:, jj])
        
            for kk in range(3):
                nu_plus[ :, jj, kk] = smooth(nu_plus[:,  jj, kk])
                nu_minus[:, jj, kk] = smooth(nu_minus[:, jj, kk])
    
    for jj in range(Nj):
        rho  += ni[:, jj] * n_contr[jj] * charge[jj]
        L    += ni[:, jj] * n_contr[jj] * charge[jj] ** 2 / mass[jj]
        
        for kk in range(3):
            J_minus[:, kk] += nu_minus[:, jj, kk] * n_contr[jj] * charge[jj]
            J_plus[ :, kk] += nu_plus[ :, jj, kk] * n_contr[jj] * charge[jj]
            G[      :, kk] += nu_plus[ :, jj, kk] * n_contr[jj] * charge[jj] ** 2 / mass[jj]
        
    return pos, Ie, W_elec, rho, J_plus, J_minus, G, L


@nb.njit(parallel=do_parallel)
def smooth(function):
    '''Smoothing function: Applies Gaussian smoothing routine across adjacent cells. 
    Assummes no contribution from ghost cells.'''
    size         = function.shape[0]
    new_function = np.zeros(size)

    for ii in nb.prange(1, size - 1):
        new_function[ii - 1] = 0.25*function[ii] + new_function[ii - 1]
        new_function[ii]     = 0.50*function[ii] + new_function[ii]
        new_function[ii + 1] = 0.25*function[ii] + new_function[ii + 1]

    # Move Ghost Cell Contributions: Periodic Boundary Condition
    new_function[1]        += new_function[size - 1]
    new_function[size - 2] += new_function[0]

    # Set ghost cell values to mirror corresponding real cell
    new_function[0]        = new_function[size - 2]
    new_function[size - 1] = new_function[1]
    return new_function


@nb.njit(parallel=do_parallel)
def manage_ghost_cells(arr):
    '''Deals with ghost cells: Moves their contributions and mirrors their counterparts.
       Works like a charm if spatial dimensions always come first in an array.'''

    arr[NX]     += arr[0]                 # Move contribution: Start to end
    arr[1]      += arr[NX + 1]            # Move contribution: End to start

    arr[NX + 1]  = arr[1]                 # Fill ghost cell: End
    arr[0]       = arr[NX]                # Fill ghost cell: Start
    
    arr[NX + 2]  = arr[2]                 # This one doesn't get used, but prevents nasty nan's from being in array.
    return arr