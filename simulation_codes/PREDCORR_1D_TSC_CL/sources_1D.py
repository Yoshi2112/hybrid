# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""
import numpy as np
import numba as nb

from simulation_parameters_1D import NX, Nj, n_contr, charge, smooth_sources, do_parallel, njit, q, ne, min_dens

@nb.jit(nopython=njit, parallel=do_parallel)
def deposit_both_moments(vel, Ie, W_elec, idx):
    '''Collect number and velocity moments in each cell, weighted by their distance
    from cell nodes.

    INPUT:
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
    
    for ii in nb.prange(vel.shape[1]):
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


@nb.jit(nopython=njit, parallel=do_parallel)
def collect_moments(vel, Ie, W_elec, idx):
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
    '''
    size    = NX + 3
    
    q_dens  = np.zeros(size)    
    Ji      = np.zeros((size, 3))

    ni, nu  = deposit_both_moments(vel, Ie, W_elec, idx)
    
    if smooth_sources == 1:
        for jj in range(Nj):
            ni[:, jj]  = smooth(ni[:, jj])
        
            for kk in range(3):
                nu[ :, jj, kk] = smooth(nu[:,  jj, kk])
    
    for jj in range(Nj):
        q_dens  += ni[:, jj] * n_contr[jj] * charge[jj]

        for kk in range(3):
            Ji[:, kk] += nu[:, jj, kk] * n_contr[jj] * charge[jj]
        
    for ii in range(size):
        if q_dens[ii] < min_dens * ne * q:
            q_dens[ii] = min_dens * ne * q
            
    return q_dens, Ji


@nb.jit(nopython=njit, parallel=do_parallel)
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


@nb.jit(nopython=njit, parallel=do_parallel)
def manage_ghost_cells(arr):
    '''Deals with ghost cells: Moves their contributions and mirrors their counterparts.
       Works like a charm if spatial dimensions always come first in an array.'''

    arr[NX]     += arr[0]                 # Move contribution: Start to end
    arr[1]      += arr[NX + 1]            # Move contribution: End to start

    arr[NX + 1]  = arr[1]                 # Fill ghost cell: End
    arr[0]       = arr[NX]                # Fill ghost cell: Start
    
    arr[NX + 2]  = arr[2]                 # This one doesn't get used, but prevents nasty nan's from being in array.
    return arr