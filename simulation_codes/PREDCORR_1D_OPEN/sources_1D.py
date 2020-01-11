# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""
import numpy as np
import numba as nb

from simulation_parameters_1D import NX, Nj, n_contr, charge, smooth_sources, q, ne, min_dens

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

    manage_ghost_cells(ni)
    manage_ghost_cells(nu)
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
        rho_c  -- Charge  density
        Ji     -- Current density
    '''
    # Zero source arrays: Test methods for speed later
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

    for ii in range(NX + 3):
        if q_dens[ii] < min_dens * ne * q:
            q_dens[ii] = min_dens * ne * q

    return


@nb.njit()
def smooth(arr, temp1D):
    '''Smoothing function: Applies Gaussian smoothing routine across adjacent cells. 
    Assummes no contribution from ghost cells.
    
    Some weird stuff going on with memory management: Does it create a new numpy array
    or not? Or new numpy instance pointing to same memory locations? Passing
    slices as function arguments seems weird. But the function works, it just might not be efficient.
    '''
    size         = arr.shape[0]
    temp1D      *= 0
             
    for ii in nb.prange(1, size - 1):
        temp1D[ii - 1] += 0.25*arr[ii]
        temp1D[ii]     += 0.50*arr[ii]
        temp1D[ii + 1] += 0.25*arr[ii]

    # Move Ghost Cell Contributions: Periodic Boundary Condition
    temp1D[1]        += temp1D[size - 1]
    temp1D[size - 2] += temp1D[0]

    # Set ghost cell values to mirror corresponding real cell
    temp1D[0]        = temp1D[size - 2]
    temp1D[size - 1] = temp1D[1]
    
    # Output smoothed array
    arr[:] = temp1D[:]
    return


@nb.njit()
def manage_ghost_cells(arr):
    '''Deals with ghost cells: Moves their contributions and mirrors their counterparts.
       Works like a charm if spatial dimensions always come first in an array.
       
       DO WE EVEN NEED THIS WITH ABC's? NO, BECAUSE CONTRIBUTIONS AREN'T PERIODIC
       DAMPING HAPPENS INSIDE FIELD UPDATE EQUATIONS? NOT A SEPARATE FUNCTION?
       MAYBE REPLACE THIS WITH A DAMPING THING I CAN CALL FROM THE E/B UPDATE'''

    arr[NX]     += arr[0]                 # Move contribution: Start to end
    arr[1]      += arr[NX + 1]            # Move contribution: End to start

    arr[NX + 1]  = arr[1]                 # Fill ghost cell: End
    arr[0]       = arr[NX]                # Fill ghost cell: Start
    
    arr[NX + 2]  = arr[2]                 # This one doesn't get used, but prevents nasty nan's from being in array.
    return