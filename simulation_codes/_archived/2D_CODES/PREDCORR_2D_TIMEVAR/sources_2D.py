# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""
import numba as nb
from simulation_parameters_2D import n_contr, charge, q, ne, min_dens

@nb.njit()
def deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu):
    '''Deposit number and velocity moment contributions from each particle
    onto E-grid nodes, weighted by their distance from grid nodes.

    INPUT:
        vel    -- [3, N]    -- Particle 3-velocities
        Ie     -- [2, N]    -- Particle leftmost to nearest E-node
        W_elec -- [2, N, 3] -- Particle TSC weighting across nearest, left,
                               and right nodes, for each dimension
        idx    -- [N]       -- Particle species identifier

    OUTPUT:
        ni     -- [NX, NY, Nj]    -- Species number moment array(size, Nj)
        nui    -- [NX, NY, Nj, 3] -- Species velocity moment array (size, Nj)
    '''
    for ii in nb.prange(vel.shape[1]):                                  # For each particle
        Ix  = Ie[0, ii]                                                 # Leftmost x-node
        Iy  = Ie[1, ii]                                                 # Leftmost y-node
        sp  = idx[ii]                                                   # Particle species id
        
        for mm in range(3):                                             # For each x-node
            for nn in range(3):                                         # For each y-node
                W = W_elec[0, ii, mm] * W_elec[1, ii, nn]               # Contribution to (mm, nn)
                
                for kk in range(3):                                     # For each velocity component
                    nu[Ix + mm, Iy + nn, sp, kk] += W * vel[kk, ii]     # Collect velocity moment
                
                ni[Ix + mm, Iy + nn, sp] += W                           # Collect density moment

    manage_E_grid_ghost_cells(ni, src=1)
    manage_E_grid_ghost_cells(nu, src=1)                                       
    return


@nb.njit()
def collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu):
    '''Moment (charge/current) collection function. Calls moment deposition
    function and converts them to real charge/current densities

    INPUT:
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier
        
    OUTPUT:
        q_dens -- Charge  density
        Ji     -- Current density
    
    INTERMEDIATE (not used outside this function):
        ni     -- Raw macroparticle number   density, per species
        nu     -- Raw macroparticle velocity density, per species
    '''
    # Zero source arrays: Test methods for speed later
    q_dens *= 0.
    Ji     *= 0.
    ni     *= 0.
    nu     *= 0.
    
    deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu)

    # Sum source contributions over species
    for jj in range(nu.shape[2]):
        q_dens  += ni[:, :, jj] * n_contr[jj] * charge[jj]

        for kk in range(3):
            Ji[:, :, kk] += nu[:, :, jj, kk] * n_contr[jj] * charge[jj]

    # Set minimum charge density as a percentage of initial electron density
    for ii in range(q_dens.shape[0]):
        for jj in range(q_dens.shape[1]):
            if q_dens[ii, jj] < min_dens * ne * q:
                q_dens[ii, jj] = min_dens * ne * q
    return


@nb.njit()
def manage_E_grid_ghost_cells(arr, src=0):
    '''
    Deals with ghost cells: Moves their contributions and mirrors
    their counterparts. If input array is for fields only (i.e. not
    source terms), existing values in ghost cells are discarded
    
    INPUT:
        arr -- Array to be managed, changed in-place
        src -- Flag to indicate if array contains source terms
    '''
    NX = arr.shape[0] - 3; NY = arr.shape[1] - 3
    
    if src == 1:                                      # Move ghost cell contributions
        arr[0: NX + 3, NY] += arr[0: NX + 3, 0     ]  # To top real,    add bottom ghost
        arr[0: NX + 3, 1 ] += arr[0: NX + 3, NY + 1]  # To bottom real, add top ghost
        arr[0: NX + 3, 2 ] += arr[0: NX + 3, NY + 2]  # To second real, add extreme top ghost (not really needed)
                                                      # "Inner", excluding top + bottom ghost cells (already applied above)
        arr[NX, 1: NY + 1] += arr[0     , 1: NY + 1]  # To right  inner real, add left inner ghost
        arr[1 , 1: NY + 1] += arr[NX + 1, 1: NY + 1]  # To left   inner real, add right inner ghost
        arr[2 , 1: NY + 1] += arr[NX + 2, 1: NY + 1]  # To second inner real, add extreme right inner ghost (not really needed)
                                                  
                                                  # Fill ghost cells
    arr[0     , 1: NY + 1] = arr[NX, 1: NY + 1]   # Left inner
    arr[NX + 1, 1: NY + 1] = arr[1 , 1: NY + 1]   # Right inner
    arr[NX + 2, 1: NY + 1] = arr[2 , 1: NY + 1]   # Extreme right inner (TSC only)
                                                  # "Whole" including side ghost cells (so no doubling up)
    arr[0: NX + 3 , 0]      = arr[0: NX + 3, NY]  # Bottom  whole row
    arr[0: NX + 3 , NY + 1] = arr[0: NX + 3, 1 ]  # Top     whole row
    arr[0: NX + 3 , NY + 2] = arr[0: NX + 3, 2 ]  # Extreme top whole row (TSC only)
    
    # Corner cells
    arr[0     , NY + 1] = arr[NX, 1 ]
    arr[0     , 0     ] = arr[NX, NY]
    arr[NX + 1, 0     ] = arr[1 , NY]
    arr[NX + 1, NY + 1] = arr[1 , 1 ]
    return