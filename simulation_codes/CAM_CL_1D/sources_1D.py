# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""

import numpy as np
import numba as nb

import particles_1D as particles

from simulation_parameters_1D import NX, dx, Nj, n_contr, charge, mass, idx_bounds, smooth_sources
from auxilliary_1D            import smooth, manage_ghost_cells, cross_product
from fields_1D                import interpolate_to_center

@nb.njit(cache=True)
def push_current(J, E, B, L, G, dt):
    '''Uses an MHD-like equation to advance the current with a moment method. 
    Could probably be shortened with loops.
    '''
    J_out        = np.zeros((NX + 2, 3))
    
    B_center     = interpolate_to_center(B)
    G_cross_B    = cross_product(G, B_center)
    
    for ii in range(3):
        J_out[:, ii] = J[:, ii] + 0.5*dt * (L * E[:, ii] + G_cross_B[:, ii]) 
    return J_out


@nb.njit(cache=True)
def collect_both_moments(part):
    '''Collect number and velocity density in each cell at each timestep, weighted by their distance
    from cell nodes.

    INPUT:
        part    -- Particle array
        weights -- Weights array
        DT      -- Timestep
        which   -- Collect 'both' (2) number and velocity densities, or 'velocity_only' (1)
    '''
    size      = NX + 2

    n_i       = np.zeros((size, Nj))
    nu_i      = np.zeros((size, Nj, 3))
    
    for jj in range(Nj):
        for ii in range(idx_bounds[jj, 0], idx_bounds[jj, 1]):
            I   = int(part[1, ii])      # Left node
            W   = part[2, ii]           # Right weight
            vel = part[3:6, ii]         # Particle velocity
    
            for kk in range(3):
                nu_i[I,     jj, kk] += (1 - W) * n_contr[jj] * vel[kk]
                nu_i[I + 1, jj, kk] +=      W  * n_contr[jj] * vel[kk]
            
            n_i[I,     jj] += (1 - W) * n_contr[jj]
            n_i[I + 1, jj] +=      W  * n_contr[jj]

    n_i  /= float(dx)
    n_i   = manage_ghost_cells(n_i)
    
    nu_i /= float(dx)
    nu_i  = manage_ghost_cells(nu_i)
    return n_i, nu_i


@nb.njit(cache=True)
def collect_velocity_moments(part):
    '''Collect number and velocity density in each cell at each timestep, weighted by their distance
    from cell nodes.

    INPUT:
        part    -- Particle array
        weights -- Weights array
        DT      -- Timestep
        which   -- Collect 'both' (2) number and velocity densities, or 'velocity_only' (1)
    '''
    size      = NX + 2

    nu_i      = np.zeros((size, Nj, 3))
    
    for jj in range(Nj):
        for ii in range(idx_bounds[jj, 0], idx_bounds[jj, 1]):
            I   = int(part[1, ii])      # Left node
            W   = part[2, ii]           # Right weight
            vel = part[3:6, ii]         # Particle velocity
    
            for kk in range(3):
                nu_i[I,     jj, kk] += (1 - W) * n_contr[jj] * vel[kk]
                nu_i[I + 1, jj, kk] +=      W  * n_contr[jj] * vel[kk]
                            
    nu_i /= float(dx)
    nu_i  = manage_ghost_cells(nu_i)
    return nu_i


@nb.njit(cache=True)
def init_collect_moments(part, DT):
    '''Primary moment collection and position advance function.

    INPUT:
        part    -- Particle array
        weights -- Weights array
        DT      -- Timestep
        init    -- Flag to indicate if this is the first time the function is called**
        
    ** At first initialization of this function, J- is equivalent to J0 and rho_0 is
    initial density (usually not returned)
    
    '''
    size    = NX + 2
    
    rho_0   = np.zeros(size)
    rho     = np.zeros(size)    
    J_plus  = np.zeros((size, 3))
    J_minus = np.zeros((size, 3))
    L       = np.zeros(size)
    G       = np.zeros((size, 3))
     
    ni_init, nu_minus    = collect_both_moments(part)
    part                 = particles.position_update(part, DT)
    ni, nu_plus          = collect_both_moments(part)
    
    if smooth_sources == 1:
        for jj in range(Nj):
            ni[:, jj]  = smooth(ni[:, jj])
        
            for kk in range(3):
                nu_plus[ :, jj, kk] = smooth(nu_plus[:,  jj, kk])
                nu_minus[:, jj, kk] = smooth(nu_minus[:, jj, kk])
    
    for jj in range(Nj):
        rho_0   += ni_init[:, jj]   * charge[jj]
        rho     += ni[:, jj]        * charge[jj]
        L       += ni[:, jj]        * charge[jj] ** 2 / mass[jj]
        
        for kk in range(3):
            J_minus[:, kk] += nu_minus[:, jj, kk] * charge[jj]
            J_plus[ :, kk] += nu_plus[ :, jj, kk] * charge[jj]
            G[      :, kk] += nu_plus[ :, jj, kk] * charge[jj] ** 2 / mass[jj]
        
    return part, rho_0, rho, J_plus, J_minus, G, L


@nb.njit(cache=True)
def collect_moments(part, DT):
    '''Primary moment collection and position advance function.

    INPUT:
        part    -- Particle array
        weights -- Weights array
        DT      -- Timestep
        init    -- Flag to indicate if this is the first time the function is called**
        
    ** At first initialization of this function, J- is equivalent to J0 and rho_0 is
    initial density (usually not returned)
    
    '''
    size    = NX + 2
    
    rho     = np.zeros(size)    
    J_plus  = np.zeros((size, 3))
    J_minus = np.zeros((size, 3))
    L       = np.zeros(size)
    G       = np.zeros((size, 3))

    nu_minus    = collect_velocity_moments(part)
    part        = particles.position_update(part, DT)
    ni, nu_plus = collect_both_moments(part)
    
    if smooth_sources == 1:
        for jj in range(Nj):
            ni[:, jj]  = smooth(ni[:, jj])
        
            for kk in range(3):
                nu_plus[ :, jj, kk] = smooth(nu_plus[:,  jj, kk])
                nu_minus[:, jj, kk] = smooth(nu_minus[:, jj, kk])
    
    for jj in range(Nj):
        rho     += ni[:, jj]        * charge[jj]
        L       += ni[:, jj]        * charge[jj] ** 2 / mass[jj]
        
        for kk in range(3):
            J_minus[:, kk] += nu_minus[:, jj, kk] * charge[jj]
            J_plus[ :, kk] += nu_plus[ :, jj, kk] * charge[jj]
            G[      :, kk] += nu_plus[ :, jj, kk] * charge[jj] ** 2 / mass[jj]
        
    return part, rho, J_plus, J_minus, G, L