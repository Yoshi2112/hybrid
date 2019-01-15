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

@nb.njit(cache=True)
def push_current(J, E, B, L, G, dt):
    '''Uses an MHD-like equation to advance the current with a moment method. Could probably be shortened with loops.
    '''
    J_out        = np.zeros(NX + 2, 3)
    G_cross_B    = cross_product(G, B)
    
    J_out[:, 0] = J[:, 0] + 0.5*dt * (L * E[:, 0] + G_cross_B[:, 0]) 
    J_out[:, 1] = J[:, 1] + 0.5*dt * (L * E[:, 1] + G_cross_B[:, 1])
    J_out[:, 2] = J[:, 2] + 0.5*dt * (L * E[:, 2] + G_cross_B[:, 2])
    return J_out


@nb.njit(cache=True)
def collect_partial_moments(part, which='both'):
    '''Collect number and velocity density in each cell at each timestep, weighted by their distance
    from cell nodes.

    INPUT:
        part    -- Particle array
        weights -- Weights array
        DT      -- Timestep
        which   -- Collect 'both' number and velocity densities, or 'velocity_only'
    '''
    size      = NX + 2

    if which == 'both':
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
        
    elif which == 'velocity_only':
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
def collect_moments(part, DT, init=False):
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
    J_plus  = np.zeros(size, 3)
    J_minus = np.zeros(size, 3)
    G       = np.zeros(size)
    L       = np.zeros(size, 3)
    
    if init == True:
        rho_0                = np.zeros(size)
        ni_init, nu_minus    = collect_partial_moments(part, which='both')
        
        for jj in range(Nj):
            rho_0 += ni_init[:, jj] * charge[jj]
            
    else:
        nu_minus             = collect_partial_moments(part, which='velocity_only')
        
    part        = particles.position_update(part, DT)
    ni, nu_plus = collect_partial_moments(part, part[2, :], which='both')
    
    if smooth_sources == 1:
        for jj in range(Nj):
            ni[:, jj]  = smooth(ni[:, jj])
        
            for kk in range(3):
                nu_plus[ :, kk, jj] = smooth(nu_plus[:,  jj, kk])
                nu_minus[:, kk, jj] = smooth(nu_minus[:, jj, kk])
    
    for jj in range(Nj):
        rho     += ni[:, jj]        * charge[jj]
        G       += ni[:, jj]        * charge[jj] ** 2 / mass[jj]
        
        for kk in range(3):
            J_minus[:, kk] += nu_minus[:, jj, kk] * charge[jj]
            J_plus[ :, kk] += nu_plus[ :, jj, kk] * charge[jj]
            L[      :, kk] += nu_plus[ :, jj, kk] * charge[jj] ** 2 / mass[jj]
        
    if init == True:
        return part, rho_0, rho, J_plus, J_minus, G, L
    else:
        return part, rho, J_plus, J_minus, G, L