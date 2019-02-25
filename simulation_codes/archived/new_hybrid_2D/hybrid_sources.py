# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""

import numpy as np
import numba as nb

from const import NX, NY, dx, dy, N
from part_params import Nj, n_contr, charge
from hybrid_auxilliary import smooth, manage_ghost_cells

@nb.njit(cache=True)
def assign_weighting(pos, I, BE):
    '''Linear weighting scheme used to interpolate particle source term contributions to
    nodes and field contributions to particle positions.
    
    INPUT:
        pos  -- Particle x,y positions
        I    -- Particle right/topmost nodes
        BE   -- Flag: Weighting factor for Magnetic (0) or Electric (1) field node
    
    Notes: Last term displaces weighting factor by half a cell by adding 0.5 for a retarded
    electric field grid (i.e. all weightings are 0.5 larger due to further distance from
    left node, and this weighting applies to the I + 1 node.
    '''
    W_x = (pos[0, :] / dx) - I[0, :] + (BE / 2.)              # Last term displaces weighting factor by half a cell by adding 0.5 for a retarded electric field grid (i.e. all weightings are 0.5 larger due to further distance from left node, and this weighting applies to the I + 1 node.)
    W_y = (pos[1, :] / dy) - I[1, :] + (BE / 2.)              # Last term displaces weighting factor by half a cell by adding 0.5 for a retarded electric field grid (i.e. all weightings are 0.5 larger due to further distance from left node, and this weighting applies to the I + 1 node.)
    
    W_out = np.stack(((1 - W_x), W_x, (1 - W_y), W_y), axis=0)  # Left, Right, 'Left' (Bottom), 'Right' (Top)
    return W_out
    
@nb.njit(cache=True)
def collect_density(part, W): 
    '''Collect charge density in each cell at each timestep, weighted by their distance
    from cell nodes on each side.
    
    INPUT:
        part -- Particle array with node locations and species types
        W    -- Particle/node E-node weightings
    '''
    n_i = np.zeros((NX + 2, NY + 2, Nj))
    
    # Collect number density of all particles
    for ii in range(N):
        idx  = int(part[2, ii])     # Species index
        Ix   = int(part[6, ii])     # Left
        Iy   = int(part[7, ii])     # Bottom nodes
        Wx   = W[0:2, ii]           # Left,   right
        Wy   = W[2:4, ii]           # Bottom, top   node weighting factors
        
        for jj in range(2):
            for kk in range(2):
                n_i[Ix + jj, Iy + kk, idx] += Wx[jj] * Wy[kk] * n_contr[idx]
    
    n_i = manage_ghost_cells(n_i, 1) / (dx*dy)         # Divide by cell size for density per unit volume
    
    for jj in range(Nj):
        n_i[:, :, jj] = smooth(n_i[:, :, jj])
        
    return n_i


@nb.njit(cache=True)
def collect_current(part, W): 
    '''Collects the ion current flow density for each species at each E-node in the simulation by J = qnV
    
    INPUT:
        part -- Particle array: Velocity, index, node
        W_in -- Weighting factor for each particle at rightmost node
    
    OUTPUT:
        J_i  -- Ion current density
    '''
    
    J_i = np.zeros((NX + 2, NY + 2, Nj, 3))    
    
    # Loop through all particles: sum velocities for each species (nV)
    for ii in range(N):
        idx = int(part[2, ii])
        Ix  = int(part[6, ii])
        Iy  = int(part[7, ii])
        Wx  = W[0:2, ii]
        Wy  = W[2:4, ii]
       
        for jj in range(2):
            for kk in range(2):
                J_i[Ix + jj, Iy + kk, idx, :] += Wx[jj] * Wy[kk] * n_contr[idx] * part[3:6, ii]
    
    for jj in range(Nj):   
        J_i[:, :, jj, :] *= charge[jj]                                # Turn the velocity into current (qnV)
    
    J_i = manage_ghost_cells(J_i, 1) / (dx*dy)                                     # Get current density
        
    for jj in range(Nj):
        for kk in range(3):
            J_i[:, :, jj, kk] = smooth(J_i[:, :, jj, kk])             # Smooth current
            
    return J_i