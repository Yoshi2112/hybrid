# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""

import numpy as np
import numba as nb

from simulation_parameters_1D import NX, dx, N, Nj, n_contr, charge, idx_bounds
from auxilliary_1D            import smooth, manage_ghost_cells

@nb.njit(cache=True)
def assign_weighting(xpos, I, BE):
    '''Linear weighting scheme used to interpolate particle source term contributions to
    nodes and field contributions to particle positions.

    INPUT:
        xpos -- Particle positions
        I    -- Particle rightmost nodes
        BE   -- Flag: Weighting factor for Magnetic (0) or Electric (1) field node

    Notes: Last term displaces weighting factor by half a cell by adding 0.5 for a retarded
    electric field grid (i.e. all weightings are 0.5 larger due to further distance from
    left node, and this weighting applies to the I + 1 node.
    '''
    W_o = ((xpos)/(dx)) - I + (BE / 2.)
    return W_o

@nb.njit(cache=True)
def calc_left_node(pos):
    node = pos / dx + 0.5                           # Leftmost (E-field) node, I
    return node.astype(nb.int32)


@nb.njit(cache=True)
def collect_partial_moments(nodes, weights, ptypes):
    '''Collect charge density in each cell at each timestep, weighted by their distance
    from cell nodes on each side.

    INPUT:
        nodes   -- Particle rightmost nodes
        weights -- Particle weighting factor
        ptypes  -- Particle species index (identifier)
    '''

    size      = NX + 2
    
    n_i       = np.zeros((size, Nj))
    nu_minus  = np.zeros((size, Nj))
    nu_plus   = np.zeros((size, Nj))

    # Collect species number density of all particles
    for jj in range(Nj):
        for ii in range(idx_bounds[jj, 0], idx_bounds[jj, 1]):
            I   = int(nodes[ii])
            W   = weights[ii]
            idx = int(ptypes[ii])
    
            n_i[I,     idx] += (1 - W) * n_contr[idx]
            n_i[I + 1, idx] +=      W  * n_contr[idx]


    n_i /= float(dx)        # Divide by cell dimensions to give densities per cubic metre
    n_i = manage_ghost_cells(n_i, 1)

    for jj in range(Nj):
        smoothed   = smooth(n_i[:, jj])
        n_i[:, jj] = smoothed

    return n_i
