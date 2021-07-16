# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:16:54 2021

@author: Yoshi
"""
import numba as nb
import numpy as np

def deposit_particles_to_grid(ploc, pweight, grid):
    '''
    ploc    : (N)   : Particle location as an index in the grid array, leftmost point
    pweight : (Nx3) : Particle weights as a float across 3 gridpoints
    grid    : (M)   : Target array gridpoints where M << N
    '''
    for ii in nb.prange(ploc.shape[0]):
        grid[ploc[ii]]     += pweight[ii, 0]
        grid[ploc[ii] + 1] += pweight[ii, 1]
        grid[ploc[ii] + 2] += pweight[ii, 2]
    return


@nb.njit(parallel=True)
def deposit_particles_to_grid_parallel(ploc, pweight, grid):
    '''
    ploc    : (N)   : Particle location as an index in the grid array, leftmost point
    pweight : (Nx3) : Particle weights as a float across 3 gridpoints
    grid    : (M)   : Target array gridpoints where M << N
    '''
    grid_threads = np.zeros((grid.shape[0], n_threads), dtype=grid.dtype)
    N_per_thread = ploc.shape[0] / n_threads     
    n_start_idxs = np.arange(n_threads)*N_per_thread

    for tt in nb.prange(n_threads):
        for ii in range(n_start_idxs[tt], n_start_idxs[tt]+N_per_thread):
            grid_threads[ploc[ii],     tt] += pweight[ii, 0]
            grid_threads[ploc[ii] + 1, tt] += pweight[ii, 1]
            grid_threads[ploc[ii] + 2, tt] += pweight[ii, 2]
    grid[:] = grid_threads.sum(axis=1)
    return 


n_particles = 128
n_threads   = 8 


n_idx_per_thread = n_particles / n_threads     
n_start_idxs     = np.arange(n_threads)*n_idx_per_thread