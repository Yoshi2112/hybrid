# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:15:59 2017

@author: iarey
"""
import numpy as np
import numba as nb
from simulation_parameters_1D import t_res, plot_res, max_sec, dx, gyfreq, lam_res

@nb.njit(cache=True)
def cross_product(A, B):
    '''Vector (cross) product between two vectors, A and B of same dimensions.

    INPUT:
        A, B -- 3D vectors (ndarrays)

    OUTPUT:
        output -- The resultant cross product with same dimensions as input vectors
    '''
    output = np.zeros(A.shape)

    output[:, 0] =    A[:, 1] * B[:, 2] - A[:, 2] * B[:, 1]
    output[:, 1] = - (A[:, 0] * B[:, 2] - A[:, 2] * B[:, 0])
    output[:, 2] =    A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]

    return output


def set_timestep(part):
    gyperiod = 2*np.pi / gyfreq                 # Gyroperiod in seconds
    ion_ts   = lam_res * gyperiod               # Timestep to resolve gyromotion
    vel_ts   = dx / (2. * np.max(part[3, :]))   # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step

    DT             = min(ion_ts, vel_ts)        # Smallest of the two
    data_dump_iter = int(t_res / DT)            # Number of iterations between dumps
    maxtime        = int(max_sec / DT) + 1      # Total number of iterations in run
    
    if plot_res == None:
        plot_dump_iter = None                   # Disable output plots
    elif plot_res == 0:
        plot_dump_iter = 1                      # Plot every iteration
    else:
        plot_dump_iter = int(plot_res / DT)     # Number of iterations between plots

    if data_dump_iter == 0:
        data_dump_iter = 1

    print 'Proton gyroperiod = %.2fs' % gyperiod
    print 'Timestep: %.4fs, %d iterations total' % (DT, maxtime)
    return DT, maxtime, data_dump_iter, plot_dump_iter


def check_timestep(qq, DT, part, maxtime, data_dump_iter, plot_dump_iter):
    gyperiod = 2*np.pi / gyfreq                 # Gyroperiod in seconds
    ion_ts   = lam_res * gyperiod               # Timestep to resolve gyromotion
    vel_ts   = (0.75*dx) / (np.max(part[3, :]))  # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step

    DT_new         = min(ion_ts, vel_ts)        # Smallest of the two

    if DT_new < DT:
        DT *= 0.5
        maxtime *= 2
        data_dump_iter *= 2
        qq *= 2
        
        if plot_dump_iter != None:
            plot_dump_iter *= 2
            
        if data_dump_iter != None:
            data_dump_iter *= 2
        print 'Timestep halved. DT = {}'.format(DT)

    return qq, DT, maxtime, data_dump_iter, plot_dump_iter

@nb.njit(cache=True)
def smooth(function):
    '''Smoothing function: Applies Gaussian smoothing routine across adjacent cells. Assummes nothing in ghost cells.'''
    size         = function.shape[0]
    new_function = np.zeros(size)

    for ii in range(1, size - 1):
        new_function[ii - 1] = 0.25*function[ii] + new_function[ii - 1]
        new_function[ii]     = 0.5*function[ii]  + new_function[ii]
        new_function[ii + 1] = 0.25*function[ii] + new_function[ii + 1]

    # Move Ghost Cell Contributions: Periodic Boundary Condition
    new_function[1]        += new_function[size - 1]
    new_function[size - 2] += new_function[0]

    # Set ghost cell values to mirror corresponding real cell
    new_function[0]        = new_function[size - 2]
    new_function[size - 1] = new_function[1]
    return new_function

@nb.njit(cache=True)
def manage_ghost_cells(arr, src):
    '''Deals with ghost cells: Moves their contributions and mirrors their counterparts.
       Works like a charm if spatial dimensions always come first in an array. Condition
       variable passed with array because ghost cell field values do not need to be moved:
       But they do need correct (mirrored) ghost cell values'''
    size = arr.shape[0]
    if src == 1:   # Move source term contributions to appropriate edge cells
        arr[size - 2]  += arr[0]                    # Move contribution: Start to end
        arr[1]         += arr[size - 1]             # Move contribution: End to start

    arr[size - 1]  = arr[1]                         # Fill ghost cell: Top
    arr[0]         = arr[size - 2]                  # Fill ghost cell: Bottom
    return arr