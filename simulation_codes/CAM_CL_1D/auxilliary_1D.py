# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:15:59 2017

@author: iarey
"""
import numpy as np
import numba as nb
from simulation_parameters_1D import t_res, plot_res, max_sec, dx, gyfreq, lam_res, charge, mass, mu0, NX

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
    gyperiod = 1 / gyfreq                       # Radian time
    ion_ts   = lam_res * gyperiod               # Timestep to resolve gyromotion
    vel_ts   = dx / (2 * np.max(part[3, :]))    # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step

    DT       = min(ion_ts, vel_ts)              # Smallest of the two
    maxtime  = int(max_sec / DT) + 1            # Total number of iterations in run

    if plot_res == 0:
        plot_dump_iter = 1                      # Plot every iteration
    else:
        plot_dump_iter = int(plot_res / DT)     # Number of iterations between plots

    if t_res == 0:
        data_dump_iter = 1                      # Dump every iteration
    else:
        data_dump_iter = int(t_res / DT)        # Number of iterations between dumps

    print 'Timestep: %.4fs, %d iterations total' % (DT, maxtime)
    return DT, maxtime, data_dump_iter, plot_dump_iter


def check_timestep(qq, DT, part, B, E, dns, maxtime, data_dump_iter, plot_dump_iter):
    max_V           = np.max(part[3, :])
    #k_wave          = np.pi / dx
    
    B_cent = np.zeros((NX + 2, 3))
    
    for ii in range(3):
        B_cent[1:-1, ii] = 0.5*(B[:-1, ii] + B[1:, ii])

    B_tot           = np.sqrt(B_cent[:, 0] ** 2 + B_cent[:, 1] ** 2 + B_cent[:, 2] ** 2)
    high_rat        = np.max(charge/mass)
    
    gyfreq          = high_rat*max(abs(B_tot))                      # Gyrofrequency
    #elecfreq        = high_rat*max(abs(E[:, 0] / max_V))            # Electron acceleration "frequency"
    #dispfreq        = (k_wave ** 2) * np.max(B_tot / (mu0 * dns))   # Dispersion frequency
    
    ion_ts          = lam_res / gyfreq                 # Timestep to resolve gyromotion
    #acc_ts          = lam_res / elecfreq               # Timestep to resolve electric field acceleration
    #dis_ts          = lam_res / dispfreq               # Timestep to resolve magnetic field dispersion
    vel_ts          = 0.50*dx / max_V                  # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step

    DT_new          = min(ion_ts, vel_ts)              # Smallest of the two (four, later)
    
    change_flag = 0
    if DT_new < DT:
        change_flag = 1
        DT         *= 0.5
        maxtime    *= 2
        qq         *= 2
        
        if plot_dump_iter != None:
            plot_dump_iter *= 2
            
        if data_dump_iter != None:
            data_dump_iter *= 2
        
    return qq, DT, maxtime, data_dump_iter, plot_dump_iter, change_flag


@nb.njit(cache=True)
def smooth(function):
    '''Smoothing function: Applies Gaussian smoothing routine across adjacent cells. 
    Assummes no contribution from ghost cells.'''
    size         = function.shape[0]
    new_function = np.zeros(size)

    for ii in range(1, size - 1):
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

@nb.njit(cache=True)
def manage_ghost_cells(arr):
    '''Deals with ghost cells: Moves their contributions and mirrors their counterparts.
       Works like a charm if spatial dimensions always come first in an array. Condition
       variable passed with array because ghost cell field values do not need to be moved:
       But they do need correct (mirrored) ghost cell values'''
    size = arr.shape[0]
    
    arr[size - 2]  += arr[0]                    # Move contribution: Start to end
    arr[1]         += arr[size - 1]             # Move contribution: End to start

    arr[size - 1]  = arr[1]                     # Fill ghost cell: Top
    arr[0]         = arr[size - 2]              # Fill ghost cell: Bottom
    return arr