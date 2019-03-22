# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:15:59 2017

@author: iarey
"""
import numpy as np
import numba as nb
from const import t_res, max_sec, dx, dy, gyfreq, NX, NY

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
    ion_ts   = 0.05 * gyperiod                  # Timestep to resolve gyromotion
    velx_ts  = dx / (2 * np.max(part[3, :]))    # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step
    vely_ts  = dy / (2 * np.max(part[4, :]))    # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step

    
    DT        = min(ion_ts, velx_ts, vely_ts)             # Smallest of the two
    framegrab = int(t_res / DT)           # Number of iterations between dumps
    maxtime   = int(max_sec / DT) + 1     # Total number of iterations to achieve desired final time
    
    if framegrab == 0:
        framegrab = 1
    
    print('Proton gyroperiod = %.2fs' % gyperiod)
    print('Timestep: %.4fs, %d iterations total' % (DT, maxtime))
    return DT, maxtime, framegrab

## FIX THIS LATER ##
def update_timestep(part, dt):
    if dx/(2*np.max(part[3:6, :])) <= dt:
        #dt  /= 2.
        #ts_history.append(qq)
        print('Timestep halved: DT = %.5fs' % dt)
        #if len(ts_history) > 7:
            #sys.exit('Timestep less than 1%% of initial. Consider parameter revision')
    return dt


@nb.njit(cache=True)
def smooth(fn):
    '''Performs a Gaussian smoothing function to a 2D array.'''
    
    new_function = np.zeros((NX + 2, NY + 2))
    
    for ii in range(1, NX + 1):
        for jj in range(1, NY + 1):
            new_function[ii, jj] = (4. / 16.) * (fn[ii, jj])                                                                        \
                                 + (2. / 16.) * (fn[ii + 1, jj]     + fn[ii - 1, jj]     + fn[ii, jj + 1]     + fn[ii, jj - 1])     \
                                 + (1. / 16.) * (fn[ii + 1, jj + 1] + fn[ii - 1, jj + 1] + fn[ii + 1, jj - 1] + fn[ii - 1, jj - 1])
   
    new_function = manage_ghost_cells(new_function, 1)        
    return new_function


@nb.njit(cache=True)
def manage_ghost_cells(arr, src):
    '''Deals with ghost cells: Moves their contributions and mirrors their counterparts.
       Works like a charm if spatial dimensions always come first in an array. Condition
       variable passed with array because ghost cell field values do not need to be moved:
       But they do need correct (mirrored) ghost cell values'''
    size = NX + 2
    if src == 1:   # Move source term contributions to appropriate edge cells
        arr[1, 1]                   += arr[size - 1, size - 1]    # TR -> BL : Move corner cell contributions
        arr[1, size - 2]            += arr[size - 1, 0]           # BR -> TL
        arr[size - 2, 1]            += arr[0, size - 1]           # TL -> BR
        arr[size - 2, size - 2]     += arr[0, 0]                  # BL -> TR
        
        arr[size - 2, 1: size - 1]  += arr[0, 1: size - 1]        # Move contribution: Bottom to top
        arr[1, 1:size - 1]          += arr[size - 1, 1: size - 1] # Move contribution: Top to bottom
        arr[1: size - 1, size - 2]  += arr[1: size - 1, 0]        # Move contribution: Left to Right
        arr[1: size - 1, 1]         += arr[1: size - 1, size - 1] # Move contribution: Right to Left
   
    arr[0, 0]                   = arr[size - 2, size - 2]         # Fill corner cell: BL
    arr[0, size - 1]            = arr[size - 2, 1]                # Fill corner cell: TL 
    arr[size - 1, 0]            = arr[1, size - 2]                # Fill corner cell: BR 
    arr[size - 1, size - 1]     = arr[1, 1]                       # Fill corner cell: TR

    arr[size - 1, 1: size - 1]  = arr[1, 1: size - 1]             # Fill ghost cell: Top
    arr[0, 1: size - 1]         = arr[size - 2, 1: size - 1]      # Fill ghost cell: Bottom
    arr[1: size - 1, 0]         = arr[1: size - 1, size - 2]      # Fill ghost cell: Left
    arr[1: size - 1, size - 1]  = arr[1: size - 1, 1]             # Fill ghost cell: Right
    return arr