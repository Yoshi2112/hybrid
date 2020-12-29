# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:53:04 2020

@author: Yoshi
"""

import numpy as np
import ctypes
import multiprocessing as mp
import random
from contextlib import closing


def init_shared(ncell):
    '''Create shared value array for processing.'''
    shared_array_base = mp.Array(ctypes.c_float, ncell, lock=False)
    return shared_array_base


def tonumpyarray(shared_array):
    '''Create numpy array from shared memory.'''
    nparray = np.frombuffer(shared_array, dtype=ctypes.c_float)
    return nparray


def init_parameters(**kwargs):
    '''Initialize parameters for processing in workers.'''
    params = dict()
    for key, value in kwargs.items():
        params[key] = value
    return params


def init_worker(shared_array_,parameters_):
    '''Initialize worker for processing.

    Args:
        shared_array_: Object returned by init_shared
        parameters_: Dictionary returned by init_parameters
    '''
    global shared_array
    global shared_parr
    global dim

    shared_array = tonumpyarray(shared_array_)
    shared_parr  = tonumpyarray(parameters_['shared_parr'])

    dim = parameters_['dimensions']
    

def worker_fun(ix):
    '''
    Function to be run inside each worker
    This is the target (main) function to parallelize
    '''
    # Redundant? These are already numpy arrays
    arr  = tonumpyarray(shared_array)
    parr = tonumpyarray(shared_parr)

    arr.shape = dim

    random.seed(ix)
    rint = random.randint(1,10)

    parr[ix] = rint

    arr[ix,...] = arr[ix,...] * rint
    

import pdb
def main():
    nrows = 100
    ncols = 10

    # Allocate shared memory
    shared_array = init_shared(nrows*ncols)
    shared_parr  = init_shared(nrows)

    # Pack parameters (Shared parameter array, dimensions) into a dict
    params = init_parameters(shared_parr=shared_parr, dimensions=(nrows,ncols))

    # Convert the shared array memory allocations to numpy arrays for easy handling
    arr  = tonumpyarray(shared_array)
    parr = tonumpyarray(params['shared_parr'])

    # Effectively a reshape operation
    arr.shape = (nrows,ncols)

    # Just fills arr with random numbers
    arr[...] = np.random.randint(1,100,size=(100,10),dtype='int16')

    
    # closing will close the pool of workers at the end of this block. Why have pool.close() explicitly then?
    with closing(mp.Pool(processes=8,                       # Number of processes
                         initializer = init_worker,         # 
                         initargs = (shared_array,params)
                         )) as pool:
        
        # Execute worker_fun with the selected argument (just a range function?)
        pool.map(worker_fun, range(arr.shape[0]))

    pool.close()
    pool.join()

    # check PARR output
    print(parr)


if __name__ == '__main__':
    main()