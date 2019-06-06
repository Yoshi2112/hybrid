# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:37:10 2019

@author: Yoshi
"""

import numba as nb
import numpy as np
from timeit import default_timer as timer

@nb.njit(fastmath=True, parallel=False)
def njit_loop(arr1, arr2):
    '''
    Basic outer product
    '''
    arr3 = np.zeros((arr1.shape[0], arr2.shape[0]))
    
    print('Starting product')
    for ii in nb.prange(arr1.shape[0]):
        for jj in range(arr2.shape[0]):
            arr3[ii, jj] = arr1[ii] * arr2[jj]
    print('Finished')
    return arr3


if __name__ == '__main__':
    inarr1 = np.arange(25000)
    inarr2 = np.arange(25000)
    
    start  = timer()
    inarr3 = njit_loop(inarr1, inarr2)
    end    = timer()
    
    print('Time elapsed: {}s'.format(end - start))