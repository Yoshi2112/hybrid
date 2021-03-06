# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:37:10 2019

@author: Yoshi
"""

import numba as nb
import numpy as np


def njit_loop(arr1, arr2):
    '''
    Basic outer product
    '''
    arr3 = np.zeros((arr1.shape[0], arr2.shape[0]))
    
    for ii in range(arr1.shape[0]):
        for jj in range(arr2.shape[0]):
            arr3[ii, jj] = arr1[ii] * arr2[jj]
    
    return arr3


if __name__ == '__main__':
    inarr1 = np.arange(100)
    inarr2 = np.arange(100)
    
    inarr3 = njit_loop(inarr1, inarr2)