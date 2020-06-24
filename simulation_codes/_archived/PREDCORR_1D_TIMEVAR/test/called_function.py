# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:53:07 2019

@author: Yoshi
"""

import numpy as np
import numba as nb

@nb.njit(cache=False)
def change_array(arr):
    N = arr.shape[0]
    
    for ii in range(N):
        arr[ii] *= 2
    print('Array changed')
    

@nb.njit(cache=False)
def change_single_val(arr):
    arr[0] += 5
    print('Single value changed')
   
#@nb.njit()
def change_slice(arr_in, arr_out):
    # Do stuff
    for ii in range(arr_in.shape[0]):
        arr_in[ii] = 0 * arr_out[ii]
        
    # Copy
    arr_in = arr_out
    return

