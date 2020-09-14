# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:51:15 2019

@author: Yoshi
"""

from numba import njit, prange
@njit(parallel=False)
def prange_test(A):
    ''' Still works even when parallel=False. prange is good to go.
    '''
    s = 0
    for i in prange(A.shape[0]):
        s += A[i]
    return s