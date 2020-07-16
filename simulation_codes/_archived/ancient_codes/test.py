# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:01:41 2017

@author: iarey
"""
import numpy as np
import numba as nb
from timeit import default_timer as timer

@nb.njit
def add_thing(a, b):
    a += b
    return a

if __name__ == '__main__':
    A = np.array([1., 2., 3.])
    B = np.ones(3)
    
    C = add_thing(A, B)
    
# =============================================================================
#     start = timer()
#     normal(part)
#     print 'Normal: {}s'.format(round(timer() - start, 6))
#     
#     start2 = timer()
#     backwards(part)
#     print 'Backwards: {}s'.format(round(timer() - start2, 6))
# =============================================================================
