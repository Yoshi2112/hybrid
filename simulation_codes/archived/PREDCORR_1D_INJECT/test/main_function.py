# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:53:07 2019

@author: Yoshi
"""

import numpy as np
import numba as nb

@nb.njit()
def set_equal(Ai, Bi):
    Bi[:] = Ai[:]
    return
    
if __name__ == '__main__':
    
    A = np.zeros(30).reshape((3, 10))
    B = np.ones(30).reshape((3, 10))
    print(A)
    print(B)
    set_equal(A, B)
    print('finished')
    print(A)
    print(B)
