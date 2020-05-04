# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:41:42 2019

@author: Yoshi
"""

import numba as nb
import numpy as np


@nb.njit()
def double_iter_var(nn):
    return nn*2


@nb.njit()
def while_loop(number, dbl):
    
    iter_var = 0; times_doubled = 0
    while iter_var < number:
        
        if iter_var == number / 2 and times_doubled < dbl:
            number = double_iter_var(number)
            times_doubled += 1
        
        iter_var += 1
        
    return iter_var


if __name__ == '__main__':
    
    sample_number = 10
    
    final = while_loop(sample_number, 1)    
    print 'Loop finished: {} iterations'.format(final)