# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 15:30:43 2017

@author: c3134027
"""
from timeit import default_timer as timer
import numpy as np

def reverse_slicing(s):
        return s[::-1]

def rkbr_uniform_set(arr, base=2):
    '''
    Works on array arr to produce fractions in (0, 1) by 
    traversing base k.
    
    Will support arrays of lengths up to at least a billion
     -- But this is super inefficient, takes up to a minute with 2 million
    '''
    # Convert ints to base k strings
    str_arr = np.zeros(arr.shape[0], dtype='U30')
    dec_arr = np.zeros(arr.shape[0], dtype=float)
    for ii in range(arr.shape[0]):
        str_arr[ii] = np.base_repr(arr[ii], base)   # Returns strings

    # Reverse string order and convert to decimal, treating as base k fraction (i.e. with 0. at front)
    for ii in range(arr.shape[0]):
        rev = reverse_slicing(str_arr[ii])

        dec_val = 0
        for jj in range(len(rev)):
            dec_val += float(rev[jj]) * (base ** -(jj + 1))
        dec_arr[ii] = dec_val
    return dec_arr

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    N_array = 262144
    BASE    = 3
    A       = np.arange(N_array)
    
    start_time = timer()
    B = rkbr_uniform_set(A, base=BASE)
    runtime = timer() - start_time
    print('Runtime {:.4f} ms'.format(runtime*1e3))
    
    # Test uniformity: Very Uniform!
    #plt.figure()
    #plt.hist(B, bins=1000)
    
    # Test Randomness: Not very random? Or just evenly spread? Not chaotic?
    plt.scatter(B[:-1], B[1:])