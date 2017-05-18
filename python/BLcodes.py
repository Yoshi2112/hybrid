# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 13:35:49 2017

@author: c3134027
"""

import numpy as np

def dec2base(num, base):
    # Find the first power of base greater than the number
    idx = 0 ;   K = base
    
    while K <= num:
        idx += 1
        K = base ** idx
    
    if idx == 0:
        idx = 1
        
    output = np.zeros(idx, dtype=int)      # Empty binary output
    numb = num
    
    for ii in range(idx):
        tar = base ** (idx - ii - 1)
        val = numb - tar
        
        if val >= 0:
            output[ii] = 1
            numb = val
    
    return output
    
def dec2base2(num, base):
    output = []    
    
    while num > 0:
        last = num%base
        output.append(last)
        num /= base

    return output[::-1]
    
def radix_two(x):
    binary = np.binary_repr(x)
    
    rev_bin  = str(binary)[::-1]
    bin_frac = '0.'+rev_bin   
    
    # Convert binary decimal to base-10 decimal:
    size   = len(bin_frac) - 2
    output = 0
    
    for i in range(1, size + 1):
        temp    = int(bin_frac[i + 1]) * (2 ** (-i))
        output += temp

    return output
    
def radix_K(x, k):
    transform = dec2base2(x, k)     
    transform = transform[::-1]    
    
    output = 0    
    size = len(transform)
        
    for ii in range(size):
        output += transform[ii] * (k ** (- (ii + 1)))

    return output
    
def basek_scramble(x, k):
    rev_k = []    
    
    while x > 0:
        last = x%k
        rev_k.append(last)
        x /= k
    
    output = 0    
    size = len(rev_k)
        
    for ii in range(size):
        output += rev_k[ii] * (k ** (- (ii + 1)))

    return output
    
if __name__ == '__main__':
    number = 5
    base   = 3

    radix_K(number, base)

