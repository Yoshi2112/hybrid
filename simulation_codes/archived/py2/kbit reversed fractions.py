# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 15:30:43 2017

@author: c3134027
"""

def base_k_brf(x, k):
    rev_k = []    
    
    while x > 0:
        last = x%k
        rev_k.append(last)
        x /= k
    
    output = 0    
    size   = len(rev_k) 
        
    for ii in range(size):
        output += rev_k[ii] * (k ** (- (ii + 1)))
    
    return output
    
if __name__ == '__main__':
    number = 3
    base   = 3

    frac = base_k_brf(number, base)
    print frac