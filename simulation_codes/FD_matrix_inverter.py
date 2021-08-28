# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 19:24:41 2021

@author: Yoshi
"""
from fractions import Fraction
import numpy as np

# Gets coefficients of Finite Difference from Taylor Series equations
# For an equation of the form Bx = C where C are the desired coefficients of
# the orders of x (i.e. all zero except f'(x)) and x is the multiple of each
# Taylor series equation needed to form the finite difference operator of that
# order.
# Need to solve x = inv(B)*C

# Test using an example from some notes somewhere
if False:
    C = np.array( [0  , 1,    0,    0])
    
    B = np.array([[1  , 1,    1,    1],
                  [1  , 0,   -1,   -2],
                  [1/2, 0,  1/2,    2],
                  [1/6, 0, -1/6, -8/6]])
    
    B_inv = np.linalg.inv(B)
    print(B_inv)
    x = np.dot(B_inv, C)
    print(x)
    
if True:
    # Note: Use the limit denominator method to prevent floating point error
    #       Change limit if expected denominator is greater
    
    # For printing: Coefficients of u_i''
    coeffs = ['', '+1', '+2', '+3', '+4']
    
    C = np.array( [0  , 1,    0,    0,    0])
    
    B = np.array([[1,    1,    1,    1,    1],
                  [0,    1,    2,    3,    4],
                  [0,  1/2,    2,  9/2,    8],
                  [0,  1/6,  4/3,  9/2, 32/3],
                  [0, 1/24,  2/3, 27/8, 32/3]])
    
    B_inv = np.linalg.inv(B)
    x = np.dot(B_inv, C)
    x_frac = [Fraction(num).limit_denominator(1000) for num in x]
    print(x)
    print('Finite difference for this system:')
    for ii in range(len(coeffs)):
        print('{:>6} * f_k{}'.format(str(x_frac[ii]), coeffs[ii]))
