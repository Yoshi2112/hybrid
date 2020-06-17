# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 21:33:52 2020

@author: Yoshi
"""

from scipy.special import erfinv
from math import erf
import numba as nb
import numpy as np

#@nb.jit(nopython=False)
def calc_erfinv(x):
    result = erfinv(x)
    return result


@nb.njit()
def calc_erf(x):
    return erf(x)


@nb.jit(nopython=True)
def test_raise():
    raise ValueError('Stuff is broked')
    return


@nb.njit()
def calc_erfinv_custom(y):
    '''
    Calculates the inverse error function erfinv() for a value in the 
    range (-1.0, 1.0). Literally just a pascal port, optimize later.
    
    Is this only accurate to single precision? Particularly around y0.
    
    This should do at least for a proof-of-method. Can do a better one later.
    It might be the accuracy of the a,b,c,d arrays (single precision constants
    give single precision answers?)
    '''
    a  = np.array([ 0.886226899, -1.645349621,  0.914624893, -0.140543331])
    b  = np.array([-2.118377725,  1.442710462, -0.329097515,  0.012229801])
    c  = np.array([-1.970840454, -1.624906493,  3.429567803,  1.641345311])
    d  = np.array([ 3.543889200,  1.637067800])
    y0 = 0.65
    
    if y < -1.0 or y > 1.0:
        raise ValueError('erfinv(y) argument out of range')

    if abs(y) == 1.0:
        x = -y*np.log(0.0)
  
    elif y < -y0:
        z = np.sqrt(-np.log((1.0+y)/2.0));
        x = -(((c[3]*z+c[2])*z+c[1])*z+c[0])/((d[1]*z+d[0])*z+1.0);

    else:
        if y<y0:
            z = y*y;
            x = y*(((a[3]*z+a[2])*z+a[1])*z+a[0])/((((b[3]*z+b[3])*z+b[1])*z+b[0])*z+1.0);

        else:
            z = np.sqrt(-np.log((1.0-y)/2.0));
            x = (((c[3]*z+c[2])*z+c[1])*z+c[0])/((d[1]*z+d[0])*z+1.0);

        # Polish x to full accuracy (Does this twice?)
        x = x - (erf(x) - y) / (2.0/np.sqrt(np.pi) * np.exp(-x*x));
        x = x - (erf(x) - y) / (2.0/np.sqrt(np.pi) * np.exp(-x*x));
    return x


def gamma_so(n, V, U):
    t1 = n * V / (2 * np.sqrt(np.pi))
    t2 = np.exp(- U ** 2 / V ** 2)
    t3 = np.sqrt(np.pi) * U / V
    t4 = 1 + erf(U / V)
    return t1 * (t2 + t3*t4)


def gamma_s(vx, n, V, U):
    t1  = n * V / (2 * np.sqrt(np.pi))
    t2  = np.exp(-       U  ** 2 / V ** 2)
    t2b = np.exp(- (vx - U) ** 2 / V ** 2)
    t3  = np.sqrt(np.pi) * U / V
    t4a = erf((vx - U) / V)
    t4  = erf(      U  / V)
    return t1 * (t2 - t2b + t3*(t4a + t4))


# Minimize this thing (e.g. Find root)
def find_root(vx, n, V, U, Rx):
    return gamma_s(vx, n, V, U) / gamma_so(n, V, U) - Rx


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Test erfinv()
    if False:
        N_val = 100000
        vals  = np.linspace(-1.0, 1.0, N_val)
        diffs = np.zeros(N_val)
        
        for ii in range(N_val):
            diffs[ii] = abs(calc_erfinv(vals[ii]) - calc_erfinv_custom(vals[ii]))
    
        plt.plot(vals, diffs)

    # Test gamma values and/or root finding. What are standard moments that I could
    # plug into this thing?