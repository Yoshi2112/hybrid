# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 17:24:21 2019

@author: Yoshi
"""
import pdb
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from timeit import default_timer as timer

@nb.njit(parallel=True)
def linterp_numba(_B, _B_cent):
    '''
    Quick and easy function to calculate B on the E-grid using scipy's cubic
    spline interpolation (mine seems to be broken). Could probably make this 
    myself and more efficient later, but need to eliminate problems!
    '''
    for _ii in nb.prange(_B_cent.shape[0]):
        _B_cent[_ii, 1] = 0.5 * (_B[_ii, 1] + _B[_ii + 1, 1])
        _B_cent[_ii, 2] = 0.5 * (_B[_ii, 2] + _B[_ii + 1, 2])
    return 

def do_interp(_B, _B_cent, _kind='linear', _sort=True):
    for _ii in range(2):
        _interp_func    = interp1d(xB, _B[:, _ii], kind=_kind, assume_sorted=_sort)
        _B_cent[:, _ii] = _interp_func(xE)
    return



def calc_quadspline(yarr):
    # Get first derivatives at B-points
    # Set derivatives at boundaries to be zero (optional later)
    deriv = np.zeros(yarr.shape[0])
    for ii in range(1, yarr.shape[0]-1):
        deriv[ii] = (yarr[ii+1] - yarr[ii-1]) / (2*dx)

    # Use derivatives to calc. interpolants
    yarr_new = np.zeros(yarr.shape[0]-1)
    for ii in range(yarr_new.shape[0]):
        yarr_new[ii] = 0.25*(yarr[ii + 1] - yarr[ii] - dx*deriv[ii]) + 0.5*dx*deriv[ii] + yarr[ii]
    return yarr_new


def calc_quadspline2(yarr):
    # From that spline_interpolation pdf
    # Even worse
    deriv = np.zeros(yarr.shape[0])
    for ii in range(yarr.shape[0] - 1):
        deriv[ii+1] = -deriv[ii] + 2*(yarr[ii + 1] - yarr[ii])/dx

    # Use derivatives to calc. interpolants
    ynew = np.zeros(yarr.shape[0]-1)
    for ii in range(ynew.shape[0]):
        ynew[ii] = 0.5*(deriv[ii+1]-deriv[ii])/dx * (dx/2)**2 + deriv[ii]*(dx/2) + yarr[ii]
    return ynew


def calc_quadspline_coeffs_matlab():
    '''
    Python port of code from:
    https://au.mathworks.com/matlabcentral/fileexchange/50622-quadratic-spline-interpolation
    '''
    x = np.arange(-5, 5, 0.5)
    y = np.exp(x)
    N = x.shape[0] - 1
    
    V = np.zeros((3*N+1))
    Z = np.zeros((3*N+1, 3*N+1))
    
    jj=0; ff=0
    for ii in np.arange(2, 2*N+2, 2):    
        Z[ii, ff  ] = x[jj]**2
        Z[ii, ff+1] = x[jj]
        Z[ii, ff+2] = 1.
        V[ii]       = y[jj]
        jj += 1
        
        Z[ii+1, ff]   = x[jj]**2
        Z[ii+1, ff+1] = x[jj]
        Z[ii+1, ff+2] = 1.
        V[ii+1]       = y[jj]
        
        ff += 3
        
    # Filling Matrix from smoothing condition
    jj =0
    ll =1
    for ii in np.arange(2*N+2, 3*N+1):
        Z[ii, jj]   = 2*x[ll]
        Z[ii, jj+1] = 1.
        Z[ii, jj+3]  = -2*x[ll]
        Z[ii, jj+4]  = -1
        
        jj += 3
        ll += 1
    
    # Adjusting the value of a1 to be zero "Linear Spline"
    Z[0, 0] = 1
    
    # Inverting and obtaining the coeffiecients, Plotting
    Coeff = np.linalg.inv(Z)*V
    
    plt.ioff()
    plt.figure()
    jj=0
    for ii in range(N):
        xrange = np.linspace(x[ii], x[ii + 1], 10)
        curve  = Coeff[jj]*xrange ** 2 + Coeff[jj+1]*xrange + Coeff[jj+2]
        jj    += 3;
        plt.plot(xrange, curve)
    plt.scatter(x, y, c='r', marker='x')
    plt.show()
    return


if __name__ == '__main__':
    dx = 1.0
    Nx = 128
    xB = np.arange(0.0, Nx*dx, dx)
    xE = np.arange(dx/2, Nx*dx - dx/2, dx)
    
    #calc_quadspline_coeffs_matlab()
    
    B      = np.zeros((xB.shape[0], 2), dtype=float)
    B_cent = np.zeros((xE.shape[0], 2), dtype=float)
    
    noise = np.random.normal(0.0, 0.0, size=Nx)
    B[:, 0] = np.sin(2*np.pi*xB / 16.)
    B[:, 1] = np.cos(2*np.pi*xB / 16.)
    
    my_quad = calc_quadspline2(B[:, 0])
    sp_quad = interp1d(xB, B[:, 0], kind='quadratic')(xE)
    analytic= np.sin(2*np.pi*xE / 16.) 
    
    my_err = np.abs(my_quad - analytic).sum()
    sp_err = np.abs(sp_quad - analytic).sum()
    
    plt.plot(xB, B[:, 0], label='raw')
    plt.plot(xE, my_quad, label='mine')
    plt.plot(xE, sp_quad, label='scipy')
    plt.plot(xE, analytic, label='anal')
    plt.legend()
    plt.show()
    
    print('My error:', my_err)
    print('SP error:', sp_err)
    
# =============================================================================
#     start = timer()
#     for xx in range(10000):
#         #linterp_numba(B, B_cent)
#         do_interp(B, B_cent, _kind='linear', _sort=False)
#     itime = round(timer() - start, 3)
#     print(f'Interpolation: {itime} seconds')
# =============================================================================
    
# =============================================================================
#     sort = True
#     
#     linear_func = interp1d(xB, B, kind='linear', assume_sorted=sort)
#     yE_linear   = linear_func(xE)
#     
#     quad_func = interp1d(xB, B, kind='quadratic', assume_sorted=sort)
#     yE_quad   = quad_func(xE)
#     
#     cubic_func = interp1d(xB, B, kind='cubic', assume_sorted=sort)
#     yE_cubic   = cubic_func(xE)
#     
#     plt.plot(xB, B, label='Data (B)')
#     plt.plot(xE, yE_linear, label='Linear (E)')
#     plt.plot(xE, yE_quad, label='Quadratic (E)')
#     plt.plot(xE, yE_cubic, label='Cubic (E)')
#     
#     plt.legend()
# =============================================================================
    
# =============================================================================
#     for ii in range(Nx):
#         plt.axvline(xB[ii], c='k', ls='--', alpha=0.5)
#         if ii < xE.shape[0]:
#             plt.axvline(xE[ii], c='r', ls='--', alpha=0.5)
# =============================================================================
        
    plt.show()