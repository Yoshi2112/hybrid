# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:24:46 2019

@author: Yoshi
"""
#import sys
#data_scripts_dir = 'F://Google Drive//Uni//PhD 2017//Data//Scripts//'
#sys.path.append(data_scripts_dir)
#from analysis_scripts import analytic_signal

import numpy as np
import numba as nb
import os, pdb
import analysis_config as cf
#import pdb
'''
Dump general processing scripts here that don't require global variables: i.e. 
they are completely self-contained or can be easily imported.

These will often be called by plotting scripts so that the main 'analysis'
script is shorter and less painful to work with

If a method requires more than a few functions, it will be split into its own 
module, i.e. get_growth_rates
'''
qp  = 1.602e-19               # Elementary charge (C)
c   = 3e8                     # Speed of light (m/s)
me  = 9.11e-31                # Mass of electron (kg)
mp  = 1.67e-27                # Mass of proton (kg)
e   = -qp                     # Electron charge (C)
mu0 = (4e-7) * np.pi          # Magnetic Permeability of Free Space (SI units)
kB  = 1.38065e-23             # Boltzmann's Constant (J/K)
e0  = 8.854e-12               # Epsilon naught - permittivity of free space





def get_thread_values():
    '''
    Function to calculate number of particles to work on each thread, and the
    start index of each batch of particles.
    '''

    n_threads    = nb.get_num_threads()
    N_per_thread = (cf.N//n_threads)*np.ones(n_threads, dtype=int)
    
    if cf.N%n_threads == 0:
        n_start_idxs = np.arange(n_threads)*N_per_thread
    else:
        leftovers = cf.N%n_threads
        for _lo in range(leftovers):
            N_per_thread[_lo] += 1
        n_start_idxs = np.asarray([np.sum(N_per_thread[0:_si]) for _si in range(0, n_threads)])
        
    # Check values (this is important)
    if N_per_thread.sum() != cf.N:
        raise ValueError('Number of particles per thread unequal to total number of particles')
        
    for _ii in range(1, n_start_idxs.shape[0] + 1):
        if _ii == n_start_idxs.shape[0]:
            n_in_thread = cf.N - n_start_idxs[-1]
        else:
            n_in_thread = n_start_idxs[_ii] - n_start_idxs[_ii - 1]
        if n_in_thread != N_per_thread[_ii - 1]:
            raise ValueError('Thread particle indices are not correct. Check this.')
    return n_threads, N_per_thread, n_start_idxs


@nb.njit(parallel=True)
def deposit_moments(vel, idx, Ie, W_elec, n_threads, N_per_thread, n_start_idxs):
    ni_threads = np.zeros((n_threads, NC, Nj), dtype=np.float64)
    nu_threads = np.zeros((n_threads, NC, Nj, 3, ), dtype=np.float64)
    for tt in nb.prange(n_threads):        
        for ii in range(n_start_idxs[tt], n_start_idxs[tt]+N_per_thread[tt]):
            if idx[ii] < Nj:
                for kk in nb.prange(3):
                    nu_threads[tt, Ie[ii],     idx[ii], kk] += W_elec[0, ii] * vel[kk, ii]
                    nu_threads[tt, Ie[ii] + 1, idx[ii], kk] += W_elec[1, ii] * vel[kk, ii]
                    nu_threads[tt, Ie[ii] + 2, idx[ii], kk] += W_elec[2, ii] * vel[kk, ii]
                
                ni_threads[tt, Ie[ii],     idx[ii]] += W_elec[0, ii]
                ni_threads[tt, Ie[ii] + 1, idx[ii]] += W_elec[1, ii]
                ni_threads[tt, Ie[ii] + 2, idx[ii]] += W_elec[2, ii]
    ni = ni_threads.sum(axis=0)
    nu = nu_threads.sum(axis=0)
    return ni, nu


def get_number_densities(pos, vel, idx):
    '''
    Function to calculate the partial moments (unweighted by charge/contribution)
    ni is equivalent to raw macroparticle density, multiply by n_contr to get
    equivalent real number density (and charge to get charge density)
    nu is equivalent to raw macroparticle flux, multiply by n_contr to get equivalent
    real number flux, and also by charge to get charge density.
    Note: Is it flux?
    
    Split into get_number_densities() and deposit_moments() so numba can do it
    '''
    # Set variables as global so numba can access them
    global NX, ND, NC, Nj, N, dx, xmax, xmin, field_periodic
    NX = cf.NX
    ND = cf.ND
    NC = cf.NC
    Nj = cf.Nj
    N = cf.N
    
    xmax = cf.xmax
    xmin = cf.xmin
    dx = cf.dx
    
    field_periodic = cf.field_periodic
    
    Ie     = np.zeros(    N,  dtype=int)
    W_elec = np.zeros((3, N), dtype=float)
    
    n_threads, N_per_thread, n_start_idxs = get_thread_values()
    assign_weighting_TSC(pos, Ie, W_elec, E_nodes=True)
    ni, nu = deposit_moments(vel, idx, Ie, W_elec, n_threads, N_per_thread, n_start_idxs)
    return ni, nu


@nb.njit()
def eval_B0x(x):
    return cf.B_eq * (1. + cf.a * x*x)


@nb.njit()
def eval_B0_particle(pos, vel, mi, qi):
    '''
    Calculates the B0 magnetic field at the position of a particle. B0x is
    non-uniform in space, and B0r (split into y,z components) is the required
    value to keep div(B) = 0
    
    These values are added onto the existing value of B at the particle location,
    Bp. B0x is simply equated since we never expect a non-zero wave field in x.
        
    Could totally vectorise this. Would have to change to give a particle_temp
    array for memory allocation or something
    
    Technically already vectorized but calculating rL would create a new array which
    would kill performance (and require the temp array). Bp would also need to be (3, N)
    '''
    Bp = np.zeros(3, dtype=np.float64)
    constant = cf.a * cf.B_eq
    Bp[0]    = eval_B0x(pos)  
    
    l_cyc    = qi*Bp[0]/mi
    Bp[1]   += constant * pos * vel[2] / l_cyc
    Bp[2]   -= constant * pos * vel[1] / l_cyc
    return Bp


def get_B0_particle(x, v, B, sp):
    
    @nb.njit()
    def get_b1(pos, mag):
        xp_mag = np.zeros(3)
        epsil  = 1e-15

        particle_transform = cf.xmax + cf.ND*cf.dx  + epsil   # Offset to account for E/B grid and damping nodes
        
        xp          = (pos + particle_transform) / cf.dx      # Shift particle position >= 0
        Ib          = int(round(xp) - 1.0)                    # Get leftmost to nearest node
        delta_left  = Ib - xp                                 # Distance from left node in grid units
    
        W0 = 0.5  * np.square(1.5 - abs(delta_left))    # Get weighting factors
        W1 = 0.75 - np.square(delta_left + 1.)
        W2 = 1.0  - W0 - W1
        
        for kk in range(3):
            xp_mag[kk] = W0 * mag[Ib, kk] + W1 * mag[Ib + 1, kk] + W2 * mag[Ib + 2, kk]
        
        return xp_mag
    
    b1 = get_b1(x, B)
    
    B0_xp    = np.zeros(3)
    B0_xp[0] = eval_B0x(x)    
    b1t      = np.sqrt(b1[0] ** 2 + b1[1] ** 2 + b1[2] ** 2)
    l_cyc    = (cf.charge[sp] / cf.mass[sp]) * (B0_xp[0] + b1t)
    
    fac      = cf.a * cf.B_eq * x / l_cyc
    B0_xp[1] = v[2] * fac
    B0_xp[2] =-v[1] * fac
    return B0_xp


@nb.njit()
def interpolate_B_to_center(bx, by, bz, zero_boundaries=False):
    ''' 
    Interpolates magnetic field from cell edges to cell centers (where the E
    field is measured). 
    
    bx, by, bz are each (time, space) ndarrays
    
    Also adds on background field in x component (for JxB calculation)
    
    VERIFIED
    '''
    n_times = bx.shape[0]
    NC      = cf.NC                                   # Number of cells
    
    y2x = np.zeros(NC + 1, dtype=nb.float64)          # Second derivatives on B grid
    y2y = np.zeros(NC + 1, dtype=nb.float64)
    y2z = np.zeros(NC + 1, dtype=nb.float64)
    
    bxi = np.zeros((n_times, NC), dtype=nb.float64)   # Interpolation on E grid
    byi = np.zeros((n_times, NC), dtype=nb.float64)
    bzi = np.zeros((n_times, NC), dtype=nb.float64)
    
    # For each time (tt): Calculate second derivative (for each component)
    for tt in range(n_times):
        y2x *= 0
        y2y *= 0
        y2z *= 0
        
        # Interior B-nodes, Centered difference
        for ii in range(1, NC):
            y2x[ii] = bx[tt, ii + 1] - 2*bx[tt, ii] + bx[tt, ii - 1]
            y2y[ii] = by[tt, ii + 1] - 2*by[tt, ii] + by[tt, ii - 1]
            y2z[ii] = bz[tt, ii + 1] - 2*bz[tt, ii] + bz[tt, ii - 1]
                
        # Edge B-nodes, Zero or Forwards/Backwards difference
        if zero_boundaries == True:
            y2x[0 ] = 0.    ;   y2y[0 ] = 0.    ;   y2z[0 ] = 0.
            y2x[NC] = 0.    ;   y2y[NC] = 0.    ;   y2z[NC] = 0.
        else:
            y2x[0]  = 2*bx[tt, 0 ] - 5*bx[tt, 1]      + 4*bx[tt, 2]      - bx[tt, 3]
            y2x[NC] = 2*bx[tt, NC] - 5*bx[tt, NC - 1] + 4*bx[tt, NC - 2] - bx[tt, NC - 3]
            
            y2y[0]  = 2*by[tt, 0 ] - 5*by[tt, 1]      + 4*by[tt, 2]      - by[tt, 3]
            y2y[NC] = 2*by[tt, NC] - 5*by[tt, NC - 1] + 4*by[tt, NC - 2] - by[tt, NC - 3]
            
            y2z[0]  = 2*bz[tt, 0 ] - 5*bz[tt, 1]      + 4*bz[tt, 2]      - bz[tt, 3]
            y2z[NC] = 2*bz[tt, NC] - 5*bz[tt, NC - 1] + 4*bz[tt, NC - 2] - bz[tt, NC - 3]
        
        # Do spline interpolation: E[ii] is bracketed by B[ii], B[ii + 1]
        for ii in range(NC):
            bxi[tt, ii] = 0.5 * (bx[tt, ii] + bx[tt, ii + 1] + (1/6) * (y2x[ii] + y2x[ii + 1]))
            byi[tt, ii] = 0.5 * (by[tt, ii] + by[tt, ii + 1] + (1/6) * (y2y[ii] + y2y[ii + 1]))
            bzi[tt, ii] = 0.5 * (bz[tt, ii] + bz[tt, ii + 1] + (1/6) * (y2z[ii] + y2z[ii + 1]))
                
    return bxi, byi, bzi


def get_electron_temp(qn):
    '''
    Calculate the electron temperature in each cell. Depends on the charge density of each cell
    and the treatment of electrons: i.e. isothermal (ie=0) or adiabatic (ie=1)
    
    qn :: (time, space)
    '''
    Te = np.zeros(qn.shape, dtype=float)
    for ii in range(qn.shape[0]):
        if cf.ie == 0:
            Te[ii, :] = np.ones(qn.shape[0]) * cf.Te0
        elif cf.ie == 1:
            gamma_e = 5./3. - 1.
            Te[ii, :] = cf.Te0 * np.power(qn[ii, :] / (qp * cf.ne), gamma_e)
    return Te


def get_grad_P(qn, te):
    '''
    Returns the electron pressure gradient (in 1D) on the E-field grid using P = nkT and 
    finite difference.
     
    INPUT:
        qn     -- Grid charge density
        te     -- Grid electron temperature
        grad_P -- Output array for electron pressure gradient
        temp   -- intermediary array used to store electron pressure, since both
                  density and temperature may vary (with adiabatic approx.)
        
    Forwards/backwards differencing at the simulation cells at the edge of the
    physical space domain. Guard cells set to zero.
    
    qn, te :: (time, space)
    '''
    Pe     = qn * kB * te / qp       # Store Pe in grad_P array for calculation

    # Central differencing, internal points
    grad_P = np.zeros(qn.shape)
    for ii in nb.prange(1, qn.shape[1] - 1):
        grad_P[:, ii] = (Pe[:, ii + 1] - Pe[:, ii - 1])
    
    # Forwards/Backwards difference at physical boundaries
    grad_P    /= (2*cf.dx)
    
    return grad_P



def get_curl_B(bx, by, bz):
    '''
    Each b component is a (time, space) ndarray. This looks fine.
    '''
    curl_B = np.zeros((bx.shape[0], bx.shape[1] - 1, 3), dtype=np.float64)
    
    for ii in nb.prange(bx.shape[1] - 1):
        curl_B[:, ii, 1] = -(bz[:, ii + 1] - bz[:, ii])
        curl_B[:, ii, 2] =   by[:, ii + 1] - by[:, ii]
    
    curl_B /= (cf.dx * mu0)

    return curl_B



def cross_product(ax, ay, az, bx, by, bz):
    '''
    Vector (cross) product between two vectors, A and B of same dimensions.
    all ai, bi are expected to be (time, space) ndarrays
    '''
    C = np.zeros((az.shape[0], az.shape[1], 3), dtype=np.float64)

    for ii in nb.prange(az.shape[0]):
        C[ii, :, 0] += ay[ii] * bz[ii]
        C[ii, :, 1] += az[ii] * bx[ii]
        C[ii, :, 2] += ax[ii] * by[ii]
        
        C[ii, :, 0] -= az[ii] * by[ii]
        C[ii, :, 1] -= ax[ii] * bz[ii]
        C[ii, :, 2] -= ay[ii] * bx[ii]
    return C



def calculate_E_components(bx, by, bz, jx, jy, jz, q_dens):
    '''
    '''
    # Need to calculate (Fatemi, 2017):
    # Ji x B / qn               Convective Term     LOOKS GOOD
    # del(p) / qn               Ambipolar term      LOOKS GOOD
    # Bx(curl B) / qn*mu0       Hall Term           LOOKS GOOD
    # This version of the code doesn't include an Ohmic term, since eta = 0

    bxi, byi, bzi = interpolate_B_to_center(bx, by, bz)
    B0  = eval_B0x(cf.E_nodes)
    for tt in range(bxi.shape[0]):
        bxi[tt] += B0
        
# =============================================================================
#     import sys
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots(2, figsize=(15, 10))
#     
#     time  = 120
#     space = bx.shape[1] // 2
#     
#     ax[0].plot(cf.B_nodes, by[ time], marker='o', c='b')
#     ax[0].plot(cf.E_nodes, byi[time], marker='x', c='r')
#     
#     ax[1].plot(by[ :, space], marker='o', c='b')
#     ax[1].plot(byi[:, space], marker='x', c='r')
#     
#     sys.exit()
# =============================================================================
    
    # Hall Term
    curl_B = get_curl_B(bx, by, bz)
    BdB    = cross_product(bxi, byi, bzi, curl_B[:, :, 0], curl_B[:, :, 1], curl_B[:, :, 2])
    
    # Ambipolar Term
    Te     = get_electron_temp(q_dens)
    grad_P = get_grad_P(q_dens, Te)                           # temp1D is now del_p term, temp3D2 slice used for computation
    grad_P/= q_dens[:]
    
    # Convective Term
    JxB  = cross_product(jx, jy, jz, bxi, byi, bzi)           # temp3De is now Ve x B term
    
    for ii in range(3):
        BdB[:, :, ii] /= q_dens[:]
        JxB[:, :, ii] /= q_dens[:]
        
    return BdB, grad_P, JxB
    #     hall, amb   , conv