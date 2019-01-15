# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:54:19 2017

@author: iarey
"""
import numpy as np
import numba as nb

from simulation_parameters_1D import NX, dx, Te0, mhd_equil, q, mu0, kB, Nj, charge, subcycles
from auxilliary_1D            import manage_ghost_cells, cross_product

@nb.njit(cache=True)
def get_curl_E(E):
    ''' Returns a vector quantity for the curl of E valid on the B-field grid
    '''
    curl = np.zeros(NX + 1, 3)
    
    curl[1, :] = E[:-1, 2]- E[1:,   2]
    curl[2, :] = E[1:,  1] - E[:-1, 1]
    return curl / dx

@nb.njit(cache=True)
def get_curl_B(B):
    ''' Returns a vector quantity for the curl of B valid on the E-field grid
    
    Note: First and last array values return zero due to relative array sizes (NX + 1 vs NX + 2)
    '''
    curl = np.zeros(NX + 2, 3)
    
    # Interior points
    curl[1, 1:-1] = B[:-1, 2]- B[1:,   2]
    curl[2, 1:-1] = B[1:,  1] - B[:-1, 1]
    return curl / dx


@nb.njit(cache=True)
def get_grad_P(charge_density, electron_temp):
    ''' Returns the electron pressure gradient (in 1D) on the B-field grid
    '''
    Pe = charge_density * kB * electron_temp / q
    
    grad       = np.zeros(NX + 1)
    grad[1:-1] = (Pe[:-1] - Pe[1:])  / dx
    grad       = interpolate_to_center(grad)
    return grad


@nb.njit(cache=True)
def interpolate_to_center(val):
    ''' Interpolates cell edge values (i.e. B-grid quantities) to cell centers (i.e. E-grid quantities)
    Note: First and last array values return zero due to relative array sizes (NX + 1 vs NX + 2)
    '''
    center          = np.zeros(NX + 2, 3)
    center[1:-1]    = 0.5*(val[:-1] + val[1:])
    return center


@nb.njit(cache=True)
def cyclic_leapfrog(B, n_i, J_i, DT):
    H  = 0.5 * DT                                               # Half-timestep
    dh = H / subcycles                                          # Subcycle timestep
    
    B1 = np.copy(B)
    B2 = np.copy(B) - dh * get_curl_E(calculate_E(B, J_i, n_i)) # Advance one copy half a timestep
    
    if subcycles == 1 or subcycles == None:                     # Return if subcycles not needed
        return B2
    
    for ii in range(subcycles - 1):             
        if ii%2 == 0:
            B1  -= 2 * dh * get_curl_E(calculate_E(B2, J_i, n_i))
        else:
            B2  -= 2 * dh * get_curl_E(calculate_E(B1, J_i, n_i))
            
    if ii%2 == 0:
        B2  -= dh * get_curl_E(calculate_E(B1, J_i, n_i))
    else:
        B1  -= dh * get_curl_E(calculate_E(B2, J_i, n_i))

    B = 0.5 * (B1 + B2)                                         # Average solutions: Could put an evaluation step here
    return B


@nb.njit(cache=True)
def calculate_E(B, J_i, n_i):
    '''Calculates the value of the electric field based on source term and magnetic field contributions, assuming constant
    electron temperature across simulation grid. This is done via a reworking of Ampere's Law that assumes quasineutrality,
    and removes the requirement to calculate the electron current. Based on equation 10 of Buchner (2003, p. 140).

    INPUT:
        B   -- Magnetic field array. Displaced from E-field array by half a spatial step.
        J_i -- Ion current density. Source term, based on particle velocities
        n_i -- Ion number density. Source term, based on particle positions

    OUTPUT:
        E_out -- Updated electric field array
    '''
    size = NX + 2

    E_out = np.zeros((size, 3))                 # Output array - new electric field
    J     = np.zeros((size, 3))                 # Ion current
    qn    = np.zeros(size,    )                 # Ion charge density
    Te    = np.ones(size) * Te0                 # Electron temperature array

    for jj in range(Nj):
        qn += charge[jj] * n_i[:, jj]           # Total charge density, sum(qj * nj)

        for kk in range(3):
            J[:, kk]  += J_i[:, jj, kk]         # Total ion current vector: J_k = qj * nj * Vj_k

    B_center = interpolate_to_center(B)
    JxB      = cross_product(J, B_center)    
    curlB    = get_curl_B(B)
    BdB      = cross_product(B_center, curlB) / mu0
    del_p    = get_grad_P(qn, Te)

    E_out[:, 0] = (- JxB[:, 0] - BdB[:, 0] - del_p ) / qn
    E_out[:, 1] = (- JxB[:, 1] - BdB[:, 1]         ) / qn
    E_out[:, 2] = (- JxB[:, 2] - BdB[:, 2]         ) / qn

    E_out[0]        = E_out[NX]
    E_out[NX + 1]   = E_out[1]
    return E_out
