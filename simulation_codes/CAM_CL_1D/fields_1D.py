# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:54:19 2017

@author: iarey
"""
import numpy as np
import numba as nb

from auxilliary_1D            import cross_product
from simulation_parameters_1D import NX, dx, Te0, q, mu0, kB, subcycles

@nb.njit()
def get_curl_E(E):
    ''' Returns a vector quantity for the curl of E valid on the B-field grid
    '''
    curl = np.zeros((NX + 1, 3))
    
    curl[:, 1] = E[:-1, 2]- E[1:,   2]
    curl[:, 2] = E[1:,  1] - E[:-1, 1]
    return curl / dx

@nb.njit()
def get_curl_B(B):
    ''' Returns a vector quantity for the curl of B valid on the E-field grid
    
    Note: First and last array values return zero due to relative array sizes (NX + 1 vs NX + 2)
    '''
    curl = np.zeros((NX + 2, 3))
    
    # Interior points
    curl[1:-1, 1] = B[:-1, 2]- B[1:,   2]
    curl[1:-1, 2] = B[1:,  1] - B[:-1, 1]
    return curl / dx


@nb.njit()
def get_grad_P(charge_density, electron_temp):
    ''' Returns the electron pressure gradient (in 1D) on the E-field grid.
    Could eventually modify this to just do a normal centred difference.
    '''
    Pe = charge_density * kB * electron_temp / q
    
    grad = np.zeros(NX + 1)
    grad = (Pe[:-1] - Pe[1:])  / dx                 # Finite difference will move it to the B-grid
    
    centered_grad = np.zeros(NX + 2)
    centered_grad[1:-1] = 0.5*(grad[:-1] + grad[1:])
    return centered_grad


@nb.njit()
def interpolate_to_center(val):
    ''' Interpolates vector cell edge values (i.e. B-grid quantities) to cell centers (i.e. E-grid quantities)
    Note: First and last array values return zero due to relative array sizes (NX + 1 vs NX + 2)
    '''
    center = np.zeros((NX + 2, 3))
    
    for ii in range(3):
        center[1:-1, ii] = 0.5*(val[:-1, ii] + val[1:, ii])
    return center


@nb.njit()
def set_periodic_boundaries(B):
    ''' Set boundary conditions for the magnetic field: Average end values and assign to first and last grid point
    '''
    end_bit = 0.5*(B[0] + B[NX])                              # Average end values (for periodic boundary condition)
    B[0]   = end_bit
    B[NX]  = end_bit
    return B


#@nb.njit()
def cyclic_leapfrog(B, n_i, J_i, DT):
    H  = 0.5 * DT                                               # Half-timestep
    dh = H / subcycles                                          # Subcycle timestep

    B1 = np.copy(B)
    B2 = np.copy(B) - dh * get_curl_E(calculate_E(B, J_i, n_i)) # Advance one copy half a timestep
    
    end_bit = 0.5*(B2[0] + B2[NX])                              # Average end values (for periodic boundary condition)
    B2[0]   = end_bit
    B2[NX]  = end_bit
    
    if subcycles == 1:                                          # Return if subcycles not needed
        return B2
    
    for ii in range(subcycles - 1):             
        if ii%2 == 0:
            B1  -= 2 * dh * get_curl_E(calculate_E(B2, J_i, n_i))
            B1   = set_periodic_boundaries(B1)
        else:
            B2  -= 2 * dh * get_curl_E(calculate_E(B1, J_i, n_i))
            B2   = set_periodic_boundaries(B2)
            
    if ii%2 == 0:
        B2  -= dh * get_curl_E(calculate_E(B1, J_i, n_i))
        B2   = set_periodic_boundaries(B2)
    else:
        B1  -= dh * get_curl_E(calculate_E(B2, J_i, n_i))
        B1   = set_periodic_boundaries(B1)

    B = 0.5 * (B1 + B2)                                         # Average solutions: Could put an evaluation step here
    
    return B


#@nb.njit()
def calculate_E(B, J, qn):
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
    Te    = np.ones(size) * Te0                 # Electron temperature array

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
