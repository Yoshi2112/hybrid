# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:54:19 2017

@author: iarey
"""
import numpy as np
import numba as nb

import auxilliary_1D as aux
from simulation_parameters_1D import NX, dx, Te0, ne, q, mu0, kB, ie, HM_amplitude, HM_frequency

@nb.njit()
def uniform_HM_field_value(T):
    '''
    INPUT:
        T -- Current simulation time
        
    OUTPUT:
        Scalar HM sinusoidal wave value (Tesla) at time T with specified wave parameters
    '''
    return HM_amplitude * np.sin(2 * np.pi * HM_frequency * T)


@nb.njit()
def get_curl_E(field, curl, DX=dx):
    ''' Returns a vector quantity for the curl of a field valid at the positions 
    between its gridpoints (i.e. curl(E) -> B-grid, etc.)
    
    INPUT:
        field    -- The 3D field to take the curl of
        DX       -- Spacing between the nodes, mostly for diagnostics. 
                    Defaults to grid spacing specified at initialization.
                 
    OUTPUT:
        curl  -- Finite-differenced solution for the curl of the input field.
        
    NOTE: This function will only work with this specific 1D hybrid code due to both 
          E and B fields having the same number of nodes (due to TSC weighting) and
         the lack of derivatives in y, z
    '''
    curl[:, 0] *= 0
    for ii in nb.prange(1, field.shape[0]):
        curl[ii, 1] = - (field[ii, 2] - field[ii - 1, 2])
        curl[ii, 2] =    field[ii, 1] - field[ii - 1, 1]

    set_periodic_boundaries(curl)
    curl /= DX
    return 


@nb.njit()
def push_B(B, E, temp3D, DT, qq, half_flag=1):
    '''
    Updated to allow time-varying background field
    
    Used as part of predictor corrector for predicing B based on an approximated
    value of E (rather than cycling with the source terms)
    
    B  -- Magnetic field array
    E  -- Electric field array
    DT -- Timestep size, in seconds
    qq  -- Current timestep, as an integer (such that qq*DT is the current simulation time in seconds)
    half_flag -- Flag to signify if pushing to a half step (e.g. 1/2 or 3/2) (1) to a full step (N + 1) (0)
    
    The half_flag can be thought of as 'not having done the full timestep yet' for N + 1/2, so 0.5*DT is
    subtracted from the "full" timestep time
    '''
    time  = (qq - 0.5*half_flag) * DT                    # Current simulation time (pushing to)
    get_curl_E(E, temp3D)

    B[:, 0] -= uniform_HM_field_value(time - 0.5*DT)     # Subtract previous HM field
    B       -= 0.5 * DT * temp3D                         # Advance using curl
    B[:, 0] += uniform_HM_field_value(time)              # Add new  HM field 
    return


@nb.njit()
def set_periodic_boundaries(arr):
    ''' 
    Set boundary conditions for the magnetic field (or values on the B-grid, i.e. curl[E]): 
     -- Average "end" values and assign to first and last grid point
     -- Set ghost (guard) cell values so TSC weighting works
    '''
    end_bit     = 0.5*(arr[1] + arr[NX+1])     # Average end values (for periodic boundary condition)
    arr[1]      = end_bit                      # Better way to do this? end_bit is a temp array, but for fields not so bad
    arr[NX+1]   = end_bit
    
    arr[0]      = arr[NX]
    arr[NX+2]   = arr[2]
    return


@nb.njit()
def curl_B_term(B, curl):
    ''' Returns a vector quantity for the curl of a field valid at the positions 
    between its gridpoints (i.e. curl(B) -> E-grid, etc.)
    
    INPUT:
        B    -- Magnetic field
                 
    OUTPUT:
        curl  -- Finite-differenced solution for the curl of the input field.
        
    NOTE: This function will only work with this specific 1D hybrid code due to both 
          E and B fields having the same number of nodes (due to TSC weighting) and
         the lack of derivatives in y, z
    '''
    curl[:, 0] *= 0
    for ii in nb.prange(1, B.shape[0]):
        curl[ii - 1, 1] = - (B[ii, 2] - B[ii - 1, 2])
        curl[ii - 1, 2] =    B[ii, 1] - B[ii - 1, 1]
    
    # Assign ghost cell values
    curl[B.shape[0] - 1] = curl[2]
    curl[B.shape[0] - 2] = curl[1]
    curl[0] = curl[B.shape[0] - 3]
    
    curl /= (dx * mu0)
    
    return 


@nb.njit()
def get_electron_temp(qn, Te):
    '''
    Calculate the electron temperature in each cell. Depends on the charge density of each cell
    and the treatment of electrons: i.e. isothermal (ie=0) or adiabatic (ie=1)
    '''
    if ie == 0:
        Te[:]     = np.ones(qn.shape[0]) * Te0
    elif ie == 1:
        gamma_e = 5./3. - 1.
        Te[:]     = Te0 * np.power(qn / (q*ne), gamma_e)
    return


@nb.njit()
def get_grad_P(qn, te, grad_P, temp):
    '''
    Returns the electron pressure gradient (in 1D) on the E-field grid using P = nkT and 
    finite difference.
     
    INPUT:
        qn -- Grid charge density
        te -- Grid electron temperature
        DX -- Grid separation, used for diagnostic purposes. Defaults to simulation dx.
        inter_type -- Linear (0) or cubic spline (1) interpolation.
        
    NOTE: Interpolation is needed because the finite differencing causes the result to be deposited on the 
    B-grid. Moving it back to the E-grid requires an interpolation. Cubic spline is desired due to its smooth
    derivatives and its higher order weighting (without the polynomial craziness)
    '''
    grad_P[:] = qn * kB * te / q       # Not actually grad P, just using this array to store Pe
                                       # Putting [:] after array points to memory locations,
                                       # and prevents deferencing

    for ii in nb.prange(1, qn.shape[0]):
        temp[ii] = (grad_P[ii] - grad_P[ii - 1])  / dx
        
    temp[0] = temp[qn.shape[0] - 3]
    aux.interpolate_to_center_cspline1D(temp, grad_P)

    grad_P[0]               = grad_P[qn.shape[0] - 3]
    grad_P[qn.shape[0] - 2] = grad_P[1]
    grad_P[qn.shape[0] - 1] = grad_P[2] 
    return


@nb.njit()
def calculate_E(B, Ji, q_dens, E, Ve, Te, temp3D, temp3D2, temp1D):
    '''Calculates the value of the electric field based on source term and magnetic field contributions, assuming constant
    electron temperature across simulation grid. This is done via a reworking of Ampere's Law that assumes quasineutrality,
    and removes the requirement to calculate the electron current. Based on equation 10 of Buchner (2003, p. 140).
    
    INPUT:
        B   -- Magnetic field array. Displaced from E-field array by half a spatial step.
        Ji  -- Ion current density. Source term, based on particle velocities
        qn  -- Charge density. Source term, based on particle positions
        
    OUTPUT:
        E   -- Updated electric field array
        Ve  -- Electron velocity moment
        Te  -- Electron temperature
    
    arr3D, arr1D are tertiary arrays used for intermediary computations
    '''

    curl_B_term(B, temp3D)                                   # temp3D is now curl B term

    Ve[:, 0] = (Ji[:, 0] - temp3D[:, 0]) / q_dens
    Ve[:, 1] = (Ji[:, 1] - temp3D[:, 1]) / q_dens
    Ve[:, 2] = (Ji[:, 2] - temp3D[:, 2]) / q_dens

    get_electron_temp(q_dens, Te)

    get_grad_P(q_dens, Te, temp1D, temp3D2[:, 0])            # temp1D is now del_p term, temp3D2 slice used for computation

    aux.interpolate_to_center_cspline3D(B, temp3D2)          # temp3d2 is now B_center

    aux.cross_product(Ve, temp3D2, temp3D)                   # temp3D is now Ve x B term

    E[:, 0]  = - temp3D[:, 0] - temp1D[:] / q_dens[:]
    E[:, 1]  = - temp3D[:, 1]
    E[:, 2]  = - temp3D[:, 2]
    
    E[0]                = E[Ji.shape[0] - 3]
    E[Ji.shape[0] - 2]  = E[1]
    E[Ji.shape[0] - 1]  = E[2]
    return 