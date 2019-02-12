# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:54:19 2017

@author: iarey
"""
import numpy as np
import numba as nb

from auxilliary_1D            import cross_product, interpolate_to_center
from simulation_parameters_1D import NX, dx, Te0, ne, q, mu0, kB, subcycles, ie


#@nb.njit()
def get_curl(field, DX=dx):
    ''' Returns a vector quantity for the curl of a field valid at the positions 
    between its gridpoints (i.e. curl(E) -> B-grid, etc.)
    
    INPUT:
        field -- The 3D field to take the curl of
        DX    -- Spacing between the nodes, mostly for diagnostics. 
                 Defaults to grid spacing specified at initialization.
                 
    OUTPUT:
        curl  -- Finite-differenced solution for the curl of the input field.
        
    NOTE: This function will only work with this specific hybrid code due to both 
          E and B fields having the same number of nodes (due to TSC weighting)
    '''
    curl = np.zeros(field.shape)
    
    for ii in nb.prange(1, field.shape[0]):
        curl[ii, 1] = - (field[ii, 2] - field[ii - 1, 2])
        curl[ii, 2] =    field[ii, 1] - field[ii - 1, 1]
    return curl / DX


@nb.njit()
def get_electron_temp(qn):
    '''
    Calculate the electron temperature in each cell. Depends on the charge density of each cell
    and the treatment of electrons: i.e. isothermal (ie=0) or adiabatic (ie=1)
    '''
    if ie == 0:
        te = np.ones(NX + 3) * Te0
    elif ie == 1:
        gamma_e = 5./3. - 1.
        te      = Te0 * np.power(qn / (q*ne), gamma_e)
    return te


@nb.njit()
def get_grad_P(qn, te):
    '''
    Returns the electron pressure gradient (in 1D) on the E-field grid.
    Could eventually modify this to just do a normal centred difference.
    
    For isothermal approximation (ie = 0): Grad P depends only on charge gradient, Te constant
    
    For adiabatic approximation (ie = 1): Grad P depends on gradients in both Te and ne (i.e. chain rule)
    '''
    
    if ie == 0: 
        grad_pe_B     = np.zeros(NX + 3)
        grad_P        = np.zeros(NX + 3)
        
        Pe = qn * kB * te / q
    
        grad_pe_B[1:NX+2] = (Pe[1:NX+2] - Pe[:NX+1])  / dx                     # Finite difference will move it to the B-grid
        grad_P[1:NX+1]    = 0.5*(grad_pe_B[1: NX+1] + grad_pe_B[2:NX+2])       # Re-interpolate to E-grid
    
    elif ie == 1:
        grad_ne  = np.zeros(NX + 3)                                            # Grads of quantities on B
        grad_te  = np.zeros(NX + 3)
        cent_gne = np.zeros(NX + 3)                                            # Grads re-centered onto E-field grid 
        cent_gte = np.zeros(NX + 3)
         
        grad_ne[1:NX+2]  = (qn[1:NX+2] - qn[:NX+1]) / (q*dx)                   # Finite differencing onto B-grid       
        grad_te[1:NX+2]  = (te[1:NX+2] - te[:NX+1]) /    dx 
        
        cent_gne[1:NX+1] = 0.5*(grad_ne[1: NX+1] + grad_ne[2:NX+2])            # Place back on E-grid (average/linear interpolation)
        cent_gte[1:NX+1] = 0.5*(grad_te[1: NX+1] + grad_te[2:NX+2])
        
        grad_P = kB * (te * cent_gne + (qn/q) * cent_gte)                      # Calculate final gradient
    return grad_P


@nb.njit()
def set_periodic_boundaries(B):
    ''' Set boundary conditions for the magnetic field: 
         -- Average "end" values and assign to first and last grid point
         -- Set ghost cell values so TSC weighting works
    '''
    end_bit = 0.5*(B[1] + B[NX+1])                              # Average end values (for periodic boundary condition)
    B[1]      = end_bit
    B[NX+1]   = end_bit
    
    B[0]      = B[NX]
    B[NX+2]   = B[2]
    return B


@nb.njit()
def cyclic_leapfrog(B, n_i, J_i, DT):

    H  = 0.5 * DT                                               # Half-timestep
    dh = H / subcycles                                          # Subcycle timestep

    B1 = np.copy(B)
    B2 = np.copy(B) - dh * get_curl(calculate_E(B, J_i, n_i)) # Advance one copy half a timestep
    B2 = set_periodic_boundaries(B2)                            # Average grid endpoints (for BC's) and fill ghost cells
    
    if subcycles == 1:                                          # Return if subcycles not needed
        return B2
    
    ii = 1                                                      # Prevents error if subcycles is 0 or 1
    for ii in range(subcycles - 1):             
        if ii%2 == 0:
            B1  -= 2 * dh * get_curl(calculate_E(B2, J_i, n_i))
            B1   = set_periodic_boundaries(B1)
        else:
            B2  -= 2 * dh * get_curl(calculate_E(B1, J_i, n_i))
            B2   = set_periodic_boundaries(B2)
            
    if ii%2 == 0:
        B2  -= dh * get_curl(calculate_E(B1, J_i, n_i))
        B2   = set_periodic_boundaries(B2)
    else:
        B1  -= dh * get_curl(calculate_E(B2, J_i, n_i))
        B1   = set_periodic_boundaries(B1)

    B = 0.5 * (B1 + B2)                                         # Average solutions: Could put an evaluation/threshold step here

    return B


@nb.njit()
def calculate_E(B, J, qn):
    '''Calculates the value of the electric field based on source term and magnetic field contributions, assuming constant
    electron temperature across simulation grid. This is done via a reworking of Ampere's Law that assumes quasineutrality,
    and removes the requirement to calculate the electron current. Based on equation 10 of Buchner (2003, p. 140).

    INPUT:
        B   -- Magnetic field array. Displaced from E-field array by half a spatial step.
        J   -- Ion current density. Source term, based on particle velocities
        qn  -- Charge density. Source term, based on particle positions

    OUTPUT:
        E_out -- Updated electric field array
    '''
    size     = NX + 3
    
    Te       = get_electron_temp(qn)

    B_center = interpolate_to_center(B)
    JxB      = cross_product(J, B_center)    
    curlB    = get_curl(B)
    BdB      = cross_product(B_center, curlB) / mu0
    del_p    = get_grad_P(qn, Te)

    E_out       = np.zeros((size, 3))                 
    E_out[:, 0] = (- JxB[:, 0] - BdB[:, 0] - del_p ) / qn
    E_out[:, 1] = (- JxB[:, 1] - BdB[:, 1]         ) / qn
    E_out[:, 2] = (- JxB[:, 2] - BdB[:, 2]         ) / qn

    E_out[0]        = E_out[NX]
    E_out[NX + 1]   = E_out[1]
    E_out[NX + 2]   = E_out[2]                                  # This doesn't really get used, but might as well
    return E_out
