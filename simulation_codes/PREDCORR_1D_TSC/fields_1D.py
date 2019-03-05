# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:54:19 2017

@author: iarey
"""
import numpy as np
import numba as nb

from particles_1D             import advance_particles_and_moments
from auxilliary_1D            import cross_product, interpolate_to_center_cspline3D, interpolate_to_center_cspline1D, interpolate_to_center_linear_1D
from simulation_parameters_1D import NX, dx, Te0, ne, q, mu0, kB, ie


@nb.njit()
def predictor_corrector(B, E_int, E_half, pos, vel, q_dens, Ie, W_elec, idx, DT):
    '''
    Isolated predictor-corrector method for easy debugging/coding. Predicts the
    electric and magnetic fields at the full timestep by using a pretend push of
    the particles. Computationally expensive, but accurate.
    
    INPUT:
        B          -- Magnetic field at N + 1/2
        E_int      -- Electric field at N
        E_half     -- Electric field at N + 1/2
        pos        -- Particle positions  at N + 1
        vel        -- Particle velocities at N + 1/2
        q_dens_adv -- Charge density at N + 1 (advanced density)
        Ji         -- Current density at N + 1/2
        Ie         -- Nearest nodes   for each particle
        W_elec     -- Node weightings for each particle
        
    OUTPUT:
        B      -- The magnetic field at N + 1
        E      -- The electric field at N + 1
        
    Note: Because position and velocity advance subroutines depend on directly modifying previous
    values, they must be copied and restored in order to ensure they don't actually move.
    '''
    E_pred          = 2.0*E_half - 1.0*E_int
    B_pred          = push_B(B, E_pred, DT)

    P, V, I, W, Q, J = advance_particles_and_moments(pos.copy(), vel.copy(), Ie.copy(), W_elec.copy(), idx, B_pred, E_pred, DT)
    
    q_dens          = 0.5*(q_dens + Q)
    B_pred          = push_B(B_pred, E_pred, DT)
    E_pred, Ve, Te  = calculate_E(B_pred, J, q_dens)
    
    E_corr          = 0.5*(E_half + E_pred)
    B_corr          = push_B(B, E_corr, DT)
    return E_corr, B_corr


@nb.njit()
def get_curl_E(field, DX=dx):
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
    curl = np.zeros(field.shape)
    
    for ii in nb.prange(1, field.shape[0]):
        curl[ii, 1] = - (field[ii, 2] - field[ii - 1, 2])
        curl[ii, 2] =    field[ii, 1] - field[ii - 1, 1]

    curl = set_periodic_boundaries(curl)
    return curl / DX


@nb.njit()
def push_B(B, E, DT):
    '''
    Used as part of predictor corrector for predicing B based on an approximated
    value of E (rather than cycling with the source terms)
    '''
    B_out = B.copy() - 0.5 * DT * get_curl_E(E)
    B_out = set_periodic_boundaries(B_out) 
    return B_out


@nb.njit()
def set_periodic_boundaries(B):
    ''' 
    Set boundary conditions for the magnetic field (or values on the B-grid, i.e. curl[E]): 
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
def get_curl_B(field, DX=dx):
    ''' Returns a vector quantity for the curl of a field valid at the positions 
    between its gridpoints (i.e. curl(B) -> E-grid, etc.)
    
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
    curl = np.zeros(field.shape)
    
    for ii in nb.prange(1, field.shape[0]):
        curl[ii - 1, 1] = - (field[ii, 2] - field[ii - 1, 2])
        curl[ii - 1, 2] =    field[ii, 1] - field[ii - 1, 1]
    
    # Assign ghost cell values
    curl[field.shape[0] - 1] = curl[2]
    curl[field.shape[0] - 2] = curl[1]
    curl[0] = curl[field.shape[0] - 3]
    return curl / DX


@nb.njit()
def get_electron_temp(qn):
    '''
    Calculate the electron temperature in each cell. Depends on the charge density of each cell
    and the treatment of electrons: i.e. isothermal (ie=0) or adiabatic (ie=1)
    '''
    if ie == 0:
        te      = np.ones(qn.shape[0]) * Te0
    elif ie == 1:
        gamma_e = 5./3. - 1.
        te      = Te0 * np.power(qn / (q*ne), gamma_e)
    return te


@nb.njit()
def get_grad_P(qn, te, DX=dx, inter_type=1):
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
    grad_pe_B     = np.zeros(qn.shape[0])
    grad_P        = np.zeros(qn.shape[0])
    Pe            = qn * kB * te / q

    for ii in nb.prange(1, qn.shape[0]):
        grad_pe_B[ii] = (Pe[ii] - Pe[ii - 1])  / DX
        
    grad_pe_B[0] = grad_pe_B[qn.shape[0] - 3]
    
    # Re-interpolate to E-grid
    if inter_type == 0:
        grad_P = interpolate_to_center_linear_1D(grad_pe_B)           
    elif inter_type == 1:
        grad_P = interpolate_to_center_cspline1D(grad_pe_B, DX=DX)

    grad_P[0]               = grad_P[qn.shape[0] - 3]
    grad_P[qn.shape[0] - 2] = grad_P[1]
    grad_P[qn.shape[0] - 1] = grad_P[2] 
    return grad_P


@nb.njit()
def calculate_E(B, J, qn, DX=dx):
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
    curlB    = get_curl_B(B, DX=DX) / mu0
        
    Ve       = np.zeros((J.shape[0], 3)) 
    Ve[:, 0] = (J[:, 0] - curlB[:, 0]) / qn
    Ve[:, 1] = (J[:, 1] - curlB[:, 1]) / qn
    Ve[:, 2] = (J[:, 2] - curlB[:, 2]) / qn
    
    Te       = get_electron_temp(qn)
    del_p    = get_grad_P(qn, Te)
    
    B_center = interpolate_to_center_cspline3D(B, DX=DX)
    VexB     = cross_product(Ve, B_center)    

    E        = np.zeros((J.shape[0], 3))                 
    E[:, 0]  = - VexB[:, 0] - del_p / qn
    E[:, 1]  = - VexB[:, 1]
    E[:, 2]  = - VexB[:, 2]

    E[0]                = E[J.shape[0] - 3]
    E[J.shape[0] - 2]   = E[1]
    E[J.shape[0] - 1]   = E[2]
    return E, Ve, Te