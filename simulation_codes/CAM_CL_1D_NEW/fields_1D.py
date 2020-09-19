# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:54:19 2017

@author: iarey
"""
import numpy as np
import numba as nb

from auxilliary_1D            import cross_product, interpolate_to_center_cspline3D, interpolate_to_center_cspline1D, interpolate_to_center_linear_1D
from simulation_parameters_1D import NX, dx, Te0, ne, q, mu0, kB, ie, e_resis


@nb.njit()
def get_curl_B(field):
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
    
    for ii in np.arange(1, field.shape[0]):
        curl[ii - 1, 1] = - (field[ii, 2] - field[ii - 1, 2])
        curl[ii - 1, 2] =    field[ii, 1] - field[ii - 1, 1]
    
    # Assign ghost cell values
    curl[field.shape[0] - 1] = curl[2]
    curl[field.shape[0] - 2] = curl[1]
    curl[0] = curl[field.shape[0] - 3]
    curl   /= dx
    return curl


@nb.njit()
def get_curl_E(field, curl):
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
    curl *= 0.
    for ii in np.arange(1, field.shape[0]):
        curl[ii, 1] = - (field[ii, 2] - field[ii - 1, 2])
        curl[ii, 2] =    field[ii, 1] - field[ii - 1, 1]

    set_periodic_boundaries(curl)
    curl /= dx
    return


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
def get_grad_P(qn, te, inter_type=1):
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

    for ii in np.arange(1, qn.shape[0]):
        grad_pe_B[ii] = (Pe[ii] - Pe[ii - 1])
        
    grad_pe_B[0] = grad_pe_B[qn.shape[0] - 3]
    
    # Re-interpolate to E-grid
    if inter_type == 0:
        grad_P = interpolate_to_center_linear_1D(grad_pe_B)           
    elif inter_type == 1:
        grad_P = interpolate_to_center_cspline1D(grad_pe_B)

    grad_P[0]               = grad_P[qn.shape[0] - 3]
    grad_P[qn.shape[0] - 2] = grad_P[1]
    grad_P[qn.shape[0] - 1] = grad_P[2] 
    grad_P /= dx
    return grad_P


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
    return


@nb.njit()
def cyclic_leapfrog(B1, B2, rho, J, curl, DT, subcycles):
    '''
    Solves for the magnetic field push by keeping two copies and subcycling between them,
    averaging them at the end of the cycle as per Matthews (1994). The source terms are
    unchanged during the subcycle step. This method damps the high frequency dispersion 
    inherent in explicit hybrid simulations.
    
    INPUT:
        B1    -- Magnetic field to update (return value comes through here)
        B2    -- Empty array for second copy
        rho_i -- Total ion charge density
        J_i   -- Total ionic current density
        DT    -- Master simulation timestep. This function advances the field by 0.5*DT
        subcycles -- The number of subcycle steps to be performed. 
    '''
    H     = 0.5 * DT
    dh    = H / subcycles
    B2[:] = B1[:]

    ## DESYNC SECOND FIELD COPY - PUSH BY DH ##
    E, Ve, Te = calculate_E(B1, J, rho)
    get_curl_E(E, curl) 
    B2       -= dh * curl
    set_periodic_boundaries(B2)                              

    ## RETURN IF NO SUBCYCLES REQUIRED ##
    if subcycles == 1:
        return B2

    ## MAIN SUBCYCLE LOOP ##
    for ii in range(subcycles - 1):             
        if ii%2 == 0:
            E, Ve, Te = calculate_E(B2, J, rho)
            get_curl_E(E, curl) 
            B1  -= 2 * dh * curl
            set_periodic_boundaries(B1)
        else:
            E, Ve, Te = calculate_E(B1, J, rho)
            get_curl_E(E, curl) 
            B2  -= 2 * dh * curl
            set_periodic_boundaries(B2)
            
    ## RESYNC FIELD COPIES ##
    if ii%2 == 0:
        E, Ve, Te = calculate_E(B2, J, rho)
        get_curl_E(E, curl) 
        B2  -= dh * curl
        set_periodic_boundaries(B2)
    else:
        E, Ve, Te = calculate_E(B1, J, rho)
        get_curl_E(E, curl) 
        B1  -= dh * curl
        set_periodic_boundaries(B1)

    ## AVERAGE FIELD SOLUTIONS: COULD PERFORM A CONVERGENCE TEST HERE IN FUTURE ##
    B1 += B2; B1 /= 2.0
    return


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
    curlB    = get_curl_B(B) / mu0
       
    Ve       = np.zeros((J.shape[0], 3)) 
    Ve[:, 0] = (J[:, 0] - curlB[:, 0]) / qn
    Ve[:, 1] = (J[:, 1] - curlB[:, 1]) / qn
    Ve[:, 2] = (J[:, 2] - curlB[:, 2]) / qn
    
    Te       = get_electron_temp(qn)
    del_p    = get_grad_P(qn, Te)
    
    B_center = interpolate_to_center_cspline3D(B)
    VexB     = cross_product(Ve, B_center)    

    E_out        = np.zeros((J.shape[0], 3))                 
    E_out[:, 0]  = - VexB[:, 0] - del_p / qn
    E_out[:, 1]  = - VexB[:, 1]
    E_out[:, 2]  = - VexB[:, 2]

    E_out       += e_resis * J

    E_out[0]                = E_out[J.shape[0] - 3]
    E_out[J.shape[0] - 2]   = E_out[1]
    E_out[J.shape[0] - 1]   = E_out[2]                                  # This doesn't really get used, but might as well
    return E_out, Ve, Te