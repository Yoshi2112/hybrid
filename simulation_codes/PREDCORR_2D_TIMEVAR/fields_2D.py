# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:54:19 2017

@author: iarey
"""
import numpy as np
import numba as nb
import pdb

import auxilliary_2D as aux
from sources_2D import manage_E_grid_ghost_cells
from simulation_parameters_2D import dx, dy, Te0, ne, q, mu0, kB, ie, HM_amplitude, HM_frequency

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
def B_grid_boundary_and_ghost_cells(arr):
    '''
    Takes quantities on the B-grid (curl E, grad_p), sets the boundary conditions (periodic)
    and fills the ghost cells
    
    Assumes that all boundary nodes (sides, corners) have already calculated values
    (This function just averages the sides/corners and copies the relevant cells)
    '''
    sx = arr.shape[0]; sy = arr.shape[1]
    
    corner_val = 0.25 * (arr[1, 1] + arr[sx - 2, 1] + arr[1, sy - 2] + arr[sx - 2, sy - 2])
    
    arr[1     , 1     ] = corner_val
    arr[1     , sy - 2] = corner_val
    arr[sx - 2, 1     ] = corner_val
    arr[sx - 2, sy - 2] = corner_val
    
    av_LR      = 0.5 * (arr[1, 2: sy - 2] + arr[sx - 2, 2: sy - 2])       # Average left + right side values 
    av_TB      = 0.5 * (arr[2: sx - 2, 1] + arr[2: sx - 2, sy - 2])       # Average top  + bottom     values in case of disparity
    arr[1, 2: sy - 2]      = av_LR
    arr[sx - 2, 2: sy - 2] = av_LR
    arr[2: sx - 2, 1]      = av_TB
    arr[2: sx - 2, sy - 2] = av_TB
    
    # Fill ghost cells
    arr[0     , 1: sy - 1] = arr[sx - 3, 1: sy - 1]    # Left inner
    arr[sx - 1, 1: sy - 1] = arr[2     , 1: sy - 1]    # Right inner
    
    arr[0: sx , 0]      = arr[0: sx, sy - 3]           # Bottom whole row
    arr[0: sx , sy - 1] = arr[0: sx, 2     ]           # Top    whole row
    return


@nb.njit()
def get_curl_E(E, curl):
    ''' Returns a vector quantity for the curl of a field valid at the positions 
    between its gridpoints (i.e. curl(E) -> B-grid, etc.)
    
    INPUT:
        field    -- The 3D field to take the curl of
        DX       -- Spacing between the nodes, mostly for diagnostics. 
                    Defaults to grid spacing specified at initialization.
                 
    OUTPUT:
        curl  -- Finite-differenced solution for the curl of the input field.
        
    NOTE: Same algorithm as curl_B_term, but loop ranges are lowered by 1 to account
        for the grid offset, and because its nicer to change the target cell address
        rather than the FD cells
        
     NX     = E.shape[0] - 3
     NX + 1 = E.shape[0] - 2
     NX + 2 = E.shape[0] - 1
     
     DOUBLE CHECK THIS
    '''
    curl[:, 0] *= 0
    for ii in nb.prange(0, E.shape[0] - 1):
        for jj in nb.prange(0, E.shape[1] - 1):
            curl[ii + 1, jj + 1, 0] =  0.5/dy*(E[ii+1,jj+1,2] - E[ii+1,jj,2] + E[ii,jj+1,2] - E[ii,jj,2])
            curl[ii + 1, jj + 1, 1] = -0.5/dx*(E[ii+1,jj+1,2] + E[ii+1,jj,2] - E[ii,jj+1,2] - E[ii,jj,2])
            curl[ii + 1, jj + 1, 2] =  0.5/dx*(E[ii+1,jj+1,1] + E[ii+1,jj,1] - E[ii,jj+1,1] - E[ii,jj,1])\
                                      -0.5/dy*(E[ii+1,jj+1,0] - E[ii+1,jj,0] + E[ii,jj+1,0] - E[ii,jj,0])
    
    B_grid_boundary_and_ghost_cells(curl)
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
    time  = (qq - 0.5*half_flag) * DT                       # Current simulation time (pushing to)
    get_curl_E(E, temp3D)

    B[:, :, 0] -= uniform_HM_field_value(time - 0.5*DT)     # Subtract previous HM field
    B          -= 0.5 * DT * temp3D                         # Advance using curl
    B[:, :, 0] += uniform_HM_field_value(time)              # Add new  HM field 
    return



@nb.njit()
def curl_B_term(B, curl):
    ''' Returns a vector quantity for the curl of a field valid at the positions 
    between its gridpoints (i.e. curl(B) -> E-grid, etc.)
    
    INPUT:
        B    -- Magnetic field values on B-grid
                 
    OUTPUT:
        curl  -- Finite-differenced solution for the curl of the input field on E-grid
        
    NOTE: This function will only work with this specific 1D hybrid code due to both 
          E and B fields having the same number of nodes (due to TSC weighting) and
         the lack of derivatives in y, z
         
     NX     = B.shape[0] - 3
     NX + 1 = B.shape[0] - 2
     NX + 2 = B.shape[0] - 1
     
     WRITE A FUNCTION TO DO THE FD?
     "TSC ONLY" lines could probably be deleted. The array locations are referenced
     in the weighting functions but their values do not contribute at all (Wx = Wy = 0).
    '''
    sx = B.shape[0]; sy = B.shape[1]                        # Array sizes (Shorthand)
    curl *= 0
    
    for ii in nb.prange(1, sx - 2):                         # E-nodes 1: NX + 1 inclusive
        for jj in nb.prange(1, sy - 2):                     # E-nodes 1: NY + 1 inclusive
            curl[ii,jj,0] =  0.5/dy*(B[ii+1,jj+1,2] - B[ii+1,jj,2] + B[ii,jj+1,2] - B[ii,jj,2])
            curl[ii,jj,1] = -0.5/dx*(B[ii+1,jj+1,2] + B[ii+1,jj,2] - B[ii,jj+1,2] - B[ii,jj,2])
            curl[ii,jj,2] =  0.5/dx*(B[ii+1,jj+1,1] + B[ii+1,jj,1] - B[ii,jj+1,1] - B[ii,jj,1])\
                            -0.5/dy*(B[ii+1,jj+1,0] - B[ii+1,jj,0] + B[ii,jj+1,0] - B[ii,jj,0])
    
    manage_E_grid_ghost_cells(curl)
    curl /= mu0
    return 


@nb.njit()
def get_electron_temp(qn, Te):
    '''
    Calculate the electron temperature in each cell. Depends on the charge density of each cell
    and the treatment of electrons: i.e. isothermal (ie=0) or adiabatic (ie=1)
    '''
    if ie == 0:
        Te[:, :] = np.ones(qn.shape[0]) * Te0
    elif ie == 1:
        gamma_e = 5./3. - 1.
        Te[:, :] = Te0 * np.power(qn / (q*ne), gamma_e)
    return


@nb.njit()
def get_grad_P(qn, te, grad_P, temp3D):
    '''
    Returns the electron pressure gradient on the E-field grid using P = nkT and 
    finite difference.
     
    INPUT:
        qn     -- Grid charge density
        te     -- Grid electron temperature
        grad_P -- Output array for solution
        temp   -- Temporary calculation array
        
    NOTE: Interpolation is needed because the finite differencing causes the result to be deposited on the 
    B-grid. Moving it back to the E-grid requires an interpolation. Cubic spline is desired due to its smooth
    derivatives and its higher order weighting (without the polynomial craziness)
    '''
    temp3D         *= 0                   # Zero temp array
    grad_P[:, :, 0] = qn * kB * te / q    # Not actually grad P, just using this array to store Pe
                                          # Putting [:] after array points to memory locations,
                                          # and prevents deferencing

    for ii in nb.prange(0, qn.shape[0] - 1):
        for jj in nb.prange(0, qn.shape[1] - 1):
            temp3D[ii + 1, jj + 1, 0] = 0.5/dx*(grad_P[ii+1,jj+1,0] + grad_P[ii+1,jj,0]
                                              - grad_P[ii,  jj+1,0] - grad_P[ii,  jj,0])
            
            temp3D[ii + 1, jj + 1, 1] = 0.5/dy*(grad_P[ii+1,jj+1,0] - grad_P[ii+1,jj,0]
                                              + grad_P[ii,  jj+1,0] - grad_P[ii,  jj,0])

    B_grid_boundary_and_ghost_cells(temp3D)            # Set B-grid BC's and fill ghost cells
    aux.linear_Bgrid_to_Egrid_scalar(temp3D, grad_P)   # Move grad_P back onto E_field grid
    return


@nb.njit()
def calculate_E(B, Ji, q_dens, E, Ve, Te, temp3Da, temp3Db, temp3Dc):
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
    
    temp3Da, temp3Db, temp3Dc are tertiary arrays used for intermediary computations
    Could eliminate at least one of these by incrementally adding to final E array,
    rather than calculating E in one go at the end
    '''    
    curl_B_term(B, temp3Da)                                         # tempA is now curl B term

    Ve[:, :, 0] = (Ji[:, :, 0] - temp3Da[:, :, 0]) / q_dens[:, :]
    Ve[:, :, 1] = (Ji[:, :, 1] - temp3Da[:, :, 1]) / q_dens[:, :]
    Ve[:, :, 2] = (Ji[:, :, 2] - temp3Da[:, :, 2]) / q_dens[:, :]   # temp3Da now free   

    get_electron_temp(q_dens, Te)

    get_grad_P(q_dens, Te, temp3Da, temp3Db)              # temp3Da is now del_p term, temp3Db used for computation

    aux.linear_Bgrid_to_Egrid_vector(B, temp3Db)          # temp3Db is now B_center

    aux.cross_product(Ve, temp3Db, temp3Dc)               # temp3Dc is now Ve x B term

    E[:, :, 0]  = - temp3Dc[:, :, 0] - temp3Da[:, :, 0] / q_dens[:, :]
    E[:, :, 1]  = - temp3Dc[:, :, 1] - temp3Da[:, :, 1] / q_dens[:, :]
    E[:, :, 2]  = - temp3Dc[:, :, 2]

    manage_E_grid_ghost_cells(E)
    return 