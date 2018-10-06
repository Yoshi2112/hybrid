# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:54:19 2017

@author: iarey
"""
import numpy as np
import numba as nb

from const             import NX, dx, Te0, mhd_equil, q, mu0, kB
from part_params       import Nj, charge
from hybrid_auxilliary import manage_ghost_cells
from hybrid_auxilliary import cross_product

@nb.njit(cache=True)
def push_B(B, E, dt):
    '''Updates the magnetic field across the simulation domain by implementing a finite difference on Faraday's Law.
    
    INPUT:
        B  --  The magnetic field array, to be updated
        E  --  The electric field array
        dt --  The simulation time cadence
    
    OUTPUT:
        B  --  The updated magnetic field
    '''
    for mm in range (1, NX + 1):
        B[mm, 0] = B[mm, 0] - (dt / 2) * 0                                                # Since derivatives for y and z go to zero
        B[mm, 1] = B[mm, 1] - (dt / 2) * ( (E[mm - 1, 2] - E[mm + 1, 2]) / (2 * dx) )
        B[mm, 2] = B[mm, 2] - (dt / 2) * ( (E[mm + 1, 1] - E[mm - 1, 1]) / (2 * dx) )
        
    B = manage_ghost_cells(B, 0)
    
    return B
    

def cubic_spline_interp(B):
    '''1D interpolation of the magnetic field values onto the E-field mesh, for use in the push_E update equation. 
    Code adapted from Numerical Recipes for Fortran 90.
    
    INPUT:
        B -- Magnetic field array
        
    OUTPUT:
        B_interp -- The magnetic field at the positions of the electric field nodes.'''
    size     = B.shape[0]
    B_interp = np.zeros((size, 3), dtype=float)	
    
    for dd in range(3):
        y2 = np.zeros(size)

        # Centered difference for y'' (y2) calculation
        for ii in range(1, size - 1):
            y2[ii] = (B[ii - 1, dd] - 2*B[ii, dd] + B[ii + 1, dd]) / (dx ** 2)
        
        y2[0]        = y2[size - 2] #B[size - 2, dd] - 2*B[0, dd]        + B[1, dd]
        y2[size - 1] = y2[1]        #B[size - 2, dd] - 2*B[size - 1, dd] + B[1, dd]

        # Actual spline calculation	
        h = dx ; a = 0.5 ; b = 0.5

        for ii in range(size - 1):
            B_interp[ii, dd] = a * B[ii, dd] + b * B[ii + 1, dd] + ((a**3 - a)*y2[ii] + (b**3 - b)*y2[ii + 1])*(h**2)/6.

    B_interp[0]        = B_interp[size - 2]
    B_interp[size - 1] = B_interp[1]
    return B_interp
   
 
def push_E_equil(B, J_i, n_i, dt):
    '''Calculates the value of the electric field based on source term and magnetic field contributions. This is done via
    a reworking of Ampere's Law that assumes quasineutrality, and removes the requirement to calculate the electron current.
    Based on equation 10 of Buchner (2003, p. 140). This version contains electron temperature variation designed to keep
    non-uniform plasma spatial distributions in equilibrium via div(P_e) = 0.
    
    INPUT:
        B   -- Magnetic field array. Displaced from E-field array by half a spatial step.
        J_i -- Average ion current per species. Source term, based on particle velocities
        n_i -- Ion number density. Source term, based on particle positions
        dt  -- Simulation time cadence
    
    OUTPUT:
        E_out -- Updated electric field array
    '''
    size = NX + 2
    
    E_out = np.zeros((size, 3))                                 # Output array - new electric field
    JxB   = np.zeros((size, 3))                                 # V cross B holder
    BdB   = np.zeros((size, 3))                                 # B cross del cross B holder     
    del_p = np.zeros((size, 3))                                 # Electron pressure tensor gradient array
    J     = np.zeros((size, 3))                                 # Total ion current
    qn    = np.zeros( size,    dtype=float)                     # Ion charge density
    
    B_i = cubic_spline_interp(B)                                # Magnetic field values interpolated onto E-field grid
    
    for jj in range(Nj):                                        # Calculate average/summations over species
        qn += charge[jj] * n_i[:, jj]                           # Total charge density, sum(qj * nj)
        
        for kk in range(3):                                     # Total ion current vector: J_k = qj * nj * Vj_k
            J[:, kk]  += J_i[:, jj, kk]   

    # MHD equilibrium thing
    Te  = [Te0 for ii in range(size)]                            # Electron temperature per cell
    nex = qn / q                                                 # Electron number density per cell (via quasineutrality)

    if mhd_equil == 1:                                           # Finite difference to give del_p = 0
        for ii in range(2, size - 1):
            Te[ii] = (4*Te[ii - 1] - Te[ii - 2]) * (1 / ( ((nex[ii + 1] - nex[ii - 1]) / nex[ii]) + 3))
    
    JxB = cross_product(J, B_i)
    
    for mm in range(1, size - 1):
        
        # B cross curl B
        BdB[mm, 0] =    B_i[mm, 1]  * ((B[mm + 1, 1] - B[mm - 1, 1]) / (2 * dx)) + B[mm, 2] * ((B[mm + 1, 2] - B[mm - 1, 2]) / (2 * dx))
        BdB[mm, 1] = (- B_i[mm, 0]) * ((B[mm + 1, 1] - B[mm - 1, 1]) / (2 * dx))
        BdB[mm, 2] = (- B_i[mm, 0]) * ((B[mm + 1, 2] - B[mm - 1, 2]) / (2 * dx))
    
        # del P
        del_p[mm, 0] = (kB / (2*dx*q)) * ( Te[mm] * (qn[mm + 1] - qn[mm - 1]) +  
                                                  qn[mm] * (Te[mm + 1] - Te[mm - 1]) )
        del_p[mm, 1] = 0
        del_p[mm, 2] = 0
   
    for ii in range(3):
        JxB[:, ii]   /=  qn
        BdB[:, ii]   /= (qn * mu0)
        del_p[:, ii] /=  qn

    E_out = - JxB - del_p - BdB                 # Final Calculation
    
    E_out = manage_ghost_cells(E_out, 0)
    Te    = manage_ghost_cells(E_out, 0)
    return E_out

@nb.njit(cache=True)
def push_E(B, J_i, n_i, dt):
    '''Calculates the value of the electric field based on source term and magnetic field contributions, assuming constant
    electron temperature across simulation grid. This is done via a reworking of Ampere's Law that assumes quasineutrality,
    and removes the requirement to calculate the electron current. Based on equation 10 of Buchner (2003, p. 140).
    
    INPUT:
        B   -- Magnetic field array. Displaced from E-field array by half a spatial step.
        J_i -- Ion current density. Source term, based on particle velocities
        n_i -- Ion number density. Source term, based on particle positions
        dt  -- Simulation time cadence
    
    OUTPUT:
        E_out -- Updated electric field array
    '''
    size = NX + 2
    
    E_out = np.zeros((size, 3))                 # Output array - new electric field
    JxB   = np.zeros((size, 3))                 # V cross B holder
    BdB   = np.zeros((size, 3))                 # B cross del cross B holder     
    del_p = np.zeros((size, 3))                 # Electron pressure tensor gradient array
    J     = np.zeros((size, 3))                 # Ion current
    qn    = np.zeros(size,    )                 # Ion charge density

    Te = np.ones(size) * Te0                    # Electron temperature array
        
    # Calculate change/current summations over each species
    for jj in range(Nj):
        qn += charge[jj] * n_i[:, jj]                                       # Total charge density, sum(qj * nj)
        
        for kk in range(3):
            J[:, kk]  += J_i[:, jj, kk]                                         # Total ion current vector: J_k = qj * nj * Vj_k

    JxB = cross_product(J, B)   
        
    for mm in range(1, size - 1):
        
        # B cross curl B
        BdB[mm, 0] =    B[mm, 1]  * ((B[mm + 1, 1] - B[mm - 1, 1]) / (2 * dx)) + B[mm, 2] * ((B[mm + 1, 2] - B[mm - 1, 2]) / (2 * dx))
        BdB[mm, 1] = (- B[mm, 0]) * ((B[mm + 1, 1] - B[mm - 1, 1]) / (2 * dx))
        BdB[mm, 2] = (- B[mm, 0]) * ((B[mm + 1, 2] - B[mm - 1, 2]) / (2 * dx))
    
        # del P
        del_p[mm, 0] = ((qn[mm + 1] - qn[mm - 1]) / (2*dx*q)) * kB * Te[mm]
        del_p[mm, 1] = 0
        del_p[mm, 2] = 0
    
    # Final Calculation
    E_out[:, 0] = (- JxB[:, 0] - del_p[:, 0] - (BdB[:, 0] / mu0)) / (qn[:])
    E_out[:, 1] = (- JxB[:, 1] - del_p[:, 1] - (BdB[:, 1] / mu0)) / (qn[:])
    E_out[:, 2] = (- JxB[:, 2] - del_p[:, 2] - (BdB[:, 2] / mu0)) / (qn[:])
    
    E_out = manage_ghost_cells(E_out, 0)
    
    return E_out
