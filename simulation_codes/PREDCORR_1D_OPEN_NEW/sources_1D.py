# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""
import numba as nb
from simulation_parameters_1D import ND, NX, Nj, n_contr, charge, q, ne, min_dens,\
                                     xmin, xmax, dx, source_smoothing


@nb.njit()
def collect_velocity_moments(pos, vel, Ie, W_elec, idx, nu, Ji):
    '''
    Collect first and second moments of macroparticle velocity distributions.
    Calculate average current density at each grid point. 

    INPUT:
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier

    OUTPUT:
        ni     -- Species number moment array(size, Nj)
        nui    -- Species velocity moment array (size, Nj)
        
    Note: To implement time-relaxation method for boundaries, it would be 
    required to copy existing values of Ji, nu temporarily (old values), collect
    new values to compute charge, then re-calculate moments as a linear weighting
    (depending on R) of the old stored moments and the new moments. This would
    cause arrays to be created, killing efficiency. Is there a way to use the 
    old_moments array? Or would that break because of the predictor-corrector
    scheme?
    '''
    nu     *= 0.
    # Deposit average velocity across all cells :: First moment
    for ii in nb.prange(vel.shape[1]):
        I   = Ie[ii]
        sp  = idx[ii]
        
        if sp >= 0:
            for kk in range(3):
                nu[I,     sp, kk] += vel[kk, ii] * (1.0 - W_elec[ii])
                nu[I + 1, sp, kk] += vel[kk, ii] *        W_elec[ii]
                
            # Simulate virtual particles in boundary ghost cells
            if abs(pos[0, ii] - xmin) < dx:
                for kk in range(3):
                    nu[I - 1, sp, kk] += vel[kk, ii] * (1.0 - W_elec[ii])
                    nu[I,     sp, kk] += vel[kk, ii] *        W_elec[ii]
                    
            elif abs(pos[0, ii] - xmax) < dx:
                for kk in range(3):
                    nu[I + 1, sp, kk] += vel[kk, ii] * (1.0 - W_elec[ii])
                    nu[I + 2, sp, kk] += vel[kk, ii] *        W_elec[ii]

    if source_smoothing == True:
        for jj in range(Nj):
            for ii in range(3):
                three_point_smoothing(nu[:, jj, ii], Ji[:,  0])

    Ji     *= 0.
    # Convert to real moment, and accumulate charge density
    for jj in range(Nj):
        for kk in range(3):
            nu[:, jj, kk] *= n_contr[jj]
            Ji[:,     kk] += nu[:, jj, kk] * charge[jj]
            
    # Set damping cell source values (last value)
    for ii in range(3):
        Ji[:ND, ii]    = Ji[ND, ii]
        Ji[ND+NX:, ii] = Ji[ND+NX - 1, ii]

    return


@nb.njit()
def collect_position_moment(pos, Ie, W_elec, idx, q_dens, ni):
    '''Collect number density in each cell, weighted by their distance
    from cell nodes.

    INPUT:
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier

    OUTPUT:
        q_dens -- Total charge density in each cell
        ni     -- Species number moment array(size, Nj)
    '''
    ni     *= 0.
    # Deposit macroparticle moment on grid
    for ii in nb.prange(Ie.shape[0]):
        I   = Ie[ii]
        sp  = idx[ii]
        
        if sp >= 0:
            ni[I,     sp] += 1.0 - W_elec[ii]
            ni[I + 1, sp] +=       W_elec[ii]
            
            # Simulate virtual particles in boundary ghost cells
            if abs(pos[0, ii] - xmin) < dx:
                ni[I - 1, sp] += (1.0 - W_elec[ii])
                ni[I,     sp] +=        W_elec[ii]
            elif abs(pos[0, ii] - xmax) < dx:
                ni[I + 1, sp] +=(1.0 - W_elec[ii])
                ni[I + 2, sp] +=       W_elec[ii]
    
    if source_smoothing == 1:
        for ii in range(Nj):
            three_point_smoothing(ni[:, ii], q_dens)
            
    q_dens *= 0.
    # Sum charge density contributions across species
    for jj in range(Nj):
        ni[:, jj] *= n_contr[jj]
        q_dens    += ni[:, jj] * charge[jj]
        
    # Set damping cell source values
    q_dens[:ND]    = q_dens[ND]
    q_dens[ND+NX:] = q_dens[ND+NX - 1]
        
    # Set density minimum
    for ii in range(q_dens.shape[0]):
        if q_dens[ii] < min_dens * ne * q:
            q_dens[ii] = min_dens * ne * q
    return


@nb.njit()
def three_point_smoothing(arr, temp):
    '''
    Three point Gaussian (1/4-1/2-1/4) smoothing function. arr, temp are both
    1D arrays of size NC = NX + 2*ND (i.e. on the E-grid)
    '''
    NC = arr.shape[0]
    
    temp *= 0.0
    for ii in range(1, NC - 1):
        temp[ii] = 0.25*arr[ii - 1] + 0.5*arr[ii] + 0.25*arr[ii + 1]
        
    temp[0]      = temp[1]
    temp[NC - 1] = temp[NC - 2]
    
    arr[:]       = temp
    return


# Pressure Tensor collection for if I actually want to collect it from the macroparticles
# =============================================================================
#     # If True: Use initial temperature/pressure for second moment
#     if Pi_use_init == True:
#         Pi[ND, sp, :, :] *= 0.0
#         
#         for jj in range(Tpar.shape[0]):
#             Pi[ND, jj, 0, 0] = density[jj] * kB * Tpar[jj]
#             Pi[ND, jj, 1, 1] = density[jj] * kB * Tperp[jj]
#             Pi[ND, jj, 2, 2] = density[jj] * kB * Tperp[jj]
#             
#             Pi[ND + NX - 1, jj, 0, 0] = density[jj] * kB * Tpar[jj]
#             Pi[ND + NX - 1, jj, 1, 1] = density[jj] * kB * Tperp[jj]
#             Pi[ND + NX - 1, jj, 2, 2] = density[jj] * kB * Tperp[jj]
#             
#     # Else: Collect pressure tensor from macroparticles at boundaries
#     else:
#         # Collect pressure tensor AT BOUNDARY CELLS ONLY :: Second moment
#         for ii in nb.prange(vel.shape[1]):
#             I   = Ie[ii]
#             sp  = idx[ii]
#             
#             # Only count specific particles: Imaginary "ghost" particles
#             if abs(pos[0, ii] - xmin) < dx:
#                 for mm in range(3):
#                     for nn in range(3):
#                         
#                         # Real macroparticle
#                         Pi[I,     sp, mm, nn] += (vel[mm, ii] - nu[I,     sp, mm]) *\
#                                                  (vel[nn, ii] - nu[I,     sp, nn]) * (1.0 - W_elec[ii])
#                                              
#                         Pi[I + 1, sp, mm, nn] += (vel[mm, ii] - nu[I + 1, sp, mm]) *\
#                                                  (vel[nn, ii] - nu[I + 1, sp, nn]) * W_elec[ii]
#                                             
#                         # Virtual macroparticle
#                         Pi[I - 1, sp, mm, nn] += (vel[mm, ii] - nu[I,     sp, mm]) *\
#                                                  (vel[nn, ii] - nu[I,     sp, nn]) * (1.0 - W_elec[ii])
#                                              
#                         Pi[I    , sp, mm, nn] += (vel[mm, ii] - nu[I + 1, sp, mm]) *\
#                                                  (vel[nn, ii] - nu[I + 1, sp, nn]) * W_elec[ii]
#                                 
#             elif abs(pos[0, ii] - xmax) < dx:
#                 for mm in range(3):
#                     for nn in range(3):
#                         # Real particle
#                         Pi[I,     sp, mm, nn] += (vel[mm, ii] - nu[I,     sp, mm]) *\
#                                                  (vel[nn, ii] - nu[I,     sp, nn]) * (1.0 - W_elec[ii])
#                                              
#                         Pi[I + 1, sp, mm, nn] += (vel[mm, ii] - nu[I + 1, sp, mm]) *\
#                                                  (vel[nn, ii] - nu[I + 1, sp, nn]) * W_elec[ii]
#                                        
#                         # Virtual macroparticle
#                         Pi[I,     sp, mm, nn] += (vel[mm, ii] - nu[I,     sp, mm]) *\
#                                                  (vel[nn, ii] - nu[I,     sp, nn]) * (1.0 - W_elec[ii])
#                                              
#                         Pi[I + 1, sp, mm, nn] += (vel[mm, ii] - nu[I + 1, sp, mm]) *\
#                                                  (vel[nn, ii] - nu[I + 1, sp, nn]) * W_elec[ii]
#         
#         # Convert to real units               
#         for jj in range(Nj):
#             Pi[:, jj, :, :] *= mass[jj] * n_contr[jj]
# =============================================================================