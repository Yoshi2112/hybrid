# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numpy as np
import numba as nb

from const          import N, dx, dy, xmax, xmin, ymin, ymax
from part_params    import charge, mass  
from hybrid_sources import assign_weighting

@nb.njit(cache=True)
def velocity_update(part, B, E, dt, WE_in, WB_in):  # Based on Appendix A of Ch5 : Hybrid Codes by Winske & Omidi.
    '''Algorithm to find the time-centered velocity of the particles given the fields and weighting factors.
    
    INPUT: 
        part -- Particle array containing current particle velocities and positions
        B    -- 3D magnetic field across domain
        E    -- 3D electric field across domain
        dt   -- Simulation time cadence
        WE_in-- Electric field node weightings
        WB_in-- Magnetic field node weightings
    '''
    for n in range(N):
        vn = part[3:6, n]                  # Existing particle velocity
        E_p = np.zeros(3)
        B_p = np.zeros(3)                  # Initialize field values
        
        # E & B fields at particle location: Node values /w weighting factors in x, y 
        Ix   = int(part[6, n])             # Nearest (leftmost) node
        Iy   = int(part[7, n])             # Nearest (bottom)   node
        Ibx  = int(part[0, n] / dx)        # Nearest (leftmost) magnetic node
        Iby  = int(part[1, n] / dy)        # Nearest (bottom)   magnetic node
        
        WEx  = WE_in[0:2, n]               # E-field weighting (x)
        WEy  = WE_in[2:4, n]               # E-field weighting (y)
        WBx  = WE_in[0:2, n]               # B-field weighting (x)
        WBy  = WB_in[2:4, n]               # B-field weighting (y)
        
        idx  = int(part[2, n])             # Particle species index
       
        # Interpolate fields to particle position
        for ii in range(2):
            for jj in range(2):
                E_p += E[Ix  + ii, Iy  + jj, 0:3] * WEx[ii] * WEy[jj]
                B_p += B[Ibx + ii, Iby + jj, 0:3] * WBx[ii] * WBy[jj]

        # Intermediate calculations
        h  = (charge[idx] * dt) / mass[idx]
        f  = 1 - (h**2) / 2 * (B_p[0]**2 + B_p[1]**2 + B_p[2]**2 )
        g  = h / 2 * (B_p[0]*vn[0] + B_p[1]*vn[1] + B_p[2]*vn[2])
        v0 = vn + (h/2)*E_p
    
        # Velocity push
        part[3, n] = f * vn[0] + h * ( E_p[0] + g * B_p[0] + (v0[1]*B_p[2] - v0[2]*B_p[1]) )
        part[4, n] = f * vn[1] + h * ( E_p[1] + g * B_p[1] - (v0[0]*B_p[2] - v0[2]*B_p[0]) )
        part[5, n] = f * vn[2] + h * ( E_p[2] + g * B_p[2] + (v0[0]*B_p[1] - v0[1]*B_p[0]) )
    return part  


def boris_velocity_update(part, B, E, dt, WE_in, WB_in): 
    '''Updates the velocities of the particles in the simulation using a Boris particle pusher, as detailed
    in Birdsall & Langdon (1985),  59-63.
    
    INPUT:
        part -- Particle array containing velocities to be updated
        B    -- Magnetic field on simulation grid
        E    -- Electric field on simulation grid
        dt   -- Simulation time cadence
        W    -- Weighting factor of particles to rightmost node
        
    OUTPUT:
        part -- Returns particle array with updated velocities
    '''

    for n in range(N):
        v_minus = np.zeros(3)                               # First velocity
        v_prime = np.zeros(3)                               # Rotation velocity
        v_plus  = np.zeros(3)                               # Second velocity
 
        vn      = part[3:6, n]                              # Existing particle velocity
        
        # E & B fields at particle location: Node values /w weighting factors in x, y 
        Ix   = int(part[6, n])             # Nearest (leftmost) node
        Iy   = int(part[7, n])             # Nearest (bottom)   node
        Ibx  = int(part[0, n] / dx)        # Nearest (leftmost) magnetic node
        Iby  = int(part[1, n] / dy)        # Nearest (bottom)   magnetic node
        
        WEx  = WE_in[0:2, n]               # E-field weighting (x)
        WEy  = WE_in[2:4, n]               # E-field weighting (y)
        WBx  = WE_in[0:2, n]               # B-field weighting (x)
        WBy  = WB_in[2:4, n]               # B-field weighting (y)
        
        idx  = int(part[2, n])             # Particle species index
       
        # Interpolate fields to particle position
        B_p = 0     ;   E_p = 0
        for ii in range(2):
            for jj in range(2):
                E_p += E[Ix  + ii, Iy  + jj, 0:3] * WEx[ii] * WEy[jj]
                B_p += B[Ibx + ii, Iby + jj, 0:3] * WBx[ii] * WBy[jj]
        
        T = (charge[idx] * dt) / (2 * mass[idx]) * B_p             # Boris variable
        S = 2*T / (1 + np.sqrt(T[0] ** 2 + T[1] ** 2 + T[2] ** 2))          # Boris variable

        # Actual Boris Method
        v_minus = vn + charge[idx] * E_p * dt / (2 * mass[idx])    
        
        v_prime[0] = v_minus[0] + (charge[idx] * dt / (2 * mass[idx])) * (v_minus[1] * B_p[2] - v_minus[2] * B_p[1])
        v_prime[1] = v_minus[1] + (charge[idx] * dt / (2 * mass[idx])) * (v_minus[0] * B_p[2] - v_minus[2] * B_p[0])
        v_prime[2] = v_minus[2] + (charge[idx] * dt / (2 * mass[idx])) * (v_minus[0] * B_p[1] - v_minus[1] * B_p[0])
        
        v_plus[0]  = v_minus[0] + (v_prime[1] * S[2] - v_prime[2] * S[1])
        v_plus[1]  = v_minus[1] + (v_prime[0] * S[2] - v_prime[2] * S[0])
        v_plus[2]  = v_minus[2] + (v_prime[0] * S[1] - v_prime[1] * S[0])
        
        part[3:6, n] = v_plus + charge[idx] * E_p * dt / (2 * mass[idx])  
    return part


@nb.njit(cache=True)    
def position_update(part, dt):
    '''Updates the position of the particles using x = x0 + vt. Also updates particle leftmost node and weighting.
    
    INPUT:
        part -- Particle array with positions to be updated
        dt   -- Time cadence of simulation
        
    OUTPUT:
        part -- Particle array with updated positions
        W    -- Particle (E-field) node weightings 
    '''
    part[0:2, :] += part[3:5, :] * dt                   # Update position: x = x0 + vt
    
    for ii in range(N):                                 # Check particle boundary conditions
        if part[0, ii] < xmin:
            part[0, ii] = xmax + part[0,ii]
            
        if part[0, ii] > xmax:
            part[0, ii] = part[0, ii] - xmax
            
        if part[1, ii] < ymin:
            part[1, ii] = ymax + part[1, ii]

        if part[1, ii] > ymax:
            part[1, ii] = part[1, ii] - ymax
    
    part[6, :] = part[0, :] / dx + 0.5 ; part[6, :] = part[6, :].astype(nb.int32)            # Ix update
    part[7, :] = part[1, :] / dy + 0.5 ; part[7, :] = part[7, :].astype(nb.int32)            # Iy update

    We_out      = assign_weighting(part[0:2, :], part[6:8, :], 1)
    Wb_out      = assign_weighting(part[0:2, :], part[6:8, :], 0)                   # Magnetic field weighting (due to E/B grid displacement) 
    return part, We_out, Wb_out