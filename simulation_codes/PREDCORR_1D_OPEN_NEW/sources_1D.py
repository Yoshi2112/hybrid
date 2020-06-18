# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""
import numba as nb
import pdb
from simulation_parameters_1D import ND, NX, Nj, n_contr, mass, charge, q, ne, min_dens,\
                                     xmin, xmax, dx


@nb.njit()
def collect_velocity_moments(pos, vel, Ie, W_elec, idx, nu, Ji, Pi):
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
    Ji     *= 0.
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

    # Set damping cell source values (zero gradient)
    for ii in range(3):
        Ji[:ND, ii]    = Ji[ND + 1, ii]
        Ji[ND+NX:, ii] = Ji[ND+NX - 2, ii]

    # Convert to real moment, and accumulate charge density
    for jj in range(Nj):
        for kk in range(3):
            nu[:, jj, kk] *= n_contr[jj]
            Ji[:,     kk] += nu[:, jj, kk] * charge[jj]

    # Collect pressure tensor AT BOUNDARY CELLS ONLY :: Second moment
    for ii in nb.prange(vel.shape[1]):
        
        # Only count specific particles
        if abs(pos[0, ii] - xmin) < dx:
            I   = Ie[ii]
            sp  = idx[ii]
            
            # For each tensor element
            for mm in range(3):
                for nn in range(3):
                    # Real particle
                    Pi[I,     sp, mm, nn] += (vel[mm, ii] - nu[I,     sp, mm]) *\
                                             (vel[nn, ii] - nu[I,     sp, nn]) * (1.0 - W_elec[ii])
                                         
                    Pi[I + 1, sp, mm, nn] += (vel[mm, ii] - nu[I + 1, sp, mm]) *\
                                             (vel[nn, ii] - nu[I + 1, sp, nn]) * W_elec[ii]
                                        
                    # Virtual particle < xmin
                    Pi[I - 1,     sp, mm, nn] += (vel[mm, ii] - nu[I,     sp, mm]) *\
                                             (vel[nn, ii] - nu[I,     sp, nn]) * (1.0 - W_elec[ii])
                                         
                    Pi[I    , sp, mm, nn] += (vel[mm, ii] - nu[I + 1, sp, mm]) *\
                                             (vel[nn, ii] - nu[I + 1, sp, nn]) * W_elec[ii]
                            
        elif abs(pos[0, ii] - xmax) < dx:
            # For each tensor element
            for mm in range(3):
                for nn in range(3):
                    # Real particle
                    Pi[I,     sp, mm, nn] += (vel[mm, ii] - nu[I,     sp, mm]) *\
                                             (vel[nn, ii] - nu[I,     sp, nn]) * (1.0 - W_elec[ii])
                                         
                    Pi[I + 1, sp, mm, nn] += (vel[mm, ii] - nu[I + 1, sp, mm]) *\
                                             (vel[nn, ii] - nu[I + 1, sp, nn]) * W_elec[ii]
                    
                    # Virtual particle > xmax
                    Pi[I,     sp, mm, nn] += (vel[mm, ii] - nu[I,     sp, mm]) *\
                                             (vel[nn, ii] - nu[I,     sp, nn]) * (1.0 - W_elec[ii])
                                         
                    Pi[I + 1, sp, mm, nn] += (vel[mm, ii] - nu[I + 1, sp, mm]) *\
                                             (vel[nn, ii] - nu[I + 1, sp, nn]) * W_elec[ii]
            
    # Convert to real units               
    for jj in range(Nj):
        Pi[:, jj, :, :] *= mass[jj] * n_contr[jj]
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
    q_dens *= 0.
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
            

    # Sum charge density contributions across species
    for jj in range(Nj):
        ni[:, jj] *= n_contr[jj]
        q_dens    += ni[:, jj] * charge[jj]
        
    # Set damping cell source values
    q_dens[:ND]    = q_dens[ND + 1]
    q_dens[ND+NX:] = q_dens[ND+NX - 2]
        
    # Set density minimum
    for ii in range(q_dens.shape[0]):
        if q_dens[ii] < min_dens * ne * q:
            q_dens[ii] = min_dens * ne * q
    return