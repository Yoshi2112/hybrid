# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:55:15 2017

@author: iarey
"""
import numba as nb
from simulation_parameters_1D import ND, NX, Nj, n_contr, charge, q, ne, min_dens,\
                                     xmin, xmax, dx, source_smoothing, field_periodic,\
                                     source_BC


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
        
    Might need to put in a check for ghost cell values?
    Or just not distribute those values to the ghost cells, since they
    get overwritten anyway. For now just leave it because I +/- 1 is simpler
    to visualise, check, and code
    '''
    epsilon = 1e-3
    nu     *= 0.
    # Deposit average velocity across all cells :: First moment
    for ii in nb.prange(vel.shape[1]):
        if idx[ii] >= 0:
            I   = Ie[ii]
            sp  = idx[ii]
        
            for kk in range(3):
                nu[I,     sp, kk] += W_elec[0, ii] * vel[kk, ii]
                nu[I + 1, sp, kk] += W_elec[1, ii] * vel[kk, ii]
                nu[I + 2, sp, kk] += W_elec[2, ii] * vel[kk, ii]
            
            if field_periodic == 0:
                # Simulate virtual particles in boundary ghost cells
                # Check if in first cell
                if pos[0, ii] - xmin < dx:
                    
                    if dx - pos[0, ii] + xmin < epsilon:        # If on inner cell boundary, don't count
                        pass
                    elif pos[0, ii] - xmin < epsilon:           # If on simulation boundary, don't count
                        pass
                    else:                                       # Otherwise, count
                        nu[I - 1, sp, kk] += W_elec[0, ii] * vel[kk, ii]
                        nu[I    , sp, kk] += W_elec[1, ii] * vel[kk, ii]
                        nu[I + 1, sp, kk] += W_elec[2, ii] * vel[kk, ii]
                
                # Check if in last cell
                elif xmax - pos[0, ii] < dx:
                    if xmax - pos[0, ii] < epsilon:             # If on simulation boundary, don't count
                        pass
                    elif dx - xmax + pos[0, ii] < epsilon:      # If on inner cell boundary, don't count
                        pass
                    else:
                        nu[I + 1, sp, kk] += W_elec[0, ii] * vel[kk, ii]
                        nu[I + 2, sp, kk] += W_elec[1, ii] * vel[kk, ii]
                        nu[I + 3, sp, kk] += W_elec[2, ii] * vel[kk, ii]

    if field_periodic == 1:
        # Copy over source terms into last cells
        pass

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
    
    if field_periodic == 0:
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
        
    Again, check the ghost cell access or remove it altogether. Also, surely
    there's a more efficient way to do this.
    '''
    epsilon = 1e-3
    ni     *= 0.
    
    # Deposit macroparticle moment on grid
    for ii in nb.prange(Ie.shape[0]):
        if idx[ii] >= 0:
            I   = Ie[ii]
            sp  = idx[ii]
        
            ni[I,     sp] += W_elec[0, ii]
            ni[I + 1, sp] += W_elec[1, ii]
            ni[I + 2, sp] += W_elec[2, ii]
            
            if field_periodic == 0:
                # Simulate virtual particles in boundary ghost cells
                # Check if in first cell
                if pos[0, ii] - xmin < dx:
                    
                    if dx - pos[0, ii] + xmin < epsilon:        # If on inner cell boundary, don't count
                        pass
                    elif pos[0, ii] - xmin < epsilon:           # If on simulation boundary, don't count
                        pass
                    else:                                       # Otherwise, count
                        ni[I - 1, sp] += W_elec[0, ii]
                        ni[I    , sp] += W_elec[1, ii]
                        ni[I + 1, sp] += W_elec[2, ii]
                   
                # Check if in last cell
                elif xmax - pos[0, ii] < dx:
                    if xmax - pos[0, ii] < epsilon:             # If on simulation boundary, don't count
                        pass
                    elif dx - xmax + pos[0, ii] < epsilon:      # If on inner cell boundary, don't count
                        pass
                    else:
                        ni[I + 1, sp] += W_elec[0, ii]
                        ni[I + 2, sp] += W_elec[1, ii]
                        ni[I + 3, sp] += W_elec[2, ii]
    
    if field_periodic == 1:
        # Copy source terms into opposite cells
        pass
    
    if source_smoothing == 1:
        for ii in range(Nj):
            three_point_smoothing(ni[:, ii], q_dens)
            
    q_dens *= 0.
    # Sum charge density contributions across species
    for jj in range(Nj):
        ni[:, jj] *= n_contr[jj]
        q_dens    += ni[:, jj] * charge[jj]
    
    
    if field_periodic == 0:
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



#%% OLD FUNCTIONS
@nb.njit()
def OPT_deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu):
    '''Collect number and velocity moments in each cell, weighted by their distance
    from cell nodes.

    INPUT:
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier

    OUTPUT:
        ni     -- Species number moment array(size, Nj)
        nui    -- Species velocity moment array (size, Nj)
        
    13/03/2020 :: Modified to ignore contributions from particles with negative
                    indices (i.e. "deactivated" particles)
                    
    07/05/2020 :: Modified to allow contributions from negatively indexed particles
                    if reflection is enabled - these *were* hot particles, now
                    counted as cold.
    '''
    for ii in nb.prange(vel.shape[1]):
        I   = Ie[ii]
        sp  = idx[ii]
        
        if sp >= 0:
            for kk in range(3):
                nu[I,     sp, kk] += W_elec[0, ii] * vel[kk, ii]
                nu[I + 1, sp, kk] += W_elec[1, ii] * vel[kk, ii]
                nu[I + 2, sp, kk] += W_elec[2, ii] * vel[kk, ii]
            
            ni[I,     sp] += W_elec[0, ii]
            ni[I + 1, sp] += W_elec[1, ii]
            ni[I + 2, sp] += W_elec[2, ii]
    return


@nb.njit()
def OPT_collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu):
    '''
    Moment (charge/current) collection function.

    INPUT:
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier
        
    OUTPUT:
        q_dens -- Charge  density
        Ji     -- Current density
        
    TERTIARY:
        ni
        nu
        temp1D
        
    Source terms in damping region set to be equal to last valid cell value. 
    Smoothing routines deleted (can be found in earlier versions) since TSC 
    weighting effectively also performs smoothing.
    
    07/05/2020 :: Changed damped source terms to give zero gradient at ND-NX
                    boundary. Remaining gradient probably a particle
                    initialization error.
                    
    28/05/2020 :: Implemented 3-point smoothing
    
    QUESTION :: The values in the green cells are still giving me pause.
    '''
    q_dens *= 0.
    Ji     *= 0.
    ni     *= 0.
    nu     *= 0.

    OPT_deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu)
    
    # Sum contributions across species
    for jj in range(Nj):
        q_dens  += ni[:, jj] * n_contr[jj] * charge[jj]

        for kk in range(3):
            Ji[:, kk] += nu[:, jj, kk] * n_contr[jj] * charge[jj]

    # Mirror source term contributions at edge back into domain: Simulates having
    # some sort of source on the outside of the physical space boundary.
    q_dens[ND]          += q_dens[ND - 1]
    q_dens[ND + NX - 1] += q_dens[ND + NX]

    offset = 0#NX//4
    for ii in range(3):
        # Mirror source term contributions
        Ji[ND, ii]          += Ji[ND - 1, ii]
        Ji[ND + NX - 1, ii] += Ji[ND + NX, ii]

        if source_BC == 0:
            # Set damping cell source values (zero gradient)
            Ji[:ND+offset, ii]    = Ji[ND + 1+offset, ii]
            Ji[ND+NX-offset:, ii] = Ji[ND+NX - 2 - offset, ii]
        elif source_BC == 1:
            # Set damping cell source values (copy last)
            Ji[:ND+offset, ii]    = Ji[ND+offset, ii]
            Ji[ND+NX-offset:, ii] = Ji[ND+NX-1-offset, ii]
        elif source_BC == 2:
            # Set damping cell source values (zero second derivative)
            Ji[:ND+offset, ii]    = 2*Ji[ND+offset,      ii] - Ji[ND + 1+offset     , ii]
            Ji[ND+NX-offset:, ii] = 2*Ji[ND+NX-1-offset, ii] - Ji[ND + NX - 2-offset, ii]
        elif source_BC == 3:
            # Set damping cell source values (Initial condition value)
            Ji[:ND+offset, ii]    *= 0.0
            Ji[ND+NX-offset:, ii] *= 0.0
        
    if source_BC == 0:
        # Set damping cell source values (zero gradient)
        q_dens[:ND+offset]    = q_dens[ND + 1+offset]
        q_dens[ND+NX-offset:] = q_dens[ND+NX - 2-offset]
    elif source_BC == 1:
        # Set damping cell source values (copy last)
        q_dens[:ND+offset]    = q_dens[ND+offset]
        q_dens[ND+NX-offset:] = q_dens[ND+NX-1-offset]
    elif source_BC == 2:
        # Set damping cell source values (zero second derivative)
        q_dens[:ND+offset]    = 2*q_dens[ND+offset]      - q_dens[ND + 1+offset]
        q_dens[ND+NX-offset:] = 2*q_dens[ND+NX-1-offset] - q_dens[ND+NX-2-offset]
    elif source_BC == 3:
        # Set damping cell source values (Initial condition value)
        q_dens[:ND+offset]    = ne * q
        q_dens[ND+NX-offset:] = ne * q
        
    # Implement smoothing filter: If enabled
    if source_smoothing == 1:
        three_point_smoothing(q_dens, ni[:, 0])
        for ii in range(3):
            three_point_smoothing(Ji[:, ii], ni[:, 0])

    # Set density minimum
    for ii in range(q_dens.shape[0]):
        if q_dens[ii] < min_dens * ne * q:
            q_dens[ii] = min_dens * ne * q
    return


        
        
    
    
    
        
        
    