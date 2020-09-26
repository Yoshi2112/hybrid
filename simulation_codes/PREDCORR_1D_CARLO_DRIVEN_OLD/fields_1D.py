# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:54:19 2017

@author: iarey
"""
import numpy as np
import numba as nb

import auxilliary_1D as aux
from simulation_parameters_1D import dx, ne, q, mu0, kB, ie, B_eq, a,         \
                                     disable_waves, E_damping, driven_freq,   \
                                     driven_ampl, ND, NX, pulse_offset,       \
                                     pulse_width, driven_k, driver_status


@nb.njit()
def eval_B0x(x):
    return B_eq * (1. + a * x**2)


@nb.njit()
def get_curl_E(E, dE):
    ''' Returns a vector quantity for the curl of a field valid at the
    positions between its gridpoints (i.e. curl(E) -> B-grid, etc.)
    
    INPUT:
        E    -- The 3D field to take the curl of
        dE   -- Finite-differenced solution for the curl of the input field.
        DX   -- Spacing between the nodes, mostly for diagnostics. 
                Defaults to grid spacing specified at initialization.
    
    Calculates value at end B-nodes by calculating the derivative at end
    E-nodes by forwards/backwards difference, then using the derivative
    at B1 to linearly interpolate to B0.    
    '''
    NC        = E.shape[0]
    dE[:, :] *= 0
    
    # Central difference middle cells. Magnetic field offset to lower numbers, dump in +1 arr higher
    for ii in nb.prange(NC - 1):
        dE[ii + 1, 1] = - (E[ii + 1, 2] - E[ii, 2])
        dE[ii + 1, 2] =    E[ii + 1, 1] - E[ii, 1]

    # Curl at E[0] : Forward/Backward difference (stored in B[0]/B[NC])
    dE[0, 1] = -(-3*E[0, 2] + 4*E[1, 2] - E[2, 2]) / 2
    dE[0, 2] =  (-3*E[0, 1] + 4*E[1, 1] - E[2, 1]) / 2
    
    dE[NC, 1] = -(3*E[NC - 1, 2] - 4*E[NC - 2, 2] + E[NC - 3, 2]) / 2
    dE[NC, 2] =  (3*E[NC - 1, 1] - 4*E[NC - 2, 1] + E[NC - 3, 1]) / 2
    
    # Linearly extrapolate to endpoints
    dE[0, 1]      -= 2*(dE[1, 1] - dE[0, 1])
    dE[0, 2]      -= 2*(dE[1, 2] - dE[0, 2])
    
    dE[NC, 1]     += 2*(dE[NC, 1] - dE[NC - 1, 1])
    dE[NC, 2]     += 2*(dE[NC, 2] - dE[NC - 1, 2])
    
    dE /= dx
    return 


@nb.njit()
def push_B(B, E, curlE, DT, qq, damping_array, half_flag=1):
    '''
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
    get_curl_E(E, curlE)

    B       -= 0.5 * DT * curlE                          # Advance using curl (apply retarding factor here?)
    
    for ii in nb.prange(1, B.shape[1]):                  # Apply damping, skipping x-axis
        B[:, ii] *= damping_array                        # Not sure if this needs to modified for half steps?
    return


@nb.njit()
def curl_B_term(B, curlB):
    ''' Returns a vector quantity for the curl of a field valid at the positions 
    between its gridpoints (i.e. curl(B) -> E-grid, etc.)
    
    INPUT:
        B     -- Magnetic field at B-nodes
        curlB -- Finite-differenced solution for curl(B) at E-nodes
    '''
    curlB[:, :] *= 0
    for ii in nb.prange(B.shape[0] - 1):
        curlB[ii, 1] = - (B[ii + 1, 2] - B[ii, 2])
        curlB[ii, 2] =    B[ii + 1, 1] - B[ii, 1]
    
    curlB /= (dx * mu0)
    return 


@nb.njit()
def get_electron_temp(qn, Te, Te0):
    '''
    Calculate the electron temperature in each cell. Depends on the charge density of each cell
    and the treatment of electrons: i.e. isothermal (ie=0) or adiabatic (ie=1)
    '''
    if ie == 0:
        Te[:] = np.ones(qn.shape[0]) * Te0
    elif ie == 1:
        gamma_e = 5./3. - 1.
        Te[:] = Te0 * np.power(qn / (q*ne), gamma_e)
    return


@nb.njit()
def get_grad_P(qn, te, grad_P, temp):
    '''
    Returns the electron pressure gradient (in 1D) on the E-field grid using P = nkT and 
    finite difference.
     
    INPUT:
        qn     -- Grid charge density
        te     -- Grid electron temperature
        grad_P -- Output array for electron pressure gradient
        temp   -- intermediary array used to store electron pressure, since both
                  density and temperature may vary (with adiabatic approx.)
        
    Forwards/backwards differencing at the simulation cells at the edge of the
    physical space domain. Guard cells set to zero. (Actually not anymore. Just take FD over all space)
    '''
    temp     *= 0; grad_P *= 0
    
    nc        = qn.shape[0]
    grad_P[:] = qn * kB * te / q       # Store Pe in grad_P array for calculation

    # Central differencing, internal points
    for ii in nb.prange(1, nc - 1):
        temp[ii] = (grad_P[ii + 1] - grad_P[ii - 1])

    temp    /= (2*dx)
    
    # Return value
    grad_P[:]    = temp[:nc]
    return


@nb.njit()
def add_J_ext(qq, Ji, DT, half_flag):
    '''
    Driven J designed as energy input into simulation. All parameters specified
    in the simulation_parameters script/file
    
    Designed as a Gaussian pulse so that things don't freak out by rising too 
    quickly. Just test with one source point at first
    
    L mode is -90 degree phase in Jz
    '''
    # Soft source wave (What t corresponds to this?)
    # Should put some sort of ramp on it?
    # Also needs to be polarised. By or Bz lagging/leading?
    phase = -90
    N_eq  = ND + NX//2
    time  = qq*DT - 0.5*half_flag*DT
    
    gaussian = np.exp(- ((time - pulse_offset)/ pulse_width) ** 2 )

    # Set new field values in array as soft source
    Ji[N_eq, 1] += driven_ampl * gaussian*np.sin(2 * np.pi * driven_freq * time)
    Ji[N_eq, 2] += driven_ampl * gaussian*np.sin(2 * np.pi * driven_freq * time + phase * np.pi / 180.)    
    return


@nb.njit()
def add_J_ext_pol(qq, Ji, DT, half_flag):
    '''
    Driven J designed as energy input into simulation. All parameters specified
    in the simulation_parameters script/file
    
    Designed as a Gaussian pulse so that things don't freak out by rising too 
    quickly. Just test with one source point at first
    
    Polarised with a LH mode only, uses five points with both w, k specified
    -- Not quite sure how to code this... do you just add a time delay (td, i.e. phase)
        to both the envelope and sin values at each point? 
        
    -- Source node as td=0, other nodes have td depending on distance from source, 
        (ii*dx) and the wave phase velocity v_ph = w/k (which are both known)
    
    P.S. A bunch of these values could be put in the simulation_parameters script.
    Optimize later (after testing shows that it actually works!)
    
    Try delay in gaussian only
    '''
    # Soft source wave (What t corresponds to this?)
    # Should put some sort of ramp on it?
    # Also needs to be polarised. By or Bz lagging/leading?
    phase = -np.pi / 2
    N_eq  = ND + NX//2
    time  = qq*DT - 0.5*half_flag*DT
    v_ph  = driven_freq / driven_k
    omega = 2 * np.pi * driven_freq
    
    for off in np.arange(-2, 3):
        x     = off*dx
        delay = x / v_ph
        gauss = driven_ampl * np.exp(- ((time - pulse_offset - delay)/ pulse_width) ** 2 )
        
        # A = A0 * sin(kx - wt + phase)
        Ji[N_eq + off, 1] += gauss * np.sin(driven_k * x - omega * time)
        Ji[N_eq + off, 2] += gauss * np.sin(driven_k * x - omega * time + phase)    
    return


@nb.njit()
def calculate_E(B, Ji, q_dens, E, Ve, Te, Te0, temp3De, temp3Db, grad_P, E_damping_array, qq, DT, half_flag):
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
    
    NOTE: Sending all but the last element in the cross_product() function seems clumsy... but it works!
    Check :: Does it dereference anything?
    
    12/06/2020 -- Added E-field damping option as per Hu & Denton (2010), Ve x B term only
    '''
    # No driven wave
    if   driver_status == 0:  
        pass
    # Single point source
    elif driver_status == 1:
        add_J_ext(qq, Ji, DT, half_flag=half_flag)
    # Multi-point source
    elif driver_status == 2:
        add_J_ext_pol(qq, Ji, DT, half_flag=half_flag)
        
    curl_B_term(B, temp3De)                                   # temp3De is now curl B term

    Ve[:, 0] = (Ji[:, 0] - temp3De[:, 0]) / q_dens
    Ve[:, 1] = (Ji[:, 1] - temp3De[:, 1]) / q_dens
    Ve[:, 2] = (Ji[:, 2] - temp3De[:, 2]) / q_dens

    get_electron_temp(q_dens, Te, Te0)

    get_grad_P(q_dens, Te, grad_P, temp3Db[:, 0])            # temp1D is now del_p term, temp3D2 slice used for computation
    aux.interpolate_edges_to_center(B, temp3Db)              # temp3db is now B_center

    aux.cross_product(Ve, temp3Db[:temp3Db.shape[0]-1, :], temp3De)                  # temp3De is now Ve x B term
    if E_damping == 1:
        temp3De *= E_damping_array
    
    E[:, 0]  = - temp3De[:, 0] - grad_P[:] / q_dens[:]
    E[:, 1]  = - temp3De[:, 1]
    E[:, 2]  = - temp3De[:, 2]
    
    # Diagnostic flag for testing
    if disable_waves == 1:   
        E *= 0.
    return 