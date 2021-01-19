# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:33:56 2021

@author: Yoshi
"""
import numpy as np
import numba as nb

do_parallel = True

@nb.njit(cache=True, parallel=do_parallel)
def assign_weighting_TSC(pos, I, W, E_nodes=True):
    '''Triangular-Shaped Cloud (TSC) weighting scheme used to distribute particle densities to
    nodes and interpolate field values to particle positions. Ref. Lipatov? Or Birdsall & Langdon?

    INPUT:
        pos     -- particle positions (x)
        I       -- Leftmost (to nearest) nodes. Output array
        W       -- TSC weights, 3xN array starting at respective I
        E_nodes -- True/False flag for calculating values at electric field
                   nodes (grid centres) or not (magnetic field, edges)
    
    The maths effectively converts a particle position into multiples of dx (i.e. nodes),
    rounded (to get nearest node) and then offset to account for E/B grid staggering and 
    to get the leftmost node. This is then offset by the damping number of nodes, ND. The
    calculation for weighting (dependent on delta_left).
    
    NOTE: The addition of `epsilon' prevents banker's rounding due to precision limits. This
          is the easiest way to get around it.
          
    NOTE2: If statements in weighting prevent double counting of particles on simulation
           boundaries. abs() with threshold used due to machine accuracy not recognising the
           upper boundary sometimes. Note sure how big I can make this and still have it
           be valid/not cause issues, but considering the scale of normal particle runs (speeds
           on the order of va) it should be plenty fine.
           
    Could vectorize this with the temp_N array, then check for particles on the boundaries (for
    manual setting)
    
    QUESTION :: Why do particles on the boundary only give 0.5 weighting? 
    ANSWER   :: It seems legit and prevents double counting of a particle on the boundary. Since it
                is a B-field node, there should be 0.5 weighted to both nearby E-nodes. In this case,
                reflection doesn't make sense because there cannot be another particle there - there is 
                only one midpoint position (c.f. mirroring contributions of all other particles due to 
                pretending there's another identical particle on the other side of the simulation boundary).
    
    UPDATE :: Periodic fields don't require weightings of 0.5 on the boundaries. Loop was copied without
               this option below because it prevents needing to evaluating field_periodic for every
               particle.
    '''
    Np         = pos.shape[0]
    epsil      = 1e-15
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 0.0
    
    particle_transform = xmax + (ND - grid_offset)*dx  + epsil  # Offset to account for E/B grid and damping nodes
    
    if field_periodic == 0:
        for ii in nb.prange(Np):
            xp          = (pos[ii] + particle_transform) / dx       # Shift particle position >= 0
            I[ii]       = int(round(xp) - 1.0)                      # Get leftmost to nearest node (Vectorize?)
            delta_left  = I[ii] - xp                                # Distance from left node in grid units
            
            if abs(pos[ii] - xmin) < 1e-10:
                I[ii]    = ND - 1
                W[0, ii] = 0.0
                W[1, ii] = 0.5
                W[2, ii] = 0.0
            elif abs(pos[ii] - xmax) < 1e-10:
                I[ii]    = ND + NX - 1
                W[0, ii] = 0.5
                W[1, ii] = 0.0
                W[2, ii] = 0.0
            else:
                W[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))  # Get weighting factors
                W[1, ii] = 0.75 - np.square(delta_left + 1.)
                W[2, ii] = 1.0  - W[0, ii] - W[1, ii]
    else:
        for ii in nb.prange(Np):
            xp          = (pos[ii] + particle_transform) / dx       # Shift particle position >= 0
            I[ii]       = int(round(xp) - 1.0)                      # Get leftmost to nearest node (Vectorize?)
            delta_left  = I[ii] - xp                                # Distance from left node in grid units

            W[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))  # Get weighting factors
            W[1, ii] = 0.75 - np.square(delta_left + 1.)
            W[2, ii] = 1.0  - W[0, ii] - W[1, ii]
    return


@nb.njit(cache=True, parallel=do_parallel)
def parmov(pos, vel, Ie, W_elec, Ib, W_mag, B, E, idx, dt):
    for ii in nb.prange(pos.shape[0]):

        # Calculate wave fields at particle position
        Ep = np.zeros(3, dtype=np.float64)  
        Bp = np.zeros(3, dtype=np.float64)

        for jj in nb.prange(3):
            for kk in nb.prange(3):
                Ep[kk] += E[Ie[ii] + jj, kk] * W_elec[jj, ii]   
                Bp[kk] += B[Ib[ii] + jj, kk] * W_mag[ jj, ii]                   

        # Calculate background field at particle position
        Bp[0] += B_eq * (1.0 + a * pos[ii] * pos[ii])
        
        constant = a * B_eq
        l_cyc    = qm_ratios[idx[ii]] * Bp[0]
        
        Bp[1] += constant * pos[ii] * vel[2, ii] / l_cyc
        Bp[2] -= constant * pos[ii] * vel[1, ii] / l_cyc

        # Start Boris Method
        qmi = 0.5 * dt * qm_ratios[idx[ii]]                             # q/m variable including dt
        T   = qmi * Bp 
        S   = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)

        # vel -> v_minus
        vel[0, ii] += qmi * Ep[0]
        vel[1, ii] += qmi * Ep[1]
        vel[2, ii] += qmi * Ep[2]
            
        # Calculate v_prime (maybe use a temp array here?)
        v_prime    = np.zeros(3, dtype=np.float64)
        v_prime[0] = vel[0, ii] + vel[1, ii] * T[2] - vel[2, ii] * T[1]
        v_prime[1] = vel[1, ii] + vel[2, ii] * T[0] - vel[0, ii] * T[2]
        v_prime[2] = vel[2, ii] + vel[0, ii] * T[1] - vel[1, ii] * T[0]
        
        # vel_minus -> vel_plus
        vel[0, ii] += v_prime[1] * S[2] - v_prime[2] * S[1]
        vel[1, ii] += v_prime[2] * S[0] - v_prime[0] * S[2]
        vel[2, ii] += v_prime[0] * S[1] - v_prime[1] * S[0]
        
        # vel_plus -> vel (updated)
        vel[0, ii] += qmi * Ep[0]
        vel[1, ii] += qmi * Ep[1]
        vel[2, ii] += qmi * Ep[2]
        
        # Update position
        pos[ii] += vel[0, ii] * dt
    
        # Check boundary conditions (This isn't the final version)
        if pos[ii] > xmax:
            pos[ii] += xmin - xmax
        elif pos[ii] < xmin:
            pos[ii] += xmax - xmin 
    return


@nb.njit(cache=False, parallel=do_parallel)
def deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu):
    '''
    Collect number and velocity moments in each cell, weighted by their distance
    from cell nodes.

    INPUT:
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier

    OUTPUT:
        ni     -- Species number moment array(size, Nj)
        nui    -- Species velocity moment array (size, Nj)
    '''
    for ii in nb.prange(vel.shape[1]):
        if idx[ii] >= 0:
            for kk in nb.prange(3):
                nu[Ie[ii],     idx[ii], kk] += W_elec[0, ii] * vel[kk, ii]
                nu[Ie[ii] + 1, idx[ii], kk] += W_elec[1, ii] * vel[kk, ii]
                nu[Ie[ii] + 2, idx[ii], kk] += W_elec[2, ii] * vel[kk, ii]
            
            ni[Ie[ii],     idx[ii]] += W_elec[0, ii]
            ni[Ie[ii] + 1, idx[ii]] += W_elec[1, ii]
            ni[Ie[ii] + 2, idx[ii]] += W_elec[2, ii]
    return


@nb.njit(cache=False)
def collect_moments(vel, Ie, W_elec, idx, q_dens, Ji):
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
    '''
    q_dens *= 0.
    Ji     *= 0.
    ni      = np.zeros((NC, Nj),    dtype=nb.float64)
    nu      = np.zeros((NC, Nj, 3), dtype=nb.float64)
    
    deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu)
    return
# =============================================================================
#     # Sum contributions across species
#     for jj in range(Nj):
#         q_dens  += ni[:, jj] * n_contr[jj] * charge[jj]
# 
#         for kk in range(3):
#             Ji[:, kk] += nu[:, jj, kk] * n_contr[jj] * charge[jj]
# 
#     if field_periodic == 0:
#         # Mirror source term contributions at edge back into domain: Simulates having
#         # some sort of source on the outside of the physical space boundary.
#         q_dens[ND]          += q_dens[ND - 1]
#         q_dens[ND + NX - 1] += q_dens[ND + NX]
#     
#         for ii in range(3):
#             # Mirror source term contributions
#             Ji[ND, ii]          += Ji[ND - 1, ii]
#             Ji[ND + NX - 1, ii] += Ji[ND + NX, ii]
#     
#             # Set damping cell source values (copy last)
#             Ji[:ND, ii]     = Ji[ND, ii]
#             Ji[ND+NX:, ii]  = Ji[ND+NX-1, ii]
#             
#         # Set damping cell source values (copy last)
#         q_dens[:ND]    = q_dens[ND]
#         q_dens[ND+NX:] = q_dens[ND+NX-1]
#     else:
#         # If homogenous, move contributions
#         q_dens[li1] += q_dens[ro1]
#         q_dens[li2] += q_dens[ro2]
#         q_dens[ri1] += q_dens[lo1]
#         q_dens[ri2] += q_dens[lo2]
#         
#         # ...and copy periodic values
#         q_dens[ro1] = q_dens[li1]
#         q_dens[ro2] = q_dens[li2]
#         q_dens[lo1] = q_dens[ri1]
#         q_dens[lo2] = q_dens[ri2]
#         
#         # ...and Fill remaining ghost cells
#         q_dens[:lo2] = q_dens[lo2]
#         q_dens[ro2:] = q_dens[ro2]
#         
#         for ii in range(3):
#             Ji[li1, ii] += Ji[ro1, ii]
#             Ji[li2, ii] += Ji[ro2, ii]
#             Ji[ri1, ii] += Ji[lo1, ii]
#             Ji[ri2, ii] += Ji[lo2, ii]
#             
#             # ...and copy periodic values
#             Ji[ro1, ii] = Ji[li1, ii]
#             Ji[ro2, ii] = Ji[li2, ii]
#             Ji[lo1, ii] = Ji[ri1, ii]
#             Ji[lo2, ii] = Ji[ri2, ii]
#             
#             # ...and Fill remaining ghost cells
#             Ji[:lo2, ii] = Ji[lo2, ii]
#             Ji[ro2:, ii] = Ji[ro2, ii]
# 
#     # Set density minimum
#     min_dens = 0.05
#     for ii in range(q_dens.shape[0]):
#         if q_dens[ii] < min_dens * ne * qp:
#             q_dens[ii] = min_dens * ne * qp
#     return
# =============================================================================



if __name__ == '__main__':
    from timeit import default_timer as timer
    
    # Define test parameters
    NP     = 10000000
    NX     = 128
    ND     = 4
    NC     = NX + 2*ND
    B_eq   = 200e-9
    a      = 0.0
    c      = 3e8 
    mp     = 1.673e-27
    qp     = 1.602e-19
    kB     = 1.381e-23
    e0     = 8.854e-12
    mu0    = np.pi*4e-7
    dx     = 1.0
    
    E_per    = np.array([5.0])
    mass     = np.array([mp])
    charge   = np.array([qp])
    density  = np.array([200e6])

    ne       = density.sum()
    vth      = np.sqrt(qp * E_per / mp)
    
    rho        = (mp*density).sum()                          # Mass density for alfven velocity calc.
    wpi        = np.sqrt(ne * qp ** 2 / (mp * e0))           # Proton   Plasma Frequency, wpi (rad/s)
    va         = B_eq / np.sqrt(mu0*rho)                     # Alfven speed at equator: Assuming pure proton plasma
    gyfreq_eq  = qp*B_eq  / mp                               # Proton Gyrofrequency (rad/s) at equator (slowest)
    dx         = 1.0 * va / gyfreq_eq                        # Alternate method of calculating dx (better for multicomponent plasmas)
    DT         = 0.02 / gyfreq_eq
    
    xmax       = NX // 2 * dx                                # Maximum simulation length, +/-ve on each side
    xmin       =-NX // 2 * dx
    Nj         = len(mass)                                   # Number of species
    n_contr    = density / NP                                # Species density contribution: Each macroparticle contributes this density to a cell

    field_periodic = 0
    
    # E-field nodes around boundaries (used for sources and E-fields)
    lo1 = ND - 1 ; lo2 = ND - 2             # Left outer (to boundary)
    ro1 = ND + NX; ro2 = ND + NX + 1        # Right outer
    
    li1 = ND         ; li2 = ND + 1         # Left inner
    ri1 = ND + NX - 1; ri2 = ND + NX - 2    # Right inner
    
    # Define arrays and initialize
    _E      = np.zeros((NC,     3), dtype=np.float64)
    _B      = np.zeros((NC + 1, 3), dtype=np.float64)
    
    pos     = np.zeros(NP     ,     dtype=np.float64)
    idx     = np.zeros(NP     ,     dtype=np.uint8)
    vel     = np.zeros((3, NP),     dtype=np.float64)
    Ie      = np.zeros(NP     ,     dtype=np.uint8)
    W_elec  = np.zeros((3, NP),     dtype=np.float64)
    Ib      = np.zeros(NP     ,     dtype=np.uint8)
    W_mag   = np.zeros((3, NP),     dtype=np.float64)
    q_dens  = np.zeros( NC,         dtype=np.float64)    
    q_dens2 = np.zeros( NC,         dtype=np.float64) 
    Ji      = np.zeros((NC, 3),     dtype=np.float64)
    #ni      = np.zeros((NC, Nj),    dtype=np.float64)
    #nu      = np.zeros((NC, Nj, 3), dtype=np.float64)
    
    
    pos[:]  = np.random.uniform(xmin, xmax, NP)
    vel[0]  = np.random.normal(0.0, vth, NP)
    
    qm_ratios = np.array([qp/mp])
    
    max_inc    = 1000
    start_time = timer()
    assign_weighting_TSC(pos, Ie, W_elec, E_nodes=True)
    for qq in range(max_inc):
        print(qq)
        parmov(pos, vel, Ie, W_elec, Ib, W_mag, _B, _E, idx, DT)

        #deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu)
        #collect_moments(vel, Ie, W_elec, idx, q_dens, Ji)
        
        #assign_weighting_TSC(pos, Ib, W_mag,  E_nodes=False)
    print('Time: {:.2f}s'.format(timer() - start_time)) 
    
    