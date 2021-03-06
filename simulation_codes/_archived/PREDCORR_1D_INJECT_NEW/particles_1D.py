# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb
import numpy as np
from   simulation_parameters_1D  import temp_type, NX, ND, dx, xmin, xmax, qm_ratios, kB,\
                                        B_eq, a, shoji_approx, particle_boundary, mass, Tper
from   sources_1D                import collect_moments

from fields_1D import eval_B0x
import pdb


@nb.njit()
def advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                  B, E, DT, q_dens_adv, Ji, ni, nu, temp1D, pc=0):
    '''
    Helper function to group the particle advance and moment collection functions
    '''
    velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, DT)
    position_update(pos, vel, idx, DT, Ie, W_elec)  
    collect_moments(vel, Ie, W_elec, idx, q_dens_adv, Ji, ni, nu, temp1D)
    return


@nb.njit()
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
    '''
    Np         = pos.shape[1]
    epsil      = 1e-15
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 0.0
    
    particle_transform = xmax + (ND - grid_offset)*dx  + epsil  # Offset to account for E/B grid and damping nodes
    
    for ii in np.arange(Np):
        xp          = (pos[0, ii] + particle_transform) / dx    # Shift particle position >= 0
        I[ii]       = int(round(xp) - 1.0)                      # Get leftmost to nearest node (Vectorize?)
        delta_left  = I[ii] - xp                                # Distance from left node in grid units
        
        if abs(pos[0, ii] - xmin) < 1e-10:
            I[ii]    = ND - 1
            W[0, ii] = 0.0
            W[1, ii] = 0.5
            W[2, ii] = 0.0
        elif abs(pos[0, ii] - xmax) < 1e-10:
            I[ii]    = ND + NX - 1
            W[0, ii] = 0.5
            W[1, ii] = 0.0
            W[2, ii] = 0.0
        else:
            W[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))  # Get weighting factors
            W[1, ii] = 0.75 - np.square(delta_left + 1.)
            W[2, ii] = 1.0  - W[0, ii] - W[1, ii]
    return


@nb.njit()
def eval_B0_particle(pos, Bp):
    '''
    Calculates the B0 magnetic field at the position of a particle. B0x is
    non-uniform in space, and B0r (split into y,z components) is the required
    value to keep div(B) = 0
    
    These values are added onto the existing value of B at the particle location,
    Bp. B0x is simply equated since we never expect a non-zero wave field in x.
        
    Could totally vectorise this. Would have to change to give a particle_temp
    array for memory allocation or something
    '''
    rL     = np.sqrt(pos[1]**2 + pos[2]**2)
    
    B0_r   = - a * B_eq * pos[0] * rL
    Bp[0]  = eval_B0x(pos[0])   
    Bp[1] += B0_r * pos[1] / rL
    Bp[2] += B0_r * pos[2] / rL
    return


@nb.njit()
def eval_B0_particle_1D(pos, vel, Bp, qm):
    '''
    Calculates the B0 magnetic field at the position of a particle. B0x is
    non-uniform in space, and B0r (split into y,z components) is the required
    value to keep div(B) = 0
    
    These values are added onto the existing value of B at the particle location,
    Bp. B0x is simply equated since we never expect a non-zero wave field in x.
        
    Could totally vectorise this. Would have to change to give a particle_temp
    array for memory allocation or something
    '''
    Bp[0]   = eval_B0x(pos[0]) 
    cyc_fac = a * B_eq * pos[0] / (qm * Bp[0])
    
    Bp[1] += vel[2] * cyc_fac
    Bp[2] -= vel[1] * cyc_fac
    return


@nb.njit()
def velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, DT):
    '''
    updates velocities using a Boris particle pusher.
    Based on Birdsall & Langdon (1985), pp. 59-63.

    INPUT:
        part -- Particle array containing velocities to be updated
        B    -- Magnetic field on simulation grid
        E    -- Electric field on simulation grid
        dt   -- Simulation time cadence
        W    -- Weighting factor of particles to rightmost node

    OUTPUT:
        None -- vel array is mutable (I/O array)
        
    Notes: Still not sure how to parallelise this: There are a lot of array operations
    Probably need to do it more algebraically? Find some way to not create those temp arrays.
    Removed the "cross product" and "field interpolation" functions because I'm
    not convinced they helped.
    
    4 temp arrays of length 3 used.
    '''
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)                     # Calculate magnetic node weights
    
    for ii in nb.prange(vel.shape[1]):  
        if idx[ii] >= 0 or particle_boundary == 1:
            qmi = 0.5 * DT * qm_ratios[idx[ii]]                                 # Charge-to-mass ration for ion of species idx[ii]
    
            # These create two new length 3 arrays
            Ep = E[Ie[ii]    , 0:3] * W_elec[0, ii]                             \
               + E[Ie[ii] + 1, 0:3] * W_elec[1, ii]                             \
               + E[Ie[ii] + 2, 0:3] * W_elec[2, ii]                             # Vector E-field at particle location
    
            Bp = B[Ib[ii]    , 0:3] * W_mag[0, ii]                              \
               + B[Ib[ii] + 1, 0:3] * W_mag[1, ii]                              \
               + B[Ib[ii] + 2, 0:3] * W_mag[2, ii]                              # b1 at particle location
                
            v_minus    = vel[:, ii] + qmi * Ep                                  # First E-field half-push
            
            # Add B0 at particle location
            #if shoji_approx == False:
            eval_B0_particle(pos[:, ii], Bp)     
            #pdb.set_trace()                               
# =============================================================================
#             else:
#                 eval_B0_particle_1D(pos[:, ii], v_minus, Bp, qm_ratios[idx[ii]])
# =============================================================================
            
            T = qmi * Bp                                                        # Vector Boris variable
            S = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)                 # Vector Boris variable
            
            v_prime    = np.zeros(3)
            v_prime[0] = v_minus[0] + v_minus[1] * T[2] - v_minus[2] * T[1]     # Magnetic field rotation
            v_prime[1] = v_minus[1] + v_minus[2] * T[0] - v_minus[0] * T[2]
            v_prime[2] = v_minus[2] + v_minus[0] * T[1] - v_minus[1] * T[0]
                    
            v_plus     = np.zeros(3)
            v_plus[0]  = v_minus[0] + v_prime[1] * S[2] - v_prime[2] * S[1]
            v_plus[1]  = v_minus[1] + v_prime[2] * S[0] - v_prime[0] * S[2]
            v_plus[2]  = v_minus[2] + v_prime[0] * S[1] - v_prime[1] * S[0]
            
            vel[:, ii] = v_plus +  qmi * Ep                                     # Second E-field half-push
    return


@nb.njit()
def position_update(pos, vel, idx, DT, Ie, W_elec):
    '''
    Updates the position of the particles using x = x0 + vt. 
    Also updates particle nearest node and weighting.

    INPUT:
        pos    -- Particle position array (Also output) 
        vel    -- Particle velocity array (Also output for reflection)
        idx    -- Particle index    array (Also output for reflection)
        DT     -- Simulation time step
        Ie     -- Particle leftmost to nearest node array (Also output)
        W_elec -- Particle weighting array (Also output)
        
    Note: This function also controls what happens when a particle leaves the 
    simulation boundary with the particle_boundary variable:
        
    particle_boundary == 0 :: Particle is "absorbed", velocity set to zero and
    index becomes negative
    
    particle_boundary == 1 :: Particle is "reflected". Cold particles are truly
    reflected (vx changes sign). Hot particles are converted to cold particles
    by folding their index negative, zeroing their parallel velocity, and
    randomly reinitializing their perpendicular velocity. 
    
    particle_boundary == 2 :: "Mario" particles that come out of the opposite
    boundary that they disappear into. Only used for homogenous B-field.    
    
    NOTE :: Currently this reinit is hardcoded for the cold proton values. More
    changes will be required if multiple hot species (helium, oxygen, etc.)
    become included. Maybe include a "cooldown" array stating at what T_per
    they'll be reinit at.
    '''
    for ii in nb.prange(pos.shape[1]):
        # Only update particles that haven't been absorbed (positive species index)
        if idx[ii] >= 0 or particle_boundary == 1:
            pos[0, ii] += vel[0, ii] * DT
            pos[1, ii] += vel[1, ii] * DT
            pos[2, ii] += vel[2, ii] * DT
            
            # Particle boundary conditions
            if (pos[0, ii] < xmin or pos[0, ii] > xmax):
                
                # Absorb
                if particle_boundary == 0:              
                    vel[:, ii] *= 0          			# Zero particle velocity
                    idx[ii]    -= 128                   # Fold index to negative values (preserves species ID)
                    
                # Reflect
                elif particle_boundary == 1:            
                    if pos[0, ii] > xmax:
                        pos[0, ii] = 2*xmax - pos[0, ii]
                    elif pos[0, ii] < xmin:
                        pos[0, ii] = 2*xmin - pos[0, ii]
                        
                    if temp_type[idx[ii]] == 0 or idx[ii] < 0:
                        # Reflect cold particle
                        vel[0, ii] *= -1.0              
                    else:
                        # Convert hot particle to cold
                        idx[ii]   -= 128                               
                        sf_per     = np.sqrt(kB *  Tper[0] /  mass[0]) # Re-init velocity with cold sf and heading away from boundary
                        vel[0, ii] = 0.0
                        vel[1, ii] = np.random.normal(0, sf_per)
                        vel[2, ii] = np.random.normal(0, sf_per)
                        
                # Mario (Periodic)
                elif particle_boundary == 2:            
                    if pos[0, ii] > xmax:
                        pos[0, ii] += xmin - xmax
                    elif pos[0, ii] < xmin:
                        pos[0, ii] += xmax - xmin    
    
    assign_weighting_TSC(pos, Ie, W_elec)
    return



# =============================================================================
# @nb.njit()
# def position_update(pos, vel, idx, DT, Ie, W_elec):
#     '''Updates the position of the particles using x = x0 + vt. 
#     Also updates particle nearest node and weighting.
# 
#     INPUT:
#         part   -- Particle array with positions to be updated
#         dt     -- Time cadence of simulation
# 
#     OUTPUT:
#         pos    -- Particle updated positions
#         W_elec -- (0) Updated nearest E-field node value and (1-2) left/centre weights
#         
#     Reflective boundaries to simulate the "open ends" that would have flux coming in from the ionosphere side.
#     
#     These equations aren't quite right for xmax != xmin, but they'll do for now
#     '''
#     for ii in nb.prange(pos.shape[1]):
#         # Only update particles that haven't been absorbed (positive species index)
#         if idx[ii] >= 0:
#             pos[0, ii] += vel[0, ii] * DT
#             pos[1, ii] += vel[1, ii] * DT
#             pos[2, ii] += vel[2, ii] * DT
#             
#             # Particle boundary conditions
#             if (pos[0, ii] < xmin or pos[0, ii] > xmax):
#                 
#                 # Absorb hot particles (maybe reinitialize later)
#                 if temp_type[idx[ii]] == 1:              
#                     vel[:, ii] *= 0          			# Zero particle velocity
#                     idx[ii]    -= 128                   # Fold index to negative values (preserves species ID)
#                     
#                 # Reflect cold particles
#                 elif temp_type[idx[ii]] == 0:            
#                     if pos[0, ii] > xmax:
#                         pos[0, ii] = 2*xmax - pos[0, ii]
#                     elif pos[0, ii] < xmin:
#                         pos[0, ii] = 2*xmin - pos[0, ii]
#                     vel[0, ii] *= -1.0
#     return
# =============================================================================