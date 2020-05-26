# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb
import numpy as np
from   simulation_parameters_1D  import temp_type, NX, ND, dx, xmin, xmax, qm_ratios, kB,\
                                        B_eq, a, mass, Tper, Tpar, homogenous, loss_cone_xmax
from   sources_1D                import collect_moments

from fields_1D import eval_B0x
from init_1D   import get_gyroangle_from_velocity



@nb.njit()
def advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, v_prime, S, T, temp_N,\
                                  B, E, DT, q_dens_adv, Ji, ni, nu, pc=0):
    '''
    Helper function to group the particle advance and moment collection functions
    '''
    velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E, v_prime, S, T, temp_N, DT)
    position_update(pos, vel, idx, DT, Ie, W_elec)  
    collect_moments(vel, Ie, W_elec, idx, q_dens_adv, Ji, ni, nu)
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
           
    Could vectorize this with the temp_N array, then check for particles on the boundaries (for
    manual setting)
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
    '''
    constant = - a * B_eq 
    Bp[0]    =   eval_B0x(pos[0])   
    Bp[1]   += constant * pos[0] * pos[1]
    Bp[2]   += constant * pos[0] * pos[2]
    return


@nb.njit()
def eval_B0_particle_1D(pos, vel, idx, Bp):
    '''
    Calculates the B0 magnetic field at the position of a particle. B0x is
    non-uniform in space, and B0r (split into y,z components) is the required
    value to keep div(B) = 0
    
    These values are added onto the existing value of B at the particle location,
    Bp. B0x is simply equated since we never expect a non-zero wave field in x.
        
    Could totally vectorise this. Would have to change to give a particle_temp
    array for memory allocation or something
    '''
    Bp[0]    =   eval_B0x(pos[0])  
    constant = a * B_eq 
    for ii in range(idx.shape[0]):
        l_cyc      = qm_ratios[idx[ii]] * Bp[0, ii]
        
        Bp[1, ii] += constant * pos[0, ii] * vel[2, ii] / l_cyc
        Bp[2, ii] -= constant * pos[0, ii] * vel[1, ii] / l_cyc
    return


@nb.njit()
def velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E, v_prime, S, T, qmi, DT):
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
    '''
    Bp *= 0
    Ep *= 0
    
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)                 # Calculate magnetic node weights
    eval_B0_particle(pos, Bp)  
    
    for ii in range(vel.shape[1]):
        qmi[ii] = 0.5 * DT * qm_ratios[idx[ii]]                         # q/m for ion of species idx[ii]
        for jj in range(3):                                             # Nodes
            for kk in range(3):                                         # Components
                Ep[kk, ii] += E[Ie[ii] + jj, kk] * W_elec[jj, ii]       # Vector E-field  at particle location
                Bp[kk, ii] += B[Ib[ii] + jj, kk] * W_mag[ jj, ii]       # Vector b1-field at particle location

    vel[:, :] += qmi[:] * Ep[:, :]                                      # First E-field half-push IS NOW V_MINUS

    T[:, :] = qmi[:] * Bp[:, :]                                               # Vector Boris variable
    S[:, :] = 2.*T[:, :] / (1. + T[0, :] ** 2 + T[1, :] ** 2 + T[2, :] ** 2)  # Vector Boris variable
    
    v_prime[0, :] = vel[0, :] + vel[1, :] * T[2, :] - vel[2, :] * T[1, :]     # Magnetic field rotation
    v_prime[1, :] = vel[1, :] + vel[2, :] * T[0, :] - vel[0, :] * T[2, :]
    v_prime[2, :] = vel[2, :] + vel[0, :] * T[1, :] - vel[1, :] * T[0, :]
            
    vel[0, :] += v_prime[1, :] * S[2, :] - v_prime[2, :] * S[1, :]
    vel[1, :] += v_prime[2, :] * S[0, :] - v_prime[0, :] * S[2, :]
    vel[2, :] += v_prime[0, :] * S[1, :] - v_prime[1, :] * S[0, :]
    
    vel[:, :] += qmi[:] * Ep[:, :]                                           # Second E-field half-push
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
    
    NOTE :: This reinitialization thing is super slow. More efficient way to code it?
            Maybe initialize array of n_ppc / 2 samples for each species, to at least vectorize it
            Count each one being used, start generating new ones if it runs out (but it shouldn't)
    '''
    pos[0, :] += vel[0, :] * DT
    pos[1, :] += vel[1, :] * DT
    pos[2, :] += vel[2, :] * DT
            
    for ii in nb.prange(pos.shape[1]):
        # Particle boundary conditions
        if (pos[0, ii] < xmin or pos[0, ii] > xmax):
            # Reinitialize
            if homogenous == False: 
                
                # Fix position
                if pos[0, ii] > xmax:
                    pos[0, ii] = 2*xmax - pos[0, ii]
                elif pos[0, ii] < xmin:
                    pos[0, ii] = 2*xmin - pos[0, ii]
                   
                # Re-initialize velocity: Vel_x sign so it doesn't go back into boundary
                sf_per     = np.sqrt(kB *  Tper[idx[ii]] /  mass[idx[ii]])
                sf_par     = np.sqrt(kB *  Tpar[idx[ii]] /  mass[idx[ii]])
                
                if temp_type[idx[ii]] == 0:
                    vel[0, ii] = np.abs(np.random.normal(0, sf_par)) * (- np.sign(pos[0, ii]))
                    vel[1, ii] =        np.random.normal(0, sf_per)
                    vel[2, ii] =        np.random.normal(0, sf_per)
                    v_perp     = np.sqrt(vel[1, ii] ** 2 + vel[2, ii] ** 2)
                else:
                    particle_PA = 0.0
                    while np.abs(particle_PA) < loss_cone_xmax:
                        vel[0, ii]  = np.abs(np.random.normal(0, sf_par)) * (- np.sign(pos[0, ii]))
                        vel[1, ii]  =        np.random.normal(0, sf_per)
                        vel[2, ii]  =        np.random.normal(0, sf_per)
                        
                        v_perp      = np.sqrt(vel[1, ii] ** 2 + vel[2, ii] ** 2)
                        particle_PA = np.arctan(v_perp / vel[0, ii])                   # Calculate particle PA's
                    
                # Don't foget : Also need to reinitialize position gyrophase (pos[1:2])
                B0x         = eval_B0x(pos[0, ii])
                gyangle     = get_gyroangle_from_velocity(vel[:, ii])
                rL          = v_perp / (qm_ratios[idx] * B0x)
                pos[1, ii]  = rL * np.cos(gyangle)
                pos[2, ii]  = rL * np.sin(gyangle)
                    
            # Mario (Periodic)
            else:            
                if pos[0, ii] > xmax:
                    pos[0, ii] += xmin - xmax
                elif pos[0, ii] < xmin:
                    pos[0, ii] += xmax - xmin    
    
    assign_weighting_TSC(pos, Ie, W_elec)
    return
