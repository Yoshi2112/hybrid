# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb
import numpy as np
from   simulation_parameters_1D  import temp_type, NX, ND, dx, xmin, xmax, qm_ratios,\
                                        B_eq, a, particle_periodic, particle_reflect, particle_open,\
                                        particle_reinit, loss_cone_xmax, \
                                        vth_perp, vth_par, inject_rate, Nj, homogenous
from   sources_1D                import collect_moments

from fields_1D import eval_B0x


@nb.njit()
def advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, v_prime, S, T, temp_N,\
                                  B, E, DT, q_dens_adv, Ji, ni, nu, mp_flux, pc=0):
    '''
    Helper function to group the particle advance and moment collection functions
    '''
    velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E, v_prime, S, T, temp_N, DT)
    position_update(pos, vel, idx, DT, Ie, W_elec, mp_flux)  
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
    
    QUESTION :: Why do particles on the boundary only give 0.5 weighting? 
    ANSWER   :: It seems legit and prevents double counting of a particle on the boundary. Since it
                is a B-field node, there should be 0.5 weighted to both nearby E-nodes. In this case,
                reflection doesn't make sense because there cannot be another particle there - there is 
                only one midpoint position (c.f. mirroring contributions of all other particles due to 
                pretending there's another identical particle on the other side of the simulation boundary).
    '''
    Np         = pos.shape[0]
    epsil      = 1e-15
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 0.0
    
    particle_transform = xmax + (ND - grid_offset)*dx  + epsil  # Offset to account for E/B grid and damping nodes
    
    for ii in np.arange(Np):
        xp          = (pos[ii] + particle_transform) / dx    # Shift particle position >= 0
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
    Bp[0]    =   eval_B0x(pos)  
    constant = a * B_eq 
    for ii in range(idx.shape[0]):
        if idx[ii] >= 0:
            l_cyc      = qm_ratios[idx[ii]] * Bp[0, ii]
            
            Bp[1, ii] += constant * pos[ii] * vel[2, ii] / l_cyc
            Bp[2, ii] -= constant * pos[ii] * vel[1, ii] / l_cyc
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
    
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)                       # Calculate magnetic node weights
    eval_B0_particle_1D(pos, vel, idx, Bp)  
    
    for ii in range(vel.shape[1]):
        if idx[ii] >= 0:
            qmi[ii] = 0.5 * DT * qm_ratios[idx[ii]]                           # q/m for ion of species idx[ii]
            for jj in range(3):                                               # Nodes
                for kk in range(3):                                           # Components
                    Ep[kk, ii] += E[Ie[ii] + jj, kk] * W_elec[jj, ii]         # Vector E-field  at particle location
                    Bp[kk, ii] += B[Ib[ii] + jj, kk] * W_mag[ jj, ii]         # Vector b1-field at particle location

    vel[:, :] += qmi[:] * Ep[:, :]                                            # First E-field half-push IS NOW V_MINUS

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
def vfx(vx, vth):
    f_vx  = np.exp(- 0.5 * (vx / vth) ** 2)
    f_vx /= vth * np.sqrt(2 * np.pi)
    return vx * f_vx


@nb.njit()
def generate_vx(vth):
    while True:
        y_uni = np.random.uniform(0, 4*vth)
        Py    = vfx(y_uni, vth)
        x_uni = np.random.uniform(0, 0.25)
        if Py >= x_uni:
            return y_uni
    

@nb.njit()
def position_update(pos, vel, idx, DT, Ie, W_elec, mp_flux):
    '''
    Updates the position of the particles using x = x0 + vt. 
    Injects particles at boundaries if particle_open == 1 using mp_flux
    Also updates particle nearest node and weighting.

    INPUT:
        pos    -- Particle position array (Also output) 
        vel    -- Particle velocity array (Also output for reflection)
        idx    -- Particle index    array (Also output for reflection)
        DT     -- Simulation time step
        Ie     -- Particle leftmost to nearest node array (Also output)
        W_elec -- Particle weighting array (Also output)
        mp_flux-- Macroparticle flux at each boundary, for each species. Accrued and used per timestep
        
    Note: This function also controls what happens when a particle leaves the 
    simulation boundary.
    
    generate_vx() always produces a positive value. Direction can be set by
    multiplying by -np.sign(pos) (i.e. positive in negative half and negative
    in positive half)
    '''
    pos     += vel[0, :] * DT
    
    # Add flux at each boundary 
    # (particles that would have entered but haven't yet been initialized)
    for kk in range(2):
        mp_flux[kk, :] += inject_rate*DT
         
    # Import Particle boundary conditions: Re-initialize if at edges
    for ii in nb.prange(pos.shape[0]):
        if (pos[ii] < xmin or pos[ii] > xmax):
            
            if particle_reinit == 1: 
                # This needs to have a flag in it to reinitialize particles in
                # pairs, to keep the quiet distribution thing going.
                
                # Reinitialize vx based on flux distribution
                vel[0, ii]  = generate_vx(vth_par[idx[ii]])
                vel[0, ii] *= -np.sign(pos[ii])
                
                # Re-initialize v_perp and check pitch angle
                if temp_type[idx[ii]] == 0:
                    vel[1, ii] = np.random.normal(0, vth_perp[idx[ii]])
                    vel[2, ii] = np.random.normal(0, vth_perp[idx[ii]])
                else:
                    particle_PA = 0.0
                    while np.abs(particle_PA) < loss_cone_xmax:
                        vel[1, ii]  = np.random.normal(0, vth_perp[idx[ii]])
                        vel[2, ii]  = np.random.normal(0, vth_perp[idx[ii]])
                        v_perp      = np.sqrt(vel[1, ii] ** 2 + vel[2, ii] ** 2)
                        
                        particle_PA = np.arctan(v_perp / vel[0, ii])
            
                # Place back inside simulation domain
                if pos[ii] < xmin:
                    pos[ii] = xmin + np.random.uniform(0, 1) * vel[0, ii] * DT
                elif pos[ii] > xmax:
                    pos[ii] = xmax + np.random.uniform(0, 1) * vel[0, ii] * DT

            elif particle_periodic == 1:  
                # Mario (Periodic)
                if pos[ii] > xmax:
                    pos[ii] += xmin - xmax
                elif pos[ii] < xmin:
                    pos[ii] += xmax - xmin 
                   
            elif particle_reflect == 1:
                # Reflect
                if pos[ii] > xmax:
                    pos[ii] = 2*xmax - pos[ii]
                elif pos[ii] < xmin:
                    pos[ii] = 2*xmin - pos[ii]
                vel[0, ii] *= -1.0
                    
            else:                
                # Disable loop: Remove particles that leave the simulation space (open boundary only)
                for ii in nb.prange(pos.shape[0]):
                    if (pos[ii] < xmin or pos[ii] > xmax):
                        pos[ii]    *= 0.0
                        vel[:, ii] *= 0.0
                        idx[ii]     = -1
        
    # Put this into its own function later? Don't bother for now.
    if particle_open == 1:
        # For each boundary
        # For each species
        # If flux > 2
        # Find two empty indices
        # Initialise pair of quiet particles
        # Subtract from total
        acc = 0
        for ii in range(2):
            for jj in range(Nj):
                while mp_flux[ii, jj] >= 2.0:
                    
                    # Find two empty particles (Yes clumsy coding but it works)
                    for kk in nb.prange(acc, pos.shape[0]):
                        if idx[kk] < 0:
                            kk1 = kk
                            acc = kk + 1
                            break
                    for kk in nb.prange(acc, pos.shape[0]):
                        if idx[kk] < 0:
                            kk2 = kk
                            acc = kk + 1
                            break

                    # Reinitialize vx based on flux distribution
                    vel[0, kk1] = generate_vx(vth_par[jj])
                    idx[kk1]    = jj
                    
                    # Re-initialize v_perp and check pitch angle
                    if temp_type[jj] == 0 or homogenous == True:
                        vel[1, kk1] = np.random.normal(0, vth_perp[jj])
                        vel[2, kk1] = np.random.normal(0, vth_perp[jj])
                    else:
                        particle_PA = 0.0
                        while np.abs(particle_PA) <= loss_cone_xmax:
                            vel[1, kk1] = np.random.normal(0, vth_perp[jj])
                            vel[2, kk1] = np.random.normal(0, vth_perp[jj])
                            v_perp      = np.sqrt(vel[1, kk1] ** 2 + vel[2, kk1] ** 2)
                            particle_PA = np.arctan(v_perp / vel[0, kk1])
                    
                    # Amount travelled (vel always +ve at first)
                    dpos = np.random.uniform(0, 1) * vel[0, kk1] * DT
                    
                    # Left boundary injection
                    if ii == 0:
                        pos[kk1]    = xmin + dpos
                        vel[0, kk1] = np.abs(vel[0, kk1])
                        
                    # Right boundary injection
                    else:
                        pos[kk1]    = xmax - dpos
                        vel[0, kk1] = -np.abs(vel[0, kk1])
                    
                    # Copy values to second particle (Same position, xvel. Opposite v_perp) 
                    idx[kk2]    = idx[kk1]
                    pos[kk2]    = pos[kk1]
                    vel[0, kk2] = vel[0, kk1]
                    vel[1, kk2] = vel[1, kk1] * -1.0
                    vel[2, kk2] = vel[2, kk1] * -1.0
                    
                    # Subtract new macroparticles from accrued flux
                    mp_flux[ii, jj] -= 2.0
                    
    assign_weighting_TSC(pos, Ie, W_elec)
    return