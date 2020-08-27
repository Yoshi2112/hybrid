# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba   as nb
import numpy   as np
import init_1D as init      
 
from   simulation_parameters_1D  import NX, ND, dx, xmin, xmax, qm_ratios, Nj, vth_par, vth_perp,\
                                        B_eq, B_xmax, a, particle_periodic, nsp_ppc, n_contr
from   sources_1D                import collect_moments

from fields_1D import eval_B0x


#@nb.njit()
def advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, v_prime, S, T, temp_N,\
                                  B, E, DT, q_dens, Ji, ni, nu, flux, pc=0):
    '''
    Helper function to group the particle advance and moment collection functions
    
    Note: use pc = 1 for predictor corrector to not collect new density. 
    Actually, E_pred at N + 3/2 (used to get E(N+1) by averaging at 1/2, 3/2) requires
    density at N + 3/2 to be known, so density at N + 2 is still required (and so
    second position push also still required).
    
    Maybe maths this later on? How did Verbonceur (2005) get around this?
    '''
    velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E, v_prime, S, T, temp_N, DT)
    position_update(pos, vel, idx, Ep, DT, Ie, W_elec, flux)  
    
    if particle_periodic == 0:
        inject_particles(pos, vel, idx, flux, DT, pc)
    
    collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu)
    return


@nb.njit()
def assign_weighting_TSC(pos, idx, I, W, E_nodes=True):
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
        if idx[ii] >= 0:
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
    
    assign_weighting_TSC(pos, idx, Ib, W_mag, E_nodes=False)            # Calculate magnetic node weights
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
def position_update(pos, vel, idx, pos_old, DT, Ie, W_elec, flux):
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
    simulation boundary.
    
    NOTE :: Check the > < and >= <= equal positions when more awake, just make sure
            they're right (not under or over triggering the condition)
            
    acc is an array accumulator that should prevent starting the search from the 
    beginning of the array each time a particle needs to be reinitialized
    
    Is it super inefficient to do two loops for the disable/enable bit? Or worse
    to have arrays all jumbled and array access all horrible? Is there a more
    efficient way to code it?
    
    To Do: Code with one loop, same acc, etc. Assume that there is enough 
    disabled particles to fulfill the needs for a single timestep (especially
    since all the 'spare' particles are at the end. Spare particles should only
    then be accessed when there was not sufficient numbers of 'normal' particles
    in domain, or they were accessed out of order. Might save a bit of time. But
    also need to check it against the 2-loop version.
    
    IDEA: Instead of searching for negative indices, assume all negatives are at 
    end of array. Sort arrays every X timesteps so 'disabled' particles are always
    at end.
    
    OR: Generate array of disabled indices at each timestep and call that instead of
    searching each time.
    
    IDEA: INSTEAD OF STORING POS_OLD, USE Ie INSTEAD, SINCE IT STORES THE CLOSEST
    CELL CENTER - IF IT WAS IN LHS BOUNDARY CELL Ie[ii] == ND (+1?)
    
    I THINK I FIXED THE ERROR :: POOR SPECIFICATION OF WHETHER OR NOT THE PARTICLE WAS
    IN CELL 2 OR NOT.
    '''
    pos_old[:, :] = pos
    
    pos[0, :] += vel[0, :] * DT
    pos[1, :] += vel[1, :] * DT
    pos[2, :] += vel[2, :] * DT
    
    if particle_periodic == 1:
        for ii in nb.prange(pos.shape[1]):           
            if pos[0, ii] > xmax:
                pos[0, ii] += xmin - xmax
            elif pos[0, ii] < xmin:
                pos[0, ii] += xmax - xmin  
    else:
        # Disable loop: Remove particles that leave the simulation space
        # Also store their flux (vx component only, n_contr can be done per species later)
        n_deleted = np.zeros((2, Nj), dtype=nb.float64)
        for ii in nb.prange(pos.shape[1]):
            if pos[0, ii] < xmin:
                flux[0, idx[ii]] += abs(vel[0, ii])
                pos[:, ii] *= 0.0
                vel[:, ii] *= 0.0
                idx[ii]     = -1
                n_deleted[0, idx[ii]]  += 1
            elif pos[0, ii] > xmax:
                flux[1, idx[ii]] += abs(vel[0, ii])
                pos[:, ii] *= 0.0
                vel[:, ii] *= 0.0
                idx[ii]     = -1
                n_deleted[1, idx[ii]]  += 1
    
    #print('Particles deleted:\n', n_deleted)
    assign_weighting_TSC(pos, idx, Ie, W_elec)
    return

#import pdb
@nb.njit()
def inject_particles(pos, vel, idx, flux, dt, pc):
    '''
    Basic injection routine. Flux is set by either:
        -- Measuring outgoing
        -- Set as initial condition by Maxwellian
        
    This code injects particles to equal that outgoing flux. Not sure
    how this is going to go with conserving moments, but we'll see.
    If this works, might be able to use Daughton conditions later. But should
    at least keep constant under static conditions.
    
    NOTE: Finding a random particle like that, do I have to change the timestep?
    What if the particle is just a little too fast? Check later. Or put in cap
    (3*vth?)
    '''
    N_retries = 10      # Set number of times to try and reinitialize a particle 
                        # This is so when the flux is low (but non-zero) the code
                        # doesn't start getting exponentially longer at the tail
                        # looking for a particle with almost zero velocity.

    # Check number of spare particles
    num_spare = (idx < 0).sum()
    if num_spare < 2 * nsp_ppc.sum():
        print('WARNING :: Less than two cells worth of spare particles remaining.')
        if num_spare == 0:
            print('WARNING :: No space particles remaining. Exiting simulation.')
            raise IndexError
    
    # Create particles one at a time until flux drops below a certain limit
    # or number of retries is reached
    acc = 0; n_created = np.zeros((2, Nj), dtype=np.float64)
    for ii in range(2):
        for jj in range(Nj):
            while flux[ii, jj] > 0:
                # Loop until not-too-energetic particle found
                vx = 3e8; new_particle = 0
                
                # Find a vx that'll fit in remaining flux
                for n_tried in range(N_retries):
                    vx       = np.random.normal(0.0, vth_par[jj])
                    if vx <= flux[ii, jj]:
                        new_particle = 1
                        break

                # If successful, load particle
                if new_particle == 1:
                    # Successful particle found, set parameters and subtract flux
                    # Find first negative idx to initialize as new particle
                    for kk in nb.prange(acc, pos.shape[1]):
                        if idx[kk] < 0:
                            acc = kk + 1
                            break
                    
                    # Decide direction of vx, and placement of particle
                    # This could probably be way optimized later.
                    if ii == 0:
                        vel[0, kk] = np.abs(vx)
                        max_pos    = vel[0, kk]*dt
                        
                        if abs(max_pos) < 0.5*dx:
                            pos[0, kk] = xmin + np.random.uniform(0, 1) * max_pos
                        else:
                            pos[0, kk] = xmin + max_pos
                            
                    else:
                        vel[0, kk] = -np.abs(vx)
                        max_pos    = vel[0, kk]*dt
                        
                        if abs(max_pos) < 0.5*dx:
                            pos[0, kk] = xmax + np.random.uniform(0, 1) * max_pos
                        else:
                            pos[0, kk] = xmax + max_pos
                        
                    vel[1, kk] = np.random.normal(0.0, vth_perp[jj])
                    vel[2, kk] = np.random.normal(0.0, vth_perp[jj])
                    
                    gyangle    = init.get_gyroangle_single(vel[:, kk])
                    rL         = np.sqrt(vel[1, kk]**2 + vel[2, kk]**2) / (qm_ratios[idx[kk]] * B_xmax)
                    pos[1, kk] = rL * np.cos(gyangle)
                    pos[2, kk] = rL * np.sin(gyangle)
                    
                    idx[kk]            = jj
                    flux[ii, jj]      -= abs(vx)
                    n_created[ii, jj] += 1
                else:
                    #print('Number of retries reached, stopping injection...')
                    break
    
    #print('Particles created:\n', n_created)
    return
