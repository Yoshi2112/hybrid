# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb
import numpy as np
from   simulation_parameters_1D  import NX, ND, dx, xmin, xmax, qm_ratios, Nj, n_contr, kB, Tpar, Tperp, temp_type,\
                                        B_eq, a, mass, particle_periodic, vth_par, B_xmax, vth_perp, loss_cone_xmax
from   sources_1D                import collect_velocity_moments, collect_position_moment

from fields_1D import eval_B0x

import init_1D as init

@nb.njit()
def advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, v_prime, S, T, temp_N,\
                                  B, E, DT, q_dens_adv, Ji, ni, nu, Pi, flux_rem, pc=0):
    '''
    Container function to group and order the particle advance and moment collection functions
    ''' 
    velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E, v_prime, S, T, temp_N, DT)
    position_update(pos, vel, idx, DT, Ie, W_elec)  

# =============================================================================
#     if particle_periodic == False and reflect_cold == False:
#         inject_particles(pos, vel, idx, ni, nu, Pi, flux_rem, DT, pc, N_lost)
# =============================================================================
    
    collect_velocity_moments(pos, vel, Ie, W_elec, idx, nu, Ji, Pi)
    collect_position_moment(pos, Ie, W_elec, idx, q_dens_adv, ni)
    return


@nb.njit()
def assign_weighting_CIC(pos, idx, I, W, E_nodes=True):
    '''Cloud-in-Cell weighting scheme used to distribute particle densities to
    nodes and interpolate field values to particle positions. Winske 1993

    INPUT:
        pos     -- particle positions (x)
        I       -- Leftmost node
        W       -- CIC weight at I + 1
        E_nodes -- True/False flag for calculating values at electric field
                   nodes (grid centres) or not (magnetic field, edges)
    
    The maths effectively converts a particle position into multiples of dx (i.e. nodes),
    rounded (to get nearest node) and then offset to account for E/B grid staggering and 
    to get the leftmost node. This is then offset by the damping number of nodes, ND. The
    calculation for weighting (dependent on delta_left).
    
    NOTE: The addition of `epsilon' prevents banker's rounding due to precision limits. This
          is the easiest way to get around it.
           
    Could vectorize this with the temp_N array, then check for particles on the boundaries (for
    manual setting)
    
    
    '''
    Np         = pos.shape[1]
    epsil      = 1e-15
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 0.0
    
    particle_transform = xmax + (ND - grid_offset)*dx  + epsil      # Offset to account for E/B grid and damping nodes
    
    for ii in np.arange(Np):
        if idx[ii] >= 0: 
            xp    = (pos[0, ii] + particle_transform) / dx    # Shift particle position >= 0
            I[ii] = int(xp)                                   # Get leftmost to nearest node (Vectorize?)
            W[ii] = xp - I[ii]                                # Distance from left node in grid units
    return


@nb.njit()
def eval_B0_particle(pos, Bp):
    '''
    Calculates the B0 magnetic field at the position of a particle. B0x is
    non-uniform in space, and B0r (split into y,z components) is the required
    value to keep div(B) = 0
    
    These values are added onto the existing value of B at the particle location,
    Bp. B0x is simply equated since we never expect a non-zero wave field in x.
    
    Maybe go back to 1D version?
    '''
    constant = - a * B_eq 
    Bp[0]    =   eval_B0x(pos[0])   
    Bp[1]   += constant * pos[0] * pos[1]
    Bp[2]   += constant * pos[0] * pos[2]
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
        
    Check for -ve indexes :: qmi set to zero and interpolation not done. This should
                            ensure the resulting velocity is zero.
    '''
    Bp *= 0
    Ep *= 0
    
    assign_weighting_CIC(pos, idx, Ib, W_mag, E_nodes=False)                # Calculate magnetic node weights
    eval_B0_particle(pos, Bp)  
    
    for ii in range(vel.shape[1]):
        if idx[ii] < 0:
            qmi[ii] = 0.0                                                   # Ensures v = 0 for dead particles
        else:
            qmi[ii] = 0.5 * DT * qm_ratios[idx[ii]]
            for kk in range(3):
                
                # Contribution of node I
                Ep[kk, ii] += E[Ie[ii] + 0, kk] * (1.0 - W_elec[ii])
                Bp[kk, ii] += B[Ib[ii] + 0, kk] * (1.0 - W_mag[ ii])

                # Contribution of node I + 1
                Ep[kk, ii] += E[Ie[ii] + 1, kk] * W_elec[ii]
                Bp[kk, ii] += B[Ib[ii] + 1, kk] * W_mag[ ii]

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
    simulation boundary. As per Daughton et al. (2006).
    '''
    #N_lost = np.zeros((2, Nj), dtype=np.int64)
    
    pos[0, :] += vel[0, :] * DT
    pos[1, :] += vel[1, :] * DT
    pos[2, :] += vel[2, :] * DT
    
    # Check Particle boundary conditions: Re-initialize if at edges
    for ii in nb.prange(pos.shape[1]):
        if idx[ii] >= 0:
            if (pos[0, ii] < xmin or pos[0, ii] > xmax):
                
# =============================================================================
#                 if pos[0, ii] < xmin:
#                     N_lost[0, idx[ii]] += 1
#                 else:
#                     N_lost[1, idx[ii]] += 1
# =============================================================================
                
                # Move particle to opposite boundary (Periodic)
                if particle_periodic == True:   
                    if pos[0, ii] > xmax:
                        pos[0, ii] += xmin - xmax
                    elif pos[0, ii] < xmin:
                        pos[0, ii] += xmax - xmin 
                
                # Random flux initialization at boundary (Open)
                else:
                    if pos[0, ii] > xmax:
                        pos[0, ii] = 2*xmax - pos[0, ii]
                    elif pos[0, ii] < xmin:
                        pos[0, ii] = 2*xmin - pos[0, ii]

                    if temp_type[idx[ii]] == 0:
                        vel[0, ii] = np.random.normal(0, vth_par[idx[ii]])
                        vel[1, ii] = np.random.normal(0, vth_perp[idx[ii]])
                        vel[2, ii] = np.random.normal(0, vth_perp[idx[ii]])
                        v_perp     = np.sqrt(vel[1, ii] ** 2 + vel[2, ii] ** 2)
                    else:
                        particle_PA = 0.0
                        while np.abs(particle_PA) < loss_cone_xmax:
                            vel[0, ii]  = np.random.normal(0, vth_par[idx[ii]])# * (-1.0) * np.sign(pos[0, ii])
                            vel[1, ii]  = np.random.normal(0, vth_perp[idx[ii]])
                            vel[2, ii]  = np.random.normal(0, vth_perp[idx[ii]])
                            v_perp      = np.sqrt(vel[1, ii] ** 2 + vel[2, ii] ** 2)
                            
                            particle_PA = np.arctan(v_perp / vel[0, ii])                   # Calculate particle PA's
                        
                    # Don't foget : Also need to reinitialize position gyrophase (pos[1:2])
                    B0x         = eval_B0x(pos[0, ii])
                    gyangle     = init.get_gyroangle_single(vel[:, ii])
                    rL          = v_perp / (qm_ratios[idx[ii]] * B0x)
                    pos[1, ii]  = rL * np.cos(gyangle)
                    pos[2, ii]  = rL * np.sin(gyangle)
                
# =============================================================================
#                 # Deactivate particle (Open, default)
#                 else: 
#                     pos[:, ii] *= 0.0
#                     vel[:, ii] *= 0.0
#                     idx[ii]    -= 128
# =============================================================================
    #print(N_lost)
    assign_weighting_CIC(pos, idx, Ie, W_elec)
    return


#%% PARTICLE INJECTION ROUTINES
#from scipy.optimize import fsolve
#from scipy.special  import erf, erfinv


@nb.njit()
def locate_spare_indices(idx, N_needed, ii_first=0):
    '''
    Function to locate N_needed number of indices that contain
    deactivated (i.e. spare) particles. 
    '''
    output_indices = np.zeros(N_needed, dtype=nb.int64)
    N_found = 0; ii = ii_first
    
    while N_found < N_needed:
        if idx[ii] < 0:
            output_indices[N_found] = ii
            N_found += 1
        ii += 1
    return output_indices, ii+1


# =============================================================================
# def gamma_so(n, V, U):
#     '''
#     Inbound flux: Phase space from 0->inf
#     '''
#     t1 = n * V / (2 * np.sqrt(np.pi))
#     t2 = np.exp(- U ** 2 / V ** 2)
#     t3 = np.sqrt(np.pi) * U / V
#     t4 = 1 + erf(U / V)
#     return t1 * (t2 + t3*t4)
# 
# 
# def gamma_s(vx, n, V, U):
#     '''
#     Inbound flux: Phase space from 0->vx
#     '''
#     t1  = n * V / (2 * np.sqrt(np.pi))
#     t2  = np.exp(-       U  ** 2 / V ** 2)
#     t2b = np.exp(- (vx - U) ** 2 / V ** 2)
#     t3  = np.sqrt(np.pi) * U / V
#     t4a = erf((vx - U) / V)
#     t4  = erf(      U  / V)
#     return t1 * (t2 - t2b + t3*(t4a + t4))
# =============================================================================


# =============================================================================
# # Minimize this thing (e.g. Find root)
# def find_root(vx, n, V, U, Rx):
#     return gamma_s(vx, n, V, U) / gamma_so(n, V, U) - Rx    
# =============================================================================

# =============================================================================
# @nb.njit()
# def inject_particles(pos, vel, idx, ni, Us, Pi, flux_rem, dt, pc, N_lost):
#     '''
#     Simplified
#     '''
#     end_cells = [ND, ND + NX - 1]
#     ii_last   = 0
#     
#     # For each boundary, use moments
#     bb = 0
#     for ii in end_cells:
#         for jj in range(Nj):
#             new_indices, ii_last = locate_spare_indices(idx, N_lost[bb, jj], ii_last)
# 
#             # For each new particle
#             for kk in range(N_lost[bb, jj]):                
#                 pp         = new_indices[kk]
#                 
#                 # Calculate vx using root finder/minimization (is this the fastest/best way?)
#                 vel[0, pp] = np.random.normal(0.0, vth_par[jj])
#                 vel[1, pp] = np.random.normal(0.0, vth_per[jj])
#                 vel[2, pp] = np.random.normal(0.0, vth_per[jj])
# 
#                 if bb == 0:
#                     vel[0, pp] =  1.0 * np.abs(vel[0, pp])
#                 else:
#                     vel[0, pp] = -1.0 * np.abs(vel[0, pp])
#                 
#                 # Set rL(y, z) off-plane using xmax value
#                 idx[pp]     = jj
#                 gyangle     = init.get_gyroangle_single(vel[:, pp])
#                 rL          = np.sqrt(vel[1, pp]**2 + vel[2, pp]**2) / (qm_ratios[idx[pp]] * B_xmax)
#                 pos[1, pp]  = rL * np.cos(gyangle)
#                 pos[2, pp]  = rL * np.sin(gyangle)
#                                 
#                 # Randomly reinitialize position so it pops into simulation domain
#                 # on position advance
#                 if ii < ni.shape[0] // 2:
#                     pos[0, pp] = np.random.uniform(xmin - vel[0, pp]*dt, xmin)
#                 else:
#                     pos[0, pp] = np.random.uniform(xmax, xmax  - vel[0, pp]*dt)
#                     
#                 # Push them (for now, since I can't seem to get the flux working)
#                 pos[:, pp] += vel[:, pp] * dt
#         bb += 1
#     return
# =============================================================================

# =============================================================================
# def inject_particles(pos, vel, idx, ni, Us, Pi, flux_rem, dt, pc, N_lost):
#     '''
#     A lot of this might be able to be replaced with numpy random functions.
#     
#     Loops through each:
#         -- Boundary (ND, ND + NX - 1)
#         -- Species
#         -- Newly injected particle
#         
#     To check :: 
#         -- Does vx have to be negative for the second boundary? Or is this
#             accounted for in the moments?
#         -- Should I put a rejection method in for a loss-cone distribution 
#             depending on ion type?
#         -- Randomize position in x up to dx/2. Depend on velocity? Or place
#             particle just prior to boundary so it moves into simulation domain
#             on position update
#     '''
#     import pdb
#     print('Injecting particles...')
#     end_cells = [ND, ND + NX - 1]
#     ii_last   = 0
#     
#     # For each boundary, use moments
#     bb = 0
#     for ii in end_cells:
#         for jj in range(Nj):
#             Ws  = 0.5 * mass[jj] * ni[ii, jj] * np.linalg.inv(Pi[ii, jj, :, :])
#             #Cs  = ni[ii, jj] * np.sqrt(np.linalg.det(Ws)) / np.pi ** 1.5
#             Vsx = np.sqrt(2 * Pi[ii, jj, 0, 0] / (mass[jj] * ni[ii, jj]))
#             
#             # Find number of (sim) particles to inject
#             
#             #maxwellian_flux      = dt * ni[ii, jj] * np.sqrt(2 * kB * (Tpar[jj] + 2*Tper[jj]) / (3*np.pi * mass[jj]))
#             
#             integrated_flux      = gamma_so(ni[ii, jj], Vsx, Us[ii, jj, 0]) * dt
#             total_flux           = integrated_flux + flux_rem[ii, jj]
#             #num_inject           = int(total_flux // n_contr[jj])
#             flux_rem[ii, jj]     = total_flux % n_contr[jj]
# 
#             new_indices, ii_last = locate_spare_indices(idx, N_lost[bb, jj], ii_last)
#             
#             # For each new particle
#             for kk in range(N_lost[bb, jj]):                
#                 Rx, Ry, Rz = np.random.uniform(size=3)
#                 
#                 # Calculate vx using root finder/minimization (is this the fastest/best way?)
#                 vx = fsolve(find_root, x0=vth_par[jj], args=(ni[ii, jj], Vsx, Us[0], Rx))
#                                       #,xtol=tol, maxfev=fev)
#                 
#                 # Calculate vy
#                 vy = Us[ii, jj, 1] * erfinv(2*Ry-1) * np.sqrt(Ws[2, 2] / (Ws[1, 1] * Ws[2, 2] - Ws[1, 2] ** 2))\
#                    + (vx - Us[ii, jj, 0]) * Pi[ii, jj, 0, 1] / Pi[ii, jj, 0, 0]
#                 
#                 # Calculate vz
#                 vz = Us[ii, jj, 0] * 1.0 / Ws[2, 2] * (np.sqrt(Ws[2, 2]) * erfinv(2*Rz-1) - (vx - Us[ii, jj, 0])*Ws[0, 2]
#                                                        - (vy - Us[ii, jj, 1])*Ws[1, 2])
#                 
#                 # Set rL(y, z) off-plane using xmax value
#                 pp          = new_indices[kk]
#                 idx[pp]     = jj
#                 gyangle     = init.get_gyroangle_single(vel[:, pp])
#                 rL          = np.sqrt(vy**2 + vz**2) / (qm_ratios[idx[pp]] * B_xmax)
#                 pos[1, pp]  = rL * np.cos(gyangle)
#                 pos[2, pp]  = rL * np.sin(gyangle)
#                 
#                 vel[0, pp]  = vx
#                 vel[1, pp]  = vy
#                 vel[2, pp]  = vz
#                 
#                 # Randomly reinitialize position so it pops into simulation domain
#                 # on position advance
#                 if ii < ni.shape[0] // 2:
#                     pos[0, pp] = np.random.uniform(xmin - vx*dt, xmin)
#                 else:
#                     pos[0, pp] = np.random.uniform(xmax, xmax  - vx*dt)
#                     
#                 # Push them (for now, since I can't seem to get the flux working)
#                 pos[:, pp] += vel[:, pp] * dt
#         bb += 1
#     return
# =============================================================================

    