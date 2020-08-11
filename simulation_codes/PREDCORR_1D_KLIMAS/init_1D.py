# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:27:33 2017

@author: iarey
"""
import numba as nb
import numpy as np
import simulation_parameters_1D as const
import save_routines as save

import particles_1D as particles
import fields_1D    as fields

from simulation_parameters_1D import dx, NX, ND, NC, N, Nj, nsp_ppc, va, B_A, dist_type,  \
                                     idx_start, idx_end, seed, vth_par, vth_perp, drift_v,  \
                                     qm_ratios, rc_hwidth, temp_type, Te0_scalar,\
                                     N_species, damping_multiplier, quiet_start


@nb.njit()
def calc_losses(v_para, v_perp, B0x, st=0):
    '''
    For arrays of parallel and perpendicular velocities, finds the number and 
    indices of particles outside the loss cone.
    '''
    alpha        = np.arctan(v_perp / v_para)                   # Calculate particle PA's
    loss_cone    = np.arcsin(np.sqrt(B0x / B_A))                # Loss cone per particle (based on B0 at particle)
    
    in_loss_cone = np.zeros(v_para.shape[0], dtype=nb.int32)
    for ii in range(v_para.shape[0]):
        if np.abs(alpha[ii]) < loss_cone[ii]:                   # Determine if particle in loss cone
            in_loss_cone[ii] = 1
    
    N_loss       = in_loss_cone.sum()                           # Count number that are
    
    # Collect the indices of those in the loss cone
    loss_idx     = np.zeros(N_loss, dtype=nb.int32)
    lc           = 0
    for ii in range(v_para.shape[0]):
        if in_loss_cone[ii] == True:
            loss_idx[lc] = ii
            lc          += 1
        
    loss_idx    += st                                           # Offset indices to account for position in master array
    return N_loss, loss_idx


@nb.njit()
def get_atan(y, x):
    '''
    Returns proper quadrant arctan in radians. Operates on single y, x
    '''
    if x > 0:
        v = np.arctan(y / x)
    elif y >= 0 and x < 0:
        v = np.pi + np.arctan(y / x)
    elif y < 0 and x < 0:
        v = -np.pi + np.arctan(y / x)
    elif y > 0 and x == 0:
        v = np.pi/2
    elif y < 0 and x == 0:
        v = -np.pi/2
        
    if v < 0:
        v += 2*np.pi
    return v
    

@nb.njit()
def get_gyroangle_array(vel):
    '''
    Vel is a (3,N) vector of 3-velocities for N-particles
    
    Calculates in radians, rounds in degrees, returns in radians.
    WHY!???
    '''
    vel_gphase = np.zeros(vel.shape[1], dtype=nb.float64)
    for ii in range(vel.shape[1]):
        vel_gphase[ii] = (get_atan(vel[2, ii], vel[1, ii]) * 180. / np.pi + 90.)%360.
    return (vel_gphase * np.pi / 180.)

@nb.njit()
def get_gyroangle_single(vel):
    '''
    Vel is a (3,N) vector of 3-velocities for N-particles
    
    Calculates in radians, rounds in degrees, returns in radians.
    WHY!???
    '''
    vel_gphase = (get_atan(vel[2], vel[1]) * 180. / np.pi + 90.)%360.
    return (vel_gphase * np.pi / 180.)


@nb.njit()
def LCD_by_rejection(pos, vel, st, en, jj):
    '''
    Takes in a Maxwellian or pseudo-maxwellian distribution. Outputs the number
    and indexes of any particle inside the loss cone
    '''
    B0x    = fields.eval_B0x(pos[0, st: en])
    N_loss = 1

    while N_loss > 0:
        v_perp      = np.sqrt(vel[1, st: en] ** 2 + vel[2, st: en] ** 2)
        
        N_loss, loss_idx = calc_losses(vel[0, st: en], v_perp, B0x, st=st)

        # Catch for a particle on the boundary : Set 90 degree pitch angle (gyrophase shouldn't overly matter)
        if N_loss == 1:
            if abs(pos[0, loss_idx[0]]) == const.xmax:
                ww = loss_idx[0]
                vel[0, loss_idx[0]] = 0.
                vel[1, loss_idx[0]] = np.sqrt(vel[0, ww] ** 2 + vel[1, ww] ** 2 + vel[2, ww] ** 2)
                vel[2, loss_idx[0]] = 0.
                N_loss = 0

        if N_loss != 0:   
            new_vx = np.random.normal(0., vth_par[ jj], N_loss)             
            new_vy = np.random.normal(0., vth_perp[jj], N_loss)
            new_vz = np.random.normal(0., vth_perp[jj], N_loss)
            
            for ii in range(N_loss):
                vel[0, loss_idx[ii]] = new_vx[ii]
                vel[1, loss_idx[ii]] = new_vy[ii]
                vel[2, loss_idx[ii]] = new_vz[ii]
    return


@nb.njit()
def uniform_gaussian_distribution_quiet():
    '''Creates an N-sampled normal distribution across all particle species within each simulation cell

    OUTPUT:
        pos -- 3xN array of particle positions. Pos[0] is uniformly distributed with boundaries depending on its temperature type
        vel -- 3xN array of particle velocities. Each component initialized as a Gaussian with a scale factor determined by the species perp/para temperature
        idx -- N   array of particle indexes, indicating which species it belongs to. Coded as an 8-bit signed integer, allowing values between +/-128
    
    Note: y,z components of particle positions intialized with identical gyrophases, since only the projection
    onto the x-axis interacts with the simulation fields. pos y,z are kept ONLY to calculate/track the Larmor radius 
    of each particle. This initial position suffers the same issue as trying to update the radial field using 
    B0r = 0 for the Larmor radius approximation, however because this is only an initial condition, at worst this
    will just cause a variation in the Larmor radius with position in x, but this will at least be conserved 
    throughout the simulation, and not drift with time.
    
    CHECK THIS LATER: BUT ITS ONLY AN INITIAL CONDITION SO IT SHOULD BE OK FOR NOW

    # Could use temp_type[jj] == 1 for RC LCD only
    '''
    pos = np.zeros((3, N), dtype=np.float64)
    vel = np.zeros((3, N), dtype=np.float64)
    idx = np.ones(N,       dtype=np.int8) * -1 # Start all particles as disabled (idx < 0)
    np.random.seed(seed)
    
    for jj in range(Nj):
        half_n = nsp_ppc[jj] // 2                     # Half particles per cell - doubled later
        if temp_type[jj] == 0:                        # Change how many cells are loaded between cold/warm populations
            NC_load = NX
        else:
            if rc_hwidth == 0 or rc_hwidth > NX//2:   # Need to change this to be something like the FWHM or something
                NC_load = NX
            else:
                NC_load = 2*rc_hwidth
        
        # Load particles in each applicable cell
        acc = 0; offset  = 0
        for ii in range(NC_load):
            # Add particle if last cell (for symmetry)
            if ii == NC_load - 1:
                half_n += 1
                offset  = 1
                
            # Particle index ranges
            st = idx_start[jj] + acc
            en = idx_start[jj] + acc + half_n
            
            # Set position for half: Analytically uniform
            for kk in range(half_n):
                pos[0, st + kk] = dx*(float(kk) / (half_n - offset) + ii)
            
            # Turn [0, NC] distro into +/- NC/2 distro
            pos[0, st: en]-= NC_load*dx/2              
            
            # Set velocity for half: Randomly Maxwellian
            vel[0, st: en] = np.random.normal(0, vth_par[jj],  half_n) +  drift_v[jj]
            vel[1, st: en] = np.random.normal(0, vth_perp[jj], half_n)
            vel[2, st: en] = np.random.normal(0, vth_perp[jj], half_n)
            idx[   st: en] = jj          # Turn particle on
            
            # Set Loss Cone Distribution: Reinitialize particles in loss cone (move to a function)
            if const.homogenous == False and temp_type[jj] == 1:
                LCD_by_rejection(pos, vel, st, en, jj)
                
# =============================================================================
#             # Quiet start : Initialize second half
#             if quiet_start == True:
#                 vel[0, en: en + half_n] = vel[0, st: en] *  1.0     # Set parallel
#             else:
#                 vel[0, en: en + half_n] = vel[0, st: en] * -1.0     # Set anti-parallel
#                 
#             pos[0, en: en + half_n] = pos[0, st: en]                # Other half, same position
#             vel[1, en: en + half_n] = vel[1, st: en] * -1.0         # Invert perp velocities (v2 = -v1)
#             vel[2, en: en + half_n] = vel[2, st: en] * -1.0
#             idx[   en: en + half_n] = jj          # Turn particle on
# =============================================================================
            
            acc                    += half_n * 2

    # Set initial Larmor radius - rL from v_perp, distributed to y,z based on velocity gyroangle
    print('Initializing particles off-axis')
    B0x         = fields.eval_B0x(pos[0, :en])
    v_perp      = np.sqrt(vel[1, :en] ** 2 + vel[2, :en] ** 2)
    gyangle     = get_gyroangle_array(vel[:, :en])
    rL          = v_perp / (qm_ratios[idx[:en]] * B0x)
    pos[1, :en] = rL * np.cos(gyangle)
    pos[2, :en] = rL * np.sin(gyangle)
    
    return pos, vel, idx


@nb.njit()
def initialize_particles():
    '''Initializes particle arrays.
    
    INPUT:
        <NONE>
        
    OUTPUT:
        pos    -- Particle position array (3, N)
        vel    -- Particle velocity array (3, N)
        Ie     -- Initial particle positions by leftmost E-field node
        W_elec -- Initial particle weights on E-grid
        Ib     -- Initial particle positions by leftmost B-field node
        W_mag  -- Initial particle weights on B-grid
        idx    -- Particle type index
    '''
    pos, vel, idx = uniform_gaussian_distribution_quiet()

    Ie         = np.zeros(N, dtype=np.uint16)
    Ib         = np.zeros(N, dtype=np.uint16)
    W_elec     = np.zeros((3, N), dtype=np.float64)
    W_mag      = np.zeros((3, N), dtype=np.float64)
    
    Bp      = np.zeros((3, N), dtype=np.float64)
    Ep      = np.zeros((3, N), dtype=np.float64)
    temp_N  = np.zeros((N),    dtype=np.float64)
    
    particles.assign_weighting_TSC(pos, idx, Ie, W_elec)
    return pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, temp_N


@nb.njit()
def set_damping_array(B_damping_array, E_damping_array, DT):
    '''Create masking array for magnetic field damping used to apply open
    boundaries. Based on applcation by Shoji et al. (2011) and
    Umeda et al. (2001)
    
    Shoji's application multiplies by the resulting field before it 
    is returned at each timestep (or each time called?), but Umeda's variant 
    includes a damping on the increment as well as the solution, with a
    different r for each (one damps, one changes the phase velocity and
    increases the "effective damping length").
    
    Also, using Shoji's parameters, mask at most damps 98.7% at end grid 
    points. Relying on lots of time spend there? Or some sort of error?
    Can just play with r value/damping region length once it doesn't explode.
    
    23/03/2020 Put factor of 0.5 in front of va to set group velocity approx.
    14/05/2020 Put factor of 0.5 in front of DT since B is only ever pushed 0.5DT
    '''
    r_damp   = np.sqrt(29.7 * 0.5 * va * (0.5 * DT / dx) / ND)   # Damping coefficient
    r_damp  *= damping_multiplier
    
    # Do B-damping array
    B_dist_from_mp  = np.abs(np.arange(NC + 1) - 0.5*NC)                # Distance of each B-node from midpoint
    for ii in range(NC + 1):
        if B_dist_from_mp[ii] > 0.5*NX:
            B_damping_array[ii] = 1. - r_damp * ((B_dist_from_mp[ii] - 0.5*NX) / ND) ** 2 
        else:
            B_damping_array[ii] = 1.0
            
    # Do E-damping array
    E_dist_from_mp  = np.abs(np.arange(NC) + 0.5 - 0.5*NC)                # Distance of each B-node from midpoint
    for ii in range(NC):
        if E_dist_from_mp[ii] > 0.5*NX:
            E_damping_array[ii] = 1. - r_damp * ((E_dist_from_mp[ii] - 0.5*NX) / ND) ** 2 
        else:
            E_damping_array[ii] = 1.0
    return


@nb.njit()
def initialize_fields():
    '''Initializes field ndarrays and sets initial values for fields based on
       parameters in config file.

    INPUT:
        <NONE>

    OUTPUT:
        B      -- Magnetic field array: Node locations on cell edges/vertices
        E_int  -- Electric field array: Node locations in cell centres
        E_half -- Electric field array: Node locations in cell centres
        Ve     -- Electron fluid velocity moment: Calculated as part of E-field update equation
        Te     -- Electron temperature          : Calculated as part of E-field update equation          
    '''
    B       = np.zeros((NC + 1, 3), dtype=nb.float64)
    E_int   = np.zeros((NC    , 3), dtype=nb.float64)
    E_half  = np.zeros((NC    , 3), dtype=nb.float64)
    
    Ve      = np.zeros((NC, 3), dtype=nb.float64)
    Te      = np.ones(  NC,     dtype=nb.float64) * Te0_scalar
    Te0     = np.ones(  NC,     dtype=nb.float64) * Te0_scalar
    return B, E_int, E_half, Ve, Te, Te0


@nb.njit()
def initialize_source_arrays():
    '''Initializes source term ndarrays. Each term is collected on the E-field grid.

    INPUT:
        <NONE>

    OUTPUT:
        q_dens  -- Total ion charge  density
        q_dens2 -- Total ion charge  density (used for averaging)
        Ji      -- Total ion current density
        ni      -- Zeroth moment : Ion number density per species (Scalar)
        nu      -- First  moment : Ion velocity "density" per species (Vector)
        Pi      -- Second moment : Ion pressure tensor per species (Tensor) (only at two cells, times two for "old/new")
    '''
    q_dens  = np.zeros( NC,            dtype=nb.float64)    
    q_dens2 = np.zeros( NC,            dtype=nb.float64) 
    Ji      = np.zeros((NC, 3),        dtype=nb.float64)
    ni      = np.zeros((NC, Nj),       dtype=nb.float64)
    nu      = np.zeros((NC, Nj, 3),    dtype=nb.float64)
    return q_dens, q_dens2, Ji, ni, nu


@nb.njit()
def initialize_tertiary_arrays():
    '''Initializes source term ndarrays. Each term is collected on the E-field grid.

    INPUT:
        <NONE>
        
    OUTPUT:
        temp3Db       -- Swap-file vector array with B-grid dimensions
        temp3De       -- Swap-file vector array with E-grid dimensions
        temp1D        -- Swap-file scalar array with E-grid dimensions
        old_particles -- Location to store old particle values (positions, velocities, weights)
                         as part of predictor-corrector routine
        old_fields    -- Location to store old B, Ji, Ve, Te field values for predictor-corrector routine
    '''
    temp3Db       = np.zeros((NC + 1, 3)     , dtype=nb.float64)
    temp3De       = np.zeros((NC    , 3)     , dtype=nb.float64)
    temp1D        = np.zeros( NC             , dtype=nb.float64) 
    old_fields    = np.zeros((NC + 1, 10)    , dtype=nb.float64)
    
    v_prime = np.zeros((3, N),      dtype=nb.float64)
    S       = np.zeros((3, N),      dtype=nb.float64)
    T       = np.zeros((3, N),      dtype=nb.float64)
        
    old_particles = np.zeros((11, N),      dtype=nb.float64)
    return old_particles, old_fields, temp3De, temp3Db, temp1D, v_prime, S, T


def set_timestep(vel, Te0):
    '''
    INPUT:
        vel -- Initial particle velocities
    OUTPUT:
        DT              -- Maximum allowable timestep (seconds)
        max_inc         -- Number of integer timesteps to get to end time
        part_save_iter  -- Number of timesteps between particle data saves
        field_save_iter -- Number of timesteps between field    data saves
    
    Note : Assumes no dispersion effects or electric field acceleration to
           be initial limiting factor. This may change for inhomogenous loading
           of particles or initial fields.
    '''
    ion_ts   = const.orbit_res / const.gyfreq               # Timestep to resolve gyromotion
    vel_ts   = 0.5 * const.dx / np.max(np.abs(vel[0, :]))   # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step 

    gyperiod = 2 * np.pi / const.gyfreq
    DT       = min(ion_ts, vel_ts)
    max_time = const.max_rev * 2 * np.pi / const.gyfreq_eq     # Total runtime in seconds
    max_inc  = int(max_time / DT) + 1                          # Total number of time steps

    if const.part_res == 0:
        part_save_iter = 1
    else:
        part_save_iter = int(const.part_res*gyperiod / DT)

    if const.field_res == 0:
        field_save_iter = 1
    else:
        field_save_iter = int(const.field_res*gyperiod / DT)

    if const.save_fields == 1 or const.save_particles == 1:
        save.store_run_parameters(DT, part_save_iter, field_save_iter, Te0)

    B_damping_array = np.ones(NC + 1, dtype=float)
    E_damping_array = np.ones(NC    , dtype=float)
    set_damping_array(B_damping_array, E_damping_array, DT)

    print('Timestep: %.4fs, %d iterations total\n' % (DT, max_inc))
    return DT, max_inc, part_save_iter, field_save_iter, B_damping_array, E_damping_array