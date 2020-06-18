# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:27:33 2017

@author: iarey
"""
import pdb
import numba as nb
import numpy as np
import simulation_parameters_1D as const
import save_routines as save
import diagnostics as diag

import particles_1D as particles
import fields_1D    as fields

from simulation_parameters_1D import dx, NX, ND, NC, N, kB, Nj, nsp_ppc, va, B_A, dist_type,  \
                                     idx_start, idx_end, seed, Tpar, Tper, mass, drift_v,  \
                                     qm_ratios, rc_hwidth, temp_type, Te0_scalar,\
                                     ne, q, N_species, damping_multiplier, quiet_start, \
                                     beta_par, beta_per, xmin, xmax


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
def LCD_by_rejection(pos, vel, sf_par, sf_per, st, en, jj):
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
            new_vx = np.random.normal(0., sf_par, N_loss)             
            new_vy = np.random.normal(0., sf_per, N_loss)
            new_vz = np.random.normal(0., sf_per, N_loss)
            
            for ii in range(N_loss):
                vel[0, loss_idx[ii]] = new_vx[ii]
                vel[1, loss_idx[ii]] = new_vy[ii]
                vel[2, loss_idx[ii]] = new_vz[ii]
    return


@nb.njit()
def check_boundary_particles(pos, vel):
    '''
    Make sure boundary particles are exactly on the boundary
    Also make sure their velocities are pointing inwards
    
    This does weird things to my moments???
    '''
    print('Checking boundary particles')
    p_thres = 0.01 # If within 1cm of boundary
    for ii in nb.prange(pos.shape[1]):
        if np.abs(pos[0, ii] - const.xmin) < p_thres:
            pos[0, ii] = const.xmin
            vel[0, ii] = np.abs(vel[0, ii])

        elif np.abs(pos[0, ii] - const.xmax) < p_thres:
            pos[0, ii] = const.xmax
            vel[0, ii] = -np.abs(vel[0, ii])

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
    idx = np.zeros(N,      dtype=np.int8)
    np.random.seed(seed)
    
    # Start all particles as disabled (idx < 0)
    idx[:] = -1.0

    for jj in range(Nj):
        idx[idx_start[jj]: idx_end[jj]] = jj          # Set particle idx
        
        sf_par = np.sqrt(kB *  Tpar[jj] /  mass[jj])  # Scale factors for velocity initialization
        sf_per = np.sqrt(kB *  Tper[jj] /  mass[jj])
        
        if dist_type[jj] == 0:                            # Uniform position distribution (incl. limitied RC)
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
                vel[0, st: en] = np.random.normal(0, sf_par, half_n) +  drift_v[jj]
                vel[1, st: en] = np.random.normal(0, sf_per, half_n)
                vel[2, st: en] = np.random.normal(0, sf_per, half_n)
    
                # Set Loss Cone Distribution: Reinitialize particles in loss cone (move to a function)
                if const.homogenous == False and temp_type[jj] == 1:
                    LCD_by_rejection(pos, vel, sf_par, sf_per, st, en, jj)
                    
                # Quiet start : Initialize second half
                if quiet_start == True:
                    vel[0, en: en + half_n] = vel[0, st: en] *  1.0     # Set parallel
                else:
                    vel[0, en: en + half_n] = vel[0, st: en] * -1.0     # Set anti-parallel
                    
                pos[0, en: en + half_n] = pos[0, st: en]                # Other half, same position
                vel[1, en: en + half_n] = vel[1, st: en] * -1.0         # Invert perp velocities (v2 = -v1)
                vel[2, en: en + half_n] = vel[2, st: en] * -1.0
                
                acc                    += half_n * 2
        
        else:
            # Gaussian position distribution
            # Remember :: N_species just gives a total number of species particles
            # which was scaled by the size of rc_hwidth
            if rc_hwidth == 0:
                rc_hwidth_norm = NX // 2
            else:
                rc_hwidth_norm = rc_hwidth
            
            half_n = N_species[jj] // 2
            sigma  = rc_hwidth_norm*dx / (2 * np.sqrt(2 * np.log(2)))
            
            st = idx_start[jj]
            en = idx_start[jj] + half_n
            
            pos[0, st: en] = np.random.normal(0, sigma, half_n)
            vel[0, st: en] = np.random.normal(0, sf_par, half_n) +  drift_v[jj]
            vel[1, st: en] = np.random.normal(0, sf_per, half_n)
            vel[2, st: en] = np.random.normal(0, sf_per, half_n)
            
            # Reinitialize particles outside simulation bounds
            for ii in range(st, en):
                while abs(pos[0, ii]) > const.xmax:
                    pos[0, st: en] = np.random.normal(0, sigma)
                    
            if const.homogenous == False and temp_type[jj] == 1:
                    LCD_by_rejection(pos, vel, sf_par, sf_per, st, en, jj)
                    
            # Initialize second half
            if quiet_start == True:
                vel[0, en: en + half_n] = vel[0, st: en]
            else:
                vel[0, en: en + half_n] = vel[0, st: en] * -1.0
                
            vel[1, en: en + half_n] = vel[1, st: en] * -1.0
            vel[2, en: en + half_n] = vel[2, st: en] * -1.0
            
            pos[0, en: en + half_n] = pos[0, st: en]            # Other half, same position
            
    
    # Set initial Larmor radius - rL from v_perp, distributed to y,z based on velocity gyroangle
    print('Initializing particles off-axis')
    B0x         = fields.eval_B0x(pos[0, :en])
    v_perp      = np.sqrt(vel[1, :en] ** 2 + vel[2, :en] ** 2)
    gyangle     = get_gyroangle_array(vel[:, :en])
    rL          = v_perp / (qm_ratios[idx[:en]] * B0x)
    pos[1, :en] = rL * np.cos(gyangle)
    pos[2, :en] = rL * np.sin(gyangle)
    
    #check_boundary_particles(pos, vel)
    
    return pos, vel, idx


@nb.njit()
def init_totally_random():
    pos = np.zeros((3, N), dtype=np.float64)
    vel = np.zeros((3, N), dtype=np.float64)
    idx = np.zeros(N,      dtype=np.int8)
    np.random.seed(seed)
    
    for jj in range(Nj):
        idx[idx_start[jj]: idx_end[jj]] = jj          # Set particle idx
        
        sf_par = np.sqrt(kB *  Tpar[jj] /  mass[jj])  # Scale factors for velocity initialization
        sf_per = np.sqrt(kB *  Tper[jj] /  mass[jj])
                
        # Particle index ranges
        st = idx_start[jj]
        en = idx_end[jj]

        pos[0, st:en]  = np.random.uniform(xmin, xmax, en-st)
        vel[0, st: en] = np.random.normal(0, sf_par, en-st) +  drift_v[jj]
        vel[1, st: en] = np.random.normal(0, sf_per, en-st)
        vel[2, st: en] = np.random.normal(0, sf_per, en-st)
            
# =============================================================================
#         # Quiet start : Initialize second half
#         if quiet_start == True:
#             vel[0, en: en + half_n] = vel[0, st: en] *  1.0     # Set parallel
#         else:
#             vel[0, en: en + half_n] = vel[0, st: en] * -1.0     # Set anti-parallel
#             
#         pos[0, en: en + half_n] = pos[0, st: en]                # Other half, same position
#         vel[1, en: en + half_n] = vel[1, st: en] * -1.0         # Invert perp velocities (v2 = -v1)
#         vel[2, en: en + half_n] = vel[2, st: en] * -1.0
# =============================================================================
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
    #pos, vel, idx = uniform_gaussian_distribution_quiet()
    pos, vel, idx = init_totally_random()
    Ie         = np.zeros(N, dtype=nb.uint16)
    Ib         = np.zeros(N, dtype=nb.uint16)
    W_elec     = np.zeros(N, dtype=nb.float64)
    W_mag      = np.zeros(N, dtype=nb.float64)
    
    Bp      = np.zeros((3, N), dtype=nb.float64)
    Ep      = np.zeros((3, N), dtype=nb.float64)
    temp_N  = np.zeros((N),    dtype=nb.float64)
    
    particles.assign_weighting_CIC(pos, idx, Ie, W_elec)
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
    Pi      = np.zeros((NC, Nj, 3, 3), dtype=nb.float64)
    return q_dens, q_dens2, Ji, ni, nu, Pi


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
        old_moments   -- Location to store old moments (13*Nj per cell). Pi Tensor 
                         seems wasteful, but since 95% or more of memory useage is particles,
                         go with the easy to code/read alternative.
    '''
    temp3Db       = np.zeros((NC + 1, 3)     , dtype=nb.float64)
    temp3De       = np.zeros((NC    , 3)     , dtype=nb.float64)
    temp1D        = np.zeros( NC             , dtype=nb.float64) 
    old_fields    = np.zeros((NC + 1, 10)    , dtype=nb.float64)
    old_moments   = np.zeros((NC    , 13, Nj), dtype=nb.float64)
    flux_rem      = np.zeros((NC    , Nj)    , dtype=nb.float64)
    
    v_prime = np.zeros((3, N),      dtype=nb.float64)
    S       = np.zeros((3, N),      dtype=nb.float64)
    T       = np.zeros((3, N),      dtype=nb.float64)
        
    old_particles = np.zeros((9, N),      dtype=nb.float64)
        
    return old_particles, old_fields, old_moments, flux_rem, temp3De, temp3Db, temp1D, v_prime, S, T


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

# =============================================================================
#     if E[:, 0].max() != 0:
#         elecfreq        = qm_ratios.max()*(np.abs(E[:, 0] / np.abs(vel).max()).max())               # Electron acceleration "frequency"
#         Eacc_ts         = freq_res / elecfreq                            
#     else:
#         Eacc_ts = ion_ts
# =============================================================================

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



if __name__ == '__main__':
    import matplotlib.pyplot as plt
   
    POS, VEL, IDX = uniform_gaussian_distribution_quiet()
    
    #POS, VEL, IDX = uniform_accounting_for_beta()
    
    V_MAG  = np.sqrt(VEL[0] ** 2 + VEL[1] ** 2 + VEL[2] ** 2) / va 
    V_PERP = np.sign(VEL[2]) * np.sqrt(VEL[1] ** 2 + VEL[2] ** 2) / va
    V_PARA = VEL[0] / va
    
    #diag.check_velocity_components_vs_space(POS, VEL, jj=1)
    #diag.plot_temperature_extremes()
    #diag.check_cell_velocity_distribution_2D(POS, VEL, node_number=None, jj=1, save=True)
    #diag.check_position_distribution(POS)
    #diag.collect_macroparticle_moments(pos, vel, idx)
    #diag.check_cell_velocity_distribution(POS, VEL, node_number=0, j=0)
    
# =============================================================================
#     jj = 1
#     
#     x = V_PERP[idx_start[jj]: idx_end[jj]]
#     y = V_PARA[idx_start[jj]: idx_end[jj]]
#     
#     plt.ioff()    
#     xmin = x.min()
#     xmax = x.max()
#     ymin = y.min()
#     ymax = y.max()
# 
#     fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
#     fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
#     ax = axs[0]
#     hb = ax.hexbin(x, y, gridsize=50, cmap='inferno')
#     ax.axis([xmin, xmax, ymin, ymax])
#     ax.set_title("F(v) :: {}".format(const.species_lbl[jj]))
#     cb = fig.colorbar(hb, ax=ax)
#     cb.set_label('counts')
#     
#     ax = axs[1]
#     hb = ax.hexbin(x, y, gridsize=50, bins='log', cmap='inferno')
#     ax.axis([xmin, xmax, ymin, ymax])
#     ax.set_title("With a log color scale")
#     cb = fig.colorbar(hb, ax=ax)
#     cb.set_label('log10(N)')
#     
#     plt.show()
# =============================================================================
    
# =============================================================================
#     # Test gyrophase transformation
#     pos_gphase  = get_atan(POS[2], POS[1]) * 180. / np.pi
#     vel_gphase  = (get_atan(VEL[2], VEL[1]) * 180. / np.pi + 90.)%360.
#     
#     dot_product = POS[1] * VEL[1] + POS[2] * VEL[2]
#     mag_a       = np.sqrt(POS[1] ** 2 + POS[2] ** 2)
#     mag_b       = np.sqrt(VEL[1] ** 2 + VEL[2] ** 2)
#     rel_angle   = np.arccos(dot_product / (mag_a * mag_b)) * 180. / np.pi
#     
#     print(dot_product.max())
#     print(rel_angle.min())
# =============================================================================
    
        
    
    
    for jj in range(const.Nj):
        if True:
            # Loss cone diagram
            fig1, ax1 = plt.subplots()
            
            ax1.scatter(V_PERP[idx_start[jj]: idx_end[jj]], V_PARA[idx_start[jj]: idx_end[jj]], s=1, c=const.temp_color[jj])

            ax1.set_title('Loss Cone Distribution :: {}'.format(const.species_lbl[jj]))
            ax1.set_ylabel('$v_\parallel (/v_A)$')
            ax1.set_xlabel('$v_\perp (/v_A)$')
            ax1.axis('equal')
     
# =============================================================================
#             ax2.scatter(POS[1, idx_start[jj]: idx_end[jj]], POS[2, idx_start[jj]: idx_end[jj]], s=1, c=const.temp_color[jj])
#             ax2.set_title('Gyroposition :: {}'.format(const.species_lbl[jj]))
#             ax2.set_ylabel('$y (m)$')
#             ax2.set_xlabel('$z (m)$')
#             ax2.axis('equal')
#             
#             ax3.scatter(POS[1, idx_start[jj]: idx_end[jj]], POS[2, idx_start[jj]: idx_end[jj]], s=1, c=const.temp_color[jj])
#             ax3.set_title('Gyrovelocity :: {}'.format(const.species_lbl[jj]))
#             ax3.set_ylabel('$v_y (m/s)$')
#             ax3.set_xlabel('$v_z (m/s)$')
#             ax3.axis('equal')
# =============================================================================
            
        if True:
            # v_mag vs. x
            fig1, ax2 = plt.subplots()
            
            ax2.scatter(POS[0, idx_start[jj]: idx_end[jj]], V_MAG[idx_start[jj]: idx_end[jj]], s=1, c=const.temp_color[jj])

            ax2.set_title('Velocity vs. Position')
            ax2.set_xlabel('Position (m)')
            ax2.set_ylabel('Velocity |v| (m/s)')
        
        if True:
            # v components vs. x (3 plots)
            fig1, ax3 = plt.subplots(3)
            
            ax3[0].scatter(POS[0, idx_start[jj]: idx_end[jj]], VEL[0, idx_start[jj]: idx_end[jj]], s=1, c=const.temp_color[jj])
            ax3[1].scatter(POS[0, idx_start[jj]: idx_end[jj]], VEL[1, idx_start[jj]: idx_end[jj]], s=1, c=const.temp_color[jj])
            ax3[2].scatter(POS[0, idx_start[jj]: idx_end[jj]], VEL[2, idx_start[jj]: idx_end[jj]], s=1, c=const.temp_color[jj])

            ax3[0].set_ylabel('$v_x$ (m/s)')
            ax3[1].set_ylabel('$v_y$ (m/s)')
            ax3[2].set_ylabel('$v_z$ (m/s)')
            
            ax3[0].set_title('Velocity Components vs. Position')
            ax3[2].set_xlabel('Position (m)')

    
# =============================================================================
#     import diagnostics       as diag
#     diag.check_velocity_distribution(VEL)
# =============================================================================
    
# =============================================================================
#     if True:
#         for jj in range(const.Nj):
#             plt.figure(jj)
#             v_perp = np.sign(VEL[2, const.idx_start[jj]:idx_end[jj]]) * \
#                      np.sqrt(VEL[1, const.idx_start[jj]:idx_end[jj]] ** 2 +
#                              VEL[2, const.idx_start[jj]:idx_end[jj]] ** 2) / const.va
#             
#             v_para = VEL[0, const.idx_start[jj]:idx_end[jj]] / const.va
#             
#             plt.scatter(v_perp, v_para, c=const.temp_color[jj], s=1)
#             plt.title(r'Total Velocity Distribution Functions (%s) :: $\alpha_L$ = %.1f$^\circ$' % (const.species_lbl[jj], const.loss_cone))
#             plt.xlabel('$v_\perp / v_A$')
#             plt.ylabel('$v_\parallel / v_A$')
#             
#             gradient = np.tan(np.pi/2 - const.loss_cone * np.pi / 180.)
#             lmin, lmax = plt.gca().get_xlim()
#             lcx  = np.linspace(lmin, lmax, 100, endpoint=True)
#             lcy1 =  gradient * lcx
#             lcy2 = -gradient * lcx
#             
#             plt.plot(lcx, lcy1, c='k', alpha=0.5, ls=':')
#             plt.plot(lcx, lcy2, c='k', alpha=0.5, ls=':')
#             
#             plt.axvline(0, c='k')
#             plt.axhline(0, c='k')
# =============================================================================
# =============================================================================
#     if False:
#         for jj in range(const.Nj):
#             plt.scatter(POS[0, const.idx_start[jj]:idx_end[jj]], 
#                         rL[const.idx_start[jj]:idx_end[jj]],
#                         c=const.temp_color[jj],
#                         label=const.species_lbl[jj], s=1)
#         plt.legend()
#         plt.title('Larmor radius with position')
#         
#     if False:
#         plt.figure()
#         for jj in range(const.Nj):
#             plt.scatter(POS[0, const.idx_start[jj]:idx_end[jj]], 
#                         VPERP_OLD[const.idx_start[jj]:idx_end[jj]],
#                         c=const.temp_color[jj],
#                         label=const.species_lbl[jj], s=4)
#         plt.legend()
#         plt.title('$v_\perp$ before transformation, with position')
#         
#         plt.figure()
#         for jj in range(const.Nj):
#             plt.scatter(POS[0, const.idx_start[jj]:idx_end[jj]], 
#                         VPERP_NEW[const.idx_start[jj]:idx_end[jj]],
#                         c=const.temp_color[jj],
#                         label=const.species_lbl[jj], s=4)
#         plt.legend()
#         plt.title('$v_\perp$ after transformation, with position')
#         
#     if False:
#         jj = 1
#         v_perp_old = np.sqrt(IDX[1] ** 2 + IDX[2] ** 2)
#         plt.scatter(POS[0, const.idx_start[jj]:idx_end[jj]], 
#                     v_perp[const.idx_start[jj]:idx_end[jj]],
#                     c='r',
#                     label=const.species_lbl[jj])
#         
#         plt.scatter(POS[0, const.idx_start[jj]:idx_end[jj]], 
#                     v_perp_old[const.idx_start[jj]:idx_end[jj]],
#                     c='k',
#                     label=const.species_lbl[jj])
#         
#         plt.legend()
#         plt.title('v_perp with position: Initial (Black) and adjusted for position (Red)')
#     
#     
#     if False:
#         jj = 1
#         plt.scatter(POS[0, const.idx_start[jj]:idx_end[jj]], 
#                     np.log10(VEL[const.idx_start[jj]:idx_end[jj]]),
#                     c='k',
#                     label='$v_\perp/v_\parallel$ old',
#                     marker='o', s=1)
#         
#         plt.scatter(POS[0, const.idx_start[jj]:idx_end[jj]], 
#                     np.log10(IDX[const.idx_start[jj]:idx_end[jj]]),
#                     c='r',
#                     label='$v_\perp/v_\parallel$ new',
#                     marker='x', s=1)
#         
#         plt.ylim(-2, 6)
#         plt.legend()
#         plt.title('perp/parallel velocity ratios before/after transformation')
# 
#     
#     #diag.check_cell_velocity_distribution(POS, VEL, j=1, node_number=0)
#     #diag.check_position_distribution(POS)
# 
# 
# =============================================================================

