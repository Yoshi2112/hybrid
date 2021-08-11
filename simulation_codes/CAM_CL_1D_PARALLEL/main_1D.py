## PYTHON MODULES ##
from timeit import default_timer as timer
from scipy.interpolate import splrep, splev
import numpy as np
import numba as nb
import sys, os, pdb

## PHYSICAL CONSTANTS ##
ECHARGE = 1.602177e-19                       # Elementary charge (C)
SPLIGHT = 2.998925e+08                       # Speed of light (m/s)
PMASS   = 1.672622e-27                       # Mass of proton (kg)
EMASS   = 9.109384e-31                       # Mass of electron (kg)
kB      = 1.380649e-23                       # Boltzmann's Constant (J/K)
e0      = 8.854188e-12                       # Epsilon naught - permittivity of free space
mu0     = (4e-7) * np.pi                     # Magnetic Permeability of Free Space (SI units)
RE      = 6.371e6                            # Earth radius in metres
B_surf  = 3.12e-5                            # Magnetic field strength at Earth surface (equatorial)

Fu_override       = True
do_parallel       = True
print_timings     = False      # Diagnostic outputs timing each major segment (for efficiency examination)
print_runtime     = True       # Flag to print runtime every 50 iterations 
adaptive_timestep = True       # Disable adaptive timestep to keep it the same as initial
adaptive_subcycling = True     # Flag (True/False) to adaptively change number of subcycles during run to account for high-frequency dispersion
default_subcycles   = 12       # Number of field subcycling steps for Cyclic Leapfrog

if not do_parallel:
    do_parallel = True
    nb.set_num_threads(1)          
#nb.set_num_threads(2)         # Uncomment to manually set number of threads, otherwise will use all available


#%% --- FUNCTIONS ---
#%% INITIALIZATION
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
def LCD_by_rejection(pos, vel, sf_par, sf_per, st, en, jj):
    '''
    Takes in a Maxwellian or pseudo-maxwellian distribution. Removes any particles
    inside the loss cone and reinitializes them using the given scale factors 
    sf_par, sf_per (the thermal velocities).
    
    Is there a better way to do this with a Monte Carlo perhaps?
    
    TODO: Check that this still works.
    '''
    B0x    = eval_B0x(pos[st: en])
    N_loss = 1

    while N_loss > 0:
        v_perp      = np.sqrt(vel[1, st: en] ** 2 + vel[2, st: en] ** 2)
        
        N_loss, loss_idx = calc_losses(vel[0, st: en], v_perp, B0x, st=st)

        # Catch for a particle on the boundary : Set 90 degree pitch angle (gyrophase shouldn't overly matter)
        if N_loss == 1:
            if abs(pos[loss_idx[0]]) == xmax:
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
def quiet_start_bimaxwellian():
    '''Initializes position, velocity, and index arrays. Position analytically
    uniform, velocity randomly sampled normal distributions using perp/para 
    scale factors. Quiet start initialized as per Birdsall and Langdon (1985).

    OUTPUT:
        pos -- 1xN array of particle positions in meters
        vel -- 3xN array of particle velocities in m/s
        idx -- N   array of particle indexes, indicating population types 
    '''
    np.random.seed(seed)
    pos = np.zeros(N, dtype=np.float64)
    vel = np.zeros((3, N), dtype=np.float64)
    idx = np.ones(N,       dtype=np.int8) * Nj

    for jj in range(Nj):
        idx[idx_start[jj]: idx_end[jj]] = jj          # Set particle idx        
        half_n = nsp_ppc[jj] // 2                     # Half particles per cell - doubled later
       
        # Change how many cells are loaded around equator
        # Currently deprecated
        if temp_type[jj] == 0:                        
            NC_load = NX
        else:
            # Need to change this to be something like the FWHM or something
            NC_load = NX
        
        # Load particles in each applicable cell
        acc = 0; offset  = 0
        for ii in range(NC_load):
            # Add particle if last cell (for symmetry, but only with open field boundaries)
            if ii == NC_load - 1 and field_periodic == 0:
                half_n += 1
                offset  = 1
                
            # Particle index ranges
            st = idx_start[jj] + acc
            en = idx_start[jj] + acc + half_n
            
            # Set position for half: Analytically uniform
            for kk in range(half_n):
                pos[st + kk] = dx*(float(kk) / (half_n - offset) + ii)
            
            # Turn [0, NC] distro into +/- NC/2 distro
            pos[st: en] -= 0.5*NC_load*dx              
            
            # Set velocity for half: Randomly Maxwellian
            vel[0, st: en] = np.random.normal(0, vth_par[ jj], half_n)  
            vel[1, st: en] = np.random.normal(0, vth_perp[jj], half_n)
            vel[2, st: en] = np.random.normal(0, vth_perp[jj], half_n)

            # Set Loss Cone Distribution: Reinitialize particles in loss cone (move to a function)
            if homogenous == 0 and temp_type[jj] == 1:
                LCD_by_rejection(pos, vel, vth_par[jj], vth_perp[jj], st, en, jj)
                
            # Quiet start : Initialize second half
            pos[en: en + half_n]    = pos[st: en]                   # Other half, same position
            vel[0, en: en + half_n] = vel[0, st: en] *  1.0         # Set parallel
            vel[1, en: en + half_n] = vel[1, st: en] * -1.0         # Invert perp velocities (v2 = -v1)
            vel[2, en: en + half_n] = vel[2, st: en] * -1.0
            
            vel[0, st: en + half_n] += drift_v[jj] * va             # Add drift offset
            
            acc                     += half_n * 2
    return pos, vel, idx


@nb.njit()
def uniform_bimaxwellian():
    '''Initializes position, velocity, and index arrays. Positions and velocities
    both initialized using appropriate numpy random distributions, cell by cell.

    OUTPUT:
        pos -- 1xN array of particle positions in meters
        vel -- 3xN array of particle velocities in m/s
        idx -- N   array of particle indexes, indicating population types 
    '''
    # Set and initialize seed
    np.random.seed(seed)
    pos   = np.zeros(N)
    vel   = np.zeros((3, N))
    idx   = np.ones(N, dtype=np.int8) * Nj

    # Initialize unformly in space, gaussian in 3-velocity
    for jj in range(Nj):
        acc = 0
        idx[idx_start[jj]: idx_end[jj]] = jj
        
        for ii in range(NX):
            n_particles = nsp_ppc[jj]

            for kk in range(n_particles):
                pos[idx_start[jj] + acc + kk] = dx*(float(kk) / n_particles + ii)
              
            vel[0, (idx_start[jj] + acc): ( idx_start[jj] + acc + n_particles)] = np.random.normal(0, vth_par[jj],  n_particles) + drift_v[jj] * va
            vel[1, (idx_start[jj] + acc): ( idx_start[jj] + acc + n_particles)] = np.random.normal(0, vth_perp[jj], n_particles)
            vel[2, (idx_start[jj] + acc): ( idx_start[jj] + acc + n_particles)] = np.random.normal(0, vth_perp[jj], n_particles)
                        
            acc += n_particles
    
    pos -= 0.5*NX*dx
    return pos, vel, idx


def initialize_particles(B, E, _mp_flux, Ji_in, rho_in):
    if quiet_start == 1:
        pos, vel, idx = quiet_start_bimaxwellian()
    else:
        pos, vel, idx = uniform_bimaxwellian()
        
    Ie         = np.zeros(N,      dtype=np.uint16)
    Ib         = np.zeros(N,      dtype=np.uint16)
    W_elec     = np.zeros((3, N), dtype=np.float64)
    W_mag      = np.zeros((3, N), dtype=np.float64)
        
# =============================================================================
#     if homogenous == False:
#         Ve_in = np.zeros((NC, 3), dtype=np.float64)
#         run_until_equilibrium(pos, vel, idx, Ie, W_elec, Ib, W_mag, B, E,
#                               Ji_in, Ve_in, rho_in,
#                           _mp_flux, hot_only=True, psave=True)
# =============================================================================
    
    assign_weighting_TSC(pos, Ie, W_elec)
    assign_weighting_TSC(pos, Ib, W_mag)
    return pos, vel, Ie, W_elec, Ib, W_mag, idx


@nb.njit()
def initialize_fields():
    '''
    Initializes field ndarrays and sets initial values for fields based on
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
    B       = np.zeros((NC + 1, 3), dtype=np.float64)
    B2      = np.zeros((NC + 1, 3), dtype=np.float64)
    B_cent  = np.zeros((NC    , 3), dtype=np.float64)
    E       = np.zeros((NC    , 3), dtype=np.float64)
    Ve      = np.zeros((NC, 3), dtype=np.float64)
    Te      = np.ones(  NC,     dtype=np.float64) * Te0_scalar
    return B, B2, B_cent, E, Ve, Te


@nb.njit()
def initialize_source_arrays():
    '''
    Initializes source term ndarrays. Each term is collected on the E-field grid.

    INPUT:
        <NONE>

    OUTPUT:
        q_dens  -- Total ion charge  density
        q_dens2 -- Total ion charge  density (used for averaging)
        Ji      -- Total ion current density
        ni      -- Ion number density per species
        nu      -- Ion velocity "density" per species
    '''
    rho_half = np.zeros(NC, dtype=np.float64)
    rho_int  = np.zeros(NC, dtype=np.float64)
    
    Ji       = np.zeros((NC, 3), dtype=np.float64)
    Ji_plus  = np.zeros((NC, 3), dtype=np.float64)
    Ji_minus = np.zeros((NC, 3), dtype=np.float64)
    J_ext    = np.zeros((NC, 3), dtype=np.float64)
    L        = np.zeros( NC,     dtype=np.float64)
    G        = np.zeros((NC, 3), dtype=np.float64)
    
    mp_flux  = np.zeros((2, Nj), dtype=np.float64)
    return rho_half, rho_int, Ji, Ji_plus, Ji_minus, J_ext, L, G, mp_flux


def run_until_equilibrium(pos, vel, idx, Ie, W_elec, Ib, W_mag, B, E,
                          mp_flux, frev=1000, hot_only=True, psave=True, save_inc=50):
    '''
    Still need to test this. Put it just after the initialization of the particles.
    Actually might want to use the real mp_flux since that'll continue once the
    waves turn on?
    
    Just use all the real arrays, since the particle arrays will be continuous
    once the fields turn on (right weightings, positions, etc.) and the wave
    fields should be empty at the start and will continue to be empty since they
    are not updated in this loop.
    
    Should be ready to test after the open particle boundaries are verified.
    
    Can be set so that only the ring current warm/hot species are set to the
    initial equilibrium, since the time taken for the cold species to do this
    is quite substantial.
    
    Is vel_ts needed? Cells don't really exist in this bottle. But you probably 
    don't want to jump too quickly through the gradient, surely there's a timestep
    limitation on that.
    
    TODO: 
        -- Put check in store_run_params() to delete if it already exists so
    this particle data doesn't alter it.
        -- Put some kind of limit on dB/dx
        -- Don't need vel_ts since cell size doesn't matter (especially for very small dx)
    '''
    psave = save_particles
    
    print('Letting particle distribution relax into static field configuration')
    #max_vx   = np.max(np.abs(vel[0, :]))
    ion_ts   = 0.1 * 2 * np.pi / gyfreq_xmax
    #vel_ts   = 0.5*dx / max_vx
    pdt      = ion_ts
    ptime    = frev / gyfreq_eq
    psteps   = int(ptime / pdt) + 1
    psim_time = 0.0

    print('Particle-only timesteps: ', psteps)
    print('Particle-push in seconds:', pdt)
    
    if psave == True:
        print('Generating data for particle equilibrium pushes')
        # Check dir
        if save_fields + save_particles == 0:
            manage_directories()
            store_run_parameters(pdt, save_inc, 0, psteps, ptime)
            
        pdata_path  = ('%s/%s/run_%d' % (drive, save_path, run_num))
        pdata_path += '/data/equil_particles/'
        if os.path.exists(pdata_path) == False:                                   # Create data directory
            os.makedirs(pdata_path)
        pnum = 0
        
    # Desync (retard) velocity here
    velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, -0.5*pdt,
                    hot_only=hot_only)
        
    for pp in range(psteps):
        # Save first and last only
        if psave == True and (pp == 0 or pp == psteps - 1):
            p_fullpath = pdata_path + 'data%05d' % pnum
            np.savez(p_fullpath, pos=pos, vel=vel, idx=idx, sim_time=psim_time)
            pnum += 1
            print('p-Particle data saved')
            
        if pp%100 == 0:
            print('Step', pp)
        
        velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, pdt,
                        hot_only=hot_only)
        position_update(pos, vel, idx, Ie, W_elec, Ib, W_mag, mp_flux, pdt,
                        hot_only=hot_only)

        # Check number of spare particles every 25 steps
        if pp > 0 and pp%100 == 0 and particle_open == 1:
            num_spare = (idx >= Nj).sum()
            if num_spare < nsp_ppc.sum():
                print('WARNING :: Less than one cell of spare particles remaining.')
                if num_spare < inject_rate.sum() * pdt * 5.0:
                    # Change this to dynamically expand particle arrays later on (adding more particles)
                    # Can do it by cell lots (i.e. add a cell's worth each time)
                    raise Exception('WARNING :: No spare particles remaining. Exiting simulation.')
                
        psim_time += pdt
    
    # Resync (advance) velocity here
    velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, 0.5*pdt,
                    hot_only=hot_only)
    
    # Dump indicator file
    if save_fields == 1 or save_particles == 1:
        efin_path = '%s/%s/run_%d/equil_finished.txt' % (drive, save_path, run_num)
        open_file = open(efin_path, 'w')
        open_file.close()
    return


@nb.njit()
def set_damping_arrays(B_damping_array, resistive_array, DT, subcycles):
    '''Create masking array for magnetic field damping used to apply open
    boundaries. Based on applcation by Shoji et al. (2011) and
    Umeda et al. (2001)
    
    Question: Which timestep to use? Subcycled one? 
    
    Still need to implement retarding array as per Umeda and E-damping as
    per Hu & Denton (2010). Code in the PREDCORR version but doesn't work
    as yet.
    '''
    # Location and thickness of damping region (in units of dx)
    damping_thickness  = damping_fraction*NC
    damping_boundary   = 0.5*NC - damping_thickness  
    
    dh       = DT / subcycles
    r_damp   = np.sqrt(29.7 * 0.5 * va * (0.5 * dh / dx) / damping_thickness)
    r_damp  *= damping_multiplier
    
    # Do B-damping array
    B_dist_from_mp  = np.abs(np.arange(NC + 1) - 0.5*NC)                # Distance of each B-node from midpoint
    for ii in range(NC + 1):
        if B_dist_from_mp[ii] > damping_boundary:
            B_damping_array[ii] = 1. - r_damp * ((B_dist_from_mp[ii] - damping_boundary) / damping_thickness) ** 2 
        else:
            B_damping_array[ii] = 1.0
            
    # Set resistivity
    if True:
        # Lower Hybrid Resonance method (source?)
        LH_res_is = 1. / (gyfreq_eq * egyfreq_eq) + 1. / wpi ** 2  # Lower Hybrid Resonance frequency, inverse squared
        LH_res    = 1. / np.sqrt(LH_res_is)                        # Lower Hybrid Resonance frequency: DID I CHECK THIS???
        max_eta   = (resis_multiplier * LH_res)  / (e0 * wpe ** 2) # Electron resistivity (using intial conditions for wpi/wpe)
    else:
        # Spitzer resistance as per B&T
        max_eta = (resis_multiplier * vei)  / (e0 * wpe ** 2)
    
    E_dist_from_mp  = np.abs(np.arange(NC) + 0.5 - 0.5*NC)
    for ii in range(NC):
        # Damping region
        if E_dist_from_mp[ii] > damping_boundary:
            resistive_array[ii] = 0.5*max_eta*(1. + np.cos(np.pi*(E_dist_from_mp[ii] - 0.5*NC) / damping_thickness))
        
        # Inside solution space
        else:
            resistive_array[ii] = 0.0
            
    # Make sure no damping arrays go negative (or that's growth!)
    for arr in [B_damping_array, resistive_array]:
        for ii in range(arr.shape[0]):
            if arr[ii] < 0.0:
                arr[ii] = 0.0
    return


def set_timestep(vel):
    '''
    Timestep limitations:
        -- Resolve ion gyromotion (controlled by orbit_res)
        -- Resolve ion velocity (<0.5dx per timestep, varies with particle temperature)
        -- Resolve B-field solution on grid (controlled by subcycling and freq_res)
        -- E-field acceleration? (not implemented)
    '''
    max_vx   = np.max(np.abs(vel[0, :]))
    ion_ts   = orbit_res / gyfreq_xmax            # Timestep to resolve gyromotion
    vel_ts   = 0.5*dx / max_vx                    # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step

    DT       = min(ion_ts, vel_ts)
    max_time = max_wcinv / gyfreq_eq              # Total runtime in seconds
    max_inc  = int(max_time / DT) + 1             # Total number of time steps

    if part_res == 0:
        part_save_iter = 1
    else:
        part_save_iter = int(part_res / (DT*gyfreq_xmax))
        if part_save_iter == 0: part_save_iter = 1

    if field_res == 0:
        field_save_iter = 1
    else:
        field_save_iter = int(field_res / (DT*gyfreq_xmax))
        if field_save_iter == 0: field_save_iter = 1

    if adaptive_subcycling == True:
        k_max      = np.pi / dx
        dispfreq   = (k_max ** 2) * B_xmax / (mu0 * ne * ECHARGE)
        dt_sc      = freq_res / dispfreq
        subcycles  = int(DT / dt_sc + 1)
        print('Number of subcycles required: {}'.format(subcycles))
    else:
        subcycles = default_subcycles
        print('Number of subcycles set at default: {}'.format(subcycles))

    if save_fields == 1 or save_particles == 1:
        store_run_parameters(DT, part_save_iter, field_save_iter, max_inc, max_time, subcycles)
    
    B_damping_array = np.ones(NC + 1, dtype=float)
    resistive_array = np.ones(NC    , dtype=float)
    set_damping_arrays(B_damping_array, resistive_array, DT, subcycles)
    
    # DIAGNOSTIC PLOT: CHECK OUT DAMPING FACTORS
    if False:
        plt.ioff()
        fig, axes = plt.subplots(2)
        axes[0].plot(B_nodes_loc/dx, B_damping_array)
        axes[0].set_ylabel('B damp')
        axes[1].plot(E_nodes_loc/dx, resistive_array)
        axes[1].set_ylabel('$\eta$')
        for ax in axes:
            ax.set_xlim(B_nodes_loc[0]/dx, B_nodes_loc[-1]/dx)
            ax.axvline(    0, color='k', ls=':', alpha=0.5)
            ax.axvline( NX/2, color='k', ls=':')
            ax.axvline(-NX/2, color='k', ls=':')
            
        plt.show()
        sys.exit()
    
    print('Timestep: %.4fs, %d iterations total\n' % (DT, max_inc))
    
    return DT, max_inc, part_save_iter, field_save_iter, subcycles,\
             B_damping_array, resistive_array


#%% PARTICLES 
@nb.njit(parallel=do_parallel)
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
    
    particle_transform = xmax + (ND - grid_offset)*dx  + epsil      # Offset to account for E/B grid and damping nodes
    
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


@nb.njit(parallel=do_parallel)
def velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, 
                    dt, hot_only=False):    
    '''
    Note: Keeping the code in case it's useful later, but commenting it out
    for speed. Also removed requirement to call Ji, Ve, q_dens (rho) because
    it makes coding the equilibrium bit easier. Also resisitive_array.
    '''
    for ii in nb.prange(pos.shape[0]):
        if temp_type[idx[ii]] == 1 or hot_only == False:
            # Calculate wave fields at particle position
            Ep = np.zeros(3, dtype=np.float64)  
            Bp = np.zeros(3, dtype=np.float64)
            #Jp  = np.zeros(3, dtype=np.float64)
            #eta = 0.0
            for jj in nb.prange(3):
                #eta += resistive_array[Ie[ii] + jj] * W_elec[jj, ii]
                for kk in nb.prange(3):
                    Ep[kk] += E[Ie[ii] + jj, kk] * W_elec[jj, ii]   
                    Bp[kk] += B[Ib[ii] + jj, kk] * W_mag[ jj, ii]   
                    #if resis_multiplier != 0.0:
                    #    Jp[kk] += (Ji[Ie[ii] + jj, kk] - Ve[Ie[ii] + jj, kk]*q_dens[Ie[ii] + jj])
                
            # Add resistivity into 'effective' E-field
            #if eta != 0.0:
            #    Ep[0] -= eta * Jp[0]
            #    Ep[1] -= eta * Jp[1]
            #    Ep[2] -= eta * Jp[2]

            # Start Boris Method
            qmi = 0.5 * dt * qm_ratios[idx[ii]]                             # q/m variable including dt
            
            # vel -> v_minus
            vel[0, ii] += qmi * Ep[0]
            vel[1, ii] += qmi * Ep[1]
            vel[2, ii] += qmi * Ep[2]
            
            # Calculate background field at particle position (using v_minus)
            Bp[0]    += B_eq * (1.0 + a * pos[ii] * pos[ii])
            constant  = a * B_eq
            l_cyc     = qm_ratios[idx[ii]] * Bp[0]
            Bp[1]    += constant * pos[ii] * vel[2, ii] / l_cyc
            Bp[2]    -= constant * pos[ii] * vel[1, ii] / l_cyc
            
            T         = qmi * Bp 
            S         = 2.*T / (1. + T[0]*T[0] + T[1]*T[1] + T[2]*T[2])
                
            # Calculate v_prime
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
    return


@nb.njit(parallel=do_parallel)
def position_update(pos, vel, idx, Ie, W_elec, Ib, W_mag, mp_flux, dt,
                    hot_only=False):
    '''
    Updates the position of the particles using x = x0 + vt. 
    Also updates particle nearest node and weighting, for E-nodes only since
    these are used for source term collection (mag only used for vel push) and
    is called much more frequently.
    Also injects particles since this is one of the four particle boundary 
    conditions.
    '''
    for ii in nb.prange(pos.shape[0]):
        if temp_type[idx[ii]] == 1 or hot_only == False:
            pos[ii] += vel[0, ii] * dt
            
            # Check if particle has left simulation and apply boundary conditions
            if (pos[ii] < xmin or pos[ii] > xmax):

                if particle_periodic == 1:  
                    idx[ii] += Nj                            
                elif particle_open == 1:                
                    pos[ii]     = 0.0
                    vel[0, ii]  = 0.0
                    vel[1, ii]  = 0.0
                    vel[2, ii]  = 0.0
                    idx[ii]     = Nj
                elif particle_reinit == 1:
                    vel[0, ii]  = 0.0
                    vel[1, ii]  = 0.0
                    vel[2, ii]  = 0.0
                    idx[ii]    += Nj                            
                else:
                    idx[ii] += Nj 
    
    apply_particle_BCs(pos, vel, idx, mp_flux, dt)
    assign_weighting_TSC(pos, Ie, W_elec)
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)
    return


@nb.njit()
def apply_particle_BCs(pos, vel, idx, _mp_flux, DT):
    # TODO: Still need to test open BCs, especially in the run_until_equil() bit
    # Also, reinit BCs are weirdly scrambled, but good flux?
    if particle_open == 1:
        inject_particles(pos, vel, idx, _mp_flux, DT)
    elif particle_reinit == 1:
        if particle_reinit == 1:
            reinit_count_flux(pos, idx, _mp_flux)
        inject_particles(pos, vel, idx, _mp_flux, DT)
    elif particle_periodic == 1:
        periodic_BC(pos, idx)
    else:
        reflective_BC(pos, vel, idx)
    return


@nb.njit(parallel=do_parallel)
def reinit_count_flux(pos, idx, _mp_flux):
    '''
    Simple function to work out where to reinitialize particles (species/side)
    Coded for serial computation since numba can't do parallel reductions with
    arrays as a target.
    
    Shouldn't be any slower than requiring the source functions to be serial,
    especially since its only an evaluation for every particle, and then a few
    more operations for a miniscule portion of those particles.
    '''
    for ii in nb.prange(idx.shape[0]):
        if idx[ii] >= Nj:
            sp = idx[ii]-Nj
            if pos[ii] > xmax:
                _mp_flux[1, sp] += 1.0
            elif pos[ii] < xmin:
                _mp_flux[0, sp] += 1.0 
    return


@nb.njit(parallel=do_parallel)
def periodic_BC(pos, idx):
    '''
    Simple function to work out where to reinitialize particles (species/side)
    Coded for serial computation since numba can't do parallel reductions with
    arrays as a target.
    
    Shouldn't be any slower than requiring the source functions to be serial,
    especially since its only an evaluation for every particle, and then a few
    more operations for a miniscule portion of those particles.
    '''
    for ii in nb.prange(idx.shape[0]):
        if idx[ii] >= Nj:
            if pos[ii] > xmax:
                pos[ii] += xmin
                pos[ii] -= xmax
            elif pos[ii] < xmin:
                pos[ii] += xmax
                pos[ii] -= xmin 
            idx[ii] -= Nj
    return


@nb.njit(parallel=do_parallel)
def reflective_BC(pos, vel, idx):
    for ii in nb.prange(idx.shape[0]):
        if idx[ii] >= Nj:
            # Reflect
            if pos[ii] > xmax:
                pos[ii] = 2*xmax - pos[ii]
            elif pos[ii] < xmin:
                pos[ii] = 2*xmin - pos[ii]
                
            vel[0, ii] *= -1.0
            idx[ii] -= Nj
    return


@nb.njit()
def inject_particles_all_species(pos, vel, idx, _mp_flux, dt):
    '''
    Control function for injection that does all species.
    This is so the actual injection function can be per-species (in case
    I want to inject just one species, such as for the equilibrium stuff)
    '''
    for jj in range(Nj):
        inject_particles_1sp(pos, vel, idx, _mp_flux, dt, jj)
    return


@nb.njit()
def inject_particles_1sp(pos, vel, idx, _mp_flux, dt, jj):        
    '''
    How to create new particles in parallel? Just test serial for now, but this
    might become my most expensive function for large N.
    
    Also need to work out how to add flux in serial (might just have to put it
    in calling function: advance_particles_and_moments())
    
    NOTE: How does this work for -0.5*DT ?? Might have to double check
    '''
    # Add flux at each boundary 
    if particle_open == 1:
        for kk in range(2):
            _mp_flux[kk, jj] += inject_rate[jj]*dt
        
    # acc used only as placeholder to mark place in array. How to do efficiently? 
    acc = 0; n_created = 0
    for ii in nb.prange(2):
        N_inject = int(_mp_flux[ii, jj] // 2)
        
        for xx in nb.prange(N_inject):
            
            # Find two empty particles (Yes clumsy coding but it works)
            for kk in nb.prange(acc, pos.shape[0]):
                if idx[kk] >= Nj:
                    kk1 = kk
                    acc = kk + 1
                    break
            for kk in nb.prange(acc, pos.shape[0]):
                if idx[kk] >= Nj:
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
            dpos = np.random.uniform(0, 1) * vel[0, kk1] * dt
            
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
            _mp_flux[ii, jj] -= 2.0
            n_created        += 2
    return


@nb.njit()
def inject_particles(pos, vel, idx, _mp_flux, DT):        
    '''
    Modified version designed to be used  for reinit conditions. Flag for 
    quiet start that halves mp_flux (since two particles are required for
                                     injection)
    
    How to create new particles in parallel? Just test serial for now, but this
    might become my most expensive function for large N.
    
    Also need to work out how to add flux in serial (might just have to put it
    in calling function: advance_particles_and_moments())
    
    NOTE: How does this work for -0.5*DT ?? Might have to double check
    '''
    # Add flux at each boundary if 'open' flux boundaries
    if particle_open == 1:
        for kk in range(2):
            _mp_flux[kk, :] += inject_rate*DT
    
    # acc used only as placeholder to mark place in array. How to do efficiently? 
    acc = 0; n_created = 0
    for ii in nb.prange(2):
        for jj in nb.prange(Nj):
        
            if quiet_start == 1:
                N_inject = int(_mp_flux[ii, jj] // 2)
            else:
                N_inject = int(_mp_flux[ii, jj])

            for xx in nb.prange(N_inject):
                
                # Find two empty particles (Yes clumsy coding but it works)
                for kk in nb.prange(acc, pos.shape[0]):
                    if idx[kk] >= Nj:
                        kk1 = kk
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
                
                # Inject at either left/right boundary
                if ii == 0:
                    pos[kk1]    = xmin + dpos
                    vel[0, kk1] = np.abs(vel[0, kk1])
                else:
                    pos[kk1]    = xmax - dpos
                    vel[0, kk1] = -np.abs(vel[0, kk1])
                _mp_flux[ii, jj] -= 1.0; n_created += 1
                
                if quiet_start == 1:
                    # Copy values to second particle (Same position, xvel. Opposite v_perp) 
                    for kk in nb.prange(acc, pos.shape[0]):
                        if idx[kk] >= Nj:
                            kk2 = kk
                            acc = kk + 1
                            break
                        
                    idx[kk2]    = idx[kk1]
                    pos[kk2]    = pos[kk1]
                    vel[0, kk2] = vel[0, kk1]
                    vel[1, kk2] = vel[1, kk1] * -1.0
                    vel[2, kk2] = vel[2, kk1] * -1.0
                    
                    # Subtract new macroparticles from accrued flux
                    _mp_flux[ii, jj] -= 1.0; n_created += 1
    #print('Particles injected:', n_created)
    return


@nb.njit()
def vfx(vx, vth):
    f_vx  = np.exp(- 0.5 * (vx / vth) ** 2)
    f_vx /= vth * np.sqrt(2 * np.pi)
    return vx * f_vx


@nb.njit()
def generate_vx(vth):
    '''
    Maybe could try batch approach? If we need X number of solutions and we do
    batches of 100 or something and just pick the first x number. Then again,
    it depends on how long this has to loop until a valid value is found... if
    it only takes a few iterations, it's not worth it. But if it takes a few
    thousand to get sufficient numbers, then maybe it'll be worth it?
    '''
    while True:
        y_uni = np.random.uniform(0, 4*vth)
        Py    = vfx(y_uni, vth)
        x_uni = np.random.uniform(0, 0.25)
        if Py >= x_uni:
            return y_uni



#%% SOURCES
@nb.njit()
def push_current(J_in, J_out, E, B_center, L, G, dt):
    '''Uses an MHD-like equation to advance the current with a moment method as 
    per Matthews (1994) CAM-CL method. Fills in ghost cells at edges (excluding very last one)
    
    INPUT:
        J  -- Ionic current (J plus)
        E  -- Electric field
        B  -- Magnetic field (offset from E by 0.5dx)
        L  -- "Lambda" MHD variable
        G  -- "Gamma"  MHD variable
        dt -- Timestep
        
    OUTPUT:
        J_plus in main() becomes J_half (same memory space)
    '''
    J_out    *= 0
    
    G_cross_B = np.zeros(E.shape, dtype=np.float64)
    for ii in np.arange(NC):
        G_cross_B[ii, 0] = G[ii, 1] * B_center[ii, 2] - G[ii, 2] * B_center[ii, 1]
        G_cross_B[ii, 1] = G[ii, 2] * B_center[ii, 0] - G[ii, 0] * B_center[ii, 2]
        G_cross_B[ii, 2] = G[ii, 0] * B_center[ii, 1] - G[ii, 1] * B_center[ii, 0]
    
    for ii in range(3):
        J_out[:, ii] = J_in[:, ii] + 0.5*dt * (L * E[:, ii] + G_cross_B[:, ii]) 
    
    # Copy periodic values
    if field_periodic == 1:
        for ii in range(3):
            # Copy edge cells
            J_out[ro1, ii] = J_out[li1, ii]
            J_out[ro2, ii] = J_out[li2, ii]
            J_out[lo1, ii] = J_out[ri1, ii]
            J_out[lo2, ii] = J_out[ri2, ii]
            
            # Fill remaining ghost cells
            J_out[:lo2, ii] = J_out[lo2, ii]
            J_out[ro2:, ii] = J_out[ro2, ii]
    return


@nb.njit(parallel=do_parallel)
def deposit_both_moments(vel, Ie, W_elec, idx, ni, nu):
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
        
    TODO:
        -- Initialize thread arrays at runtime (preallocate memory)
        -- Calculate N_per_thread and n_start_idxs at runtime
        -- Check if this works for non-multiples of the thread count
           i.e. N_per_thread may need to be an array with particle counts since
           we can't assume that it is constant for each thread.
    '''
    ni_threads = np.zeros((n_threads, NC, Nj), dtype=np.float64)
    nu_threads = np.zeros((n_threads, NC, Nj, 3, ), dtype=np.float64)
    for tt in nb.prange(n_threads):        
        for ii in range(n_start_idxs[tt], n_start_idxs[tt]+N_per_thread[tt]):
            if idx[ii] < Nj:
                for kk in nb.prange(3):
                    nu_threads[tt, Ie[ii],     idx[ii], kk] += W_elec[0, ii] * vel[kk, ii]
                    nu_threads[tt, Ie[ii] + 1, idx[ii], kk] += W_elec[1, ii] * vel[kk, ii]
                    nu_threads[tt, Ie[ii] + 2, idx[ii], kk] += W_elec[2, ii] * vel[kk, ii]
                
                ni_threads[tt, Ie[ii],     idx[ii]] += W_elec[0, ii]
                ni_threads[tt, Ie[ii] + 1, idx[ii]] += W_elec[1, ii]
                ni_threads[tt, Ie[ii] + 2, idx[ii]] += W_elec[2, ii]
    ni[:, :]    = ni_threads.sum(axis=0)
    nu[:, :, :] = nu_threads.sum(axis=0)
    return


@nb.njit(parallel=do_parallel)
def deposit_velocity_moments(vel, Ie, W_elec, idx, nu):
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
        
    TODO:
        -- Initialize thread arrays at runtime (preallocate memory)
        -- Calculate N_per_thread and n_start_idxs at runtime
        -- Check if this works for non-multiples of the thread count
           i.e. N_per_thread may need to be an array with particle counts since
           we can't assume that it is constant for each thread.
    '''
    nu_threads = np.zeros((n_threads, NC, Nj, 3, ), dtype=np.float64)
    for tt in nb.prange(n_threads):        
        for ii in range(n_start_idxs[tt], n_start_idxs[tt]+N_per_thread[tt]):
            if idx[ii] < Nj:
                for kk in nb.prange(3):
                    nu_threads[tt, Ie[ii],     idx[ii], kk] += W_elec[0, ii] * vel[kk, ii]
                    nu_threads[tt, Ie[ii] + 1, idx[ii], kk] += W_elec[1, ii] * vel[kk, ii]
                    nu_threads[tt, Ie[ii] + 2, idx[ii], kk] += W_elec[2, ii] * vel[kk, ii]
    nu[:, :, :] = nu_threads.sum(axis=0)
    return


@nb.njit()
def manage_source_term_boundaries(arr):
    '''
    If numba doesn't like the different possible shapes of arr
    just use a loop in the calling function to work each component 
    and this becomes for 1D arrays only
    '''
    if field_periodic == 0:
        # Mirror on open boundary
        arr[ND]          += arr[ND - 1]
        arr[ND + NX - 1] += arr[ND + NX]
        
        # ...and Fill remaining ghost cells
        arr[:li1] = arr[li1]
        arr[ro1:] = arr[ri1]
    else:
        # If periodic, move contributions
        arr[li1] += arr[ro1]
        arr[li2] += arr[ro2]
        arr[ri1] += arr[lo1]
        arr[ri2] += arr[lo2]
        
        # ...and copy periodic values
        arr[ro1] = arr[li1]
        arr[ro2] = arr[li2]
        arr[lo1] = arr[ri1]
        arr[lo2] = arr[ri2]
        
        # ...and Fill remaining ghost cells
        arr[:lo2] = arr[lo2]
        arr[ro2:] = arr[ro2]
    return


@nb.njit()
def init_collect_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, rho_0, rho, J_init, J_plus,
                         L, G, mp_flux, dt):
    '''Moment collection and position advance function. Specifically used at initialization or
    after timestep synchronization.

    INPUT:
        pos    -- Particle positions (x)
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier
        DT     -- Timestep for position advance
        
    OUTPUT:
        pos     -- Advanced particle positions
        Ie      -- Updated leftmost to nearest E-nodes
        W_elec  -- Updated TSC weighting coefficients
        rho_0   -- Charge  density at initial time (p0)
        rho     -- Charge  density at +0.5 timestep
        J_init  -- Current density at initial time (J0)
        J_plus  -- Current density at +0.5 timestep
        G       -- "Gamma"  MHD variable for current advance : Current-like
        L       -- "Lambda" MHD variable for current advance :  Charge-like
        
    TODO: Incorporate velocity advance into this and rename function to "particles and moments"
    Advance would be:
        -- Advance velocity v0 -> v1
        -- Collect moments at x1/2
        -- Advance position
        -- Collect moments at x3/2
        -- Post-process to get charge/current densities
    This may allow it to be better optimised/parallelised, and to separate the
    particle from the field actions.
    '''
    ni       = np.zeros((NC, Nj), dtype=np.float64)
    ni_init  = np.zeros((NC, Nj), dtype=np.float64)
    nu_init  = np.zeros((NC, Nj, 3), dtype=np.float64)
    nu_plus  = np.zeros((NC, Nj, 3), dtype=np.float64)

    rho_0   *= 0.0
    rho     *= 0.0
    J_init  *= 0.0
    J_plus  *= 0.0
    L       *= 0.0
    G       *= 0.0
                         
    deposit_both_moments(vel, Ie, W_elec, idx, ni_init, nu_init)      # Collects sim_particles/cell/species
    position_update(pos, vel, idx, Ie, W_elec, Ib, W_mag, mp_flux, dt)
    deposit_both_moments(vel, Ie, W_elec, idx, ni, nu_plus)

    if source_smoothing == 1:
        for jj in range(Nj):
            ni[:, jj]  = smooth(ni[:, jj])
        
            for kk in range(3):
                nu_plus[:, jj, kk] = smooth(nu_plus[:,  jj, kk])
                nu_init[:, jj, kk] = smooth(nu_init[:, jj, kk])
    
    # Sum contributions across species
    for jj in range(Nj):
        rho_0   += ni_init[:, jj]   * n_contr[jj] * charge[jj]
        rho     += ni[:, jj]        * n_contr[jj] * charge[jj]
        L       += ni[:, jj]        * n_contr[jj] * charge[jj] ** 2 / mass[jj]
        
        for kk in range(3):
            J_init[:, kk]  += nu_init[:, jj, kk] * n_contr[jj] * charge[jj]
            J_plus[ :, kk] += nu_plus[:, jj, kk] * n_contr[jj] * charge[jj]
            G[      :, kk] += nu_plus[:, jj, kk] * n_contr[jj] * charge[jj] ** 2 / mass[jj]

    manage_source_term_boundaries(rho_0)
    manage_source_term_boundaries(rho)
    manage_source_term_boundaries(L)
    for ii in range(3):
        manage_source_term_boundaries(J_init[:, ii])
        manage_source_term_boundaries(J_plus[:, ii])
        manage_source_term_boundaries(G[:, ii])

    for ii in range(rho_0.shape[0]):
        if rho_0[ii] < min_dens * ne * ECHARGE:
            rho_0[ii] = min_dens * ne * ECHARGE
            
        if rho[ii] < min_dens * ne * ECHARGE:
            rho[ii] = min_dens * ne * ECHARGE
    return


@nb.njit()
def collect_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, rho, J_minus, J_plus, L, G, mp_flux, dt):
    '''
    Moment collection and position advance function.

    INPUT:
        pos    -- Particle positions (x)
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier
        DT     -- Timestep for position advance
        
    OUTPUT:
        pos     -- Advanced particle positions
        Ie      -- Updated leftmost to nearest E-nodes
        W_elec  -- Updated TSC weighting coefficients
        rho     -- Charge  density at +0.5 timestep
        J_plus  -- Current density at +0.5 timestep
        J_minus -- Current density at initial time (J0)
        G       -- "Gamma"  MHD variable for current advance
        L       -- "Lambda" MHD variable for current advance    
    '''
    ni       = np.zeros((NC, Nj), dtype=np.float64)
    nu_plus  = np.zeros((NC, Nj, 3), dtype=np.float64)
    nu_minus = np.zeros((NC, Nj, 3), dtype=np.float64)
    
    rho      *= 0.0
    J_minus  *= 0.0
    J_plus   *= 0.0
    L        *= 0.0
    G        *= 0.0
    
    deposit_velocity_moments(vel, Ie, W_elec, idx, nu_minus)
    position_update(pos, vel, idx, Ie, W_elec, Ib, W_mag, mp_flux, dt)
    deposit_both_moments(vel, Ie, W_elec, idx, ni, nu_plus)
    
    if source_smoothing == 1:
        for jj in range(Nj):
            ni[:, jj]  = smooth(ni[:, jj])
        
            for kk in range(3):
                nu_plus[ :, jj, kk] = smooth(nu_plus[:,  jj, kk])
                nu_minus[:, jj, kk] = smooth(nu_minus[:, jj, kk])
    
    for jj in range(Nj):
        rho  += ni[:, jj] * n_contr[jj] * charge[jj]
        L    += ni[:, jj] * n_contr[jj] * charge[jj] ** 2 / mass[jj]
        
        for kk in range(3):
            J_minus[:, kk] += nu_minus[:, jj, kk] * n_contr[jj] * charge[jj]
            J_plus[ :, kk] += nu_plus[ :, jj, kk] * n_contr[jj] * charge[jj]
            G[      :, kk] += nu_plus[ :, jj, kk] * n_contr[jj] * charge[jj] ** 2 / mass[jj]
        
    manage_source_term_boundaries(rho)
    manage_source_term_boundaries(L)
    for ii in range(3):
        manage_source_term_boundaries(J_minus[:, ii])
        manage_source_term_boundaries(J_plus[:, ii])
        manage_source_term_boundaries(G[:, ii])
        
    for ii in range(rho.shape[0]):
        if rho[ii] < min_dens * ne * ECHARGE:
            rho[ii] = min_dens * ne * ECHARGE  
    return


@nb.njit()
def smooth(function):
    '''
    Smoothing function: Applies Gaussian smoothing routine across adjacent cells. 
    Assummes no contribution from ghost cells.
    '''
    size         = function.shape[0]
    new_function = np.zeros(size, dtype=np.float64)

    for ii in np.arange(1, size - 1):
        new_function[ii - 1] = 0.25*function[ii] + new_function[ii - 1]
        new_function[ii]     = 0.50*function[ii] + new_function[ii]
        new_function[ii + 1] = 0.25*function[ii] + new_function[ii + 1]

    # Move Ghost Cell Contributions: Periodic Boundary Condition
    new_function[1]        += new_function[size - 1]
    new_function[size - 2] += new_function[0]

    # Set ghost cell values to mirror corresponding real cell
    new_function[0]        = new_function[size - 2]
    new_function[size - 1] = new_function[1]
    return new_function


#%% FIELDS
@nb.njit()
def eval_B0x(x):
    return B_eq * (1. + a * x**2)


@nb.njit()
def get_curl_B(B):
    ''' Returns a vector quantity for the curl of a field valid at the positions 
    between its gridpoints (i.e. curl(B) -> E-grid, etc.)
    
    INPUT:
        field    -- The 3D field to take the curl of
        DX       -- Spacing between the nodes, mostly for diagnostics. 
                    Defaults to grid spacing specified at initialization.
                 
    OUTPUT:
        curl  -- Finite-differenced solution for the curl of the input field.
        
    NOTE: This function will only work with this specific 1D hybrid code due to both 
          E and B fields having the same number of nodes (due to TSC weighting) and
         the lack of derivatives in y, z
    '''
    curlB = np.zeros((NC, 3), dtype=nb.float64)
    for ii in nb.prange(B.shape[0] - 1):
        curlB[ii, 1] = - (B[ii + 1, 2] - B[ii, 2])
        curlB[ii, 2] =    B[ii + 1, 1] - B[ii, 1]
    return curlB/(dx*mu0)


@nb.njit()
def get_curl_E(E, dE):
    ''' 
    Returns a vector quantity for the curl of a field valid at the positions 
    between its gridpoints (i.e. curl(E) -> B-grid, etc.)
    
    INPUT:
        field    -- The 3D field to take the curl of
        DX       -- Spacing between the nodes, mostly for diagnostics. 
                    Defaults to grid spacing specified at initialization.
                 
    OUTPUT:
        curl  -- Finite-differenced solution for the curl of the input field.
    '''   
    dE *= 0.
    for ii in np.arange(1, E.shape[0]):
        dE[ii, 1] = - (E[ii, 2] - E[ii - 1, 2])
        dE[ii, 2] =    E[ii, 1] - E[ii - 1, 1]
        
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
def get_electron_temp(qn, te):
    '''
    Calculate the electron temperature in each cell. Depends on the charge density of each cell
    and the treatment of electrons: i.e. isothermal (ie=0) or adiabatic (ie=1)
    '''
    if ie == 0:
        te[:]      = np.ones(qn.shape[0]) * Te0_scalar
    elif ie == 1:
        gamma_e = 5./3. - 1.
        te[:]      = Te0_scalar * np.power(qn / (ECHARGE*ne), gamma_e)
    return


def get_grad_P(qn, te):
    '''
    Returns the electron pressure gradient (in 1D) on the E-field grid using P = nkT and 
    finite difference.
    
    INPUT:
        qn -- Grid charge density
        te -- Grid electron temperature
        DX -- Grid separation, used for diagnostic purposes. Defaults to simulation dx.
        inter_type -- Linear (0) or cubic spline (1) interpolation.
        
    NOTE: Interpolation is needed because the finite differencing causes the result to be deposited on the 
    B-grid. Moving it back to the E-grid requires an interpolation. Cubic spline is desired due to its smooth
    derivatives and its higher order weighting (without the polynomial craziness)
    '''
    grad_pe_B     = np.zeros(NC + 1, dtype=np.float64)
    Pe            = qn * kB * te / ECHARGE

    # Center points
    for ii in np.arange(1, qn.shape[0]):
        grad_pe_B[ii] = (Pe[ii] - Pe[ii - 1])/dx
            
    # Set endpoints (there should be no gradients here anyway, but just to be safe)
    grad_pe_B[0]  = grad_pe_B[1]
    grad_pe_B[NC] = grad_pe_B[NC - 1]
        
    # Re-interpolate to E-grid
    coeffs = splrep(B_nodes_loc, grad_pe_B)
    grad_P = splev( E_nodes_loc, coeffs)
    return Pe, grad_P


@nb.njit()
def apply_boundary(B, B_damp):
    if field_periodic == 0:
        for ii in nb.prange(1, B.shape[1]):
            B[:, ii] *= B_damp
    else:
        for ii in nb.prange(1, B.shape[1]):
            # Boundary value (should be equal)
            end_bit = 0.5 * (B[ND, ii] + B[ND + NX, ii])

            B[ND,      ii] = end_bit
            B[ND + NX, ii] = end_bit
            
            B[ND - 1, ii]  = B[ND + NX - 1, ii]
            B[ND - 2, ii]  = B[ND + NX - 2, ii]
            
            B[ND + NX + 1, ii] = B[ND + 1, ii]
            B[ND + NX + 2, ii] = B[ND + 2, ii]
    return


#@nb.njit()
def cyclic_leapfrog(B1, B2, B_center, rho, Ji, J_ext, E, Ve, Te, DT, subcycles,
                    B_damp, resistive_array, sim_time):
    '''
    Solves for the magnetic field push by keeping two copies and subcycling between them,
    averaging them at the end of the cycle as per Matthews (1994). The source terms are
    unchanged during the subcycle step. This method damps the high frequency dispersion 
    inherent in explicit hybrid simulations.
    
    INPUT:
        B1    -- Magnetic field to update (return value comes through here)
        B2    -- Empty array for second copy
        rho_i -- Total ion charge density
        J_i   -- Total ionic current density
        DT    -- Master simulation timestep. This function advances the field by 0.5*DT
        subcycles -- The number of subcycle steps to be performed. 
        
    22/02/2021 :: Applied damping field to each subcycle. Does this damping array
            need to account for subcycling in the DX/DT bit? Test later.
            
    28/02/2021 :: Added advancement of sim_time within this function. The global
            "clock" now follows the development of the magnetic field.
            Resync doesn't require a sim_time increment since the field solution
            is already at 0.5*DT and this solution is used to advance the second
            field copy for averaging.
    '''
    if disable_waves:
        return 0.0
    
    curl  = np.zeros((NC + 1, 3), dtype=np.float64)
    H     = 0.5 * DT
    dh    = H / subcycles
    B2[:] = B1[:]

    ## DESYNC SECOND FIELD COPY - PUSH BY DH ##
    calculate_E(B1, B_center, Ji, J_ext, rho, E, Ve, Te, resistive_array, sim_time)
    get_curl_E(E, curl) 
    B2       -= dh * curl
    apply_boundary(B2, B_damp)
    get_B_cent(B2, B_center)
    sim_time += dh

    ## RETURN IF NO SUBCYCLES REQUIRED ##
    if subcycles == 1:
        B1[:] = B2[:]
        return sim_time

    ## MAIN SUBCYCLE LOOP ##
    for ii in range(subcycles - 1):             
        if ii%2 == 0:
            calculate_E(B2, B_center, Ji, J_ext, rho, E, Ve, Te, resistive_array, sim_time)
            get_curl_E(E, curl) 
            B1  -= 2 * dh * curl
            apply_boundary(B1, B_damp)
            get_B_cent(B1, B_center)
            sim_time += dh
        else:
            calculate_E(B1, B_center, Ji, J_ext, rho, E, Ve, Te, resistive_array, sim_time)
            get_curl_E(E, curl) 
            B2  -= 2 * dh * curl
            apply_boundary(B2, B_damp)
            get_B_cent(B2, B_center)
            sim_time += dh

    ## RESYNC FIELD COPIES ##
    if ii%2 == 0:
        calculate_E(B2, B_center, Ji, J_ext, rho, E, Ve, Te, resistive_array, sim_time)
        get_curl_E(E, curl) 
        B2  -= dh * curl
        apply_boundary(B2, B_damp)
        get_B_cent(B2, B_center)
    else:
        calculate_E(B1, B_center, Ji, J_ext, rho, E, Ve, Te, resistive_array, sim_time)
        get_curl_E(E, curl) 
        B1  -= dh * curl
        apply_boundary(B1, B_damp)
        get_B_cent(B1, B_center)

    ## AVERAGE FIELD SOLUTIONS: COULD PERFORM A CONVERGENCE TEST HERE IN FUTURE ##
    B1 += B2; B1 /= 2.0
    return sim_time


@nb.njit()
def get_J_ext(J_ext, sim_time):
    '''
    Driven J designed as energy input into simulation. All parameters specified
    in the simulation_parameters script/file
    
    Designed as a Gaussian pulse so that things don't freak out by rising too 
    quickly. Just test with one source point at first
    
    NEED TO ADJUST SIM_TIME TO ACCOUNT FOR SUBCYCLING -- NOT YET DONE
    PROBABLY JUST NEED TO PASS A TIME ARGUMENT AROUND THE FIELD FUNCTIONS
    TO KEEP TRACK OF WHAT POINT IN TIME IT IS
    '''
    # Soft source wave (What t corresponds to this?)
    # Should put some sort of ramp on it?
    # Also needs to be polarised. By or Bz lagging/leading?
    phase = -90
    N_eq  = ND + NX//2

    gaussian = np.exp(- ((sim_time - pulse_offset)/ pulse_width) ** 2 )

    # Set new field values in array as soft source
    J_ext[N_eq, 1] = driven_ampl*gaussian*np.sin(2 * np.pi * driven_freq * sim_time)
    J_ext[N_eq, 2] = driven_ampl*gaussian*np.sin(2 * np.pi * driven_freq * sim_time + phase * np.pi / 180.)    
    return


@nb.njit()
def get_J_ext_pol(sim_time):
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
    '''
    # Soft source wave (What t corresponds to this?)
    # Should put some sort of ramp on it?
    # Also needs to be polarised. By or Bz lagging/leading?
    J_ext = np.zeros((NC, 3), dtype=np.float64)
    phase = -np.pi / 2
    N_eq  = ND + NX//2
    time  = sim_time
    v_ph  = driven_freq / driven_k
    
    for off in np.arange(-2, 3):
        delay = off*dx / v_ph
        gauss = driven_ampl * np.exp(- ((time - pulse_offset - delay)/ pulse_width) ** 2 )
        
        J_ext[N_eq + off, 1] += gauss * np.sin(2 * np.pi * driven_freq * (time - delay))
        J_ext[N_eq + off, 2] += gauss * np.sin(2 * np.pi * driven_freq * (time - delay) + phase)    
    return J_ext


#@nb.njit()
def calculate_E(B, B_center, Ji, J_ext, qn, E, Ve, Te, resistive_array, sim_time):
    '''Calculates the value of the electric field based on source term and magnetic field contributions, assuming constant
    electron temperature across simulation grid. This is done via a reworking of Ampere's Law that assumes quasineutrality,
    and removes the requirement to calculate the electron current. Based on equation 10 of Buchner (2003, p. 140).
    INPUT:
        B   -- Magnetic field array. Displaced from E-field array by half a spatial step.
        J   -- Ion current density. Source term, based on particle velocities
        qn  -- Charge density. Source term, based on particle positions
    OUTPUT:
        E_out -- Updated electric field array
    '''
    # Calculate J_ext at same instance as Ji
    if pol_wave == 0:
        J_ext[:, :] *= 0.0
    elif pol_wave == 1:
        get_J_ext(J_ext, sim_time)
    elif pol_wave == 2:
        get_J_ext_pol(J_ext, sim_time)
        
    curlB = get_curl_B(B)
       
    Ve[:, 0] = (Ji[:, 0] + J_ext[:, 0] - curlB[:, 0]) / qn
    Ve[:, 1] = (Ji[:, 1] + J_ext[:, 1] - curlB[:, 1]) / qn
    Ve[:, 2] = (Ji[:, 2] + J_ext[:, 2] - curlB[:, 2]) / qn
    
    get_electron_temp(qn, Te)
    Pe, del_p = get_grad_P(qn, Te)
    
    VexB     = np.zeros((NC, 3), dtype=np.float64)  
    for ii in np.arange(NC):
        VexB[ii, 0] = Ve[ii, 1] * B_center[ii, 2] - Ve[ii, 2] * B_center[ii, 1]
        VexB[ii, 1] = Ve[ii, 2] * B_center[ii, 0] - Ve[ii, 0] * B_center[ii, 2]
        VexB[ii, 2] = Ve[ii, 0] * B_center[ii, 1] - Ve[ii, 1] * B_center[ii, 0]
    
    # When I finally get around to adding E_damping back
    #if E_damping == 1 and field_periodic == 0:
    #    temp3De *= E_damping_array
        
    E[:, 0]  = - VexB[:, 0] - del_p / qn
    E[:, 1]  = - VexB[:, 1]
    E[:, 2]  = - VexB[:, 2]

    # Add resistivity
    if resis_multiplier != 0:
        E[:, 0] += resistive_array[:] * (Ji[:, 0] - qn[:]*Ve[:, 0] + J_ext[:, 0])
        E[:, 1] += resistive_array[:] * (Ji[:, 1] - qn[:]*Ve[:, 1] + J_ext[:, 0])
        E[:, 2] += resistive_array[:] * (Ji[:, 2] - qn[:]*Ve[:, 2] + J_ext[:, 0])
    
    # Copy periodic values
    if field_periodic == 1:
        for ii in range(3):
            # Copy edge cells
            E[ro1, ii] = E[li1, ii]
            E[ro2, ii] = E[li2, ii]
            E[lo1, ii] = E[ri1, ii]
            E[lo2, ii] = E[ri2, ii]
            
            # Fill remaining ghost cells
            E[:lo2, ii] = E[lo2, ii]
            E[ro2:, ii] = E[ro2, ii]
            
    # Diagnostic flag for testing
    if disable_waves == 1:
        E *= 0.    
    return


#%% AUXILLIARY FUNCTIONS
# =============================================================================
# @nb.njit()
# def get_B_at_center(B, zero_boundaries=True):
#     ''' 
#     Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
#     with a 3D array (e.g. B). Second derivative y2 is calculated on the B-grid, with
#     forwards/backwards difference used for endpoints. (i.e. y2 at data points)
#     
#     interp has one more gridpoint than required just because of the array used. interp[-1]
#     should remain zero.
#     
#     This might be able to be done without the intermediate y2 array since the interpolated
#     points don't require previous point values.
#     
#     As long as B-grid is filled properly in the push_B() routine, this shouldn't have to
#     vary for homogenous boundary conditions
#     
#     ADDS B0 TO X-AXIS ON TOP OF INTERPOLATION
#     '''
#     y2      = np.zeros((NC + 1, 3), dtype=np.float64)
#     interp  = np.zeros((NC    , 3), dtype=np.float64)
#     
#     # Calculate second derivative
#     for jj in range(1, B.shape[1]):
#         
#         # Interior B-nodes, Centered difference
#         for ii in range(1, NC):
#             y2[ii, jj] = B[ii + 1, jj] - 2*B[ii, jj] + B[ii - 1, jj]
#                 
#         # Edge B-nodes, Forwards/Backwards difference
#         if zero_boundaries == True:
#             y2[0 , jj] = 0.
#             y2[NC, jj] = 0.
#         else:
#             y2[0,  jj] = 2*B[0 ,    jj] - 5*B[1     , jj] + 4*B[2     , jj] - B[3     , jj]
#             y2[NC, jj] = 2*B[NC,    jj] - 5*B[NC - 1, jj] + 4*B[NC - 2, jj] - B[NC - 3, jj]
#         
#     # Do spline interpolation: E[ii] is bracketed by B[ii], B[ii + 1]
#     for jj in range(1, B.shape[1]):
#         for ii in range(NC):
#             interp[ii, jj] = 0.5 * (B[ii, jj] + B[ii + 1, jj] + (1/6) * (y2[ii, jj] + y2[ii + 1, jj]))
#     
#     # Add B0x to interpolated array
#     for ii in range(NC):
#         interp[ii, 0] = eval_B0x(E_nodes[ii])
#     return interp
# =============================================================================


def get_B_cent(B, B_center):
    '''
    Quick and easy function to calculate B on the E-grid using scipy's cubic
    spline interpolation (mine seems to be broken). Could probably make this 
    myself and more efficient later, but need to eliminate problems!
    
    TODO: Change this to a quadratic fit using poly1D. Also, time.
    '''
    B_center *= 0.0
    for jj in range(1, 3):
        coeffs          = splrep(B_nodes_loc, B[:, jj])
        B_center[:, jj] = splev( E_nodes_loc, coeffs)
    B_center[:, 0] = eval_B0x(E_nodes_loc)
    return


#@nb.njit()
def check_timestep(qq, DT, pos, vel, idx, Ie, W_elec, B, B_center, E, dns, 
                   max_inc, part_save_iter, field_save_iter, loop_save_iter,
                   subcycles, manual_trip=0):
    '''
    Check that simulation quantities still obey timestep limitations. Reduce
    timestep for particle violations or increase subcycling for field violations.
    
    To Do:
        -- How many subcycles should be allowed to occur before the timestep itself
            is shorted? Maybe once subcycle > 100, trip timestep and half subcycles.
            Would need to make sure they didn't autochange back. Think about this.
            
    manual_trip :: Diagnostic flag to manually increase/decrease timestep/subcycle
       -1  :: Disables all timestep checks
        0  :: Normal (Auto)
        1  :: Halve timestep
        2  :: Double timestep
        3  :: Double subcycles
        4  :: Half subcycles
        
    '''
    if manual_trip < 0: # Return without change or check
        return qq, DT, max_inc, part_save_iter, field_save_iter, loop_save_iter, 0, subcycles
    max_Vx          = np.abs(vel[0, :]).max()
    max_V           = np.abs(vel      ).max()

    B_tot           = np.sqrt(B_center[:, 0] ** 2 + B_center[:, 1] ** 2 + B_center[:, 2] ** 2)
    high_rat        = qm_ratios.max()
    
    local_gyfreq    = high_rat  * np.abs(B_tot).max()      
    ion_ts          = orbit_res / local_gyfreq
    
    if E[:, 0].max() != 0:
        elecfreq    = high_rat * (np.abs(E[:, 0] / max_V)).max()
        freq_ts     = freq_res / elecfreq                            
    else:
        freq_ts     = ion_ts
    
    vel_ts          = 0.75*dx / max_Vx
    DT_part         = min(freq_ts, vel_ts, ion_ts)
    
    # Reduce timestep
    change_flag       = 0
    if (DT_part < 0.9*DT and manual_trip == 0) or manual_trip == 1:
        position_update(pos, vel, idx, Ie, W_elec, -0.5*DT)
        
        change_flag      = 1
        DT              *= 0.5
        max_inc         *= 2
        qq              *= 2
        part_save_iter  *= 2
        field_save_iter *= 2
        loop_save_iter  *= 2
        print('Timestep halved. Syncing particle velocity/position with DT =', DT)
    
    # Increase timestep
    # DISABLED until I work out how to do the injection for half a timestep when this changes
    # Idea: Maybe implement time_change on next iteration? Or is it possible to push velocity
    # by half a timestep to sync that way?
    elif (DT_part >= 4.0*DT and qq%2 == 0 and part_save_iter%2 == 0 and field_save_iter%2 == 0 and max_inc%2 == 0 and manual_trip == 0)\
        or manual_trip == 2:
        position_update(pos, vel, idx, Ie, W_elec, -0.5*DT)
        
        change_flag       = 1
        DT               *= 2.0
        max_inc         //= 2
        qq              //= 2
        part_save_iter  //= 2
        field_save_iter //= 2
        loop_save_iter  //= 2
        
        print('Timestep doubled. Syncing particle velocity/position with DT =', DT)

    if adaptive_subcycling == 1:
        k_max           = np.pi / dx
        dispfreq        = (k_max ** 2) * (B_tot / (mu0 * dns)).max()             # Dispersion frequency
        dt_sc           = freq_res / dispfreq
        new_subcycles   = int(DT / dt_sc + 1)
        
        if (subcycles < 0.75*new_subcycles and manual_trip == 0) or manual_trip == 3:                                       
            subcycles *= 2
            print('Number of subcycles per timestep doubled to', subcycles)
            
        if (subcycles > 3.0*new_subcycles and subcycles%2 == 0 and manual_trip == 0) or manual_trip == 4:                                      
            subcycles //= 2
            print('Number of subcycles per timestep halved to', subcycles)
            
        if subcycles >= 2000:
            subcycles = 2000
            sys.exit('Maximum number of subcycles reached :: Simulation aborted')

    return qq, DT, max_inc, part_save_iter, field_save_iter, loop_save_iter, change_flag, subcycles


#%% SAVE FUNCTIONS
def manage_directories():
    from shutil import rmtree
    print('Checking directories...')
    if (save_particles == 1 or save_fields == 1) == True:
        if os.path.exists('%s/%s' % (drive, save_path)) == False:
            os.makedirs('%s/%s' % (drive, save_path))                        # Create master test series directory
            print('Master directory created')

        path = ('%s/%s/run_%d' % (drive, save_path, run_num))          

        if os.path.exists(path) == False:
            os.makedirs(path)
            print('Run directory created')
        else:
            print('Run directory already exists')
            overwrite_flag = input('Overwrite? (Y/N) \n')
            if overwrite_flag.lower() == 'y':
                rmtree(path)
                os.makedirs(path)
            elif overwrite_flag.lower() == 'n':
                sys.exit('Program Terminated: Change run_num in simulation_parameters_1D')
            else:
                sys.exit('Unfamiliar input: Run terminated for safety')
    return


def store_run_parameters(dt, part_save_iter, field_save_iter, max_inc, max_time, subcycles):
    import pickle
    d_path = '%s/%s/run_%d/data/' % (drive, save_path, run_num)     # Set main dir for data
    f_path = d_path + '/fields/'
    p_path = d_path + '/particles/'

    for folder in [d_path, f_path, p_path]:
        if os.path.exists(folder) == False:                               # Create data directories
            os.makedirs(folder)

    Bc       = np.zeros((NC + 1, 3), dtype=np.float64)
    Bc[:, 0] = B_eq * (1 + a * B_nodes_loc**2)
    
    # Save simulation parameters to file (Some unused, copied from PREDCORR)
    params = dict([('seed', seed),
                   ('Nj', Nj),
                   ('dt', dt),
                   ('max_inc', max_inc),
                   ('max_time', max_time),
                   ('NX', NX),
                   ('ND', ND),
                   ('NC', NC),
                   ('N' , N),
                   ('dxm', dxm),
                   ('dx', dx),
                   ('L', L_val),
                   ('B_eq', B_eq),
                   ('xmax', xmax),
                   ('xmin', xmin),
                   ('B_xmax', B_xmax),
                   ('a', a),
                   ('theta_xmax', 0.0),
                   ('theta_L', 0.0),
                   ('loss_cone', 0.0),
                   ('loss_cone_xmax', 0.0),
                   ('r_A', 0.0),
                   ('lat_A', 0.0),
                   ('B_A', 0.0),
                   ('rc_hwidth', None),
                   ('ne', ne),
                   ('Te0', Te0_scalar),
                   ('ie', ie),
                   ('theta', 0.0),
                   ('part_save_iter', part_save_iter),
                   ('field_save_iter', field_save_iter),
                   ('max_wcinv', max_wcinv),
                   ('resis_multiplier', resis_multiplier),
                   ('freq_res', freq_res),
                   ('orbit_res', orbit_res),
                   ('run_desc', run_description),
                   ('method_type', 'CAM_CL_PARABOLIC_PARALLEL'),
                   ('particle_shape', 'TSC'),
                   ('field_periodic', field_periodic),
                   ('run_time', None),
                   ('loop_time', None),
                   ('homogeneous', homogenous),
                   ('particle_periodic', particle_periodic),
                   ('particle_reflect', particle_reflect),
                   ('particle_reinit', particle_reinit),
                   ('disable_waves', disable_waves),
                   ('source_smoothing', source_smoothing),
                   ('E_damping', 0),
                   ('quiet_start', quiet_start),
                   ('num_threads', nb.get_num_threads()),
                   ('subcycles', subcycles),
                   ('beta_flag', beta_flag),
                   ('damping_multiplier', damping_multiplier),
                   ('damping_fraction', damping_fraction),
                   ('pol_wave', pol_wave),
                   ('driven_freq', driven_freq),
                   ('driven_ampl', driven_ampl),
                   ('driven_k', driven_k),
                   ('pulse_offset', pulse_offset),
                   ('pulse_width', pulse_width),
                   ])

    with open(d_path + 'simulation_parameters.pckl', 'wb') as f:
        pickle.dump(params, f)
        f.close()
        
    print('Simulation parameters saved')
    
    # Save particle parameters to file (Need to change things into vth)
    p_file = os.path.join(d_path, 'particle_parameters')
    np.savez(p_file, idx_start   = idx_start,
                     idx_end     = idx_end,
                     species_lbl = species_lbl,
                     temp_color  = temp_color,
                     temp_type   = temp_type,
                     dist_type   = dist_type,
                     mass        = mass,
                     charge      = charge,
                     drift_v     = drift_v*va,
                     nsp_ppc     = nsp_ppc,
                     N_species   = N_species,
                     density     = density,
                     vth_par     = vth_par,
                     vth_perp    = vth_perp,
                     Tpar        = T_par,
                     Tperp       = T_perp,
                     E_par       = E_par,
                     E_perp      = E_perp,
                     anisotropy  = anisotropy,
                     Bc          = Bc,
                     Te0         = None)
    
    print('Particle parameters saved')
    return


def save_field_data(dt, field_save_iter, qq, Ji, E, B, Ve, Te, dns, sim_time,
                    damping_array, resistive_array):
    #print('Saving field data')
    d_path = '%s/%s/run_%d/data/fields/' % (drive, save_path, run_num)
    r      = qq / field_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, E = E[:, 0:3], B = B[:, 0:3], Ji = Ji,
                         dns = dns, Ve = Ve[:, 0:3], Te = Te,
                         sim_time = sim_time,
                         damping_array = damping_array,
                         resistive_array=resistive_array)
    if print_runtime == True:
        print('Field data saved')
    return
    
    
def save_particle_data(dt, part_save_iter, qq, pos, vel, idx, sim_time):
    #print('Saving particle data')
    d_path = '%s/%s/run_%d/data/particles/' % (drive, save_path, run_num)
    r      = qq / part_save_iter

    d_filename = 'data%05d' % r
    d_fullpath = os.path.join(d_path, d_filename)
    np.savez(d_fullpath, pos=pos, vel=vel, idx=idx, sim_time = sim_time)
    if print_runtime == True:
        print('Particle data saved')
    return


def add_runtime_to_header(runtime, loop_time):
    import pickle
    d_path = ('%s/%s/run_%d/data/' % (drive, save_path, run_num))     # Data path
    
    h_name = os.path.join(d_path, 'simulation_parameters.pckl')         # Header file path
    f      = open(h_name, 'rb')                                         # Open header file
    params = pickle.load(f)                                             # Load variables from header file into dict
    f.close()  
    
    params['run_time'] = runtime
    params['loop_time'] = loop_time
    
    # Re-save
    with open(d_path + 'simulation_parameters.pckl', 'wb') as f:
        pickle.dump(params, f)
        f.close()
        print('Run time appended to simulation header file')
    return


import matplotlib.pyplot as plt
def diagnostic_field_plot(B, E, q_dens, Ji, Ve, Te,
                          B_damping_array, qq, DT, sim_time):
    '''
    Check field grid arrays, probably at every timestep
    '''
    print('Generating diagnostic plot for timestep', qq)
    # Check dir
    diagnostic_path = drive + save_path + 'run_{}/diagnostic_plots/'.format(run_num)
    if os.path.exists(diagnostic_path) == False:                                   # Create data directory
        os.makedirs(diagnostic_path)
    
    ## Initialize plots and prepare plotspace
    plt.ioff()
    fontsize = 14; fsize = 12; lpad = 20
    fig, axes = plt.subplots(5, ncols=3, figsize=(20,10), sharex=True)
    fig.patch.set_facecolor('w')   
    axes[0, 0].set_title('Diagnostics :: Grid Ouputs ::: {}[{}] :: {:.4f}s'.format(save_path.split('/')[2], run_num, round(sim_time, 4)),
                         fontsize=fontsize+4, family='monospace')

    background_B = eval_B0x(E_nodes_loc)
    
    axes[0, 0].plot(B_nodes_loc / dx, B_damping_array, color='k', label=r'$r_D(x)$') 
    axes[1, 0].plot(B_nodes_loc / dx, B[:, 1]*1e9,     color='b', label=r'$B_y$') 
    axes[2, 0].plot(B_nodes_loc / dx, B[:, 2]*1e9,     color='g', label=r'$B_z$')
    axes[3, 0].plot(E_nodes_loc / dx, E[:, 1]*1e3, color='b', label=r'$E_y$')
    axes[4, 0].plot(E_nodes_loc / dx, E[:, 2]*1e3, color='g', label=r'$E_z$')

    axes[0, 1].plot(E_nodes_loc / dx, q_dens,   color='k', label=r'$n_e$')
    axes[1, 1].plot(E_nodes_loc / dx, Ve[:, 1], color='b', label=r'$V_{ey}$')
    axes[2, 1].plot(E_nodes_loc / dx, Ve[:, 2], color='g', label=r'$V_{ez}$')
    axes[3, 1].plot(E_nodes_loc / dx, Ji[:, 1], color='b', label=r'$J_{iy}$' )
    axes[4, 1].plot(E_nodes_loc / dx, Ji[:, 2], color='g', label=r'$J_{iz}$' )
    
    axes[0, 2].axhline(Te0_scalar, c='k', alpha=0.5, ls='--')
    axes[0, 2].plot(E_nodes_loc / dx, Te, color='r',          label=r'$T_e$')
    axes[1, 2].plot(E_nodes_loc / dx, Ve[:, 0], color='r',    label=r'$V_{ex}$')
    axes[2, 2].plot(E_nodes_loc / dx, Ji[:, 0], color='r',    label=r'$J_{ix}$' )
    axes[3, 2].plot(E_nodes_loc / dx, E[:, 0]*1e3, color='r', label=r'$E_x$')
    axes[4, 2].plot(B_nodes_loc / dx, B[:, 0]*1e9, color='r',     label=r'$B_{wx}$')
    axes[4, 2].plot(E_nodes_loc / dx, background_B, color='k', ls='--',    label=r'$B_{0x}$')
    

    axes[0, 0].set_ylabel('$r_D(x)$'     , rotation=0, labelpad=lpad, fontsize=fsize)
    axes[1, 0].set_ylabel('$B_y$\n(nT)'  , rotation=0, labelpad=lpad, fontsize=fsize)
    axes[2, 0].set_ylabel('$B_z$\n(nT)'  , rotation=0, labelpad=lpad, fontsize=fsize)
    axes[3, 0].set_ylabel('$E_y$\n(mV/m)', rotation=0, labelpad=lpad, fontsize=fsize)
    axes[4, 0].set_ylabel('$E_z$\n(mV/m)', rotation=0, labelpad=lpad, fontsize=fsize)
    
    axes[0, 1].set_ylabel('$n_e$\n$(cm^{-1})$', fontsize=fsize, rotation=0, labelpad=lpad)
    axes[1, 1].set_ylabel('$V_{ey}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
    axes[2, 1].set_ylabel('$V_{ez}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
    axes[3, 1].set_ylabel('$J_{iy}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
    axes[4, 1].set_ylabel('$J_{iz}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
    
    axes[0, 2].set_ylabel('$T_e$\n(eV)'     , fontsize=fsize, rotation=0, labelpad=lpad)
    axes[1, 2].set_ylabel('$V_{ex}$\n(m/s)' , fontsize=fsize, rotation=0, labelpad=lpad)
    axes[2, 2].set_ylabel('$J_{ix}$'        , fontsize=fsize, rotation=0, labelpad=lpad)
    axes[3, 2].set_ylabel('$E_x$\n(mV/m)'   , fontsize=fsize, rotation=0, labelpad=lpad)
    axes[4, 2].set_ylabel('$B_x$\n(nT)'     , fontsize=fsize, rotation=0, labelpad=lpad)
    
    fig.align_labels()
            
    for ii in range(3):
        axes[4, ii].set_xlabel('Position (m/dx)')
        for jj in range(5):
            axes[jj, ii].set_xlim(B_nodes_loc[0] / dx, B_nodes_loc[-1] / dx)
            axes[jj, ii].axvline(-NX//2, c='k', ls=':', alpha=0.5)
            axes[jj, ii].axvline( NX//2, c='k', ls=':', alpha=0.5)
            axes[jj, ii].ticklabel_format(axis='y', useOffset=False)
            axes[jj, ii].grid()
    
    plt.tight_layout(pad=1.0, w_pad=1.8)
    fig.subplots_adjust(hspace=0.125)
    plt.savefig(diagnostic_path + 'diag_field_{:07}'.format(qq), 
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close('all')
    return

def dump_to_file(pos, vel, E_int, Ve, Te, B, Ji, q_dens, qq, folder='parallel', print_particles=False):
    import os
    np.set_printoptions(threshold=sys.maxsize)
    
    dirpath = drive + save_path + '/{}/timestep_{:05}/'.format(folder, qq) 
    if os.path.exists(dirpath) == False:
        os.makedirs(dirpath)
        
    print('Dumping arrays to file')
    if print_particles == True:
        with open(dirpath + 'pos.txt', 'w') as f:
            print(pos, file=f)
        with open(dirpath + 'vel.txt', 'w') as f:
            print(vel, file=f)
    with open(dirpath + 'E.txt', 'w') as f:
        print(E_int, file=f)
    with open(dirpath + 'Ve.txt', 'w') as f:
        print(Ve, file=f)
    with open(dirpath + 'Te.txt', 'w') as f:
        print(Te, file=f)
    with open(dirpath + 'B.txt', 'w') as f:
        print(B, file=f)
    with open(dirpath + 'Ji.txt', 'w') as f:
        print(Ji, file=f)
    with open(dirpath + 'rho.txt', 'w') as f:
        print(q_dens, file=f)

    np.set_printoptions(threshold=1000)
    return


def load_run_params():
    global drive, save_path, run_num, save_particles, save_fields, seed, homogenous, particle_periodic,\
        particle_reflect, particle_reinit, field_periodic, disable_waves, source_smoothing, quiet_start,\
        E_damping, damping_fraction, damping_multiplier, resis_multiplier, NX, max_wcinv, dxm, ie,\
            orbit_res, freq_res, part_res, field_res, run_description, particle_open, ND, NC,\
                lo1, lo2, ro1, ro2, li1, li2, ri1, ri2,\
                adaptive_timestep, adaptive_subcycling, default_subcycles
            
    print('LOADING RUNFILE: {}'.format(run_input))
    with open(run_input, 'r') as f:
        drive             = f.readline().split()[1]        # Drive letter or path for portable HDD e.g. 'E:/' or '/media/yoshi/UNI_HD/'
        save_path         = f.readline().split()[1]        # Series save dir   : Folder containing all runs of a series
        run_num           = f.readline().split()[1]        # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)
    
        save_particles    = int(f.readline().split()[1])   # Save data flag    : For later analysis
        save_fields       = int(f.readline().split()[1])   # Save plot flag    : To ensure hybrid is solving correctly during run
        seed              = f.readline().split()[1]        # RNG Seed          : Set to enable consistent results for parameter studies
        
        homogenous        = int(f.readline().split()[1])   # Set B0 to homogenous (as test to compare to parabolic)
        particle_periodic = int(f.readline().split()[1])   # Set particle boundary conditions to periodic
        particle_reflect  = int(f.readline().split()[1])   # Set particle boundary conditions to reflective
        particle_reinit   = int(f.readline().split()[1])   # Set particle boundary conditions to reinitialize
        field_periodic    = int(f.readline().split()[1])   # Set field boundary to periodic (False: Absorbtive Boundary Conditions)
        disable_waves     = int(f.readline().split()[1])   # Zeroes electric field solution at each timestep
        source_smoothing  = int(f.readline().split()[1])   # Smooth source terms with 3-point Gaussian filter
        quiet_start       = int(f.readline().split()[1])   # Flag to use quiet start
        E_damping         = int(f.readline().split()[1])   # Damp E in a manner similar to B for ABCs
        damping_fraction  = float(f.readline().split()[1]) # Fraction of solution domain (on each side) that includes damping    
        damping_multiplier= float(f.readline().split()[1]) # Multiplies the r-factor to increase/decrease damping rate.
        resis_multiplier  = float(f.readline().split()[1]) # Fraction of Lower Hybrid resonance used for resistivity calculation
    
        NX        = int(f.readline().split()[1])           # Number of cells - doesn't include ghost/damping cells
        max_wcinv = float(f.readline().split()[1])         # Simulation runtime, in multiples of the ion gyroperiod (in seconds)
        dxm       = float(f.readline().split()[1])         # Number of ion inertial lengths per dx
        
        ie        = int(f.readline().split()[1])           # Adiabatic electrons. 0: off (constant), 1: on.
          
        orbit_res = float(f.readline().split()[1])         # Orbit resolution
        freq_res  = float(f.readline().split()[1])         # Frequency resolution     : Fraction of angular frequency for multiple cyclical values
        part_res  = float(f.readline().split()[1])         # Data capture resolution in gyroperiod fraction: Particle information
        field_res = float(f.readline().split()[1])         # Data capture resolution in gyroperiod fraction: Field information
    
        run_description = f.readline()                     # Commentary to attach to runs, helpful to have a quick description
    
    # Set number of ghost cells, count total
    ND = 2
    NC = NX + 2*ND
    
    # E-field nodes around boundaries (used for sources and E-fields)
    lo1 = ND - 1 ; lo2 = ND - 2             # Left outer (to boundary)
    ro1 = ND + NX; ro2 = ND + NX + 1        # Right outer
    
    li1 = ND         ; li2 = ND + 1         # Left inner
    ri1 = ND + NX - 1; ri2 = ND + NX - 2    # Right inner

    # Check BCs
    if field_periodic == 1 and not disable_waves:
        if particle_periodic == 0:
            print('Periodic field compatible only with periodic particles.')
            particle_periodic = 1
            particle_reflect = particle_reinit = 0
    
    particle_open = 0
    if particle_reflect + particle_reinit + particle_periodic == 0:
        particle_open = 1
        
    # Hardcode because I keep forgetting to change this
    if os.name == 'posix':
        drive = '/home/c3134027/'
    
    # Set run number
    if args['run_num'] != -1:
        run_num = args['run_num']
    elif run_num != '-':
        run_num = int(run_num)
    else:
        if os.path.exists(drive + save_path) == False:
            run_num = 0
        else:
            run_num = len(os.listdir(drive + save_path))
    
    if seed == '-':
        seed = None
    else:
        seed = int(seed)
    
    if field_periodic == 1 and damping_multiplier != 0:
        damping_multiplier = 0.0
        
    if disable_waves == True:
        print('-- Wave solutions disabled, removing subcycles --')
        adaptive_timestep = adaptive_subcycling = 0
        default_subcycles = 1
    return


def load_plasma_params():
    global species_lbl, temp_color, temp_type, dist_type, nsp_ppc, mass, charge, \
        drift_v, density, anisotropy, E_perp, E_e, beta_flag, L_val, B_eq, B_xmax_ovr,\
        qm_ratios, N, idx_start, idx_end, Nj, N_species, B_eq, ne, density, \
        E_par, Te0_scalar, vth_perp, vth_par, T_par, T_perp, vth_e, vei,\
        wpi, wpe, va, gyfreq_eq, egyfreq_eq, dx, n_contr, min_dens, xmax, xmin,\
            inject_rate, k_max
        
    print('LOADING PLASMA: {}'.format(plasma_input))
    with open(plasma_input, 'r') as f:
        species_lbl = np.array(f.readline().split()[1:])
        
        temp_color = np.array(f.readline().split()[1:])
        temp_type  = np.array(f.readline().split()[1:], dtype=int)
        dist_type  = np.array(f.readline().split()[1:], dtype=int)
        nsp_ppc    = np.array(f.readline().split()[1:], dtype=int)
        
        mass       = np.array(f.readline().split()[1:], dtype=float)
        charge     = np.array(f.readline().split()[1:], dtype=float)
        drift_v    = np.array(f.readline().split()[1:], dtype=float)
        density    = np.array(f.readline().split()[1:], dtype=float)*1e6
        anisotropy = np.array(f.readline().split()[1:], dtype=float)
        
        # Particle energy: If beta == 1, energies are in beta. If not, they are in eV                                    
        E_perp     = np.array(f.readline().split()[1:], dtype=float)
        E_e        = float(f.readline().split()[1])
        beta_flag  = int(f.readline().split()[1])
    
        L_val     = float(f.readline().split()[1])           # Field line L shell, used to calculate parabolic scale factor.
        B_eq      = f.readline().split()[1]                  # Magnetic field at equator in T: '-' for L-determined value :: 'Exact' value in node ND + NX//2
        B_xmax_ovr= f.readline().split()[1]                  # Magnetic field at simulation boundaries (excluding damping cells). Overrides 'a' calculation.
    
    if B_eq == '-':
        B_eq = (B_surf / (L_val ** 3))                       # Magnetic field at equator, based on L value
    else:
        B_eq = float(B_eq)
    
    ### -- Normalization of density override (e.g. Fu, Winkse)
    if Fu_override == True:
        rat        = 5
        ne         = (rat*B_eq)**2 * e0 / EMASS
        density    = np.array([0.05, 0.94, 0.01])*ne
    ### 
    
    charge    *= ECHARGE                                     # Cast species charge to Coulomb
    mass      *= PMASS                                       # Cast species mass to kg
    qm_ratios  = np.divide(charge, mass)                     # q/m ratio for each species
    
    # DOUBLE ARRAYS THAT MIGHT BE ACCESSED BY DEACTIVATED PARTICLES
    qm_ratios  = np.concatenate((qm_ratios, qm_ratios))
    temp_type  = np.concatenate((temp_type, temp_type))
    
    ne         = density.sum()                               # Electron number density
    rho        = (mass*density).sum()                        # Mass density for alfven velocity calc.
    wpi        = np.sqrt((density * charge ** 2 / (mass * e0)).sum())            # Proton Plasma Frequency, wpi (rad/s)
    wpe        = np.sqrt(ne * ECHARGE ** 2 / (EMASS * e0))   # Electron Plasma Frequency, wpi (rad/s)
    va         = B_eq / np.sqrt(mu0*rho)                    # Alfven speed at equator: Assuming pure proton plasma
    gyfreq_eq  = ECHARGE*B_eq  / PMASS                       # Proton Gyrofrequency (rad/s) at equator (slowest)
    egyfreq_eq = ECHARGE*B_eq  / EMASS                       # Electron Gyrofrequency (rad/s) at equator (slowest)
    #va         = B_eq / np.sqrt(mu0*mass[1]*density[1])      # Hard-coded to be 'cold proton alfven velocity' (Shoji et al, 2013)
    dx         = dxm * va / gyfreq_eq                        # Alternate method of calculating dx (better for multicomponent plasmas)
    n_contr    = density / nsp_ppc                           # Species density contribution: Each macroparticle contributes this SI density to a cell
    min_dens   = 0.05                                        # Minimum charge density in a cell
    xmax       = NX / 2 * dx                                 # Maximum simulation length, +/-ve on each side
    xmin       =-NX / 2 * dx
    k_max      = np.pi / dx                                  # Maximum permissible wavenumber in system (SI???)

    E_par       = E_perp / (anisotropy + 1)     # Parallel species energy
    if beta_flag == 0:
        # Input energies in eV
        Te0_scalar = ECHARGE * E_e / kB
        vth_perp   = np.sqrt(charge *  E_perp /  mass)    # Perpendicular thermal velocities
        vth_par    = np.sqrt(charge *  E_par  /  mass)    # Parallel thermal velocities
        T_par      = E_par  * 11603.
        T_perp     = E_perp * 11603.
    else:
        # Input energies in terms of beta (Generally only used for Winske/Gary/Fu stuff... invalid in general?)
        kbt_par    = E_par  * (B_eq ** 2) / (2 * mu0 * ne)
        kbt_per    = E_perp * (B_eq ** 2) / (2 * mu0 * ne)
        Te0_scalar = E_e    * (B_eq ** 2) / (2 * mu0 * ne * kB)
        vth_perp   = np.sqrt(kbt_per /  mass)                # Perpendicular thermal velocities
        vth_par    = np.sqrt(kbt_par /  mass)                # Parallel thermal velocities
        T_par      = kbt_par / kB
        T_perp     = kbt_per / kB
    
    # This will change with Te0 which means the resistance will change. Complicated!
    vth_e      = np.sqrt(kB*Te0_scalar/EMASS)
    vei        = np.sqrt(2.) * wpe**4 / (64.*np.pi*ne*vth_e**3) # Ion-Electron collision frequency

    # Number of sim particles for each species, total
    Nj        = len(mass)                                    # Number of species
    N_species = nsp_ppc * NX
    if field_periodic == 0:
        N_species += 2   
    
    # Add number of spare particles proportional to percentage of total (50% standard, high but safe)
    if particle_open == 1:
        spare_ppc  = N_species.sum() * 0.5
    else:
        spare_ppc  = 0
    N = N_species.sum() + int(spare_ppc)
    
    # Calculate injection rate for open boundaries
    if particle_open == 1:
        inject_rate = nsp_ppc * (vth_par / dx) / np.sqrt(2 * np.pi)
    else:
        inject_rate = 0.0
    
    idx_start  = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
    idx_end    = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order
    return


def load_wave_driver_params():
    global pol_wave, driven_freq, driven_ampl, driven_k, pulse_offset, pulse_width
    
    print('LOADING DRIVER: {}'.format(driver_input))
    with open(driver_input, 'r') as f:
        pol_wave     = int(np.array(f.readline().split()[1]))
        driven_freq  = float(np.array(f.readline().split()[1]))
        driven_ampl  = float(np.array(f.readline().split()[1]))
        pulse_offset = float(np.array(f.readline().split()[1]))
        pulse_cycle  = float(np.array(f.readline().split()[1]))
    
    # Define pulse width in terms of wave frequency
    # i.e. pump X cycles of a wave at driven_freq
    wave_period = 1. / driven_freq
    pulse_width = pulse_cycle*wave_period
    
    species_plasfreq_sq   = (density * charge ** 2) / (mass * e0)
    species_gyrofrequency = np.divide(charge, mass) * B_eq
    
    # Looks right!
    driven_rad = driven_freq * 2 * np.pi
    driven_k   = (driven_rad / SPLIGHT) ** 2
    driven_k  *= 1 - (species_plasfreq_sq / (driven_rad * (driven_rad - species_gyrofrequency))).sum()
    driven_k   = np.sqrt(driven_k)
    return


def calculate_background_magnetic_field():
    global a, B_xmax, gyfreq_xmax, loss_cone_eq, loss_cone_xmax, lambda_L, theta_xmax,\
        B_A, r_A, lat_A, B_nodes_loc, E_nodes_loc
    if homogenous == 1:
        a      = 0
        B_xmax = B_eq
        
        # Also need to set any numeric values
        B_A            = 0.0
        loss_cone_eq   = 0.0
        loss_cone_xmax = 0.0
        theta_xmax     = 0.0
        lambda_L       = 0.0
        lat_A          = 0.0
        r_A            = 0.0
    else:    
        a          = 4.5 / (L_val*RE)**2
        r_A        = 120e3
        lat_A      = np.arccos(np.sqrt((RE + r_A)/(RE*L_val)))    # Anchor latitude in radians
        B_A        = B_eq * np.sqrt(4 - 3*np.cos(lat_A) ** 2)\
                    / (np.cos(lat_A) ** 6)                        # Magnetic field at anchor point
                    
        # GENERAL PARABOLIC FIELD
        if B_xmax_ovr == '-':
            a      = 4.5 / (L_val*RE)**2
            B_xmax = B_eq * (1 + a*xmax**2)
        else:
            B_xmax = float(B_xmax_ovr)
            a      = (B_xmax / B_eq - 1) / xmax**2
            
        lambda_L       = np.arccos(np.sqrt(1.0 / L_val))                # MLAT of anchor point
        loss_cone_eq   = np.arcsin(np.sqrt(B_eq   / B_A))*180 / np.pi   # Equatorial loss cone in degrees
        loss_cone_xmax = np.arcsin(np.sqrt(B_xmax / B_A))               # Boundary loss cone in radians
        theta_xmax     = 0.0                                            # NOT REALLY ANY WAY TO TELL MLAT WITH THIS METHOD
       
    B_nodes_loc  = (np.arange(NC + 1) - NC // 2)       * dx             # B grid points position in space
    E_nodes_loc  = (np.arange(NC)     - NC // 2 + 0.5) * dx             # E grid points position in space
    gyfreq_xmax  = ECHARGE*B_xmax/ PMASS                                # Proton Gyrofrequency (rad/s) at boundary (highest)
    return


def print_summary_and_checks():    
    print('')
    print('Run Started')
    print('Run Series         : {}'.format(save_path.split('//')[-1]))
    print('Run Number         : {}'.format(run_num))
    print('# threads used     : {}'.format(n_threads))
    print('Field save flag    : {}'.format(save_fields))
    print('Particle save flag : {}\n'.format(save_particles))
    
    print('Sim domain length  : {:5.2f}R_E'.format(2 * xmax / RE))
    print('Density            : {:5.2f}cc'.format(ne / 1e6))
    print('Equatorial B-field : {:5.2f}nT'.format(B_eq*1e9))
    print('Boundary   B-field : {:5.2f}nT'.format(B_xmax*1e9))
    print('Iono.      B-field : {:5.2f}mT'.format(B_A*1e6))
    print('Equat. Loss cone   : {:<5.2f} degrees  '.format(loss_cone_eq))
    print('Bound. Loss cone   : {:<5.2f} degrees  '.format(loss_cone_xmax * 180. / np.pi))
    print('Maximum MLAT (+/-) : {:<5.2f} degrees  '.format(theta_xmax * 180. / np.pi))
    print('Iono.   MLAT (+/-) : {:<5.2f} degrees\n'.format(lambda_L * 180. / np.pi))
    
    print('Equat. Gyroperiod: : {}s'.format(round(2. * np.pi / gyfreq_eq, 3)))
    print('Inverse rad gyfreq : {}s'.format(round(1 / gyfreq_eq, 3)))
    print('Maximum sim time   : {}s ({} gyroperiods)\n'.format(round(max_wcinv / gyfreq_eq, 2), 
                                                               round(max_wcinv/(2*np.pi), 2)))    
    print('{} spatial cells, 2x{} damped cells'.format(NX, ND))
    print('{} cells total'.format(NC))
    print('{} particles total\n'.format(N))
    
    if theta_xmax > lambda_L:
        print('ABORT : SIMULATION DOMAIN LONGER THAN FIELD LINE')
        sys.exit()
    
    if particle_periodic + particle_reflect + particle_reinit > 1:
        print('ABORT : ONLY ONE PARTICLE BOUNDARY CONDITION ALLOWED')
        sys.exit()
            
    if  os.name != 'posix':
        os.system("title Hybrid Simulation :: {} :: Run {}".format(save_path.split('//')[-1], run_num))
    return


def get_thread_values():
    '''
    Function to calculate number of particles to work on each thread, and the
    start index of each batch of particles.
    '''
    global do_parallel, n_threads, N_per_thread, n_start_idxs
    
    # Count available threads
    if do_parallel:
        n_threads = nb.get_num_threads()
    else:
        # Parallel optimization makes functions run faster even on 1 thread
        n_threads = 1
        do_parallel = True

    N_per_thread = (N//n_threads)*np.ones(n_threads, dtype=int)
    if N%n_threads == 0:
        n_start_idxs = np.arange(n_threads)*N_per_thread
    else:
        leftovers = N%n_threads
        for _lo in range(leftovers):
            N_per_thread[_lo] += 1
        n_start_idxs = np.asarray([np.sum(N_per_thread[0:_si]) for _si in range(0, n_threads)])
        
    # Check values (this is important)
    if N_per_thread.sum() != N:
        raise ValueError('Number of particles per thread unequal to total number of particles')
        
    for _ii in range(1, n_start_idxs.shape[0] + 1):
        if _ii == n_start_idxs.shape[0]:
            n_in_thread = N - n_start_idxs[-1]
        else:
            n_in_thread = n_start_idxs[_ii] - n_start_idxs[_ii - 1]
        if n_in_thread != N_per_thread[_ii - 1]:
            raise ValueError('Thread particle indices are not correct. Check this.')
    return 


#%% --- MAIN ---

#################################
### FILENAMES AND DIRECTORIES ###
#################################

#### Read in command-line arguments, if present
import argparse as ap
parser = ap.ArgumentParser()
parser.add_argument('-r', '--runfile'   , default='_run_params.run', type=str)
parser.add_argument('-p', '--plasmafile', default='_plasma_params.plasma', type=str)
parser.add_argument('-d', '--driverfile', default='_driver_params.txt'   , type=str)
parser.add_argument('-n', '--run_num'   , default=-1, type=int)
args = vars(parser.parse_args())

# Check root directory (change if on RCG)
if os.name == 'posix':
    root_dir = os.path.dirname(sys.path[0])
else:
    root_dir = '..'
    
# Set input .run and .plasma files
run_input    = root_dir +  '/run_inputs/' + args['runfile']
plasma_input = root_dir +  '/run_inputs/' + args['plasmafile']
driver_input = root_dir +  '/run_inputs/' + args['driverfile']
if Fu_override == True:
        plasma_input = root_dir +  '/run_inputs/' + '_Fu_test.plasma'

load_run_params()
if __name__ == '__main__':
    manage_directories()
load_plasma_params()
load_wave_driver_params()
calculate_background_magnetic_field()
get_thread_values()
print_summary_and_checks()


#%%#####################
### START SIMULATION ###
########################
if __name__ == '__main__':
    start_time = timer()
    
    _B, _B2, _B_CENT, _E, _VE, _TE = initialize_fields()
    _RHO_HALF, _RHO_INT, _Ji,      \
    _Ji_PLUS, _Ji_MINUS, _J_EXT,   \
    _L, _G, _MP_FLUX               = initialize_source_arrays()
    _POS, _VEL, _IE, _W_ELEC, _IB, \
    _W_MAG, _IDX                   = initialize_particles(_B, _E, _MP_FLUX, _Ji, _RHO_INT)
    _DT, _MAX_INC, _PART_SAVE_ITER,\
    _FIELD_SAVE_ITER, _SUBCYCLES,  \
    _B_DAMP, _RESIS_ARR            = set_timestep(_VEL)    
    
    print('Loading initial state...\n')
    init_collect_moments(_POS, _VEL, _IE, _W_ELEC, _IB, _W_MAG, _IDX, _RHO_INT, _RHO_HALF,
                         _Ji, _Ji_PLUS, _L, _G, _MP_FLUX, 0.5*_DT)
    get_B_cent(_B, _B_CENT)
    calculate_E(_B, _B_CENT, _Ji, _J_EXT, _RHO_HALF, _E, _VE, _TE, _RESIS_ARR, 0.0)

    _LOOP_TIMES     = np.zeros(_MAX_INC-1, dtype=float)
    _LOOP_SAVE_ITER = 1

    # Put init into qq = 0 and save as usual, qq = 1 will be at t = dt
    # Need to change this so the initial state gets saved?
    # WARNING :: Accruing sim_time like this leads to numerical error accumulation at the LSB.
    _QQ = 0; _SIM_TIME = 0.0
    print('Starting loop...')
    while _QQ < _MAX_INC:
        loop_start = timer()
        if print_timings:
            print('')
            print(f'Loop {_QQ}:')
        
        if adaptive_timestep == 1 and disable_waves == 0:  
            CHECK_start = timer()
            _QQ, _DT, _MAX_INC, _PART_SAVE_ITER, _FIELD_SAVE_ITER, _LOOP_SAVE_ITER, _CHANGE_FLAG, _SUBCYCLES =\
                check_timestep(_QQ, _DT, _POS, _VEL, _IDX, _IE, _W_ELEC, _B, _B_CENT, _E, _RHO_INT, 
                               _MAX_INC, _PART_SAVE_ITER, _FIELD_SAVE_ITER, _LOOP_SAVE_ITER,
                               _SUBCYCLES)
            CHECK_time = round(timer() - CHECK_start, 3)
            if print_timings: print(f'CHECK TIME: {CHECK_time}')
            
            # Collect new moments and desync position and velocity. Reset damping array.
            CHNGE_start = timer()
            if _CHANGE_FLAG == 1:
                # If timestep was doubled, do I need to consider 0.5dt's worth of
                # new particles? Maybe just disable the doubling until I work this out
                init_collect_moments(_POS, _VEL, _IE, _W_ELEC, _IB, _W_MAG, _IDX,  
                         _RHO_INT, _RHO_HALF, _Ji, _Ji_PLUS, _L, _G, _MP_FLUX, 0.5*_DT)
                
                set_damping_arrays(_B_DAMP, _RESIS_ARR, _DT, _SUBCYCLES)
            CHNGE_time = round(timer() - CHNGE_start, 3) 
            if print_timings: print(f'CHNGE TIME: {CHNGE_time}')
        
        #######################
        ###### MAIN LOOP ######
        #######################
        LEAP1_start = timer()
        _SIM_TIME = cyclic_leapfrog(_B, _B2, _B_CENT, _RHO_INT, _Ji, _J_EXT, _E, _VE, _TE,
                                    _DT, _SUBCYCLES, _B_DAMP, _RESIS_ARR, _SIM_TIME)
        calculate_E(_B, _B_CENT, _Ji, _J_EXT, _RHO_HALF,
                    _E, _VE, _TE, _RESIS_ARR, _SIM_TIME)
        LEAP1_time = round(timer() - LEAP1_start, 3)

        CAMEL_start = timer()
        push_current(_Ji_PLUS, _Ji, _E, _B_CENT, _L, _G, _DT)
        calculate_E(_B, _B_CENT, _Ji, _J_EXT, _RHO_HALF,
                    _E, _VE, _TE, _RESIS_ARR, _SIM_TIME)
        CAMEL_time = round(timer() - CAMEL_start, 3)
        
        VELAD_start = timer()
        velocity_update(_POS, _VEL, _IE, _W_ELEC, _IB, _W_MAG, _IDX, _B, _E, _DT)
        VELAD_time = round(timer() - VELAD_start, 3)

        # Store pc(1/2) here while pc(3/2) is collected
        _RHO_INT[:]  = _RHO_HALF[:]
        
        MOMAD_start = timer()
        collect_moments(_POS, _VEL, _IE, _W_ELEC, _IB, _W_MAG, _IDX, 
                              _RHO_HALF, _Ji_MINUS, _Ji_PLUS, _L, _G, _MP_FLUX, _DT)
        MOMAD_time = round(timer() - MOMAD_start, 3)
        
        _RHO_INT += _RHO_HALF
        _RHO_INT /= 2.0
        _Ji[:]    = 0.5 * (_Ji_PLUS  +  _Ji_MINUS)

        LEAP2_start = timer()
        _SIM_TIME = cyclic_leapfrog(_B, _B2, _B_CENT, _RHO_INT, _Ji, _J_EXT, _E, _VE, _TE,
                                    _DT, _SUBCYCLES, _B_DAMP, _RESIS_ARR, _SIM_TIME)
        calculate_E(_B, _B_CENT, _Ji, _J_EXT, _RHO_INT,
                    _E, _VE, _TE, _RESIS_ARR, _SIM_TIME)
        LEAP2_time = round(timer() - LEAP2_start, 3)
        
        loop_diag = round(timer() - loop_start, 3)
        
        if print_timings:
            print(f'LEAP1 TIME: {LEAP1_time}')
            print(f'CAMEL TIME: {CAMEL_time}')
            print(f'VELAD TIME: {MOMAD_time}')
            print(f'MOMAD TIME: {MOMAD_time}')
            print(f'LEAP2 TIME: {LEAP2_time}')
            print(f'Total Loop: {loop_diag}')

        ########################
        ##### OUTPUT DATA  #####
        ########################
        if _QQ%_PART_SAVE_ITER == 0 and save_particles == 1:
            save_particle_data(_DT, _PART_SAVE_ITER, _QQ, _POS, _VEL, _IDX, _SIM_TIME)

        if _QQ%_FIELD_SAVE_ITER == 0 and save_fields == 1:
            save_field_data(_DT, _FIELD_SAVE_ITER, _QQ, _Ji, _E, _B, _VE, _TE,
                            _RHO_INT, _SIM_TIME, _B_DAMP, _RESIS_ARR)
        
        if _QQ%100 == 0:
            running_time = int(timer() - start_time)
            hrs          = running_time // 3600
            rem          = running_time %  3600
            
            mins         = rem // 60
            sec          = rem %  60
            
            print('Step {} of {} :: Current runtime {:02}:{:02}:{:02}'.format(_QQ, _MAX_INC, hrs, mins, sec))

        if _QQ%_LOOP_SAVE_ITER == 0:
            _LOOP_TIME = round(timer() - loop_start, 4)
            _LOOP_IDX  = _QQ // _LOOP_SAVE_ITER
            _LOOP_TIMES[_LOOP_IDX-1] = _LOOP_TIME
            
        _QQ += 1
        
    runtime = round(timer() - start_time,2) 
    print('Run complete : {} s'.format(runtime))
    if save_fields == 1 or save_particles == 1:
        add_runtime_to_header(runtime, _LOOP_TIMES[1:].mean())
        fin_path = '%s/%s/run_%d/run_finished.txt' % (drive, save_path, run_num)
        with open(fin_path, 'w') as open_file:
            pass