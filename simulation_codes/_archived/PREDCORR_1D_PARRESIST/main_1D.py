## PYTHON MODULES ##
import numpy as np
import numba as nb
import os, sys, pdb
import pickle
from shutil import rmtree
from timeit import default_timer as timer
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

## PHYSICAL CONSTANTS ##
q      = 1.602177e-19                       # Elementary charge (C)
c      = 2.998925e+08                       # Speed of light (m/s)
mp     = 1.672622e-27                       # Mass of proton (kg)
me     = 9.109384e-31                       # Mass of electron (kg)
kB     = 1.380649e-23                       # Boltzmann's Constant (J/K)
e0     = 8.854188e-12                       # Epsilon naught - permittivity of free space
mu0    = (4e-7) * np.pi                     # Magnetic Permeability of Free Space (SI units)
RE     = 6.371e6                            # Earth radius in metres
B_surf = 3.12e-5                            # Magnetic field strength at Earth surface (equatorial)

# A few internal flags
Fu_override       = False       # Override to allow density to be calculated as a ratio of frequencies
adaptive_timestep = True       # Disable adaptive timestep to keep it the same as initial
do_parallel       = True       # Flag to use available threads to parallelize particle functions
print_timings     = False      # Diagnostic outputs timing each major segment (for efficiency examination)
print_runtime     = True       # Flag to print runtime every 50 iterations 

if not do_parallel:
    nb.set_num_threads(1)          
#nb.set_num_threads(1)         # Uncomment to manually set number of threads, otherwise will use all available

### ##
#%% INITIALIZATION
### ##
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
    sf_par, sf_per (the thermal velocities) as a Bi-Maxwellian distribution.
    
    Is there a better way to do this with a Monte Carlo perhaps?
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
    idx = np.ones(N,       dtype=np.int8) * -1

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
    idx   = np.ones(N, dtype=np.int8) * -1

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



def initialize_particles(B, E, _mp_flux):
    '''
    Initializes particle arrays. Selects which velocity/position function to
    use for intialization, runs equilibrium loop if required, and calculates
    the initial weighting functions for the initial particle positions.
    
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
    if quiet_start == 1:
        pos, vel, idx = quiet_start_bimaxwellian()
    else:
        pos, vel, idx = uniform_bimaxwellian()

    Ie         = np.zeros(N,      dtype=np.uint16)
    Ib         = np.zeros(N,      dtype=np.uint16)
    W_elec     = np.zeros((3, N), dtype=np.float64)
    W_mag      = np.zeros((3, N), dtype=np.float64)
    
    if homogenous == False:
        run_until_equilibrium(pos, vel, idx, Ie, W_elec, Ib, W_mag, B, E,
                          _mp_flux, hot_only=True, psave=True)
    
    assign_weighting_TSC(pos, Ie, W_elec)
    assign_weighting_TSC(pos, Ib, W_mag)
    return pos, vel, Ie, W_elec, Ib, W_mag, idx


@nb.njit()
def set_damping_array(B_damping_array, E_damping_array, DT):
    '''
    Create masking array for magnetic field damping used to apply open
    boundaries. Based on applcation by Shoji et al. (2011) and
    Umeda et al. (2001).
    
    Equivalent to introducing a magnetic current into Faraday's law (Birdsall
    and Langdon, 1985, Section 15-11b). Basically the same as Taflove and Hagness, 
    equation 3.7 with Jm = sigma* x H.
    
    E_damping_array applied to Hall Term of E-field update as per Hu & Denton (2009)
     - Might need a factor to make this stronger than the B-damping
    '''
    # Fraction of simulation domain also damped (in addition to damping region)
    frac_encroach    = 0.10
    interior_damp    = frac_encroach*NX
    damping_boundary = 0.5*NX - interior_damp
    
    # Damping coefficient
    r_damp   = np.sqrt(29.7 * 0.5 * va * (0.5 * DT / dx) / ND)   
    r_damp  *= damping_multiplier
    
    # Do B-damping array
    B_dist_from_mp  = np.abs(np.arange(NC + 1) - 0.5*NC)
    for ii in range(NC + 1):
        if B_dist_from_mp[ii] > damping_boundary:
            B_damping_array[ii] = 1. - r_damp * ((B_dist_from_mp[ii] - damping_boundary) / ND) ** 2 
        else:
            B_damping_array[ii] = 1.0
            
    # Do E-damping array
    E_dist_from_mp  = np.abs(np.arange(NC) + 0.5 - 0.5*NC)
    for ii in range(NC):
        if E_dist_from_mp[ii] > damping_boundary:
            E_damping_array[ii] = 1. - r_damp * ((E_dist_from_mp[ii] - damping_boundary) / ND) ** 2 
        else:
            E_damping_array[ii] = 1.0
            
    for da in [B_damping_array, E_damping_array]:
        for _ii in range(da.shape[0]):
            if da[_ii] < 0.0:
                da[_ii] = 0.0
    return


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
    B_cent  = np.zeros((NC    , 3), dtype=np.float64)
    E_int   = np.zeros((NC    , 3), dtype=np.float64)
    E_half  = np.zeros((NC    , 3), dtype=np.float64)
    
    Ve_int  = np.zeros((NC, 3), dtype=np.float64)
    Ve_half = np.zeros((NC, 3), dtype=np.float64)
    Te      = np.ones(  NC,     dtype=np.float64) * Te0_scalar
    return B, B_cent, E_int, E_half, Ve_int, Ve_half, Te


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
    q_dens  = np.zeros( NC,         dtype=np.float64)    
    q_dens2 = np.zeros( NC,         dtype=np.float64) 
    Ji_int  = np.zeros((NC, 3),     dtype=np.float64)
    Ji_half = np.zeros((NC, 3),     dtype=np.float64)
    J_ext   = np.zeros((NC, 3),     dtype=np.float64)
    return q_dens, q_dens2, Ji_int, Ji_half, J_ext


@nb.njit()
def initialize_tertiary_arrays():
    '''
    Initializes source term ndarrays. Each term is collected on the E-field grid.

    INPUT:
        <NONE>
        
    OUTPUT:
        temp3Db       -- Swap-file vector array with B-grid dimensions
        temp3De       -- Swap-file vector array with E-grid dimensions
        temp1D        -- Swap-file scalar array with E-grid dimensions
        old_particles -- Location to store old particle values (positions, velocities, weights)
                         as part of predictor-corrector routine
        old_fields    -- Location to store old B, Ji, Ve, Te field values for predictor-corrector routine
        mp_flux       -- Tracking variable designed to accrue the flux at each timestep (in terms of macroparticles
                             at each boundary and for each species) and trigger an injection if >= 2.
    '''
    temp3Db       = np.zeros((NC + 1, 3),  dtype=np.float64)
    temp3De       = np.zeros((NC    , 3),  dtype=np.float64)
    temp1D        = np.zeros( NC    ,      dtype=np.float64) 
    old_fields    = np.zeros((NC + 1, 10), dtype=np.float64)
 
    old_particles = np.zeros((13, N),      dtype=np.float64)
    mp_flux       = np.zeros((2 , Nj),     dtype=np.float64)
        
    return old_particles, old_fields, temp3De, temp3Db, temp1D, mp_flux


def set_timestep(vel):
    '''
    INPUT:
        vel             -- Initial particle velocities
    OUTPUT:
        DT              -- Maximum allowable timestep (seconds)
        max_inc         -- Number of integer timesteps to get to end time
        part_save_iter  -- Number of timesteps between particle data saves
        field_save_iter -- Number of timesteps between field    data saves
    
    Note : Assumes no dispersion effects or electric field acceleration to
           be initial limiting factor. This may change for inhomogenous loading
           of particles or initial fields.
           
    To do : 
        - Actually put a Courant condition check in here
    '''
    if disable_waves == 0:
        ion_ts = dxm * orbit_res / gyfreq_xmax        # Timestep to highest resolve gyromotion
    else:
        ion_ts = 0.25 / gyfreq_xmax                   # If no waves, 20 points per revolution (~4 per wcinv)
        
    vel_ts = 0.5 * dx / np.max(np.abs(vel[0, :]))     # Timestep to satisfy particle CFL: <0.5dx per timestep
    
    DT       = min(ion_ts, vel_ts)                    # Timestep as smallest of options
    max_time = max_wcinv / gyfreq_eq                  # Total runtime in seconds
    max_inc  = int(max_time / DT) + 1                 # Total number of time steps
    
    if part_res == 0:
        part_save_iter = 1
    else:
        part_save_iter = int(part_res / (DT*gyfreq_eq))

    if field_res == 0:
        field_save_iter = 1
    else:
        field_save_iter = int(field_res / (DT*gyfreq_eq))

    if save_fields == 1 or save_particles == 1:
        store_run_parameters(DT, part_save_iter, field_save_iter, max_inc, max_time)
    
    B_damping_array = np.ones(NC + 1, dtype=float)
    E_damping_array = np.ones(NC    , dtype=float)
    set_damping_array(B_damping_array, E_damping_array, DT)
    
    # DIAGNOSTIC PLOT: CHECK OUT DAMPING FACTORS
    if False:
        plt.ioff()
        fig, axes = plt.subplots(2)
        axes[0].plot(B_nodes_loc/dx, B_damping_array)
        axes[0].set_ylabel('B damp')
        axes[1].plot(E_nodes_loc/dx, E_damping_array)
        axes[1].set_ylabel('E damp')
        for ax in axes:
            ax.set_xlim(B_nodes_loc[0]/dx, B_nodes_loc[-1]/dx)
            ax.axvline(    0, color='k', ls=':', alpha=0.5)
            ax.axvline( NX/2, color='k', ls=':')
            ax.axvline(-NX/2, color='k', ls=':')
        plt.show()
        sys.exit()
        
    
    print('Timestep: %.4fs, %d iterations total\n' % (DT, max_inc))
    return DT, max_inc, part_save_iter, field_save_iter, B_damping_array, E_damping_array



def run_until_equilibrium(pos, vel, idx, Ie, W_elec, Ib, W_mag, B, E_int, Ji, Ve, q_dens,
                          _mp_flux, frev=1000, hot_only=True, psave=True, save_inc=50):
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
    
    TODO: Set particle boundary conditions to be species specific - i.e. reflect
    cold populations, but let hot populations have an open boundary.
    
    Is vel_ts needed? Cells don't really exist in this bottle. But you probably 
    don't want to jump too quickly through the gradient, surely there's a timestep
    limitation on that.
    
    To do: Put check in store_run_params() to delete if it already exists so
    this particle data doesn't alter it.
    '''
    psave = save_particles
    
    print('Letting particle distribution relax into static field configuration')
    max_vx   = np.max(np.abs(vel[0, :]))
    ion_ts   = orbit_res * 2 * np.pi / gyfreq_xmax
    vel_ts   = 0.5*dx / max_vx                    # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step
    pdt      = min(ion_ts, vel_ts)
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
            
        pdata_path  = ('%s/%s/run_%d' % (drive, save_path, run))
        pdata_path += '/data/equil_particles/'
        if os.path.exists(pdata_path) == False:                                   # Create data directory
            os.makedirs(pdata_path)
        pnum = 0
        
    # Desync (retard) velocity here
    parmov(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E_int, Ji, Ve, q_dens,
           -0.5*pdt, _mp_flux, vel_only=True, hot_only=hot_only)
    for pp in range(psteps):

        # Save first and last only
        if psave == True and (pp == 0 or pp == psteps - 1):
            p_fullpath = pdata_path + 'data%05d' % pnum
            np.savez(p_fullpath, pos=pos, vel=vel, idx=idx, sim_time=psim_time)
            pnum += 1
            print('p-Particle data saved')
            
        if pp%500 == 0:
            print('Step', pp)
        
        for jj in range(Nj):
            parmov(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E_int, Ji, Ve, q_dens,
                   pdt, _mp_flux, hot_only=hot_only)
            
            if hot_only == False or temp_type[jj] == 1:
                if particle_open == True:
                    inject_particles_1sp(pos, vel, idx, _mp_flux, pdt, jj)

        # Check number of spare particles every 25 steps
        if pp%100 == 0 and particle_open == 1:
            num_spare = (idx < 0).sum()
            if num_spare < nsp_ppc.sum():
                print('WARNING :: Less than one cell of spare particles remaining.')
                if num_spare < inject_rate.sum() * DT * 5.0:
                    # Change this to dynamically expand particle arrays later on (adding more particles)
                    # Can do it by cell lots (i.e. add a cell's worth each time)
                    raise Exception('WARNING :: No spare particles remaining. Exiting simulation.')
                
        psim_time += pdt
    
    # Resync (advance) velocity here
    parmov(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E_int, Ji, Ve, q_dens,
           0.5*pdt, _mp_flux, vel_only=True, hot_only=hot_only)
    
    # Dump indicator file
    if save_fields == 1 or save_particles == 1:
        efin_path = '%s/%s/run_%d/equil_finished.txt' % (drive, save_path, run)
        open_file = open(efin_path, 'w')
        open_file.close()
    return

### ##
#%% PARTICLES
### ##
def advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx,\
                                  B, E, DT, rho_in, rho_out, Ji_in, Ji_out, J_ext, Ve,
                                  _mp_flux, qq, pc=0):
    '''
    Helper function to group the particle advance and moment collection functions
    
    Also evaluates external current so that this is synchronised to when the
    ion current is collected (N + 1/2, N+ 3/2)
    
    rho_in is used for resistivity calculation in parmov (N)
    rho_out holds the value calculated after the particles move (N + 1)
    Not overwritten because both rho_in and rho_out are needed for the average
    at (N + 1/2)
    
    Ji, Ve treated similarly - int needed for particle pushes, J_half
    is where the actual velocity moments are calculated.
    '''
    parmov(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, Ji_in, Ve, rho_in, DT,
           _mp_flux, vel_only=False)

    # Particle injector goes here
    if particle_open == 1 or particle_reinit == 1:
        if particle_reinit == 1:
            reinit_count_flux(pos, idx, _mp_flux)
        inject_particles(pos, vel, idx, _mp_flux, DT)
    
    assign_weighting_TSC(pos, Ie, W_elec)
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)

    J_time = (qq + 0.5 + pc) * DT
    collect_moments(vel, Ie, W_elec, idx, rho_out, Ji_out, J_ext, J_time)
    return


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
def assign_weighting_CIC(pos, I, W, E_nodes=True):
    '''Assigns weighting function based on 1st order Cloud-in-Cell type particle shape
    
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
def parmov(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, Ji, Ve, q_dens, DT,
           mp_flux, vel_only=False, hot_only=False):
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
        
    Notes:
        -- Particle boundary conditions arranged in order of probability of use.
        -- mp_flux is used to count the particles leaving the boundaries,
        then call the injection routine at the end of the loop. Thus, the only
        difference between true 'open/injection' and 'reinit' boundary conditions
        is that the flux is added at a constant rate and for open, but is reactive
        to internal conditions for reinit. Technically for a high enough particle
        count (and low enough wave amplitude) these should be identical,
        but they're definitely not going to be in practice.
        -- Resistivity calculation is done via calculation of an "effective E",
        which is proportional to the total current (via Ve, q_dens), external 
        current (J_ext) and existing ion current (Ji). These are based on the 
        moments from last iteration, since that's the last time E was evaluated,
        so it should be ok
    '''
    for ii in nb.prange(pos.shape[0]):
        if temp_type[idx[ii]] == 1 or hot_only == False:
            # Calculate wave fields at particle position
            Ep = np.zeros(3, dtype=np.float64)  
            Bp = np.zeros(3, dtype=np.float64)
            Jp = np.zeros(3, dtype=np.float64)
            
            for jj in nb.prange(3):
                for kk in nb.prange(3):
                    # Calculate fields and moments at particle location
                    Ep[kk] += E[Ie[ii] + jj, kk] * W_elec[jj, ii]   
                    Bp[kk] += B[Ib[ii] + jj, kk] * W_mag[ jj, ii]  
                    
                    if eta != 0.0:
                        Jp[kk] += (Ji[Ie[ii] + jj, kk] - Ve[Ie[ii] + jj, kk]*q_dens[Ie[ii] + jj])
                    
            # Add resistivity into 'effective' E-field
            if eta != 0.0:
                Ep[0] -= eta * Jp[0]
                Ep[1] -= eta * Jp[1]
                Ep[2] -= eta * Jp[2]
    
            # Calculate background field at particle position
            Bp[0]   += B_eq * (1.0 + a * pos[ii] * pos[ii])
            constant = a * B_eq
            l_cyc    = qm_ratios[idx[ii]] * Bp[0]
            Bp[1]   += constant * pos[ii] * vel[2, ii] / l_cyc
            Bp[2]   -= constant * pos[ii] * vel[1, ii] / l_cyc
    
            # Start Boris Method
            qmi = 0.5 * DT * qm_ratios[idx[ii]]                             # q/m variable including dt
            T   = qmi * Bp 
            S   = 2.*T / (1. + T[0]*T[0] + T[1]*T[1] + T[2]*T[2])
    
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
            
            if vel_only == False:
                # Update position
                pos[ii] += vel[0, ii] * DT
            
                # Check if particle has left simulation and apply boundary conditions
                if (pos[ii] < xmin or pos[ii] > xmax):
    
                    if particle_periodic == 1:  
                        # Mario (Periodic)
                        if pos[ii] > xmax:
                            pos[ii] += xmin - xmax
                        elif pos[ii] < xmin:
                            pos[ii] += xmax - xmin 
                            
                    elif particle_open == 1:                
                        # Open: Deactivate particles that leave the simulation space
                        pos[ii]     = 0.0
                        vel[0, ii]  = 0.0
                        vel[1, ii]  = 0.0
                        vel[2, ii]  = 0.0
                        idx[ii]     = -1
                            
                    elif particle_reinit == 1: 
                        # Reinit: Deactivate particle, counted later
                        vel[0, ii]  = 0.0
                        vel[1, ii]  = 0.0
                        vel[2, ii]  = 0.0
                        idx[ii]    -= 128                            
                    else:
                        # Reflect
                        if pos[ii] > xmax:
                            pos[ii] = 2*xmax - pos[ii]
                        elif pos[ii] < xmin:
                            pos[ii] = 2*xmin - pos[ii]
                            
                        vel[0, ii] *= -1.0
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
        if idx[ii] < 0:
            sp = idx[ii]+128
            if pos[ii] > xmax:
                _mp_flux[1, sp] += 1.0
            elif pos[ii] < xmin:
                _mp_flux[0, sp] += 1.0 
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
    
    TODO: 
        After checking that the normal inject_particles() works fine with the
        new QS flag, modify this function to be basically identical to that 
        one with the new flag put in.
        
        THEN, we can finally test the new vA params
    '''
    # Add flux at each boundary 
    if particle_open == 1:
        for kk in range(2):
            _mp_flux[kk, jj] += inject_rate[jj]*dt
        
    # acc used only as placeholder to mark place in array. How to do efficiently? 
    acc = 0; n_created = 0
    for ii in nb.prange(2):
        if quiet_start == 1:
            N_inject = int(_mp_flux[ii, jj] // 2)
        else:
            N_inject = int(_mp_flux[ii, jj])
        
        for xx in nb.prange(N_inject):
            
            # Find two empty particles (Yes clumsy coding but it works)
            for kk in nb.prange(acc, pos.shape[0]):
                if idx[kk] < 0:
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
            dpos = np.random.uniform(0, 1) * vel[0, kk1] * dt
            
            # Left/Right boundary injection
            if ii == 0:
                pos[kk1]    = xmin + dpos
                vel[0, kk1] = np.abs(vel[0, kk1])
            else:
                pos[kk1]    = xmax - dpos
                vel[0, kk1] = -np.abs(vel[0, kk1])
            _mp_flux[ii, jj] -= 1.0
            n_created        += 1
            
            if quiet_start == 1:
                for kk in nb.prange(acc, pos.shape[0]):
                    if idx[kk] < 0:
                        kk2 = kk
                        acc = kk + 1
                        break
                
                # Copy values to second particle (Same position, xvel. Opposite v_perp) 
                idx[kk2]    = idx[kk1]
                pos[kk2]    = pos[kk1]
                vel[0, kk2] = vel[0, kk1]
                vel[1, kk2] = vel[1, kk1] * -1.0
                vel[2, kk2] = vel[2, kk1] * -1.0
                
                # Subtract new macroparticles from accrued flux
                _mp_flux[ii, jj] -= 1.0
                n_created        += 1
    return


@nb.njit()
def inject_particles(pos, vel, idx, _mp_flux, DT):        
    '''
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
                    if idx[kk] < 0:
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
                        if idx[kk] < 0:
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

### ##
#%% SOURCES
### ##
@nb.njit(parallel=False)
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
        
    NOTES: This needs manual work to better parallelize due to multiple threads
    wanting to access the same array. Probably need n_thread copies worth and 
    add them all at the end.
    
    Also, enabling parallel seems to introduce slight numerical noise in the last
    few bits. Not sure why, but I'm sure its fine. Need macroscale testing.
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


@nb.njit(parallel=True)
def deposit_moments_to_grid_parallel(vel, Ie, W_elec, idx, ni, nu):
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
    # Each thread needs a copy of nu, ni
    # This would be the code run on each thread for some subset of vel.shape[1]
    ni_threads = np.zeros((n_threads, NC, Nj), dtype=np.float64)
    nu_threads = np.zeros((n_threads, NC, Nj, 3, ), dtype=np.float64)
    
    for tt in nb.prange(n_threads):        
        for ii in range(n_start_idxs[tt], n_start_idxs[tt]+N_per_thread[tt]):
            if idx[ii] >= 0:
                for kk in nb.prange(3):
                    nu_threads[tt, Ie[ii],     idx[ii], kk] += W_elec[0, ii] * vel[kk, ii]
                    nu_threads[tt, Ie[ii] + 1, idx[ii], kk] += W_elec[1, ii] * vel[kk, ii]
                    nu_threads[tt, Ie[ii] + 2, idx[ii], kk] += W_elec[2, ii] * vel[kk, ii]
                
                ni_threads[tt, Ie[ii],     idx[ii]] += W_elec[0, ii]
                ni_threads[tt, Ie[ii] + 1, idx[ii]] += W_elec[1, ii]
                ni_threads[tt, Ie[ii] + 2, idx[ii]] += W_elec[2, ii]
                
    # Sum across threads
    ni[:, :]    = ni_threads.sum(axis=0)
    nu[:, :, :] = nu_threads.sum(axis=0)
    return


@nb.njit()
def collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, J_ext, J_time):
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
    
    Especially for the parallel version, might be again more efficient to initialize
    ni, nu at runtime, instead of dynamic creation.
    '''
    q_dens *= 0.
    Ji     *= 0.
    
    ni = np.zeros((NC, Nj),    dtype=np.float64)
    nu = np.zeros((NC, Nj, 3), dtype=np.float64)
        
    # Parallel moment/histogram collection or not
    if False:
        deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu)
    else:
        deposit_moments_to_grid_parallel(vel, Ie, W_elec, idx, ni, nu)

    # Sum contributions across species
    for jj in range(Nj):
        q_dens  += ni[:, jj] * n_contr[jj] * charge[jj]

        for kk in range(3):
            Ji[:, kk] += nu[:, jj, kk] * n_contr[jj] * charge[jj]

    if field_periodic == 0:
        # Mirror source term contributions at edge back into domain: Simulates having
        # some sort of source on the outside of the physical space boundary.
        q_dens[ND]          += q_dens[ND - 1]
        q_dens[ND + NX - 1] += q_dens[ND + NX]
    
        for ii in range(3):
            # Mirror source term contributions
            Ji[ND, ii]          += Ji[ND - 1, ii]
            Ji[ND + NX - 1, ii] += Ji[ND + NX, ii]
    
            # Set damping cell source values (copy last)
            Ji[:ND, ii]     = Ji[ND, ii]
            Ji[ND+NX:, ii]  = Ji[ND+NX-1, ii]
            
        # Set damping cell source values (copy last)
        q_dens[:ND]    = q_dens[ND]
        q_dens[ND+NX:] = q_dens[ND+NX-1]
    else:
        # If homogenous, move contributions
        q_dens[li1] += q_dens[ro1]
        q_dens[li2] += q_dens[ro2]
        q_dens[ri1] += q_dens[lo1]
        q_dens[ri2] += q_dens[lo2]
        
        # ...and copy periodic values
        q_dens[ro1] = q_dens[li1]
        q_dens[ro2] = q_dens[li2]
        q_dens[lo1] = q_dens[ri1]
        q_dens[lo2] = q_dens[ri2]
        
        # ...and Fill remaining ghost cells
        q_dens[:lo2] = q_dens[lo2]
        q_dens[ro2:] = q_dens[ro2]
        
        for ii in range(3):
            Ji[li1, ii] += Ji[ro1, ii]
            Ji[li2, ii] += Ji[ro2, ii]
            Ji[ri1, ii] += Ji[lo1, ii]
            Ji[ri2, ii] += Ji[lo2, ii]
            
            # ...and copy periodic values
            Ji[ro1, ii] = Ji[li1, ii]
            Ji[ro2, ii] = Ji[li2, ii]
            Ji[lo1, ii] = Ji[ri1, ii]
            Ji[lo2, ii] = Ji[ri2, ii]
            
            # ...and Fill remaining ghost cells
            Ji[:lo2, ii] = Ji[lo2, ii]
            Ji[ro2:, ii] = Ji[ro2, ii]
 
    # Implement smoothing filter: If enabled
    if source_smoothing == 1:
        three_point_smoothing(q_dens, ni[:, 0])
        for ii in range(3):
            three_point_smoothing(Ji[:, ii], ni[:, 0])

    # Set density minimum
    for ii in range(q_dens.shape[0]):
        if q_dens[ii] < min_dens * ne * q:
            q_dens[ii] = min_dens * ne * q
            
    # Calculate J_ext at same instance as Ji
    if pol_wave == 0:
        J_ext[:, :] *= 0.0
    elif pol_wave == 1:
        get_J_ext(J_ext, J_time)
    elif pol_wave == 2:
        get_J_ext_pol(J_ext, J_time)
    return


@nb.njit()
def three_point_smoothing(arr, temp):
    '''
    Three point Gaussian (1/4-1/2-1/4) smoothing function. arr, temp are both
    1D arrays of size NC = NX + 2*ND (i.e. on the E-grid)
    
    NOT IMPLEMENTED FOR HOMOGENOUS CONDITIONS YET
        --- Smooth spatial values only
        --- Do ghost cell move/copy in temp array
        --- Overwrite main array afterwards
    '''
    NC = arr.shape[0]
    
    temp *= 0.0
    for ii in range(1, NC - 1):
        temp[ii] = 0.25*arr[ii - 1] + 0.5*arr[ii] + 0.25*arr[ii + 1]
        
    temp[0]      = temp[1]
    temp[NC - 1] = temp[NC - 2]
    
    arr[:]       = temp
    return


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
def get_J_ext_pol(J_ext, sim_time):
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
    phase = -np.pi / 2
    N_eq  = ND + NX//2
    time  = sim_time
    v_ph  = driven_freq / driven_k
    
    for off in np.arange(-2, 3):
        delay = off*dx / v_ph
        gauss = driven_ampl * np.exp(- ((time - pulse_offset - delay)/ pulse_width) ** 2 )
        
        J_ext[N_eq + off, 1] += gauss * np.sin(2 * np.pi * driven_freq * (time - delay))
        J_ext[N_eq + off, 2] += gauss * np.sin(2 * np.pi * driven_freq * (time - delay) + phase)    
    return


### ##
#%% FIELDS
### ##

@nb.njit()
def eval_B0x(x):
    return B_eq * (1. + a * x**2)


def get_B_cent(_B, _B_cent):
    '''
    Quick and easy function to calculate B on the E-grid using scipy's cubic
    spline interpolation (mine seems to be broken). Could probably make this 
    myself and more efficient later, but need to eliminate problems!
    '''
    for jj in range(1, 3):
        coeffs         = splrep(B_nodes_loc, _B[:, jj])
        _B_cent[:, jj] = splev( E_nodes_loc, coeffs)
    _B_cent[:, 0] = eval_B0x(E_nodes_loc)
    return


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

    B -= 0.5 * DT * curlE                                # Advance using curl
    
    if field_periodic == 0:
        for ii in nb.prange(1, B.shape[1]):              # Apply damping, skipping x-axis
            B[:, ii] *= damping_array                    # Not sure if this needs to modified for half steps?
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
def get_electron_temp(qn, Te):
    '''
    Calculate the electron temperature in each cell. Depends on the charge density of each cell
    and the treatment of electrons: i.e. isothermal (ie=0) or adiabatic (ie=1)
    '''
    if ie == 0:
        Te[:] = np.ones(qn.shape[0]) * Te0_scalar
    elif ie == 1:
        gamma_e = 5./3. - 1.
        Te[:] = Te0_scalar * np.power(qn / (q*ne), gamma_e)
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
    physical space domain.
    
    Maybe check this at some point, or steal the one from the CAM_CL code.
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
def calculate_E(B, B_center, Ji, J_ext, q_dens, E, Ve, Te, temp3De, temp3Db, grad_P, E_damping_array):
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
    
    22/02/2021 -- Removed cross product because surely that B term is fucking with things.
    '''
    curl_B_term(B, temp3De)                                   # temp3De is now curl B term

    # Calculate Ve
    Ve[:, 0] = (Ji[:, 0] + J_ext[:, 0] - temp3De[:, 0]) / q_dens
    Ve[:, 1] = (Ji[:, 1] + J_ext[:, 1] - temp3De[:, 1]) / q_dens
    Ve[:, 2] = (Ji[:, 2] + J_ext[:, 2] - temp3De[:, 2]) / q_dens

    get_electron_temp(q_dens, Te)
    get_grad_P(q_dens, Te, grad_P, temp3Db[:, 0])
    
    # Calculate Ve x B
    temp3De[:, 0] += Ve[:, 1] * B_center[:, 2]
    temp3De[:, 1] += Ve[:, 2] * B_center[:, 0]
    temp3De[:, 2] += Ve[:, 0] * B_center[:, 1]
    
    temp3De[:, 0] -= Ve[:, 2] * B_center[:, 1]
    temp3De[:, 1] -= Ve[:, 0] * B_center[:, 2]
    temp3De[:, 2] -= Ve[:, 1] * B_center[:, 0]
    
    if E_damping == 1 and field_periodic == 0:
        temp3De *= E_damping_array
    
    # E = - (Ve x B) - nabla(pe)/rho_c
    E[:, 0]  = - temp3De[:, 0] - grad_P[:] / q_dens[:]
    E[:, 1]  = - temp3De[:, 1]
    E[:, 2]  = - temp3De[:, 2]
    
    # Add resistivity
    E[:, 0] += eta * (Ji[:, 0] - q_dens[:]*Ve[:, 0] + J_ext[:, 0])
    E[:, 1] += eta * (Ji[:, 1] - q_dens[:]*Ve[:, 1] + J_ext[:, 0])
    E[:, 2] += eta * (Ji[:, 2] - q_dens[:]*Ve[:, 2] + J_ext[:, 0])

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


### ##
#%% AUXILLIARY FUNCTIONS 
### ##

# =============================================================================
# @nb.njit()
# def interpolate_edges_to_center(B, interp, zero_boundaries=True):
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
#     y2      = np.zeros(B.shape, dtype=nb.float64)
#     interp *= 0.
#     mx      = B.shape[0] - 1
#     
#     # Calculate second derivative
#     for jj in range(1, B.shape[1]):
#         
#         # Interior B-nodes, Centered difference
#         for ii in range(1, mx):
#             y2[ii, jj] = B[ii + 1, jj] - 2*B[ii, jj] + B[ii - 1, jj]
#                 
#         # Edge B-nodes, Forwards/Backwards difference
#         if zero_boundaries == True:
#             y2[0 , jj] = 0.
#             y2[mx, jj] = 0.
#         else:
#             y2[0,  jj] = 2*B[0 ,    jj] - 5*B[1     , jj] + 4*B[2     , jj] - B[3     , jj]
#             y2[mx, jj] = 2*B[mx,    jj] - 5*B[mx - 1, jj] + 4*B[mx - 2, jj] - B[mx - 3, jj]
#         
#     # Do spline interpolation: E[ii] is bracketed by B[ii], B[ii + 1]
#     for jj in range(1, B.shape[1]):
#         for ii in range(mx):
#             interp[ii, jj] = 0.5 * (B[ii, jj] + B[ii + 1, jj] + (1/6) * (y2[ii, jj] + y2[ii + 1, jj]))
#     
#     # Add B0x to interpolated array
#     for ii in range(mx):
#         interp[ii, 0] = eval_B0x(E_nodes[ii])
#     return
# =============================================================================


# =============================================================================
# @nb.njit()
# def interpolate_centers_to_edge(E, interp, zero_boundaries=False):
#     '''
#     As above, but interpolating center values (E) to edge positions (B)
#     
#     Might need forward/backwards difference for interpolation boundary cells
#     at ii = 0, NC
#     '''
#     y2      = np.zeros(E.shape, dtype=np.float64)
#     interp *= 0.
#     mx      = E.shape[0]
#     
#     # Calculate y2 at E-field data points
#     for jj in range(E.shape[1]):
#         
#         # Interior E-nodes, Centered difference
#         for ii in range(1, mx - 1):
#             y2[ii, jj] = E[ii + 1, jj] - 2*E[ii, jj] + E[ii - 1, jj]
#                 
#         # Edge E-nodes, Forwards/Backwards difference
#         if zero_boundaries == True:
#             y2[0 ,     jj] = 0.
#             y2[mx - 1, jj] = 0.
#         else:
#             y2[0,      jj] = 2*E[0     , jj] - 5*E[1     , jj] + 4*E[2     , jj] - E[3     , jj]
#             y2[mx - 1, jj] = 2*E[mx - 1, jj] - 5*E[mx - 2, jj] + 4*E[mx - 3, jj] - E[mx - 4, jj]
# 
#     # Return to test y2
#     #y2 /= (dx ** 2)
#     #return y2
#     
#     # Do spline interpolation: B[ii] is bracketed by E[ii - 1], E[ii]
#     # Center points only
#     for jj in range(E.shape[1]):
#         for ii in range(1, mx):
#             interp[ii, jj] = 0.5 * (E[ii - 1, jj] + E[ii, jj] + (1/6) * (y2[ii - 1, jj] + y2[ii, jj]))
#     
#     if field_periodic == True:
#         for jj in range(E.shape[1]):
#             interp[0,  jj] = interp[mx - 1, jj]
#             interp[mx, jj] = interp[1, jj]
#     return
# =============================================================================

def get_thread_values():
    '''
    Function to calculate number of particles to work on each thread, and the
    start index of each batch of particles.
    '''
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
    return N_per_thread, n_start_idxs

@nb.njit(parallel=do_parallel)
def get_max_vx(vel):
    return np.abs(vel[0]).max()


@nb.njit()
def check_timestep(pos, vel, B, E, q_dens, Ie, W_elec, Ib, W_mag, B_center, Ji_in, Ve, rho_in, \
                     qq, DT, max_inc, part_save_iter, field_save_iter, loop_save_iter,
                     idx, damping_array, mp_flux):
    '''
    Evaluates all the things that could cause a violation of the timestep:
        - Magnetic field dispersion (switchable in param file since this can be tiny)
        - Gyromotion resolution
        - Ion velocity (Don't cross more than half a cell in a timestep)
        - Electric field acceleration
        
    When a violating condition found, velocity is advanced by 0.5DT (since this happens
    at the top of a loop anyway). The assumption is that the timestep isn't violated by
    enough to cause instant instability (each criteria should have a little give), which 
    should be valid except under extreme instability. The timestep is then halved and all
    time-dependent counters and quantities are doubled. Velocity is then retarded back
    half a timestep to de-sync back into a leapfrog scheme.
    
    Also evaluates if a timestep is unnneccessarily too small, which can sometimes happen
    after wave-particle interactions are complete and energetic particles are slower. This
    criteria is higher in order to provide a little hysteresis and prevent constantly switching
    timesteps.
    
    Shoji code blowing up because of Eacc_ts - what is this and does it matter?
    
    TODO: 
        - Re-evaluate damping array when timestep changes
    '''
    B_magnitude = np.sqrt(B_center[ND:ND+NX+1, 0] ** 2 +
                          B_center[ND:ND+NX+1, 1] ** 2 +
                          B_center[ND:ND+NX+1, 2] ** 2)
    gyfreq = qm_ratios.max() * B_magnitude.max()     
    ion_ts = dxm * orbit_res / gyfreq
    max_V  = get_max_vx(vel)
    
    if False:#E[:, 0].max() != 0:
        elecfreq = qm_ratios.max()*(np.abs(E[:, 0] / max_V).max())    # E-field acceleration "frequency"
        Eacc_ts  = freq_res / elecfreq                            
    else:
        Eacc_ts  = ion_ts
    
    vel_ts          = 0.60 * dx / max_V                                      # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than 'half' a cell in one time step
    DT_part         = min(Eacc_ts, vel_ts, ion_ts)                           # Smallest of the allowable timesteps
    
    if DT_part < 0.9*DT:

        parmov(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, Ji_in, Ve, rho_in, 0.5*DT,
               mp_flux, vel_only=True)    # Re-sync vel/pos       
        
        DT         *= 0.5
        max_inc    *= 2
        qq         *= 2
        
        field_save_iter *= 2
        part_save_iter  *= 2
        loop_save_iter  *= 2

        parmov(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, Ji_in, Ve, rho_in, -0.5*DT, mp_flux, vel_only=True)   # De-sync vel/pos 
        print('Timestep halved. Syncing particle velocity...')
    return qq, DT, max_inc, part_save_iter, field_save_iter, loop_save_iter, damping_array


### ##
#%% SAVE ROUTINES 
### ##
def manage_directories():
    print('Checking directories...')
    if (save_particles == 1 or save_fields == 1) == True:
        if os.path.exists('%s/%s' % (drive, save_path)) == False:
            print('Creating master directory %s/%s' % (drive, save_path))
            os.makedirs('%s/%s' % (drive, save_path))                        # Create master test series directory
            print('Master directory created')

        path = ('%s/%s/run_%d' % (drive, save_path, run))          # Set root run path (for images)
        
        if os.path.exists(path) == False:
            print('Creating run directory %s' % (path))
            os.makedirs(path)
            print('Run directory created')
        else:
            print('Run directory already exists')
            overwrite_flag = input('Overwrite? (Y/N) \n')
            if overwrite_flag.lower() == 'y':
                rmtree(path)
                os.makedirs(path)
            elif overwrite_flag.lower() == 'n':
                sys.exit('Program Terminated: Change run number')
            else:
                sys.exit('Unfamiliar input: Run terminated for safety')
    return


def store_run_parameters(dt, part_save_iter, field_save_iter, max_inc, max_time):
    d_path = ('%s/%s/run_%d/data/' % (drive, save_path, run))    # Set path for data
    f_path = d_path + '/fields/'
    p_path = d_path + '/particles/'
    
    for folder in [d_path, f_path, p_path]:
        if os.path.exists(folder) == False:                               # Create data directories
            os.makedirs(folder)
    
    Bc       = np.zeros((NC + 1, 3), dtype=np.float64)
    Bc[:, 0] = B_eq * (1 + a * B_nodes_loc**2)

    # Single parameters
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
                   ('L', L), 
                   ('B_eq', B_eq),
                   ('xmax', xmax),
                   ('xmin', xmin),
                   ('B_xmax', B_xmax),
                   ('a', a),
                   ('theta_xmax', theta_xmax),
                   ('theta_L', lambda_L),
                   ('loss_cone', loss_cone_eq),
                   ('loss_cone_xmax', loss_cone_xmax*180./np.pi),
                   ('r_A', r_A),
                   ('lat_A', lat_A),
                   ('B_A', B_A),
                   ('rc_hwidth', None),
                   ('ne', ne),
                   ('Te0', Te0_scalar),
                   ('ie', ie),
                   ('theta', 0.0),
                   ('part_save_iter', part_save_iter),
                   ('field_save_iter', field_save_iter),
                   ('max_wcinv', max_wcinv),
                   ('LH_frac', 0.0),
                   ('freq_res', freq_res),
                   ('orbit_res', orbit_res),
                   ('run_desc', run_description),
                   ('method_type', 'PREDCORR_PARABOLIC_PARALLEL'),
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
                   ('E_damping', E_damping),
                   ('quiet_start', quiet_start),
                   ('num_threads', nb.get_num_threads()),
                   ('subcycles', 1),
                   ('damping_multiplier', damping_multiplier)
                   ])

    with open(d_path + 'simulation_parameters.pckl', 'wb') as f:
        pickle.dump(params, f)
        f.close()
        print('Simulation parameters saved')
    
    # Particle values: Array parameters
    p_file = d_path + 'particle_parameters'
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
                     density     = density,
                     N_species   = N_species,
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


def save_field_data(sim_time, dt, field_save_iter, qq, Ji, E, B, Ve, Te, dns, damping_array, E_damping_array):
    d_path   = '%s/%s/run_%d/data/fields/' % (drive, save_path, run)
    r        = qq / field_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, E = E[:, 0:3], B = B[:, 0:3],   Ji = Ji[:, 0:3],
                       dns = dns,      Ve = Ve[:, 0:3], Te = Te,
                       sim_time = sim_time,
                       damping_array = damping_array, E_damping_array=E_damping_array)
    if print_runtime == True:
        print('Field data saved')
    return
    
    
def save_particle_data(sim_time, dt, part_save_iter, qq, pos, vel, idx):
    d_path   = '%s/%s/run_%d/data/particles/' % (drive, save_path, run)
    r        = qq / part_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, pos = pos, vel = vel, idx=idx, sim_time = sim_time)
    if print_runtime == True:
        print('Particle data saved')
    return
    
    
def add_runtime_to_header(runtime, loop_time):
    d_path = ('%s/%s/run_%d/data/' % (drive, save_path, run))     # Data path
    
    h_name = os.path.join(d_path, 'simulation_parameters.pckl')         # Header file path
    f      = open(h_name, 'rb')                                         # Open header file
    params = pickle.load(f)                                             # Load variables from header file into dict
    f.close()  
    
    params['run_time']  = runtime
    params['loop_time'] = loop_time
    
    # Re-save
    with open(d_path + 'simulation_parameters.pckl', 'wb') as f:
        pickle.dump(params, f)
        f.close()
        print('Run time appended to simulation header file')
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


def diagnostic_field_plot(B, E_half, q_dens, Ji, Ve, Te,
                          B_damping_array, qq, DT, sim_time):
    '''
    Check field grid arrays, probably at every timestep
    '''
    print('Generating diagnostic plot for timestep', qq)
    # Check dir
    diagnostic_path = drive + save_path + 'run_{}/diagnostic_plots/'.format(run)
    if os.path.exists(diagnostic_path) == False:                                   # Create data directory
        os.makedirs(diagnostic_path)
    
    ## Initialize plots and prepare plotspace
    plt.ioff()
    fontsize = 14; fsize = 12; lpad = 20
    fig, axes = plt.subplots(5, ncols=3, figsize=(20,10), sharex=True)
    fig.patch.set_facecolor('w')   
    axes[0, 0].set_title('Diagnostics :: Grid Ouputs ::: {}[{}] :: {:.4f}s'.format(save_path.split('/')[2], run, round(sim_time, 4)),
                         fontsize=fontsize+4, family='monospace')

    background_B = eval_B0x(E_nodes_loc)
    
    axes[0, 0].plot(B_nodes_loc / dx, B_damping_array, color='k', label=r'$r_D(x)$') 
    axes[1, 0].plot(B_nodes_loc / dx, B[:, 1]*1e9,     color='b', label=r'$B_y$') 
    axes[2, 0].plot(B_nodes_loc / dx, B[:, 2]*1e9,     color='g', label=r'$B_z$')
    axes[3, 0].plot(E_nodes_loc / dx, E_int[:, 1]*1e3, color='b', label=r'$E_y$')
    axes[4, 0].plot(E_nodes_loc / dx, E_int[:, 2]*1e3, color='g', label=r'$E_z$')

    axes[0, 1].plot(E_nodes_loc / dx, q_dens,   color='k', label=r'$n_e$')
    axes[1, 1].plot(E_nodes_loc / dx, Ve[:, 1], color='b', label=r'$V_{ey}$')
    axes[2, 1].plot(E_nodes_loc / dx, Ve[:, 2], color='g', label=r'$V_{ez}$')
    axes[3, 1].plot(E_nodes_loc / dx, Ji[:, 1], color='b', label=r'$J_{iy}$' )
    axes[4, 1].plot(E_nodes_loc / dx, Ji[:, 2], color='g', label=r'$J_{iz}$' )
    
    axes[0, 2].axhline(Te0_scalar, c='k', alpha=0.5, ls='--')
    axes[0, 2].plot(E_nodes_loc / dx, Te, color='r',          label=r'$T_e$')
    axes[1, 2].plot(E_nodes_loc / dx, Ve[:, 0], color='r',    label=r'$V_{ex}$')
    axes[2, 2].plot(E_nodes_loc / dx, Ji[:, 0], color='r',    label=r'$J_{ix}$' )
    axes[3, 2].plot(E_nodes_loc / dx, E_int[:, 0]*1e3, color='r', label=r'$E_x$')
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


### ##
#%% MAIN FUNCTIONS
### ##
@nb.njit(parallel=do_parallel)
def store_old(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, Ji, Ve, Te, old_particles, old_fields): 
    '''
    Stores current values in arrays in old_arrays for P/C method
    '''
    for ii in nb.prange(pos.shape[0]):
        old_particles[0   , ii] = pos[ii]
    for ii in nb.prange(pos.shape[0]):    
        old_particles[1   , ii] = vel[0, ii]
    for ii in nb.prange(pos.shape[0]):
        old_particles[2   , ii] = vel[1, ii]
    for ii in nb.prange(pos.shape[0]):
        old_particles[3   , ii] = vel[2, ii]
    for ii in nb.prange(pos.shape[0]):
        old_particles[4   , ii] = Ie[ii]
    for ii in nb.prange(pos.shape[0]):
        old_particles[5   , ii] = W_elec[0, ii]
    for ii in nb.prange(pos.shape[0]):
        old_particles[6   , ii] = W_elec[1, ii]
    for ii in nb.prange(pos.shape[0]):
        old_particles[7   , ii] = W_elec[2, ii]
    for ii in nb.prange(pos.shape[0]):
        old_particles[8   , ii] = Ib[ii]
    for ii in nb.prange(pos.shape[0]):
        old_particles[9   , ii] = W_mag[0, ii]
    for ii in nb.prange(pos.shape[0]):
        old_particles[10  , ii] = W_mag[1, ii]
    for ii in nb.prange(pos.shape[0]):
        old_particles[11  , ii] = W_mag[2, ii]
    for ii in nb.prange(pos.shape[0]):
        old_particles[12  , ii] = idx[ii]
    
    old_fields[:,   0:3]  = B
    old_fields[:NC, 3:6]  = Ji
    old_fields[:NC, 6:9]  = Ve
    old_fields[:NC,   9]  = Te
    return

@nb.njit(parallel=do_parallel)
def restore_old(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, Ji, Ve, Te, old_particles, old_fields):  
    '''
    Restores old values from old_arrays at conclusion of P/C method
    '''
    for ii in nb.prange(pos.shape[0]):
        pos[ii]       = old_particles[0 , ii]
    for ii in nb.prange(pos.shape[0]):
        vel[0, ii]    = old_particles[1 , ii]
    for ii in nb.prange(pos.shape[0]):
        vel[1, ii]    = old_particles[2 , ii]
    for ii in nb.prange(pos.shape[0]):
        vel[2, ii]    = old_particles[3 , ii]
    for ii in nb.prange(pos.shape[0]):
        Ie[ii]        = old_particles[4 , ii]
    for ii in nb.prange(pos.shape[0]):
        W_elec[0, ii] = old_particles[5 , ii]
    for ii in nb.prange(pos.shape[0]):
        W_elec[1, ii] = old_particles[6 , ii]
    for ii in nb.prange(pos.shape[0]):
        W_elec[2, ii] = old_particles[7 , ii]
    for ii in nb.prange(pos.shape[0]):
        Ib[ii]        = old_particles[8 , ii]
    for ii in nb.prange(pos.shape[0]):
        W_mag[0, ii]  = old_particles[9 , ii]
    for ii in nb.prange(pos.shape[0]):
        W_mag[1, ii]  = old_particles[10, ii]
    for ii in nb.prange(pos.shape[0]):
        W_mag[2, ii]  = old_particles[11, ii]
    for ii in nb.prange(pos.shape[0]):
        idx[ii]       = old_particles[12, ii]
    
    B[:]  = old_fields[:,   0:3]
    Ji[:] = old_fields[:NC, 3:6]
    Ve[:] = old_fields[:NC, 6:9]
    Te[:] = old_fields[:NC,   9]
    return


def main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag,
              B, B_cent, E_int, E_half, q_dens, q_dens_adv, Ji_int, Ji_half, J_ext, mp_flux,
              Ve_int, Ve_half, Te, temp3De, temp3Db, temp1D, old_particles, old_fields,
              B_damping_array, E_damping_array, qq, DT, max_inc, part_save_iter,
              field_save_iter, loop_save_iter):
    '''
    Main loop separated from __main__ function, since this is the actual computation bit.
    '''
    # Check timestep (Maybe only check every few. Set in main body)
    if adaptive_timestep == True and qq%1 == 0 and disable_waves == 0:
        qq, DT, max_inc, part_save_iter, field_save_iter, loop_save_iter, damping_array \
        = check_timestep(pos, vel, B, E_int, q_dens, Ie, W_elec, Ib, W_mag, B_cent, Ji_int, Ve_int, q_dens,
                         qq, DT, max_inc, part_save_iter, field_save_iter, loop_save_iter,
                         idx, mp_flux, B_damping_array)
    
    # Move particles, collect moments, deal with particle boundaries
    # Current temporal position of the velocity moment (same as J_ext)
    # q_dens is at N, q_dens_adv returned at N + 1
    advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx,\
                                  B, E_int, DT, q_dens, q_dens_adv,
                                  Ji_int, Ji_half, J_ext, Ve_int,
                                  mp_flux, qq, pc=0)
    
    # Average to get charge density at N + 1/2
    q_dens *= 0.5
    q_dens += 0.5 * q_dens_adv
    
    if disable_waves == 0:   
        # Push B from N to N + 1/2 and calculate E(N + 1/2)
        push_B(B, E_int, temp3Db, DT, qq, B_damping_array, half_flag=1)
        get_B_cent(B, B_cent)
        calculate_E(B, B_cent, Ji_half, J_ext, q_dens, E_half, Ve_half, Te,
                    temp3De, temp3Db, temp1D, E_damping_array)
        
        ###################################
        ### PREDICTOR CORRECTOR SECTION ###
        ###################################
        # Store old values
        mp_flux_old = mp_flux.copy()
        store_old(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, Ji_half, Ve_half, Te,
                  old_particles, old_fields)
        
        # Predict E-field fields (and moments?)
        E_int  *= -1.0; E_int  +=  2.0 * E_half
        Ji_int *= -1.0; Ji_int +=  2.0 * Ji_half
        Ve_int *= -1.0; Ve_int +=  2.0 * Ve_half
        
        push_B(B, E_int, temp3Db, DT, qq, B_damping_array, half_flag=0)
    
        # Advance particles to obtain source terms at N + 3/2
        # q_dens_adv is at N + 1, q_dens returned at N + 2
        advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx,\
                                      B, E_int, DT, q_dens_adv, q_dens,
                                      Ji_int, Ji_half, J_ext, Ve_int,
                                      mp_flux, qq, pc=1)

        # Average to get charge density at N + 3/2
        # Check this (make sure array values are where they should be)
        q_dens *= 0.5;    q_dens += 0.5 * q_dens_adv
    
        # Compute predicted fields at N + 3/2, advance J_ext too
        push_B(B, E_int, temp3Db, DT, qq + 1, B_damping_array, half_flag=1)
        get_B_cent(B, B_cent)
        calculate_E(B, B_cent, Ji_int, J_ext, q_dens, E_int, Ve_int, Te,
                    temp3De, temp3Db, temp1D, E_damping_array)
        
        # Determine corrected fields (and moments?) at N + 1 
        E_int  *= 0.5;  E_int += 0.5 * E_half   
        Ji_int *= 0.5; Ji_int += 0.5 * Ji_half
        Ve_int *= 0.5; Ve_int += 0.5 * Ve_half
    
        # Restore old values and push B-field final time
        restore_old(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, Ji_half, Ve_half, Te,
                    old_particles, old_fields)       
    
        push_B(B, E_int, temp3Db, DT, qq, B_damping_array, half_flag=0)   # Advance the original B
        get_B_cent(B, B_cent)
        
        # Reset q_dens to top of loop, N + 1
        q_dens[:] = q_dens_adv
        mp_flux   = mp_flux_old.copy()
        
    # Check number of spare particles every 25 steps
    if qq%1 == 0 and particle_open == 1:
        num_spare = (idx < 0).sum()
        if num_spare < nsp_ppc.sum():
            print('WARNING :: Less than one cell of spare particles remaining.')
            if num_spare < inject_rate.sum() * DT * 5.0:
                # Change this to dynamically expand particle arrays later on (adding more particles)
                # Can do it by cell lots (i.e. add a cell's worth each time)
                print('num_spare = ', num_spare)
                print('inject_rate = ', inject_rate.sum() * DT * 5.0)
                raise Exception('WARNING :: No spare particles remaining. Exiting simulation.')
        
    return qq, DT, max_inc, part_save_iter, field_save_iter, loop_save_iter





### ##
### MAIN GLOBAL CONTROL
### ##

    
#################################
### FILENAMES AND DIRECTORIES ###
#################################

#### Read in command-line arguments, if present
import argparse as ap
parser = ap.ArgumentParser()
parser.add_argument('-r', '--runfile'   , default='_run_params.run', type=str)
parser.add_argument('-p', '--plasmafile', default='_plasma_params.plasma', type=str)
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

if Fu_override == True:
    plasma_input = root_dir +  '/run_inputs/' + '_Fu_test.plasma'

###########################
### LOAD RUN PARAMETERS ###
###########################
with open(run_input, 'r') as f:
    drive             = f.readline().split()[1]        # Drive letter or path for portable HDD e.g. 'E:/' or '/media/yoshi/UNI_HD/'
    save_path         = f.readline().split()[1]        # Series save dir   : Folder containing all runs of a series
    run               = f.readline().split()[1]        # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)

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
    damping_multiplier= float(f.readline().split()[1]) # Multiplies the r-factor to increase/decrease damping rate.
    damping_frac_in   = float(f.readline().split()[1]) # Fraction of solution domain (on each side) that includes damping

    NX        = int(f.readline().split()[1])           # Number of cells - doesn't include ghost/damping cells
    ND        = int(f.readline().split()[1])           # Damping region length on each side of simulation domain
    max_wcinv = float(f.readline().split()[1])         # Simulation runtime, in multiples of the ion gyroperiod (in seconds)
    dxm       = float(f.readline().split()[1])         # Number of ion inertial lengths per dx
    
    ie        = int(f.readline().split()[1])           # Adiabatic electrons. 0: off (constant), 1: on.
      
    orbit_res = float(f.readline().split()[1])         # Orbit resolution
    freq_res  = float(f.readline().split()[1])         # Frequency resolution     : Fraction of angular frequency for multiple cyclical values
    part_res  = float(f.readline().split()[1])         # Data capture resolution in gyroperiod fraction: Particle information
    field_res = float(f.readline().split()[1])         # Data capture resolution in gyroperiod fraction: Field information

    run_description = f.readline()                     # Commentary to attach to runs, helpful to have a quick description

# Hardcode because I keep forgetting to change this
if os.name == 'posix':
    drive = '/home/c3134027/'

# Set run number
if args['run_num'] != -1:                              # Check CLI, or
    run = args['run_num']
elif run != '-':                                       # Check input file, else
    run = int(run)
else:                                                  # Autoset
    if os.path.exists(drive + save_path) == False:
        run = 0
    else:
        run = len(os.listdir(drive + save_path))
    print('Run number AUTOSET to ', run)

if seed == '-':
    seed = None
else:
    seed = int(seed)

if __name__ == '__main__':
    manage_directories()

#######################################
### LOAD PARTICLE/PLASMA PARAMETERS ###
#######################################
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

    L         = float(f.readline().split()[1])           # Field line L shell, used to calculate parabolic scale factor.
    B_eq      = f.readline().split()[1]                  # Magnetic field at equator in T: '-' for L-determined value :: 'Exact' value in node ND + NX//2
    B_xmax_ovr= f.readline().split()[1]                  # Magnetic field at simulation boundaries (excluding damping cells). Overrides 'a' calculation.

charge *= q                                              # Cast species charge to Coulomb
mass   *= mp                                             # Cast species mass to kg

#####################################
### DERIVED SIMULATION PARAMETERS ###
#####################################
if ND < 2:
    ND = 2                                               # Set minimum (used for array addresses)
    
if B_eq == '-':
    B_eq = (B_surf / (L ** 3))                           # Magnetic field at equator, based on L value
else:
    B_eq = float(B_eq)
    
### -- Normalization of density override (e.g. Fu, Winkse)
if Fu_override == True:
    rat        = 5
    ne         = (rat*B_eq)**2 * e0 / me
    density    = np.array([0.05, 0.94, 0.01])*ne
### 

NC          = NX + 2*ND                     # Total number of cells
ne          = density.sum()                 # Electron number density
E_par       = E_perp / (anisotropy + 1)     # Parallel species energy

if field_periodic == 1:
    if particle_periodic == 0:
        print('Periodic field compatible only with periodic particles.')
        particle_periodic = 1
        particle_reflect = particle_reinit = 0

particle_open = 0
if particle_reflect + particle_reinit + particle_periodic == 0:
    particle_open = 1

if beta_flag == 0:
    # Input energies in eV
    Te0_scalar = q * E_e / kB
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

rho        = (mass*density).sum()                        # Mass density for alfven velocity calc.
wpi        = np.sqrt((density * charge ** 2 / (mass * e0)).sum())            # Proton   Plasma Frequency, wpi (rad/s)
va         = B_eq / np.sqrt(mu0*rho)                     # Alfven speed at equator: Assuming pure proton plasma
#va         = B_eq / np.sqrt(mu0*mass[1]*density[1])      # Hard-coded to be 'cold proton alfven velocity' (Shoji et al, 2013)
gyfreq_eq  = q*B_eq  / mp                                # Proton Gyrofrequency (rad/s) at equator (slowest)
dx         = dxm * va / gyfreq_eq                        # Alternate method of calculating dx (better for multicomponent plasmas)
#dx2        = dxm * c / wpi

xmax       = NX / 2 * dx                                 # Maximum simulation length, +/-ve on each side
xmin       =-NX / 2 * dx
Nj         = len(mass)                                   # Number of species
n_contr    = density / nsp_ppc                           # Species density contribution: Each macroparticle contributes this SI density to a cell
min_dens   = 0.05

# Number of sim particles for each species, total
N_species = nsp_ppc * NX
if field_periodic == 0:
    N_species += 2   

# Add number of spare particles proportional to percentage of total (50% standard, high but safe)
if particle_open == 1:
    spare_ppc  = N_species.sum() * 0.5
else:
    spare_ppc  = 0
N = N_species.sum() + int(spare_ppc)

idx_start  = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
idx_end    = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order

############################
### MAGNETIC FIELD STUFF ###
############################
B_nodes_loc   = (np.arange(NC + 1) - NC // 2)       * dx     # B grid points position in space
E_nodes_loc  = (np.arange(NC)     - NC // 2 + 0.5) * dx      # E grid points position in space

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
    # DIPOLE STUFF THAT I MADE
# =============================================================================
#         print('Calculating length of field line...')
#                                                                       # Ionospheric anchor point (loss zone/max mirror point) - "Below 100km" - Baumjohann, Basic Space Plasma Physics
#         N_fl   = 1e5                                                                # Number of points to calculate field line length (higher is more accurate)
#         lat0   = np.arccos(np.sqrt((RE + r_A)/(RE*L)))                              # Latitude for this L value (at ionosphere height)
#         h      = 2.0*lat0/float(N_fl)                                               # Step size of lambda (latitude)
#         f_len  = 0.0
#         for ii in range(int(N_fl)):
#             lda        = ii*h - lat0                                                # Lattitude for this step
#             f_len     += L*RE*np.cos(lda)*np.sqrt(4.0 - 3.0*np.cos(lda) ** 2) * h   # Field line length accruance
#         print('Field line length = {:.2f} RE'.format(f_len/RE))
#         print('Simulation length = {:.2f} RE'.format(2*xmax/RE))
#         
#         if xmax > f_len / 2:
#             sys.exit('Simulation length longer than field line. Aboring...')
#         
#         print('Finding simulation boundary MLAT...')
#         dlam   = 1e-5                                            # Latitude increment in radians
#         fx_len = 0.0; ii = 1                                     # Arclength/increment counters
#         while fx_len < xmax:
#             theta_xmax = dlam * ii                                                             # Current latitude
#             d_len      = L * RE * np.cos(theta_xmax) * np.sqrt(4.0 - 3.0*np.cos(theta_xmax) ** 2) * dlam # Length increment
#             fx_len    += d_len                                                                 # Accrue arclength
#             ii        += 1                                                                     # Increment counter
#     
#         r_xmax      = L * RE * np.cos(theta_xmax) ** 2                                      # Radial distance of simulation boundary
#         
#         # Magnetic field intensity at boundary : Calculate or manually set
#         if B_xmax_ovr == '-':
#             B_xmax = B_eq*np.sqrt(4 - 3*np.cos(theta_xmax)**2)/np.cos(theta_xmax)**6       
#         else:
#             B_xmax = float(B_xmax_ovr)
#             
#         a           = (B_xmax / B_eq - 1) / xmax ** 2                                       # Parabolic scale factor: Fitted to B_eq, B_xmax
#                                                   # Lattitude of Earth's surface at this L
#     
# =============================================================================
    # BASICS THAT I SHOULD HAVE STARTED WITH
    a          = 4.5 / (L*RE)**2
    r_A        = 120e3
    lat_A      = np.arccos(np.sqrt((RE + r_A)/(RE*L)))       # Anchor latitude in radians
    B_A        = B_eq * np.sqrt(4 - 3*np.cos(lat_A) ** 2)\
                / (np.cos(lat_A) ** 6)                        # Magnetic field at anchor point
                
    # GENERAL PARABOLIC FIELD
    if B_xmax_ovr == '-':
        a      = 4.5 / (L*RE)**2
        B_xmax = B_eq * (1 + a*xmax**2)
    else:
        B_xmax = float(B_xmax_ovr)
        a      = (B_xmax / B_eq - 1) / xmax**2
        
    lambda_L       = np.arccos(np.sqrt(1.0 / L))                    # MLAT of anchor point
    loss_cone_eq   = np.arcsin(np.sqrt(B_eq   / B_A))*180 / np.pi   # Equatorial loss cone in degrees
    loss_cone_xmax = np.arcsin(np.sqrt(B_xmax / B_A))               # Boundary loss cone in radians
    theta_xmax     = 0.0                                            # NOT REALLY ANY WAY TO TELL MLAT WITH THIS METHOD
    
gyfreq_xmax = q*B_xmax/ mp                                # Proton Gyrofrequency (rad/s) at boundary (highest)
k_max       = np.pi / dx                                  # Maximum permissible wavenumber in system (SI???)
qm_ratios   = np.divide(charge, mass)                     # q/m ratio for each species

# Calculate injection rate for open boundaries
if particle_open == 1:
    inject_rate = nsp_ppc * (vth_par / dx) / np.sqrt(2 * np.pi)
else:
    inject_rate = 0.0

# E-field nodes around boundaries (used for sources and E-fields)
lo1 = ND - 1 ; lo2 = ND - 2             # Left outer (to boundary)
ro1 = ND + NX; ro2 = ND + NX + 1        # Right outer

li1 = ND         ; li2 = ND + 1         # Left inner
ri1 = ND + NX - 1; ri2 = ND + NX - 2    # Right inner

# Count available threads
if do_parallel:
    n_threads = nb.get_num_threads()
else:
    n_threads = 1
    do_parallel = True
N_per_thread, n_start_idxs = get_thread_values()

###
### Resistivity calculation (Constant for now, spatially varying later)
###
eta = 0.0

###
### DRIVEN WAVE STUFF (Load this from file eventually)
###
pol_wave    = 1         # 0: No wave, 1: Single point source, 2: Multi point source

# DRIVEN B PARAMS: Sine part
driven_freq = 1.0       # Driven wave frequency in Hz standard 2.2
driven_ampl = 25e-7     # Driven wave amplitude in A/m (I think?) Standard 50e-7

# Gaussian part
pulse_offset = 5.0      # Pulse center time (s)
pulse_width  = 1.0      # Pulse width (proportional to 2*std. 3*width decayed to 0.0123%) 

species_plasfreq_sq   = (density * charge ** 2) / (mass * e0)
species_gyrofrequency = qm_ratios * B_eq

# Looks right!
driven_rad = driven_freq * 2 * np.pi
driven_k   = (driven_rad / c) ** 2
driven_k  *= 1 - (species_plasfreq_sq / (driven_rad * (driven_rad - species_gyrofrequency))).sum()
driven_k   = np.sqrt(driven_k)



##############################
### INPUT TESTS AND CHECKS ###
##############################
print('Run Started')
print('Run Series         : {}'.format(save_path.split('//')[-1]))
print('Run Number         : {}'.format(run))
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
    
if field_periodic == 1 and damping_multiplier != 0:
    damping_multiplier = 0.0
    
if  os.name != 'posix':
    os.system("title Hybrid Simulation :: {} :: Run {}".format(save_path.split('//')[-1], run))


#####################
#%% -- GLOBAL MAIN --
#####################
if __name__ == '__main__':
    start_time = timer()
        
    # Initialize simulation: Allocate memory and set time parameters
    B, B_cent, E_int, E_half, Ve_int, Ve_half, Te       = initialize_fields()
    q_dens, q_dens_adv, Ji_int, Ji_half, J_ext          = initialize_source_arrays()
    old_particles, old_fields, temp3De, temp3Db, temp1D,\
                                               mp_flux  = initialize_tertiary_arrays()
    pos, vel, Ie, W_elec, Ib, W_mag, idx                = initialize_particles(B, E_int, mp_flux)

    # Collect initial moments and save initial state
    get_B_cent(B, B_cent)
    collect_moments(vel, Ie, W_elec, idx, q_dens, Ji_int, J_ext, 0.0) 

    DT, max_inc, part_save_iter, field_save_iter, B_damping_array, E_damping_array\
        = set_timestep(vel)

    calculate_E(B, B_cent, Ji_int, J_ext, q_dens, E_int, Ve_int, Te,
                temp3De, temp3Db, temp1D, E_damping_array)
    
    if save_particles == 1:
        save_particle_data(0, DT, part_save_iter, 0, pos, vel, idx)
        
    if save_fields == 1:
        save_field_data(0, DT, field_save_iter, 0, Ji_int, E_int,\
                             B, Ve_int, Te, q_dens, B_damping_array, E_damping_array)

    loop_times = np.zeros(max_inc-1, dtype=float)
    loop_save_iter = 1
    
    # Retard velocity
    print('Retarding velocity...')
    parmov(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E_int, Ji_int, Ve_int, q_dens, -0.5*DT, mp_flux, vel_only=True)
    
    qq       = 1;    time_sec = DT
    print('Starting main loop...')
    #part_save_iter = 1; field_save_iter = 1
    while qq < max_inc:
        
        ### DIAGNOSTICS :: MAYBE PUT UNDER A FLAG AT SOME POINT
        #dump_to_file(pos, vel, E_int, Ve, Te, B, Ji, q_dens, qq, folder='serial', print_particles=False)
        #diagnostic_field_plot(B, E_half, q_dens, Ji, Ve, Te, B_damping_array, qq, DT, sim_time)
        loop_start = timer()
        qq, DT, max_inc, part_save_iter, field_save_iter, loop_save_iter =         \
        main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag,                             \
              B, B_cent, E_int, E_half, q_dens, q_dens_adv, Ji_int, Ji_half, J_ext, mp_flux,      \
              Ve_int, Ve_half, Te, temp3De, temp3Db, temp1D, old_particles, old_fields,            \
              B_damping_array, E_damping_array, qq, DT, max_inc, part_save_iter,
              field_save_iter, loop_save_iter)
            
        if qq%part_save_iter == 0 and save_particles == 1:
            save_particle_data(time_sec, DT, part_save_iter, qq, pos,
                                    vel, idx)
            
        if qq%field_save_iter == 0 and save_fields == 1:
            save_field_data(time_sec, DT, field_save_iter, qq, Ji_int, E_int,
                                 B, Ve_int, Te, q_dens, B_damping_array, E_damping_array)
        
        if qq%100 == 0 and print_runtime == True:            
            running_time = int(timer() - start_time)
            hrs          = running_time // 3600
            rem          = running_time %  3600
            
            mins         = rem // 60
            sec          = rem %  60
            
            print('Step {} of {} :: Current runtime {:02}:{:02}:{:02}'.format(qq, max_inc, hrs, mins, sec))
        
        if qq%loop_save_iter == 0:
            loop_time = round(timer() - loop_start, 4)
            loop_idx  = qq // loop_save_iter
            loop_times[loop_idx-1] = loop_time

            if print_timings == True:
                print('Loop {}  time: {}s\n'.format(qq, loop_time))
        
        if qq == 1:
            print('First loop complete.')
        qq       += 1
        time_sec += DT

    runtime = round(timer() - start_time,2)
    
    if save_fields == 1 or save_particles == 1:
        add_runtime_to_header(runtime, loop_times[1:].mean())
        fin_path = '%s/%s/run_%d/run_finished.txt' % (drive, save_path, run)
        with open(fin_path, 'w') as open_file:
            pass
    print("Time to execute program: {0:.2f} seconds".format(runtime))
    print('Average loop time: {0:.4f} seconds'.format(loop_times[1:].mean()))