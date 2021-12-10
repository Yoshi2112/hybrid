## PYTHON MODULES ##
from timeit import default_timer as timer
import numpy as np
import numba as nb
import sys, os
from scipy.interpolate import splrep, splev

'''
CAM_CL version of the code where everything's stripped down to its basics until
it becomes stable. It should be able to recreate the run4 of the 20% He data
runs the same as the Predictor/Corrector code does.

Keep stripping until we have a MWE of the CAM_CL. Do periodic runs just to check.
'''

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

# A few internal flags
do_parallel         = True
adaptive_timestep   = True       # Disable adaptive timestep to keep it the same as initial
print_timings       = False      # Diagnostic outputs timing each major segment (for efficiency examination)
print_runtime       = True       # Flag to print runtime every 50 iterations 
adaptive_subcycling = True       # Flag (True/False) to adaptively change number of subcycles during run to account for high-frequency dispersion

if not do_parallel:
    do_parallel = True
    nb.set_num_threads(1)          
nb.set_num_threads(4)         # Uncomment to manually set number of threads, otherwise will use all available


#%% --- FUNCTIONS ---
#%% INITIALIZATION
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


def initialize_particles(B, E, mp_flux):
    if quiet_start == 1:
        pos, vel, idx = quiet_start_bimaxwellian()
    else:
        pos, vel, idx = uniform_bimaxwellian()
        
    Ie         = np.zeros(N,      dtype=np.uint16)
    Ib         = np.zeros(N,      dtype=np.uint16)
    W_elec     = np.zeros((3, N), dtype=np.float64)
    W_mag      = np.zeros((3, N), dtype=np.float64)
    
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
    
    # Set initial values
    Te           = np.ones(NC, dtype=np.float64) * Te0_scalar
    B_cent[:, 0] = B_eq
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
    L        = np.zeros( NC,     dtype=np.float64)
    G        = np.zeros((NC, 3), dtype=np.float64)
    
    mp_flux  = np.zeros((2, Nj), dtype=np.float64)
    return rho_half, rho_int, Ji, Ji_plus, Ji_minus, L, G, mp_flux



def set_timestep(vel):
    '''
    Timestep limitations:
        -- Resolve ion gyromotion (controlled by orbit_res)
        -- Resolve ion velocity (<0.5dx per timestep, varies with particle temperature)
        -- Resolve B-field solution on grid (controlled by subcycling and freq_res)
        -- E-field acceleration? (not implemented)
        
    Problems:
        -- Reducing dx means that the timestep will be shortened by same vx
        -- Reducing dx also increases dispersion. Is there a maximum number of s/c?
        
    To do: 
        -- After initial calculation of DT, calculate number of required subcycles.
        -- If greater than some preset value, change timestep instead.
    '''
    max_vx   = np.max(np.abs(vel[0, :]))
    ion_ts   = orbit_res / gyfreq_eq              # Timestep to resolve gyromotion
    vel_ts   = 0.5*dx / max_vx                    # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step

    DT       = min(ion_ts, vel_ts)
    max_time = max_wcinv / gyfreq_eq              # Total runtime in seconds
    
    if adaptive_subcycling == True:
        # b1 factor accounts for increase in total field due to wave growth
        # Without this, s/c count doubles as soon as waves start to grow
        # which unneccessarily slows the simulation
        b1_fac     = 1.2
        k_max      = np.pi / dx
        dispfreq   = (k_max ** 2) * B_eq*b1_fac / (mu0 * ne * ECHARGE)
        dt_sc      = freq_res / dispfreq
        subcycles  = int(DT / dt_sc + 1)
        
        # Set subcycles to maximum, set timestep to match s/c loop length
        if subcycles > init_max_subcycle:
            print(f'Subcycles required ({subcycles}) greater than defined max ({init_max_subcycle})')
            print(f'Number of subcycles set at default init_max: {init_max_subcycle}')
            print('Resetting timestep to match subcycle loop size')
            DT = init_max_subcycle * dt_sc
            subcycles = init_max_subcycle
            
    else:
        subcycles = default_subcycles
        print('Number of subcycles set at default: {}'.format(subcycles))
    
    if part_dumpf == 0:
        part_save_iter = 1
    else:
        part_save_iter = int(part_dumpf / (DT*gyfreq_eq))
        if part_save_iter == 0: part_save_iter = 1

    if field_dumpf == 0:
        field_save_iter = 1
    else:
        field_save_iter = int(field_dumpf / (DT*gyfreq_eq))
        if field_save_iter == 0: field_save_iter = 1
        
    max_inc = int(max_time / DT) + 1
    if save_fields == 1 or save_particles == 1:
        store_run_parameters(DT, part_save_iter, field_save_iter, max_inc, max_time, subcycles)
    
    if DT < 1e-2:
        print('Timestep: %.3es with %d subcycles' % (DT, subcycles))
    else:
        print('Timestep: %.3fs with %d subcycles' % (DT, subcycles))
    print(f'{max_inc} iterations total\n')
    return DT, max_inc, part_save_iter, field_save_iter, subcycles


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
            for jj in nb.prange(3):
                for kk in nb.prange(3):
                    Ep[kk] += E[Ie[ii] + jj, kk] * W_elec[jj, ii]   
                    Bp[kk] += B[Ib[ii] + jj, kk] * W_mag[ jj, ii]   

            # Start Boris Method
            qmi = 0.5 * dt * qm_ratios[idx[ii]]                             # q/m variable including dt
            
            # vel -> v_minus
            vel[0, ii] += qmi * Ep[0]
            vel[1, ii] += qmi * Ep[1]
            vel[2, ii] += qmi * Ep[2]
            
            # Calculate background field at particle position (using v_minus)
            # Could probably make this more efficient for a=0
            Bp[0]    += B_eq
            constant  = 0.
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
    
    periodic_BC(pos, idx)
    assign_weighting_TSC(pos, Ie, W_elec)
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)
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
    
    Note: This function may only work because xmin = -xmax. Make more generic.
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
def get_max_vx(vel):
    return np.abs(vel[0]).max()


@nb.njit(parallel=do_parallel)
def get_max_v(vel):
    max_vx = np.abs(vel[0]).max()
    max_vy = np.abs(vel[1]).max()
    max_vz = np.abs(vel[2]).max()
    return max_vx, max_vy, max_vz

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
    nu_threads = np.zeros((n_threads, NC, Nj, 3), dtype=np.float64)
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
    '''
    nu_threads = np.zeros((n_threads, NC, Nj, 3), dtype=np.float64)
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
                         
    deposit_both_moments(vel, Ie, W_elec, idx, ni_init, nu_init)
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


#@nb.njit()
def advance_particles_and_collect_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E,
                    rho_int, rho_half, Ji, Ji_minus, Ji_plus, L, G, mp_flux, dt):
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
    
    rho_int[:] = rho_half[:] # Store pc(1/2) here while pc(3/2) is collected
    rho_half  *= 0.0
    Ji_minus  *= 0.0
    Ji_plus   *= 0.0
    L         *= 0.0
    G         *= 0.0   
    
    velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, dt)
    
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
        rho_half += ni[:, jj] * n_contr[jj] * charge[jj]
        L        += ni[:, jj] * n_contr[jj] * charge[jj] ** 2 / mass[jj]
        
        for kk in range(3):
            Ji_minus[:, kk] += nu_minus[:, jj, kk] * n_contr[jj] * charge[jj]
            Ji_plus[ :, kk] += nu_plus[ :, jj, kk] * n_contr[jj] * charge[jj]
            G[       :, kk] += nu_plus[ :, jj, kk] * n_contr[jj] * charge[jj] ** 2 / mass[jj]
        
    manage_source_term_boundaries(rho_half)
    manage_source_term_boundaries(L)
    for ii in range(3):
        manage_source_term_boundaries(Ji_minus[:, ii])
        manage_source_term_boundaries(Ji_plus[:, ii])
        manage_source_term_boundaries(G[:, ii])
        
    for ii in range(rho_half.shape[0]):
        if rho_half[ii] < min_dens * ne * ECHARGE:
            rho_half[ii] = min_dens * ne * ECHARGE  
            
    rho_int += rho_half
    rho_int /= 2.0
    Ji[:]    = 0.5 * (Ji_plus  +  Ji_minus)
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
@nb.njit(parallel=False)
def get_curl_B_4thOrder(B):
    '''
    Same as other function, but uses 4th order finite difference in the bulk.
    Gets deposited on the E-grid
    '''
    nc    = B.shape[0] - 1
    curlB = np.zeros((B.shape[0] - 1, 3), dtype=nb.float64)
    
    # Do 4th order for bulk
    for ii in nb.prange(1, B.shape[0] - 2):
        curlB[ii, 1] = -B[ii - 1, 2] + 27*B[ii, 2] - 27*B[ii + 1, 2] + B[ii + 2, 2]
        curlB[ii, 2] =  B[ii - 1, 1] - 27*B[ii, 1] + 27*B[ii + 1, 1] - B[ii + 2, 1]
    
    curlB /= 24.*dx
    
    # Do second order for interior points (LHS/RHS)
    curlB[0, 1] =  (-B[1, 2] + B[0, 2])/dx
    curlB[0, 2] =  ( B[1, 1] - B[0, 1])/dx
    
    curlB[nc - 1, 1] = (-B[nc, 2] + B[nc - 1, 2])/dx
    curlB[nc - 1, 2] = ( B[nc, 1] - B[nc - 1, 1])/dx
    return curlB


@nb.njit()
def get_curl_E_4thOrder(E, dE):
    ''' 
    Same as normal function, but 4th order solution for bulk. Gets dumped on B-grid.
    
    This is mega messy. Is this really necessary? Get your BCs under control, man.
    '''   
    nc  = E.shape[0] 
    dE *= 0.
    for ii in nb.prange(2, nc - 1):
        dE[ii, 1] = - E[ii - 2, 2] + 27*E[ii - 1, 2] - 27*E[ii, 2] + E[ii + 1, 2]
        dE[ii, 2] =   E[ii - 2, 1] - 27*E[ii - 1, 1] + 27*E[ii, 1] - E[ii + 1, 1]
    dE /= 24.
    
    # 2nd order solution for interior B points (not edge) (LHS/RHS)
    dE[1, 1] = (-E[1, 2] + E[0, 2])
    dE[1, 2] = ( E[1, 1] - E[0, 1])
    
    dE[nc - 1, 1] = (-E[nc - 1, 2] + E[nc - 2, 2])
    dE[nc - 1, 2] = ( E[nc - 1, 1] - E[nc - 2, 1])
        
    # Curl at E[0] : Forward/Backward difference (stored in B[0]/B[NC])
    dE[0, 1] = -(-3*E[0, 2] + 4*E[1, 2] - E[2, 2]) / 2
    dE[0, 2] =  (-3*E[0, 1] + 4*E[1, 1] - E[2, 1]) / 2
    
    dE[nc, 1] = -(3*E[nc - 1, 2] - 4*E[nc - 2, 2] + E[nc - 3, 2]) / 2
    dE[nc, 2] =  (3*E[nc - 1, 1] - 4*E[nc - 2, 1] + E[nc - 3, 1]) / 2
    
    # Linearly extrapolate to endpoints
    dE[0, 1]      -= 2*(dE[1, 1] - dE[0, 1])
    dE[0, 2]      -= 2*(dE[1, 2] - dE[0, 2])
    
    dE[nc, 1]     += 2*(dE[nc, 1] - dE[nc - 1, 1])
    dE[nc, 2]     += 2*(dE[nc, 2] - dE[nc - 1, 2])

    dE /= dx
    return


@nb.njit()
def get_electron_temp(qn, te):
    '''
    Calculate the electron temperature in each cell. Depends on the charge density of each cell
    and the treatment of electrons: i.e. isothermal (ie=0) or adiabatic (ie=1)
    '''
    gamma_e = 5./3. - 1.
    te[:] = Te0_scalar * np.power(qn / (ECHARGE*ne), gamma_e)
    return


@nb.njit(parallel=False)
def get_grad_P(qn, te):
    '''
    Returns the electron pressure gradient (in 1D) on the E-field grid using
    P = nkT and finite difference.
    
    INPUT:
        qn -- Grid charge density
        te -- Grid electron temperature
        
    NOTE: Interpolation is needed because the finite differencing causes the
    result to be deposited on the B-grid. Moving it back to the E-grid requires
    an interpolation.
    '''
    grad_pe       = np.zeros(NC    , dtype=np.float64)
    grad_pe_B     = np.zeros(NC + 1, dtype=np.float64)
    grad_pe[:]    = qn[:] * kB * te[:] / ECHARGE

    # Loop center points, set endpoints for no gradients (just to be safe)
    for ii in np.arange(1, qn.shape[0]):
        grad_pe_B[ii] = (grad_pe[ii] - grad_pe[ii - 1])/dx
    grad_pe_B[0]  = grad_pe_B[1]
    grad_pe_B[NC] = grad_pe_B[NC - 1]
    
    interpolate_cell_centre_4thOrder(grad_pe_B, grad_pe)
    return grad_pe


@nb.njit(parallel=False)
def apply_boundary(B):
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
def cyclic_leapfrog(B1, B2, B_center, rho, Ji, E, Ve, Te, dt, subcycles,
                    sim_time):
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
    '''
    H     = 0.5 * dt
    dh    = H / subcycles
    
    curl  = np.zeros((NC + 1, 3), dtype=np.float64)
    B2[:] = B1[:]

    ## DESYNC SECOND FIELD COPY - PUSH BY DH ##
    ## COUNTS AS ONE SUBCYCLE ##
    calculate_E(B1, B_center, Ji, rho, E, Ve, Te, sim_time)
    get_curl_E_4thOrder(E, curl) 
    B2       -= dh * curl
    apply_boundary(B2)
    get_B_cent(B2, B_center)
    sim_time += dh

    ## RETURN IF NO SUBCYCLES REQUIRED ##
    if subcycles == 1:
        B1[:] = B2[:]
        return sim_time+H
    
    ## MAIN SUBCYCLE LOOP ##
    ii = 1
    while ii < subcycles:
        if ii%2 == 1:
            calculate_E(B2, B_center, Ji, rho, E, Ve, Te, sim_time)
            get_curl_E_4thOrder(E, curl) 
            B1  -= 2 * dh * curl
            apply_boundary(B1)
            get_B_cent(B1, B_center)
            sim_time += dh
        else:
            calculate_E(B1, B_center, Ji, rho, E, Ve, Te, sim_time)
            get_curl_E_4thOrder(E, curl) 
            B2  -= 2 * dh * curl
            apply_boundary(B2)
            get_B_cent(B2, B_center)
            sim_time += dh
            
        ii += 1

    ## RESYNC FIELD COPIES ##
    ## DOESN'T COUNT AS A SUBCYCLE ##
    if ii%2 == 0:
        calculate_E(B2, B_center, Ji, rho, E, Ve, Te, sim_time)
        get_curl_E_4thOrder(E, curl) 
        B2  -= dh * curl
        apply_boundary(B2)
        get_B_cent(B2, B_center)
    else:
        calculate_E(B1, B_center, Ji, rho, E, Ve, Te, sim_time)
        get_curl_E_4thOrder(E, curl) 
        B1  -= dh * curl
        apply_boundary(B1)
        get_B_cent(B1, B_center)
    
    ## AVERAGE FOR OUTPUT ##
    B1 += B2; B1 /= 2.0
    
    # Calculate final values
    get_B_cent(B1, B_center)
    calculate_E(B1, B_center, Ji, rho, E, Ve, Te, sim_time)
    return sim_time


@nb.njit()
def calculate_E(B, B_center, Ji, qn, E, Ve, Te, sim_time):
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
    curlB  = get_curl_B_4thOrder(B)
    curlB /= mu0
       
    Ve[:, 0] = (Ji[:, 0] - curlB[:, 0]) / qn
    Ve[:, 1] = (Ji[:, 1] - curlB[:, 1]) / qn
    Ve[:, 2] = (Ji[:, 2] - curlB[:, 2]) / qn
    
    if ie == 1:
        get_electron_temp(qn, Te)
    del_p = get_grad_P(qn, Te)
    
    VexB     = np.zeros((NC, 3), dtype=np.float64)  
    for ii in np.arange(NC):
        VexB[ii, 0] = Ve[ii, 1] * B_center[ii, 2] - Ve[ii, 2] * B_center[ii, 1]
        VexB[ii, 1] = Ve[ii, 2] * B_center[ii, 0] - Ve[ii, 0] * B_center[ii, 2]
        VexB[ii, 2] = Ve[ii, 0] * B_center[ii, 1] - Ve[ii, 1] * B_center[ii, 0]
        
    E[:, 0]  = - VexB[:, 0] - del_p / qn
    E[:, 1]  = - VexB[:, 1]
    E[:, 2]  = - VexB[:, 2]

    # Copy edge cells
    E[ro1, :] = E[li1, :]
    E[ro2, :] = E[li2, :]
    E[lo1, :] = E[ri1, :]
    E[lo2, :] = E[ri2, :]
    
    # Fill remaining ghost cells
    E[:lo2, :] = E[lo2, :]
    E[ro2:, :] = E[ro2, :] 
    return


#%% AUXILLIARY FUNCTIONS
@nb.njit(parallel=False)
def get_B_cent(B, _B_cent):
    '''
    Quick and dirty linear interpolation so I have a working code
    But this is going to kill the order of my solutions
    Need at least a quadratic spline fit for true second-order solution
    
    Modified to use the higher-order interpolation
    '''
    interpolate_cell_centre_4thOrder(B[:, 1], _B_cent[:, 1])
    interpolate_cell_centre_4thOrder(B[:, 2], _B_cent[:, 2])
    return


@nb.njit(parallel=False)
def interpolate_cell_centre_4thOrder(edge_arr, interp):
    '''
    Uses equation derived in the TIMCOM model (looks like just a cubic spline?)
    
    http://coda.oc.ntu.edu.tw/coda/research/timcom/FRAME/fourth.html
    
    Does just a 1D array. Use 4th order interpolation on bulk values. 
    Use linear for edges (because easy)
    
    Seems to only be second order?
    '''
    nc = edge_arr.shape[0]-1
    
    for ii in nb.prange(1, nc-1):
        interp[ii] = -edge_arr[ii-1] + 7.*edge_arr[ii] + 7.*edge_arr[ii+1] - edge_arr[ii+2]
    interp /= 12.
    
    interp[0]    = 0.5*(edge_arr[0]  + edge_arr[1])
    interp[nc-1] = 0.5*(edge_arr[nc] + edge_arr[nc-1]) 
    return


#@nb.njit()
def check_timestep(qq, DT, pos, vel, idx, Ie, W_elec, Ib, W_mag, mp_flux, B, B_center, E, dns, 
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
    
    To do:
        -- Calculate number of required subcycles first. If greater than some
            predetermined limit, half timestep instead
    '''
    if manual_trip < 0: # Return without change or check
        return qq, DT, max_inc, part_save_iter, field_save_iter, loop_save_iter, 0, subcycles
    
    max_vx, max_vy, max_vz = get_max_v(vel)
    max_V = max(max_vx, max_vy, max_vz)
    
    B_tot           = np.sqrt(B_center[:, 0] ** 2 + B_center[:, 1] ** 2 + B_center[:, 2] ** 2)
    high_rat        = qm_ratios.max()
    
    local_gyfreq    = high_rat  * np.abs(B_tot).max()      
    ion_ts          = orbit_res / local_gyfreq
    
    if E[:, 0].max() != 0:
        elecfreq    = high_rat * (np.abs(E[:, 0] / max_V)).max()
        freq_ts     = freq_res / elecfreq                            
    else:
        freq_ts     = ion_ts
    
    vel_ts          = 0.75*dx / max_vx
    DT_part         = min(freq_ts, vel_ts, ion_ts)
    
    # Check subcycles to see if DT_part needs to be changed instead
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
            
        if subcycles > max_subcycles:
            subcycles = max_subcycles
            print(f'Number of subcycles exceeding maximum, setting to {max_subcycles}')
            print( 'Modifying timestep...')
            DT_part = 0.5*DT
    
    # Reduce timestep
    change_flag       = 0
    if (DT_part < 0.9*DT and manual_trip == 0) or manual_trip == 1:
        #(pos, vel, idx, Ie, W_elec, Ib, W_mag, mp_flux, dt, hot_only=False)
        position_update(pos, vel, idx, Ie, W_elec, Ib, W_mag, mp_flux, -0.5*DT)
        
        change_flag      = 1
        DT              *= 0.5
        max_inc         *= 2
        qq              *= 2
        part_save_iter  *= 2
        field_save_iter *= 2
        loop_save_iter  *= 2
        if DT < 1e-2:
            print('Timestep halved to: %.3es with %d subcycles' % (DT, subcycles))
        else:
            print('Timestep halved to: %.3fs with %d subcycles' % (DT, subcycles))
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
    Bc[:, 0] = B_eq
    
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
                   ('B_xmax', B_eq),
                   ('a', 0),
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
                   ('resis_multiplier', 0),
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
                   ('damping_multiplier', 0),
                   ('damping_fraction', 0),
                   ('pol_wave', 0),
                   ('driven_freq', 0),
                   ('driven_ampl', 0),
                   ('driven_k', 0),
                   ('pulse_offset', 0),
                   ('pulse_width', 0),
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


def save_field_data(dt, field_save_iter, qq, Ji, E, B, Ve, Te, dns, sim_time):

    d_path = '%s/%s/run_%d/data/fields/' % (drive, save_path, run_num)
    r      = qq / field_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, E=E, B=B, Ji=Ji, dns=dns, Ve=Ve, Te=Te,
                         sim_time=sim_time,
                         damping_array=None,
                         resistive_array=None)
    return
    

def save_particle_data(dt, part_save_iter, qq, sim_time, 
                       pos, vel, idx, Ji, E, B, Ve, Te, dns, 
                       damping_array, resistive_array):
    d_path = '%s/%s/run_%d/data/particles/' % (drive, save_path, run_num)
    r      = qq / part_save_iter

    d_filename = 'data%05d' % r
    d_fullpath = os.path.join(d_path, d_filename)
    np.savez(d_fullpath, pos=pos, vel=vel, idx=idx, sim_time=sim_time,
             E=E, B=B, Ji=Ji, dns=dns, Ve=Ve, Te=Te,
             damping_array=None,
             resistive_array=None)
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


def load_run_params():
    global drive, save_path, run_num, save_particles, save_fields, seed, homogenous, particle_periodic,\
        particle_reflect, particle_reinit, field_periodic, disable_waves, source_smoothing, quiet_start,\
        NX, max_wcinv, dxm, ie, E_damping, damping_fraction, resis_multiplier, \
            orbit_res, freq_res, part_dumpf, field_dumpf, run_description, particle_open, ND, NC,\
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
        part_dumpf  = float(f.readline().split()[1])         # Data capture resolution in gyroperiod fraction: Particle information
        field_dumpf = float(f.readline().split()[1])         # Data capture resolution in gyroperiod fraction: Field information
    
        run_description = f.readline()                     # Commentary to attach to runs, helpful to have a quick description
    
    particle_periodic = 1
    field_periodic = 1
    
    # Set number of ghost cells, count total
    ND = 2
    NC = NX + 2*ND
    
    # E-field nodes around boundaries (used for sources and E-fields)
    lo1 = ND - 1 ; lo2 = ND - 2             # Left outer (to boundary)
    ro1 = ND + NX; ro2 = ND + NX + 1        # Right outer
    
    li1 = ND         ; li2 = ND + 1         # Left inner
    ri1 = ND + NX - 1; ri2 = ND + NX - 2    # Right inner
        
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
            k_max, inv_dx, inv_mu0
        
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
    
    charge    *= ECHARGE                                     # Cast species charge to Coulomb
    mass      *= PMASS                                       # Cast species mass to kg
    qm_ratios  = np.divide(charge, mass)                     # q/m ratio for each species
    
    # DOUBLE ARRAYS THAT MIGHT BE ACCESSED BY DEACTIVATED PARTICLES
    qm_ratios  = np.concatenate((qm_ratios, qm_ratios))
    temp_type  = np.concatenate((temp_type, temp_type))

    mass_dens  = (mass*density).sum()
    va         = B_eq / np.sqrt(mu0*mass_dens)
    ne         = density.sum()                               # Electron number density
    wpi        = np.sqrt((density * charge ** 2 / (mass * e0)).sum())            # Proton Plasma Frequency, wpi (rad/s)
    wpe        = np.sqrt(ne * ECHARGE ** 2 / (EMASS * e0))   # Electron Plasma Frequency, wpi (rad/s)
    gyfreq_eq  = ECHARGE*B_eq  / PMASS                       # Proton Gyrofrequency (rad/s) at equator (slowest)
    egyfreq_eq = ECHARGE*B_eq  / EMASS                       # Electron Gyrofrequency (rad/s) at equator (slowest)
    dx         = dxm * va / gyfreq_eq                        # Alternate method of calculating dx (better for multicomponent plasmas)
    n_contr    = density / nsp_ppc                           # Species density contribution: Each macroparticle contributes this SI density to a cell
    min_dens   = 0.05                                        # Minimum charge density in a cell
    xmax       = NX / 2 * dx                                 # Maximum simulation length, +/-ve on each side
    xmin       =-NX / 2 * dx
    k_max      = np.pi / dx                                  # Maximum permissible wavenumber in system (SI???)

    # Inverse values so that I can avoid divisions
    inv_dx  = 1./dx
    inv_mu0 = 1./mu0 

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

    N = N_species.sum()
    
    idx_start  = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
    idx_end    = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order
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
    
    print('Equat. Gyroperiod: : {}s'.format(round(2. * np.pi / gyfreq_eq, 3)))
    print('Inverse rad gyfreq : {}s'.format(round(1 / gyfreq_eq, 3)))
    print('Maximum sim time   : {}s ({} gyroperiods)\n'.format(round(max_wcinv / gyfreq_eq, 2), 
                                                               round(max_wcinv/(2*np.pi), 2)))    
    print('{} spatial cells, 2x{} damped cells'.format(NX, ND))
    print('{} cells total'.format(NC))
    print('{} particles total\n'.format(N))
            
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
parser.add_argument('-r', '--runfile'     , default='_run_params.run', type=str)
parser.add_argument('-p', '--plasmafile'  , default='_plasma_params.plasma', type=str)
parser.add_argument('-d', '--driverfile'  , default='_driver_params.txt'   , type=str)
parser.add_argument('-n', '--run_num'     , default=-1, type=int)
parser.add_argument('-s', '--subcycle'    , default=16, type=int)
parser.add_argument('-m', '--max_subcycle', default=256, type=int)
parser.add_argument('-M', '--init_max_subcycle', default=16, type=int)
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

# Set anything else useful before file input load
default_subcycles = args['subcycle']
max_subcycles     = args['max_subcycle']
init_max_subcycle = args['init_max_subcycle']

load_run_params()
if __name__ == '__main__':
    manage_directories()
load_plasma_params()
get_thread_values()



#%%#####################
### START SIMULATION ###
########################
if __name__ == '__main__':
    print_summary_and_checks()

    start_time = timer()
    
    _B, _B2, _B_CENT, _E, _VE, _TE = initialize_fields()
    _RHO_HALF, _RHO_INT, _Ji,      \
    _Ji_PLUS, _Ji_MINUS,    \
    _L, _G, _MP_FLUX               = initialize_source_arrays()
    _POS, _VEL, _IE, _W_ELEC, _IB, \
    _W_MAG, _IDX                   = initialize_particles(_B, _E, _MP_FLUX)
    _DT, _MAX_INC, _PART_SAVE_ITER,\
    _FIELD_SAVE_ITER, _SUBCYCLES   = set_timestep(_VEL)    
        
    print('Loading initial state...')
    init_collect_moments(_POS, _VEL, _IE, _W_ELEC, _IB, _W_MAG, _IDX, _RHO_INT, _RHO_HALF,
                         _Ji, _Ji_PLUS, _L, _G, _MP_FLUX, 0.5*_DT)
    get_B_cent(_B, _B_CENT)
    calculate_E(_B, _B_CENT, _Ji, _RHO_HALF, _E, _VE, _TE, 0.0)
    
    print('Saving initial state...\n')
    if save_particles == 1:
        save_particle_data(_DT, _PART_SAVE_ITER, 0, 0.0, _POS, _VEL, _IDX,
                           _Ji, _E, _B, _VE, _TE, _RHO_INT)

    if save_fields == 1:
        save_field_data(_DT, _FIELD_SAVE_ITER, 0, _Ji, _E, _B, _VE, _TE,
                        _RHO_INT, 0.0)

    _LOOP_TIMES     = np.zeros(_MAX_INC-1, dtype=float)
    _LOOP_SAVE_ITER = 1

    _QQ = 1; _SIM_TIME = 0.0
    print('Starting loop...')
    while _QQ < _MAX_INC:
        
        if adaptive_timestep == 1 and disable_waves == 0:  
            _QQ, _DT, _MAX_INC, _PART_SAVE_ITER, _FIELD_SAVE_ITER, _LOOP_SAVE_ITER, _CHANGE_FLAG, _SUBCYCLES =\
                check_timestep(_QQ, _DT, _POS, _VEL, _IDX, _IE, _W_ELEC, _IB, _W_MAG, _MP_FLUX,
                               _B, _B_CENT, _E, _RHO_INT, 
                               _MAX_INC, _PART_SAVE_ITER, _FIELD_SAVE_ITER, _LOOP_SAVE_ITER,
                               _SUBCYCLES)
            
            # Collect new moments and desync position and velocity. Reset damping array.
            if _CHANGE_FLAG == 1:
                # If timestep was doubled, do I need to consider 0.5dt's worth of
                # new particles? Maybe just disable the doubling until I work this out
                init_collect_moments(_POS, _VEL, _IE, _W_ELEC, _IB, _W_MAG, _IDX,  
                         _RHO_INT, _RHO_HALF, _Ji, _Ji_PLUS, _L, _G, _MP_FLUX, 0.5*_DT)
        
        #######################
        ###### MAIN LOOP ######
        #######################
        
        # First field advance to N + 1/2
        _SIM_TIME = cyclic_leapfrog(_B, _B2, _B_CENT, _RHO_INT, _Ji, _E, _VE, _TE,
                                    _DT, _SUBCYCLES, _SIM_TIME)

        # CAM part
        push_current(_Ji_PLUS, _Ji, _E, _B_CENT, _L, _G, _DT)
        calculate_E(_B, _B_CENT, _Ji, _RHO_HALF,
                    _E, _VE, _TE, _SIM_TIME)
        
        # Particle advance, moment calculation
        advance_particles_and_collect_moments(_POS, _VEL, _IE, _W_ELEC, _IB, _W_MAG,
                                              _IDX, _B, _E, _RHO_INT, _RHO_HALF, _Ji,
                                              _Ji_MINUS, _Ji_PLUS, _L, _G, _MP_FLUX, _DT)
        
        # Second field advance to N + 1
        _SIM_TIME = cyclic_leapfrog(_B, _B2, _B_CENT, _RHO_INT, _Ji, _E, _VE, _TE,
                                    _DT, _SUBCYCLES, _SIM_TIME)
        
        ########################
        ##### OUTPUT DATA  #####
        ########################
        if _QQ%_PART_SAVE_ITER == 0 and save_particles == 1:
            save_particle_data(_DT, _PART_SAVE_ITER, _QQ, _SIM_TIME, _POS, _VEL, _IDX,
                               _Ji, _E, _B, _VE, _TE, _RHO_INT)

        if _QQ%_FIELD_SAVE_ITER == 0 and save_fields == 1:
            save_field_data(_DT, _FIELD_SAVE_ITER, _QQ, _Ji, _E, _B, _VE, _TE,
                            _RHO_INT, _SIM_TIME)
        
        if _QQ%100 == 0 and print_runtime:
            running_time = int(timer() - start_time)
            hrs          = running_time // 3600
            rem          = running_time %  3600
            
            mins         = rem // 60
            sec          = rem %  60
            
            pcent = round(float(_QQ) / float(_MAX_INC) * 100., 2)
            print('{:5.2f}% :: Step {} of {} :: Current runtime {:02}:{:02}:{:02}'.format(
                                                   pcent, _QQ, _MAX_INC, hrs, mins, sec))
            
        _QQ += 1
        
    runtime = round(timer() - start_time,2) 
    print('Run complete : {} s'.format(runtime))