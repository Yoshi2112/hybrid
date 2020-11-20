## PYTHON MODULES ##
import numpy as np
import numba as nb
import os, sys, pdb
import pickle
from shutil import rmtree
from timeit import default_timer as timer

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

### ##
### INITIALIZATION
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
    Takes in a Maxwellian or pseudo-maxwellian distribution. Outputs the number
    and indexes of any particle inside the loss cone
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
def uniform_gaussian_distribution_quiet():
    '''Creates an N-sampled normal distribution across all particle species within each simulation cell

    OUTPUT:
        pos -- 1xN array of particle positions. Pos[0] is uniformly distributed with boundaries depending on its temperature type
        vel -- 3xN array of particle velocities. Each component initialized as a Gaussian with a scale factor determined by the species perp/para temperature
        idx -- N   array of particle indexes, indicating which species it belongs to. Coded as an 8-bit signed integer, allowing values between +/-128
    
    New code: Removed all 3D position things because we won't need it for long. Check this later, since its easy to change
            Also removed all references to dist_type since initializing particles in the middle is stupid.
    '''
    pos = np.zeros(N, dtype=np.float64)
    vel = np.zeros((3, N), dtype=np.float64)
    idx = np.ones(N,       dtype=np.int8) * -1
    np.random.seed(seed)

    for jj in range(Nj):
        idx[idx_start[jj]: idx_end[jj]] = jj          # Set particle idx        
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
            pos[st: en]-= NC_load*dx/2              
            
            # Set velocity for half: Randomly Maxwellian
            vel[0, st: en] = np.random.normal(0, vth_par[ jj], half_n)  
            vel[1, st: en] = np.random.normal(0, vth_perp[jj], half_n)
            vel[2, st: en] = np.random.normal(0, vth_perp[jj], half_n)

            # Set Loss Cone Distribution: Reinitialize particles in loss cone (move to a function)
            if homogenous == 0 and temp_type[jj] == 1:
                LCD_by_rejection(pos, vel, vth_par[jj], vth_perp[jj], st, en, jj)
                
            # Quiet start : Initialize second half
            if quiet_start == 1:
                vel[0, en: en + half_n] = vel[0, st: en] *  1.0     # Set parallel
            else:
                vel[0, en: en + half_n] = vel[0, st: en] * -1.0     # Set anti-parallel
                
            pos[en: en + half_n]    = pos[st: en]                   # Other half, same position
            vel[1, en: en + half_n] = vel[1, st: en] * -1.0         # Invert perp velocities (v2 = -v1)
            vel[2, en: en + half_n] = vel[2, st: en] * -1.0
            
            vel[0, st: en + half_n] += drift_v[jj] * va             # Add drift offset
            
            acc                    += half_n * 2
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
    
    Ie         = np.zeros(N,      dtype=nb.uint16)
    Ib         = np.zeros(N,      dtype=nb.uint16)
    W_elec     = np.zeros((3, N), dtype=nb.float64)
    W_mag      = np.zeros((3, N), dtype=nb.float64)
    
    Bp         = np.zeros((3, N), dtype=nb.float64)
    Ep         = np.zeros((3, N), dtype=nb.float64)
    temp_N     = np.zeros((N),    dtype=nb.float64)
    
    assign_weighting_TSC(pos, Ie, W_elec)
    return pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, temp_N


@nb.njit()
def set_damping_array(B_damping_array, E_damping_array, DT):
    '''Create masking array for magnetic field damping used to apply open
    boundaries. Based on applcation by Shoji et al. (2011) and
    Umeda et al. (2001)
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
        ni      -- Ion number density per species
        nu      -- Ion velocity "density" per species
    '''
    q_dens  = np.zeros( NC,         dtype=nb.float64)    
    q_dens2 = np.zeros( NC,         dtype=nb.float64) 
    Ji      = np.zeros((NC, 3),     dtype=nb.float64)
    ni      = np.zeros((NC, Nj),    dtype=nb.float64)
    nu      = np.zeros((NC, Nj, 3), dtype=nb.float64)
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
        mp_flux       -- Tracking variable designed to accrue the flux at each timestep (in terms of macroparticles
                             at each boundary and for each species) and trigger an injection if >= 2.
    '''
    temp3Db       = np.zeros((NC + 1, 3),  dtype=nb.float64)
    temp3De       = np.zeros((NC    , 3),  dtype=nb.float64)
    temp1D        = np.zeros( NC    ,      dtype=nb.float64) 
    old_fields    = np.zeros((NC + 1, 10), dtype=nb.float64)
    
    v_prime = np.zeros((3, N),      dtype=nb.float64)
    S       = np.zeros((3, N),      dtype=nb.float64)
    T       = np.zeros((3, N),      dtype=nb.float64)
        
    old_particles = np.zeros((9, N),      dtype=nb.float64)
    mp_flux       = np.zeros((2, Nj),     dtype=nb.float64)
        
    return old_particles, old_fields, temp3De, temp3Db, temp1D, v_prime, S, T, mp_flux


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
    ion_ts   = orbit_res / gyfreq               # Timestep to resolve gyromotion
    vel_ts   = 0.5 * dx / np.max(np.abs(vel[0, :]))   # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step 
    
    DT       = min(ion_ts, vel_ts)
    max_time = max_rev * 2 * np.pi / gyfreq_eq     # Total runtime in seconds
    max_inc  = int(max_time / DT) + 1                          # Total number of time steps

    gyperiod = 2 * np.pi / gyfreq
    if part_res == 0:
        part_save_iter = 1
    else:
        part_save_iter = int(part_res*gyperiod / DT)

    if field_res == 0:
        field_save_iter = 1
    else:
        field_save_iter = int(field_res*gyperiod / DT)

    if save_fields == 1 or save_particles == 1:
        store_run_parameters(DT, part_save_iter, field_save_iter, Te0)

    B_damping_array = np.ones(NC + 1, dtype=float)
    E_damping_array = np.ones(NC    , dtype=float)
    set_damping_array(B_damping_array, E_damping_array, DT)

    print('Timestep: %.4fs, %d iterations total\n' % (DT, max_inc))
    return DT, max_inc, part_save_iter, field_save_iter, B_damping_array, E_damping_array

### ##
### PARTICLES
### ##
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
    
    particle_transform = xmax + (ND - grid_offset)*dx  + epsil  # Offset to account for E/B grid and damping nodes
    
    if field_periodic == 0:
        for ii in np.arange(Np):
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
        for ii in np.arange(Np):
            xp          = (pos[ii] + particle_transform) / dx       # Shift particle position >= 0
            I[ii]       = int(round(xp) - 1.0)                      # Get leftmost to nearest node (Vectorize?)
            delta_left  = I[ii] - xp                                # Distance from left node in grid units

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
    Bp[0]    = eval_B0x(pos)  
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
    for kk in range(2):
        mp_flux[kk, :] += inject_rate*DT
         
    # Import Particle boundary conditions: Re-initialize if at edges
    for ii in nb.prange(pos.shape[0]):
        if (pos[ii] < xmin or pos[ii] > xmax):
            if particle_reinit == 1: 
                
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
                n_deleted = 0
                for ii in nb.prange(pos.shape[0]):
                    if (pos[ii] < xmin or pos[ii] > xmax):
                        pos[ii]    *= 0.0
                        vel[:, ii] *= 0.0
                        idx[ii]     = -1
                        n_deleted  += 1
        
    # Put this into its own function later? Don't bother for now.
    if particle_open == 1:
        acc = 0; n_created = 0
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
                    n_created       += 2
                    
    #print('Number created:', n_created)
    assign_weighting_TSC(pos, Ie, W_elec)
    return

### ##
### SOURCES
### ##
@nb.njit()
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
    '''
    for ii in nb.prange(vel.shape[1]):
        I   = Ie[ii]
        sp  = idx[ii]
        
        if sp >= 0:
            for kk in range(3):
                nu[I,     sp, kk] += W_elec[0, ii] * vel[kk, ii]
                nu[I + 1, sp, kk] += W_elec[1, ii] * vel[kk, ii]
                nu[I + 2, sp, kk] += W_elec[2, ii] * vel[kk, ii]
            
            ni[I,     sp] += W_elec[0, ii]
            ni[I + 1, sp] += W_elec[1, ii]
            ni[I + 2, sp] += W_elec[2, ii]
    return


@nb.njit()
def collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu):
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
    '''
    q_dens *= 0.
    Ji     *= 0.
    ni     *= 0.
    nu     *= 0.
    
    deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu)

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


### ##
### FIELDS
### ##

@nb.njit()
def eval_B0x(x):
    return B_eq * (1. + a * x**2)


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
def get_electron_temp(qn, Te, Te0):
    '''
    Calculate the electron temperature in each cell. Depends on the charge density of each cell
    and the treatment of electrons: i.e. isothermal (ie=0) or adiabatic (ie=1)
    '''
    if ie == 0:
        Te[:] = np.ones(qn.shape[0]) * Te0
    elif ie == 1:
        gamma_e = 5./3. - 1.
        Te[:] = Te0 * np.power(qn / (q*ne), gamma_e)
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
def calculate_E(B, Ji, q_dens, E, Ve, Te, Te0, temp3De, temp3Db, grad_P, E_damping_array):
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
    '''
    curl_B_term(B, temp3De)                                   # temp3De is now curl B term

    Ve[:, 0] = (Ji[:, 0] - temp3De[:, 0]) / q_dens
    Ve[:, 1] = (Ji[:, 1] - temp3De[:, 1]) / q_dens
    Ve[:, 2] = (Ji[:, 2] - temp3De[:, 2]) / q_dens

    get_electron_temp(q_dens, Te, Te0)

    get_grad_P(q_dens, Te, grad_P, temp3Db[:, 0])             # temp1D is now del_p term, temp3D2 slice used for computation
    interpolate_edges_to_center(B, temp3Db)                   # temp3db is now B_center

    cross_product(Ve, temp3Db[:temp3Db.shape[0]-1, :], temp3De)                  # temp3De is now Ve x B term
    if E_damping == 1 and field_periodic == 0:
        temp3De *= E_damping_array
    
    E[:, 0]  = - temp3De[:, 0] - grad_P[:] / q_dens[:]
    E[:, 1]  = - temp3De[:, 1]
    E[:, 2]  = - temp3De[:, 2]
        
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
### AUXILLIARY FUNCTIONS 
### ##
@nb.njit()
def cross_product(A, B, C):
    '''
    Vector (cross) product between two vectors, A and B of same dimensions.

    INPUT:
        A, B -- 3D vectors (ndarrays)

    OUTPUT:
        C -- The resultant cross product with same dimensions as input vectors
    '''
    C[:, 0] += A[:, 1] * B[:, 2]
    C[:, 1] += A[:, 2] * B[:, 0]
    C[:, 2] += A[:, 0] * B[:, 1]
    
    C[:, 0] -= A[:, 2] * B[:, 1]
    C[:, 1] -= A[:, 0] * B[:, 2]
    C[:, 2] -= A[:, 1] * B[:, 0]
    return


@nb.njit()
def interpolate_edges_to_center(B, interp, zero_boundaries=False):
    ''' 
    Used for interpolating values on the B-grid to the E-grid (for E-field calculation)
    with a 3D array (e.g. B). Second derivative y2 is calculated on the B-grid, with
    forwards/backwards difference used for endpoints.
    
    interp has one more gridpoint than required just because of the array used. interp[-1]
    should remain zero.
    
    This might be able to be done without the intermediate y2 array since the interpolated
    points don't require previous point values.
    
    ADDS B0 TO X-AXIS ON TOP OF INTERPOLATION
    '''
    y2      = np.zeros(B.shape, dtype=nb.float64)
    interp *= 0.
    
    # Calculate second derivative
    for jj in range(1, B.shape[1]):
        
        # Interior B-nodes, Centered difference
        for ii in range(1, NC):
            y2[ii, jj] = B[ii + 1, jj] - 2*B[ii, jj] + B[ii - 1, jj]
                
        # Edge B-nodes, Forwards/Backwards difference
        if zero_boundaries == True:
            y2[0 , jj] = 0.
            y2[NC, jj] = 0.
        else:
            y2[0,  jj] = 2*B[0 ,    jj] - 5*B[1     , jj] + 4*B[2     , jj] - B[3     , jj]
            y2[NC, jj] = 2*B[NC,    jj] - 5*B[NC - 1, jj] + 4*B[NC - 2, jj] - B[NC - 3, jj]
        
    # Do spline interpolation: E[ii] is bracketed by B[ii], B[ii + 1]
    for jj in range(1, B.shape[1]):
        for ii in range(NC):
            interp[ii, jj] = 0.5 * (B[ii, jj] + B[ii + 1, jj] + (1/6) * (y2[ii, jj] + y2[ii + 1, jj]))
    
    # Add B0x to interpolated array
    for ii in range(NC):
        interp[ii, 0] = eval_B0x(E_nodes[ii])
    
    # This bit could be removed to allow B0x to vary in green cells naturally
    # interp[:ND,      0] = interp[ND,    0]
    # interp[ND+NX+1:, 0] = interp[ND+NX, 0]
    return


@nb.njit()
def check_timestep(pos, vel, B, E, q_dens, Ie, W_elec, Ib, W_mag, B_center, Ep, Bp, v_prime, S, T, temp_N,\
                     qq, DT, max_inc, part_save_iter, field_save_iter, idx, damping_array):
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
    '''
    interpolate_edges_to_center(B, B_center)
    B_magnitude     = np.sqrt(B_center[ND:ND+NX+1, 0] ** 2 +
                              B_center[ND:ND+NX+1, 1] ** 2 +
                              B_center[ND:ND+NX+1, 2] ** 2)
    gyfreq          = qm_ratios.max() * B_magnitude.max()     
    ion_ts          = orbit_res / gyfreq
    
    if E[:, 0].max() != 0:
        elecfreq        = qm_ratios.max()*(np.abs(E[:, 0] / np.abs(vel).max()).max())               # Electron acceleration "frequency"
        Eacc_ts         = freq_res / elecfreq                            
    else:
        Eacc_ts = ion_ts

    vel_ts          = 0.60 * dx / np.abs(vel[0, :]).max()                        # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than 'half' a cell in one time step
    DT_part         = min(Eacc_ts, vel_ts, ion_ts)                      # Smallest of the allowable timesteps
    
    if DT_part < 0.9*DT:

        velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E, v_prime, S, T,temp_N,0.5*DT)    # Re-sync vel/pos       

        DT         *= 0.5
        max_inc    *= 2
        qq         *= 2
        
        field_save_iter *= 2
        part_save_iter *= 2

        velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E, v_prime, S, T,temp_N,-0.5*DT)   # De-sync vel/pos 
        print('Timestep halved. Syncing particle velocity...')

    return qq, DT, max_inc, part_save_iter, field_save_iter, damping_array


### ##
### SAVE ROUTINES 
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
                sys.exit('Program Terminated: Change run in simulation_parameters_1D')
            else:
                sys.exit('Unfamiliar input: Run terminated for safety')
    return


def store_run_parameters(dt, part_save_iter, field_save_iter, Te0):
    d_path = ('%s/%s/run_%d/data/' % (drive, save_path, run))    # Set path for data
    f_path = d_path + '/fields/'
    p_path = d_path + '/particles/'
    
    manage_directories()

    for folder in [d_path, f_path, p_path]:
        if os.path.exists(folder) == False:                               # Create data directories
            os.makedirs(folder)
    
    Bc       = np.zeros((NC + 1, 3), dtype=np.float64)
    Bc[:, 0] = B_eq * (1 + a * B_nodes**2)

    # Single parameters
    params = dict([('seed', seed),
                   ('Nj', Nj),
                   ('dt', dt),
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
                   ('loss_cone_xmax', loss_cone_xmax),
                   ('r_A', r_A),
                   ('lat_A', lat_A),
                   ('B_A', B_A),
                   ('rc_hwidth', rc_hwidth),
                   ('ne', ne),
                   ('Te0', Te0_scalar),
                   ('ie', ie),
                   ('part_save_iter', part_save_iter),
                   ('field_save_iter', field_save_iter),
                   ('max_rev', max_rev),
                   ('freq_res', freq_res),
                   ('orbit_res', orbit_res),
                   ('run_desc', run_description),
                   ('method_type', 'PREDCORR_PARABOLIC_ONESCRIPT'),
                   ('particle_shape', 'TSC'),
                   ('field_periodic', field_periodic),
                   ('run_time', None),
                   ('homogeneous', homogenous),
                   ('particle_periodic', particle_periodic),
                   ('particle_reflect', particle_reflect),
                   ('particle_reinit', particle_reinit),
                   ('disable_waves', disable_waves),
                   ('source_smoothing', source_smoothing),
                   ('E_damping', E_damping),
                   ('quiet_start', quiet_start)
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
                     Tpar        = None,
                     Tperp       = None,
                     vth_par     = vth_par,
                     vth_perp    = vth_perp,
                     Bc          = Bc,
                     Te0         = Te0)
    print('Particle data saved')
    return


def save_field_data(sim_time, dt, field_save_iter, qq, Ji, E, B, Ve, Te, dns, damping_array, E_damping_array):
    d_path   = '%s/%s/run_%d/data/fields/' % (drive, save_path, run)
    r        = qq / field_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, E = E[:, 0:3], B = B[:, 0:3],   J = Ji[:, 0:3],
                       dns = dns,      Ve = Ve[:, 0:3], Te = Te, sim_time = sim_time,
                       damping_array = damping_array, E_damping_array=E_damping_array)
    print('Field data saved')
    
    
def save_particle_data(sim_time, dt, part_save_iter, qq, pos, vel, idx):
    d_path   = '%s/%s/run_%d/data/particles/' % (drive, save_path, run)
    r        = qq / part_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, pos = pos, vel = vel, idx=idx, sim_time = sim_time)
    print('Particle data saved')
    
    
def add_runtime_to_header(runtime):
    d_path = ('%s/%s/run_%d/data/' % (drive, save_path, run))     # Data path
    
    h_name = os.path.join(d_path, 'simulation_parameters.pckl')         # Header file path
    f      = open(h_name, 'rb')                                         # Open header file
    params = pickle.load(f)                                             # Load variables from header file into dict
    f.close()  
    
    params['run_time'] = runtime
    
    # Re-save
    with open(d_path + 'simulation_parameters.pckl', 'wb') as f:
        pickle.dump(params, f)
        f.close()
        print('Run time appended to simulation header file')
    return


def dump_to_file(pos, vel, E, Ve, Te, B, Ji, rho, qq, suff='', print_particles=False):
    import os
    np.set_printoptions(threshold=sys.maxsize)
    
    dirpath = drive + save_path + '/run_{}/ts_{:05}/'.format(run, qq, suff) 
    if os.path.exists(dirpath) == False:
        os.makedirs(dirpath)
        
    print('Dumping arrays to file')
    if print_particles == True:
        with open(dirpath + 'pos{}.txt'.format(suff), 'w') as f:
            print(pos, file=f)
        with open(dirpath + 'vel{}.txt'.format(suff), 'w') as f:
            print(vel, file=f)
    with open(dirpath + 'E{}.txt'.format(suff), 'w') as f:
        print(E, file=f)
    with open(dirpath + 'Ve{}.txt'.format(suff), 'w') as f:
        print(Ve, file=f)
    with open(dirpath + 'Te{}.txt'.format(suff), 'w') as f:
        print(Te, file=f)
    with open(dirpath + 'B{}.txt'.format(suff), 'w') as f:
        print(B, file=f)
    with open(dirpath + 'J{}.txt'.format(suff), 'w') as f:
        print(Ji, file=f)

    np.set_printoptions(threshold=1000)
    return

### ##
### MAIN LOOP
### ##
@nb.njit()
def main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag, Ep, Bp, v_prime, S, T,temp_N,                      \
              B, E_int, E_half, q_dens, q_dens_adv, Ji, ni, nu, mp_flux,       \
              Ve, Te, Te0, temp3De, temp3Db, temp1D, old_particles, old_fields,\
              B_damping_array, E_damping_array, qq, DT, max_inc, part_save_iter, field_save_iter):
    '''
    Main loop separated from __main__ function, since this is the actual computation bit.
    Could probably be optimized further, but I wanted to njit() it.
    The only reason everything's not njit() is because of the output functions.
    
    Future: Come up with some way to loop until next save point
    
    Thoughts: declare a variable steps_to_go. Output all time variables at return
    to resync everything, and calculate steps to next stop.
    If no saves, steps_to_go = max_inc
    '''
    # Check timestep
    if adaptive_timestep == True:
        qq, DT, max_inc, part_save_iter, field_save_iter, damping_array \
        = check_timestep(pos, vel, B, E_int, q_dens, Ie, W_elec, Ib, W_mag, temp3De, Ep, Bp, v_prime, S, T,temp_N,\
                         qq, DT, max_inc, part_save_iter, field_save_iter, idx, B_damping_array)
            
    # Move particles, collect moments
    advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, v_prime, S, T,temp_N,\
                                            B, E_int, DT, q_dens_adv, Ji, ni, nu, mp_flux)
    
    # Average N, N + 1 densities (q_dens at N + 1/2)
    q_dens *= 0.5
    q_dens += 0.5 * q_dens_adv
    
    # Push B from N to N + 1/2
    push_B(B, E_int, temp3Db, DT, qq, B_damping_array, half_flag=1)
    
    # Calculate E at N + 1/2
    calculate_E(B, Ji, q_dens, E_half, Ve, Te, Te0, temp3De, temp3Db, temp1D, E_damping_array)

    ###################################
    ### PREDICTOR CORRECTOR SECTION ###
    ###################################

    # Store old values
    mp_flux_old           = mp_flux.copy()
    old_particles[0  , :] = pos
    old_particles[1:4, :] = vel
    old_particles[4  , :] = Ie
    old_particles[5:8, :]    = W_elec
    old_particles[8  , :]  = idx
    
    old_fields[:,   0:3]  = B
    old_fields[:NC, 3:6]  = Ji
    old_fields[:NC, 6:9]  = Ve
    old_fields[:NC,   9]  = Te
    
    # Predict fields
    E_int *= -1.0
    E_int +=  2.0 * E_half
    
    push_B(B, E_int, temp3Db, DT, qq, B_damping_array, half_flag=0)

    # Advance particles to obtain source terms at N + 3/2
    advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, v_prime, S, T,temp_N,\
                                            B, E_int, DT, q_dens, Ji, ni, nu, mp_flux, pc=1)
    
    q_dens *= 0.5;    q_dens += 0.5 * q_dens_adv
    
    # Compute predicted fields at N + 3/2
    push_B(B, E_int, temp3Db, DT, qq + 1, B_damping_array, half_flag=1)
    calculate_E(B, Ji, q_dens, E_int, Ve, Te, Te0, temp3De, temp3Db, temp1D, E_damping_array)
    
    # Determine corrected fields at N + 1 
    E_int *= 0.5;    E_int += 0.5 * E_half

    # Restore old values: [:] allows reference to same memory (instead of creating new, local instance)
    pos[:]    = old_particles[0  , :]
    vel[:]    = old_particles[1:4, :]
    Ie[:]     = old_particles[4  , :]
    W_elec[:] = old_particles[5:8, :]
    idx[:]    = old_particles[8  , :]
    
    B[:]      = old_fields[:,   0:3]
    Ji[:]     = old_fields[:NC, 3:6]
    Ve[:]     = old_fields[:NC, 6:9]
    Te[:]     = old_fields[:NC,   9]
    
    push_B(B, E_int, temp3Db, DT, qq, B_damping_array, half_flag=0)   # Advance the original B

    q_dens[:] = q_dens_adv
    mp_flux   = mp_flux_old.copy()
    
    # Check number of spare particles every 25 steps
    if qq%25 == 0 and particle_open == 1:
        num_spare = (idx < 0).sum()
        if num_spare < nsp_ppc.sum():
            print('WARNING :: Less than one cell of spare particles remaining.')
            if num_spare < inject_rate.sum() * DT * 5.0:
                # Change this to dynamically expand particle arrays later on (adding more particles)
                # Can do it by cell lots (i.e. add a cell's worth each time)
                raise Exception('WARNING :: No space particles remaining. Exiting simulation.')
                
    
    return qq, DT, max_inc, part_save_iter, field_save_iter



### ##
### MAIN GLOBAL CONTROL
### ##

# A few internal flags
event_inputs      = False      # Can be set for lists of input files for easy batch-runs
adaptive_timestep = True       # Disable adaptive timestep if you hate when it doubles

#################################
### FILENAMES AND DIRECTORIES ###
#################################
plasma_list = ['/run_inputs/from_data/H_ONLY/plasma_params_20130725_213004105000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213050105000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213221605000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213248105000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213307605000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213406605000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213703105000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213907605000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_214026105000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_214105605000_H_ONLY.txt']

if os.name == 'posix':
    root_dir     = os.path.dirname(sys.path[0])
else:
    root_dir     = '..'
run_input = root_dir +  '/run_inputs/run_params.txt'

###########################
### LOAD RUN PARAMETERS ###
###########################
with open(run_input, 'r') as f:
    drive             = f.readline().split()[1]        # Drive letter or path for portable HDD e.g. 'E:/' or '/media/yoshi/UNI_HD/'
    save_path         = f.readline().split()[1]        # Series save dir   : Folder containing all runs of a series
    run               = f.readline().split()[1]        # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)

    save_particles    = int(f.readline().split()[1])   # Save data flag    : For later analysis
    save_fields       = int(f.readline().split()[1])   # Save plot flag    : To ensure hybrid is solving correctly during run
    seed              = int(f.readline().split()[1])   # RNG Seed          : Set to enable consistent results for parameter studies
    cpu_affin         = f.readline().split()[1]        # Set CPU affinity for run as list. Set as None to auto-assign. 

    homogenous        = int(f.readline().split()[1])   # Set B0 to homogenous (as test to compare to parabolic)
    particle_periodic = int(f.readline().split()[1])   # Set particle boundary conditions to periodic
    particle_reflect  = int(f.readline().split()[1])   # Set particle boundary conditions to reflective
    particle_reinit   = int(f.readline().split()[1])   # Set particle boundary conditions to reinitialize
    field_periodic    = int(f.readline().split()[1])   # Set field boundary to periodic (False: Absorbtive Boundary Conditions)
    disable_waves     = int(f.readline().split()[1])   # Zeroes electric field solution at each timestep
    te0_equil         = int(f.readline().split()[1])   # Initialize te0 to be in equilibrium with density
    source_smoothing  = int(f.readline().split()[1])   # Smooth source terms with 3-point Gaussian filter
    E_damping         = int(f.readline().split()[1])   # Damp E in a manner similar to B for ABCs
    quiet_start       = int(f.readline().split()[1])   # Flag to use quiet start (False :: semi-quiet start)
    radix_loading     = int(f.readline().split()[1])   # Load particles with reverse-radix scrambling sets (not implemented in this version)
    damping_multiplier= float(f.readline().split()[1]) # Multiplies the r-factor to increase/decrease damping rate.

    NX        = int(f.readline().split()[1])           # Number of cells - doesn't include ghost cells
    ND        = int(f.readline().split()[1])           # Damping region length: Multiple of NX (on each side of simulation domain)
    max_rev   = float(f.readline().split()[1])         # Simulation runtime, in multiples of the ion gyroperiod (in seconds)
    dxm       = float(f.readline().split()[1])         # Number of c/wpi per dx (Ion inertial length: anything less than 1 isn't "resolvable" by hybrid code, anything too much more than 1 does funky things to the waveform)
    r_A       = float(f.readline().split()[1])         # Ionospheric anchor point (loss zone/max mirror point) - "Below 100km" - Baumjohann, Basic Space Plasma Physics
    
    ie        = int(f.readline().split()[1])           # Adiabatic electrons. 0: off (constant), 1: on.
    min_dens  = float(f.readline().split()[1])         # Allowable minimum charge density in a cell, as a fraction of ne*q
    rc_hwidth = f.readline().split()[1]                # Ring current half-width in number of cells (2*hwidth gives total cells with RC) 
      
    orbit_res = float(f.readline().split()[1])         # Orbit resolution
    freq_res  = float(f.readline().split()[1])         # Frequency resolution     : Fraction of angular frequency for multiple cyclical values
    part_res  = float(f.readline().split()[1])         # Data capture resolution in gyroperiod fraction: Particle information
    field_res = float(f.readline().split()[1])         # Data capture resolution in gyroperiod fraction: Field information

    run_description = f.readline()                     # Commentary to attach to runs, helpful to have a quick description

# Override because I keep forgetting to change this
if os.name == 'posix':
    drive = '/home/c3134027/'

# Load run num from file, autoset if necessary
if run == '-':
    if os.path.exists(drive + save_path) == False:
        run = 0
    else:
        run = len(os.listdir(drive + save_path))
    print('Run number AUTOSET to ', run)
else:
    run = int(run)

# Set plasma parameter file
if event_inputs == False:
    plasma_input = root_dir +  '/run_inputs/plasma_params.txt'
else:
    plasma_input = root_dir +  plasma_list[run]
print('LOADING PLASMA: {}'.format(plasma_input))
    
#######################################
### LOAD PARTICLE/PLASMA PARAMETERS ###
#######################################
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
    E_per      = np.array(f.readline().split()[1:], dtype=float)
    E_e        = float(f.readline().split()[1])
    beta_flag  = int(f.readline().split()[1])

    L         = float(f.readline().split()[1])           # Field line L shell
    B_eq      = f.readline().split()[1]                  # Initial magnetic field at equator: None for L-determined value (in T) :: 'Exact' value in node ND + NX//2

charge    *= q                                           # Cast species charge to Coulomb
mass      *= mp                                          # Cast species mass to kg

#####################################
### DERIVED SIMULATION PARAMETERS ###
#####################################
if ND < 2:
    ND = 2                                  # Set minimum (used for array addresses)

NC          = NX + 2*ND                     # Total number of cells
ne          = density.sum()                 # Electron number density
E_par       = E_per / (anisotropy + 1)      # Parallel species energy

if field_periodic == 1:
    if particle_periodic == 0:
        print('Periodic field compatible only with periodic particles.')
        particle_periodic = 1
        particle_reflect = particle_reinit = 0

particle_open = 0
if particle_reflect + particle_reinit + particle_periodic == 0:
    particle_open = 1
    
if B_eq == '-':
    B_eq      = (B_surf / (L ** 3))         # Magnetic field at equator, based on L value
else:
    B_eq = float(B_eq)
    
if rc_hwidth == '-':
    rc_hwidth = 0
    
if beta_flag == 0:
    # Input energies in eV
    beta_per   = None
    Te0_scalar = q * E_e / kB
    vth_perp   = np.sqrt(q * charge *  E_per /  mass)    # Perpendicular thermal velocities
    vth_par    = np.sqrt(q * charge *  E_par /  mass)    # Parallel thermal velocities
else:
    # Input energies in terms of beta
    kbt_par    = E_par * B_eq ** 2 / (2 * mu0 * ne)
    kbt_per    = E_per * B_eq ** 2 / (2 * mu0 * ne)
    Te0_scalar = E_e   * B_eq ** 2 / (2 * mu0 * ne * kB)
    vth_perp   = np.sqrt(kbt_per /  mass)                # Perpendicular thermal velocities
    vth_par    = np.sqrt(kbt_par /  mass)                # Parallel thermal velocities

rho        = (mass*density).sum()                        # Mass density for alfven velocity calc.
wpi        = np.sqrt(ne * q ** 2 / (mp * e0))            # Proton   Plasma Frequency, wpi (rad/s)
va         = B_eq / np.sqrt(mu0*rho)                     # Alfven speed at equator: Assuming pure proton plasma
gyfreq_eq  = q*B_eq  / mp                                # Proton Gyrofrequency (rad/s) at equator (slowest)
dx         = va / gyfreq_eq                              # Alternate method of calculating dx (better for multicomponent plasmas)
#dx         = dxm * c / wpi                               # Spatial cadence, based on ion inertial length
xmax       = NX // 2 * dx                                # Maximum simulation length, +/-ve on each side
xmin       =-NX // 2 * dx
Nj         = len(mass)                                   # Number of species
n_contr    = density / nsp_ppc                           # Species density contribution: Each macroparticle contributes this density to a cell

# Number of sim particles for each species, total
N_species = nsp_ppc * NX
if field_periodic == 0:
    N_species += 2   

# Add number of spare particles proportional to # cells worth
if particle_open == 1:
    spare_ppc  = 5*nsp_ppc.copy()
else:
    spare_ppc  = np.zeros(Nj, dtype=int)
N = N_species.sum() + spare_ppc.sum()

idx_start  = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
idx_end    = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order
        
############################
### MAGNETIC FIELD STUFF ###
############################
B_nodes  = (np.arange(NC + 1) - NC // 2)       * dx      # B grid points position in space
E_nodes  = (np.arange(NC)     - NC // 2 + 0.5) * dx      # E grid points position in space

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
else:
    print('Calculating length of field line...')
    N_fl   = 1e5                                                                # Number of points to calculate field line length (higher is more accurate)
    lat0   = np.arccos(np.sqrt((RE + r_A)/(RE*L)))                              # Latitude for this L value (at ionosphere height)
    h      = 2.0*lat0/float(N_fl)                                               # Step size of lambda (latitude)
    f_len  = 0.0
    for ii in range(int(N_fl)):
        lda        = ii*h - lat0                                                # Lattitude for this step
        f_len     += L*RE*np.cos(lda)*np.sqrt(4.0 - 3.0*np.cos(lda) ** 2) * h   # Field line length accruance
    print('Field line length = {:.2f} RE'.format(f_len/RE))
    print('Simulation length = {:.2f} RE'.format(2*xmax/RE))
    
    if xmax > f_len / 2:
        sys.exit('Simulation length longer than field line. Aboring...')
        
    print('Finding simulation boundary MLAT...')
    dlam   = 1e-5                                            # Latitude increment in radians
    fx_len = 0.0; ii = 1                                     # Arclength/increment counters
    while fx_len < xmax:
        lam_i   = dlam * ii                                                             # Current latitude
        d_len   = L * RE * np.cos(lam_i) * np.sqrt(4.0 - 3.0*np.cos(lam_i) ** 2) * dlam     # Length increment
        fx_len += d_len                                                                 # Accrue arclength
        ii     += 1                                                                     # Increment counter

    theta_xmax  = lam_i                                                                 # Latitude of simulation boundary
    r_xmax      = L * RE * np.cos(theta_xmax) ** 2                                      # Radial distance of simulation boundary
    B_xmax      = B_eq*np.sqrt(4 - 3*np.cos(theta_xmax)**2)/np.cos(theta_xmax)**6       # Magnetic field intensity at boundary
    a           = (B_xmax / B_eq - 1) / xmax ** 2                                       # Parabolic scale factor: Fitted to B_eq, B_xmax
    lambda_L    = np.arccos(np.sqrt(1.0 / L))                                           # Lattitude of Earth's surface at this L

    lat_A      = np.arccos(np.sqrt((RE + r_A)/(RE*L)))       # Anchor latitude in radians
    B_A        = B_eq * np.sqrt(4 - 3*np.cos(lat_A) ** 2)\
               / (np.cos(lat_A) ** 6)                        # Magnetic field at anchor point
    
    loss_cone_eq   = np.arcsin(np.sqrt(B_eq   / B_A))*180 / np.pi   # Equatorial loss cone in degrees
    loss_cone_xmax = np.arcsin(np.sqrt(B_xmax / B_A))               # Boundary loss cone in radians

gyfreq     = q*B_xmax/ mp                                # Proton Gyrofrequency (rad/s) at boundary (highest)
k_max      = np.pi / dx                                  # Maximum permissible wavenumber in system (SI???)
qm_ratios  = np.divide(charge, mass)                     # q/m ratio for each species

if particle_open == 1:
    inject_rate = nsp_ppc * (vth_par / dx) / np.sqrt(2 * np.pi)
else:
    inject_rate = nsp_ppc * 0.0

# E-field nodes around boundaries (used for sources and E-fields)
lo1 = ND - 1 ; lo2 = ND - 2             # Left outer (to boundary)
ro1 = ND + NX; ro2 = ND + NX + 1        # Right outer

li1 = ND         ; li2 = ND + 1         # Left inner
ri1 = ND + NX - 1; ri2 = ND + NX - 2    # Right inner
    
##############################
### INPUT TESTS AND CHECKS ###
##############################
print('Run Started')
print('Run Series         : {}'.format(save_path.split('//')[-1]))
print('Run Number         : {}'.format(run))
print('Field save flag    : {}'.format(save_fields))
print('Particle save flag : {}\n'.format(save_particles))

print('Sim domain length  : {:5.2f}R_E'.format(2 * xmax / RE))
print('Density            : {:5.2f}cc'.format(ne / 1e6))
print('Equatorial B-field : {:5.2f}nT'.format(B_eq*1e9))
print('Maximum    B-field : {:5.2f}nT'.format(B_xmax*1e9))
print('Iono.      B-field : {:5.2f}mT'.format(B_A*1e6))
print('Equat. Loss cone   : {:<5.2f} degrees  '.format(loss_cone_eq))
print('Bound. Loss cone   : {:<5.2f} degrees  '.format(loss_cone_xmax * 180. / np.pi))
print('Maximum MLAT (+/-) : {:<5.2f} degrees  '.format(theta_xmax * 180. / np.pi))
print('Iono.   MLAT (+/-) : {:<5.2f} degrees\n'.format(lambda_L * 180. / np.pi))

print('Equat. Gyroperiod: : {}s'.format(round(2. * np.pi / gyfreq, 3)))
print('Inverse rad gyfreq : {}s'.format(round(1 / gyfreq, 3)))
print('Maximum sim time   : {}s ({} gyroperiods)\n'.format(round(max_rev * 2. * np.pi / gyfreq_eq, 2), max_rev))

print('{} spatial cells, 2x{} damped cells'.format(NX, ND))
print('{} cells total'.format(NC))
print('{} particles total\n'.format(N))

if cpu_affin != '-':
    if len(cpu_affin) == 1:
        cpu_affin = [int(cpu_affin)]        
    else:
        cpu_affin = list(map(int, cpu_affin.split(',')))
    
    import psutil
    run_proc = psutil.Process()
    run_proc.cpu_affinity(cpu_affin)
    print('CPU affinity for run (PID {}) set to :: {}'.format(run_proc.pid, ', '.join(map(str, run_proc.cpu_affinity()))))
else:
    print('CPU affinity not set.')

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

########################
### START SIMULATION ###
########################
#sys.exit()
if __name__ == '__main__':
    start_time = timer()
    
    # Initialize simulation: Allocate memory and set time parameters
    pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp,temp_N = initialize_particles()
    B, E_int, E_half, Ve, Te, Te0                       = initialize_fields()
    q_dens, q_dens_adv, Ji, ni, nu                      = initialize_source_arrays()
    old_particles, old_fields, temp3De, temp3Db, temp1D,\
                                v_prime, S, T, mp_flux  = initialize_tertiary_arrays()
    
    # Collect initial moments and save initial state
    collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu) 

    DT, max_inc, part_save_iter, field_save_iter, B_damping_array, E_damping_array\
        = set_timestep(vel, Te0)
        
    calculate_E(B, Ji, q_dens, E_int, Ve, Te, Te0, temp3De, temp3Db, temp1D, E_damping_array)
    
    if save_particles == 1:
        save_particle_data(0, DT, part_save_iter, 0, pos, vel, idx)
        
    if save_fields == 1:
        save_field_data(0, DT, field_save_iter, 0, Ji, E_int,\
                             B, Ve, Te, q_dens, B_damping_array, E_damping_array)

    # Retard velocity
    print('Retarding velocity...')
    velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E_int, v_prime, S, T, temp_N, -0.5*DT)

    qq       = 1;    sim_time = DT
    print('Starting main loop...')
    while qq < max_inc:
        qq, DT, max_inc, part_save_iter, field_save_iter =                                \
        main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag, Ep, Bp, v_prime, S, T, temp_N,\
              B, E_int, E_half, q_dens, q_dens_adv, Ji, ni, nu, mp_flux,                  \
              Ve, Te, Te0, temp3De, temp3Db, temp1D, old_particles, old_fields,           \
              B_damping_array, E_damping_array, qq, DT, max_inc, part_save_iter, field_save_iter)

        if qq%part_save_iter == 0 and save_particles == 1:
            save_particle_data(sim_time, DT, part_save_iter, qq, pos,
                                    vel, idx)
            
        if qq%field_save_iter == 0 and save_fields == 1:
            save_field_data(sim_time, DT, field_save_iter, qq, Ji, E_int,
                                 B, Ve, Te, q_dens, B_damping_array, E_damping_array)
        
        if qq%100 == 0:            
            running_time = int(timer() - start_time)
            hrs          = running_time // 3600
            rem          = running_time %  3600
            
            mins         = rem // 60
            sec          = rem %  60
            
            print('Step {} of {} :: Current runtime {:02}:{:02}:{:02}'.format(qq, max_inc, hrs, mins, sec))
        
        qq       += 1
        sim_time += DT
        
        if qq == 2:
            print('First loop complete.')
            
    runtime = round(timer() - start_time,2)
    
    if save_fields == 1 or save_particles == 1:
        add_runtime_to_header(runtime)
    print("Time to execute program: {0:.2f} seconds".format(runtime))