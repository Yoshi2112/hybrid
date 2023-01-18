# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:09:28 2023

@author: iarey
"""
import numpy as np
import numba as nb


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
def initialize_velocity_LCD_elimination(pos, vel, sf_par, sf_per, st, en, jj):
    '''
    Takes in a Maxwellian or pseudo-maxwellian distribution. Removes any particles
    inside the loss cone and reinitializes them using the given scale factors 
    sf_par, sf_per (the thermal velocities).
    
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


def initialize_velocity_LCD_MonteCarlo():
    '''
    This would only be called for a non-homogeneous open-boundary simulation
    since homogenous = no loss cone and periodic = no field gradient.
    '''
    print('Initialising loss-cone distribution')
    def PFLCD(x, y, jj):
        '''
        Partially-filled loss-cone distribution as per eqns 6.41-6.42 of
        Baumjohann (1997). Can take single values or 1D arrays for x, y
        
        INPUTS:
        x :: Parallel velocity variable
        y :: Perpendicular velocity variable
        
        OUTPUTS:
        fx :: Value of the distribution function at point (x,y)
        
        GLOBALS:
        vth_para :: Parallel thermal velocity (m/s)
        vth_perp :: Perpendicular thermal velocity (m/s)
        density  :: Total density of the distribution (/m3)
        beta     :: Determines the slope of the distribution in the loss cone
        delta    :: Determines loss cone fullness
        
        QUESTION: How do beta, delta relate to the size of the loss cone in degrees?
        '''
        # Set as constant, maybe set as variable later
        lcd_delta = 0.0
        lcd_beta  = 0.1
        
        exp1 = lcd_delta*np.exp(- y ** 2 / (2*vth_perp[jj]**2))
        exp2 = np.exp(- y ** 2 / (2*vth_perp[jj]**2)) - np.exp(- y ** 2 / (2*lcd_beta*vth_perp[jj]**2))
        gx   = exp1 + ((1 - lcd_delta) / (1 - lcd_beta)) * exp2
        
        fx  = density[jj] / (8*np.pi**3*vth_para[jj]*vth_perp[jj]**2) * np.exp(- x ** 2 / (2*vth_para[jj]**2))
        fx *= gx
        return fx
    
    def generate_PFLCD_distribution(jj, n_samples=1000, n_vtherm=4):
        '''
        Randomly generates n samples of a PFLCD by Monte-Carlo rejection method.
        
        jj        -- Species to generate for (defines thermal velocities, density)
        n_vtherm  -- Defines width of sampled distribution, up to 4 vth is >99%
        n_samples -- Number of samples to collect from distribution
        
        Breaks v_perp (cy) into vy, vz by randomly initialising in gyroangle
        '''
        xmin, xmax = -n_vtherm*vth_para[jj], n_vtherm*vth_para[jj]
        ymin, ymax = -n_vtherm*vth_perp[jj], n_vtherm*vth_perp[jj]
        
        print('Checking distribution max value...')
        test_n = 1000
        test_x = np.linspace(xmin, xmax, test_n)
        test_y = np.linspace(ymin, ymax, test_n)
        P_max = 0.0
        for mm in range(test_n):
            for nn in range(test_n):
                test_P = PFLCD(test_x[mm], test_y[nn], jj)
                if test_P > P_max: P_max = test_P
        P_max *= 1.005   # Pad a little to make up for inaccuracy in test sampling
        
        print('Creating LCD distribution...')
        n_batch = 5*n_samples
        dist_x = np.zeros(n_samples, dtype=np.float32)
        dist_y = np.zeros(n_samples, dtype=np.float32)
        acc = 0
        while acc < n_samples:
            cx = np.random.uniform(xmin, xmax, n_batch)     # Sample
            cy = np.random.uniform(ymin, ymax, n_batch)
            cP = PFLCD(cx, cy, jj)                          # Evaluate
            z  = np.random.uniform(0., P_max, n_batch)       # Generate a uniform distribution between 0 and Py_max, z

            # If z < P(x,y) then accept sample, otherwise reject
            for ii in range(n_batch):
                if z[ii] < cP[ii]:
                    dist_x[acc] = cx[ii]
                    dist_y[acc] = cy[ii]
                    acc += 1
                    
                    if acc == n_samples:
                        print('Finished.')
                        return dist_x, dist_y
        raise Exception('You should never get here')
        
    np.random.seed(seed)
    vel = np.zeros((3, N), dtype=np.float64)
    for jj in range(Nj): 
        
        n_init = N_species[jj]
        if quiet_start == 1:                          # Determine how many particles are initialized randomly
            n_init //= 2                              # For quiet start, half are copies with -v_perp
            
        # Particle index ranges
        st = idx_start[jj]
        en = idx_start[jj] + n_init
          
        # Set Loss Cone Distribution for hot particles, Maxwellian for cold
        if temp_type[jj] == 1:
            vpara, vperp = generate_PFLCD_distribution(jj, n_samples=n_init)
            gyangles = np.random.uniform(0.0, 2*np.pi, n_init)
            
            vel[0, st: en] = vpara + drift_v[jj]*va
            vel[1, st: en] = vperp * np.sin(gyangles)
            vel[2, st: en] = vperp * np.cos(gyangles)
        else:
            vel[0, st: en] = np.random.normal(0, vth_para[jj], n_init) + drift_v[jj]*va
            vel[1, st: en] = np.random.normal(0, vth_perp[jj], n_init)
            vel[2, st: en] = np.random.normal(0, vth_perp[jj], n_init)
            
        # Quiet start : Initialize second half
        if quiet_start == 1:
            vel[0, en: en + n_init] = vel[0, st: en]                # Set parallel
            vel[1, en: en + n_init] = vel[1, st: en] * -1.0         # Invert perp velocities (v2 = -v1)
            vel[2, en: en + n_init] = vel[2, st: en] * -1.0
    return vel


# =============================================================================
# def reverse_radix_quiet_start_uniform():
#     '''
#     Need to make this faster
#     
#     Creates an N-sampled normal distribution in 3D velocity space that is 
#     uniform in a 1D configuration space. Function uses analytic sampling of
#     distribution function and reverse-radix shuffling to ensure randomness.
# 
#     OUTPUT:
#         pos --  N array of uniformly distributed particle positions
#         vel -- 3xN array of gaussian particle velocities, giving a Maxwellian in |v|
#         idx -- N array of particle indexes, indicating which species it belongs to
#     '''
#     print('Initialising particle distributions with Bit-Reversed Radix algorithm')
#     print('Please wait...')
#     def rkbr_uniform_set(arr, base=2):
#         '''
#         Works on array arr to produce fractions in (0, 1) by 
#         traversing base k.
#         
#         Will support arrays of lengths up to at least a billion
#          -- But this is super inefficient, takes up to a minute with 2 million
#          
#         Parallelise?
#         '''
#         def reverse_slicing(s):
#             return s[::-1]
#         
#         # Convert ints to base k strings
#         str_arr = np.zeros(arr.shape[0], dtype='U30')
#         dec_arr = np.zeros(arr.shape[0], dtype=float)
#         for ii in range(arr.shape[0]):
#             str_arr[ii] = np.base_repr(arr[ii], base)   # Returns strings
#     
#         # Reverse string order and convert to decimal, treating as base k fraction (i.e. with 0. at front)
#         for ii in range(arr.shape[0]):
#             rev = reverse_slicing(str_arr[ii])
#     
#             dec_val = 0
#             for jj in range(len(rev)):
#                 dec_val += float(rev[jj]) * (base ** -(jj + 1))
#             dec_arr[ii] = dec_val
#         return dec_arr 
# 
#     # Set and initialize seed
#     np.random.seed(seed)
#     pos = np.zeros(N, dtype=np.float64)
#     vel = np.zeros((3, N), dtype=np.float64)
#     idx = np.ones(N,       dtype=np.int8) * Nj
#     
#     for jj in range(Nj):
#         half_n = N_species[jj] // 2                     # Half particles of species - doubled later
#                 
#         st = idx_start[jj]
#         en = idx_start[jj] + half_n
#         
#         # Set position
#         for kk in range(half_n):
#             pos[st + kk] = 2*xmax*(float(kk) / (half_n - 1))
#         pos[st: en]-= xmax              
#         
#         # Set velocity for half: Randomly Maxwellian
#         arr     = np.arange(half_n)
#         
#         R_vr    = rkbr_uniform_set(arr+1, base=2)
#         R_theta = rkbr_uniform_set(arr  , base=3) 
#         R_vrx   = rkbr_uniform_set(arr+1, base=5)
#         
#         vr      = vth_perp[jj] * np.sqrt(-2 * np.log(R_vr ))
#         vrx     = vth_par[ jj] * np.sqrt(-2 * np.log(R_vrx))
#         theta   = R_theta * np.pi * 2
# 
#         vel[0, st: en] = vrx * np.sin(theta) +  drift_v[jj]
#         vel[1, st: en] = vr  * np.sin(theta)
#         vel[2, st: en] = vr  * np.cos(theta)
#         idx[   st: en] = jj
#         
#         # Quiet Start: Other half, same position, parallel velocity, opposite v_perp
#         pos[   en: en + half_n] = pos[   st: en]                
#         vel[0, en: en + half_n] = vel[0, st: en] *  1.0
#         vel[1, en: en + half_n] = vel[1, st: en] * -1.0
#         vel[2, en: en + half_n] = vel[2, st: en] * -1.0
#         
#         idx[st: idx_end[jj]] = jj
#     print('Particles initialised.')
#     return pos, vel, idx
# =============================================================================


@nb.njit()
def initialize_velocity_bimaxwellian():
    '''Initializes position, velocity, and index arrays. Positions and velocities
    both initialized using appropriate numpy random distributions, cell by cell.

    OUTPUT:
        pos -- 1xN array of particle positions in meters
        vel -- 3xN array of particle velocities in m/s
        idx -- N   array of particle indexes, indicating population types 
    '''
    # Set and initialize seed
    np.random.seed(seed)
    vel   = np.zeros((3, N), dtype=np.float64)

    # Initialize unformly in space, gaussian in 3-velocity
    for jj in range(Nj):
        n_init = N_species[jj]
        if quiet_start == 1:
            n_init //= 2
        
        st = idx_start[jj]
        en = idx_start[jj] + n_init
                  
        vel[0, st: en] = np.random.normal(0, vth_para[jj], n_init) + drift_v[jj] * va
        vel[1, st: en] = np.random.normal(0, vth_perp[jj], n_init)
        vel[2, st: en] = np.random.normal(0, vth_perp[jj], n_init)
        
        if quiet_start == 1:
            vel[0, en: en + n_init] =      vel[0, st: en]
            vel[1, en: en + n_init] = -1.0*vel[1, st: en]
            vel[2, en: en + n_init] = -1.0*vel[2, st: en]
    return vel


def initialise_position_uniform():
    '''
    Initializes an analytically uniform distribution per cell (for better
    consistency). Considerations:
        -- For the quiet start, only half are initialized and half are copies
        -- For open boundary conditions, an extra particle (2 for quiet) is placed
           at xmax. This is because for periodic, xmin=xmax represents the same
           location.
    '''
    print('Initializing uniform distribution')
    pos = np.zeros(N, dtype=np.float64)
    idx = np.ones( N, dtype=np.int8) * Nj

    for jj in range(Nj):
        idx[idx_start[jj]: idx_end[jj]] = jj
        
        n_init = nsp_ppc[jj]
        if quiet_start == 1:
            n_init //= 2

        acc = 0; offset  = 0
        for ii in range(NX):
            if ii == NX - 1 and field_periodic == 0:
                n_init += 1
                offset  = 1
            
            st = idx_start[jj] + acc
            en = idx_start[jj] + acc + n_init
            
            for kk in range(n_init):
                pos[st + kk] = dx*(float(kk) / (n_init - offset) + ii)
            pos[st: en] -= 0.5*NX*dx              
            
            acc += n_init
    
        if quiet_start == 1:
            st = idx_start[jj]
            en = idx_start[jj] + acc
            
            pos[en: en+acc] = pos[st:en]  
            
    return pos, idx


def initialize_particles():
    '''
    TODO Why is there noise at t=0??
    '''
    pos, idx = initialise_position_uniform()
    
    if homogenous == 1:
        vel = initialize_velocity_bimaxwellian()
    else:
        vel = initialize_velocity_LCD_MonteCarlo()
        
    Ie         = np.zeros(N,      dtype=np.uint16)
    Ib         = np.zeros(N,      dtype=np.uint16)
    W_elec     = np.zeros((3, N), dtype=np.float64)
    W_mag      = np.zeros((3, N), dtype=np.float64)
    
    assign_weighting_TSC(pos, Ie, W_elec)
    assign_weighting_TSC(pos, Ib, W_mag)
    
    # Calculate memory used:
    nbytes = 0
    nbytes += pos.nbytes
    nbytes += vel.nbytes
    nbytes += Ie.nbytes
    nbytes += W_elec.nbytes
    nbytes += Ib.nbytes
    nbytes += W_mag.nbytes
    nbytes += idx.nbytes
    ngbytes = nbytes / 1024 / 1024 / 1024
    print(f'Memory used by particle arrays: {ngbytes:.3f} GB')
    return pos, vel, Ie, W_elec, Ib, W_mag, idx


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