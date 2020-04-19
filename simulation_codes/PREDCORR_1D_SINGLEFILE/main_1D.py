## PYTHON MODULES ##
import sys
import numpy as np
import numba as nb
import pickle
from shutil import rmtree
import os
from timeit import default_timer as timer


#%% MAIN FUNCTION
def main():
    start_time = timer()
    
    # Initialize simulation: Allocate memory and set time parameters
    pos, vel, Ie, W_elec, Ib, W_mag, idx                = initialize_particles()
    B, E_int, E_half, Ve, Te                            = initialize_fields()
    q_dens, q_dens_adv, Ji, ni, nu                      = initialize_source_arrays()
    old_particles, old_fields, temp3De, temp3Db, temp1D = initialize_tertiary_arrays()
    
    # Collect initial moments and save initial state
    collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu, temp1D) 
    calculate_E(B, Ji, q_dens, E_int, Ve, Te, temp3De, temp3Db, temp1D)
    
    DT, max_inc, part_save_iter, field_save_iter, damping_array = set_timestep(vel, E_int)

    if save_particles == 1:
        save_particle_data(0, DT, part_save_iter, 0, pos, vel, idx)
        
    if save_fields == 1:
        save_field_data(0, DT, field_save_iter, 0, Ji, E_int, B, Ve, Te, q_dens, damping_array)
    
    # Retard velocity
    velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E_int, -0.5*DT)
    
    qq       = 1;    sim_time = DT
    print('Starting main loop...')
    while qq < max_inc:
        
        qq, DT, max_inc, part_save_iter, field_save_iter =                \
        main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag,                   \
              B, E_int, E_half, q_dens, q_dens_adv, Ji, ni, nu,           \
              Ve, Te, temp3De, temp3Db, temp1D, old_particles, old_fields,\
              damping_array, qq, DT, max_inc, part_save_iter, field_save_iter)

        if qq%part_save_iter == 0 and save_particles == 1:
            save_particle_data(sim_time, DT, part_save_iter, qq, pos,
                                    vel, idx)
            
        if qq%field_save_iter == 0 and save_fields == 1:
            save_field_data(sim_time, DT, field_save_iter, qq, Ji, E_int,
                                 B, Ve, Te, q_dens, damping_array)
        
        if qq%100 == 0:
            running_time = int(timer() - start_time)
            hrs          = running_time // 3600
            rem          = running_time %  3600
            
            mins         = rem // 60
            sec          = rem %  60
            print('Step {} of {} :: Current runtime {:02}:{:02}:{:02}'.format(qq, max_inc, hrs, mins, sec))
            
        qq       += 1
        sim_time += DT
        
    print("Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2)))
    return


#%% INITIALISATION
#@nb.njit()
def calc_losses(v_para, v_perp, st=0):
    '''
    For arrays of parallel and perpendicular velocities, finds the number and 
    indices of particles outside the loss cone.
    
    Calculation of in_loss_cone not compatible with njit(). Recode later if you want.
    '''
    alpha        = np.arctan(v_perp / v_para) * 180. / np.pi    # Calculate particle PA's
    in_loss_cone = (abs(alpha) < loss_cone)                     # Determine if particle in loss cone
    N_loss       = in_loss_cone.sum()                           # Count number that are
    loss_idx     = np.where(in_loss_cone == True)[0]            # Find their indices
    loss_idx    += st                                           # Offset indices to account for position in master array
    return N_loss, loss_idx


#@nb.njit()
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

    for jj in range(Nj):
        idx[idx_start[jj]: idx_end[jj]] = jj          # Set particle idx
                
        half_n = nsp_ppc[jj] // 2                     # Half particles per cell - doubled later
        sf_par = np.sqrt(kB *  Tpar[jj] /  mass[jj])  # Scale factors for velocity initialization
        sf_per = np.sqrt(kB *  Tper[jj] /  mass[jj])
       
        if temp_type[jj] == 0:                        # Change how many cells are loaded between cold/warm populations
            NC_load = NX
        else:
            if rc_hwidth == 0:
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

            # Set Loss Cone Distribution: Reinitialize particles in loss cone
            B0x = eval_B0x(pos[0, st: en])
            if homogenous == False:
                N_loss = N_species[jj]
                
                while N_loss > 0:
                    v_mag       = vel[0, st: en] ** 2 + vel[1, st: en] ** 2 + vel[2, st: en] ** 2
                    v_perp      = np.sqrt(vel[1, st: en] ** 2 + vel[2, st: en] ** 2)
                    v_perp_eq   = v_perp * np.sqrt(B_eq / B0x)
                    v_para_eq   = np.sqrt(v_mag - v_perp_eq ** 2)

                    N_loss, loss_idx = calc_losses(v_para_eq, v_perp_eq, st=st)
                    
                    # Catch for a particle on the boundary : Set 90 degree pitch angle (gyrophase shouldn't overly matter)
                    if N_loss == 1:
                        if abs(pos[0, loss_idx[0]]) == xmax:
                            ww = loss_idx[0]
                            vel[0, loss_idx[0]] = 0.
                            vel[1, loss_idx[0]] = np.sqrt(vel[0, ww] ** 2 + vel[1, ww] ** 2 + vel[2, ww] ** 2)
                            vel[2, loss_idx[0]] = 0.
                            N_loss = 0
                                        
                    if N_loss != 0:                        
                        vel[0, loss_idx] = np.random.normal(0., sf_par, N_loss)
                        vel[1, loss_idx] = np.random.normal(0., sf_per, N_loss)
                        vel[2, loss_idx] = np.random.normal(0., sf_per, N_loss)
            else:
                v_perp      = np.sqrt(vel[1, st: en] ** 2 + vel[2, st: en] ** 2)
        
            pos[1, st: en]  = v_perp / (qm_ratios[jj] * B0x)    # Set initial Larmor radius   
            
            vel[0, en: en + half_n] = vel[0, st: en] * -1.0     # Invert velocities (v2 = -v1)
            vel[1, en: en + half_n] = vel[1, st: en] * -1.0
            vel[2, en: en + half_n] = vel[2, st: en] * -1.0
            pos[1, en: en + half_n] = pos[1, st: en] * -1.0     # Move gyrophase 180 degrees (doesn't do anything)
            
            pos[0, en: en + half_n] = pos[0, st: en]            # Other half, same position
            
            acc                    += half_n * 2
        
    return pos, vel, idx


#@nb.njit()
def initialize_particles():
    '''Initializes particle arrays.
    
    INPUT:
        <NONE>
        
    OUTPUT:
        pos    -- Particle position array (1, N)
        vel    -- Particle velocity array (3, N)
        Ie     -- Initial particle positions by leftmost E-field node
        W_elec -- Initial particle weights on E-grid
        Ib     -- Initial particle positions by leftmost B-field node
        W_mag  -- Initial particle weights on B-grid
        idx    -- Particle type index
    '''
    pos, vel, idx = uniform_gaussian_distribution_quiet()
    
    Ie         = np.zeros(N,      dtype=np.uint16)
    Ib         = np.zeros(N,      dtype=np.uint16)
    W_elec     = np.zeros((3, N), dtype=np.float64)
    W_mag      = np.zeros((3, N), dtype=np.float64)
    
    assign_weighting_TSC(pos, Ie, W_elec)
    return pos, vel, Ie, W_elec, Ib, W_mag, idx


@nb.njit()
def set_damping_array(damping_array, DT):
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
    '''
    dist_from_mp  = np.abs(np.arange(NC + 1) - 0.5*NC)          # Distance of each B-node from midpoint
    r_damp        = np.sqrt(29.7 * 0.5 * va / ND * (DT / dx))   # Damping coefficient
    
    for ii in range(NC + 1):
        if dist_from_mp[ii] > 0.5*NX:
            damping_array[ii] = 1. - r_damp * ((dist_from_mp[ii] - 0.5*NX) / ND) ** 2 
        else:
            damping_array[ii] = 1.0
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
    B       = np.zeros((NC + 1, 3), dtype=np.float64)
    E_int   = np.zeros((NC    , 3), dtype=np.float64)
    E_half  = np.zeros((NC    , 3), dtype=np.float64)

    # Set initial B0
    B[:, 0] = Bc[:, 0]
    B[:, 1] = Bc[:, 1]
    B[:, 2] = Bc[:, 2]
    
    Ve      = np.zeros((NC, 3), dtype=np.float64)
    Te      = np.ones(  NC,     dtype=np.float64) * Te0
    
    return B, E_int, E_half, Ve, Te


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
        old_fields   -- Location to store old B, Ji, Ve, Te field values for predictor-corrector routine
    '''
    temp3Db       = np.zeros((NC + 1, 3),  dtype=nb.float64)
    temp3De       = np.zeros((NC    , 3),  dtype=nb.float64)
    temp1D        = np.zeros( NC    ,      dtype=nb.float64) 
    old_fields    = np.zeros((NC + 1, 10), dtype=nb.float64)
    
    old_particles = np.zeros((10, N),      dtype=nb.float64)
        
    return old_particles, old_fields, temp3De, temp3Db, temp1D


def set_equilibrium_te0(qdens):
    '''
    Modifies the initial Te array to allow grad(P_e) = grad(nkT) = 0
    
    Iterative? Analytic? WORK IN PROGRESS :: DOESN'T WORK YET
    '''
    LC             = NX + ND - 1
    qdens_gradient = np.zeros(NC    , dtype=np.float64)
    temp           = np.zeros(NC + 1, dtype=np.float64)
    
    # Get density gradient:
    
    # Central differencing, internal points
    for ii in nb.prange(ND + 1, LC - 1):
        qdens_gradient[ii] = (qdens[ii + 1] - qdens[ii - 1])
    
    # Forwards/Backwards difference at physical boundaries
    qdens_gradient[ND] = -3*qdens[ND] + 4*qdens[ND + 1] - qdens[ND + 2]
    qdens_gradient[LC] =  3*qdens[LC] - 4*qdens[LC - 1] + qdens[LC - 2]
    qdens_gradient    /= (2*dx)
    
    # Work out equilibrium temperature
    te0_arr = np.ones(NC, dtype=np.float64) * Te0
    
    for ii in range(ND, ND + NX):
        te0_arr[ii + 1] = - 2.0 * dx * te0_arr[ii] * qdens_gradient[ii] * qdens[ii] + te0_arr[ii - 1]
        
    get_grad_P(qdens, te0_arr, qdens_gradient, temp)
    return


def set_timestep(vel, E):
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
    vel_ts   = 0.5 * dx / np.max(vel[0, :])     # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step 

    if E[:, 0].max() != 0:
        elecfreq        = qm_ratios.max()*(np.abs(E[:, 0] / vel.max()).max())               # Electron acceleration "frequency"
        Eacc_ts         = freq_res / elecfreq                            
    else:
        Eacc_ts = ion_ts

    gyperiod = 2 * np.pi / gyfreq
    DT       = min(ion_ts, vel_ts, Eacc_ts)
    max_time = max_rev * 2 * np.pi / gyfreq_eq     # Total runtime in seconds
    max_inc  = int(max_time / DT) + 1                          # Total number of time steps

    if part_res == 0:
        part_save_iter = 1
    else:
        part_save_iter = int(part_res*gyperiod / DT)

    if field_res == 0:
        field_save_iter = 1
    else:
        field_save_iter = int(field_res*gyperiod / DT)

    if save_fields == 1 or save_particles == 1:
        store_run_parameters(DT, part_save_iter, field_save_iter)

    damping_array = np.ones(NC + 1)
    set_damping_array(damping_array, DT)

    print('Timestep: %.4fs, %d iterations total\n' % (DT, max_inc))
    return DT, max_inc, part_save_iter, field_save_iter, damping_array


#%% SAVE FUNCTIONS
def store_run_parameters(dt, part_save_iter, field_save_iter):
    d_path = ('%s/%s/run_%d/data/' % (drive, save_path, run))    # Set path for data
    f_path = d_path + '/fields/'
    p_path = d_path + '/particles/'
    
    for folder in [d_path, f_path, p_path]:
        if os.path.exists(folder) == False:                               # Create data directories
            os.makedirs(folder)
    
    if periodic == True:
        rstring = 'periodic'
    else:
        if reflect == True:
            rstring = 'reflective'
        else:
            rstring = 'absorptive'
        
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
                   ('r_damp', 0.0),
                   ('L', L), 
                   ('B_eq', B_eq),
                   ('xmax', xmax),
                   ('xmin', xmin),
                   ('B_xmax', B_xmax),
                   ('a', a),
                   ('theta_xmax', theta_xmax),
                   ('rc_hwidth', rc_hwidth),
                   ('ne', ne),
                   ('Te0', Te0),
                   ('ie', ie),
                   ('part_save_iter', part_save_iter),
                   ('field_save_iter', field_save_iter),
                   ('max_rev', max_rev),
                   ('freq_res', freq_res),
                   ('orbit_res', orbit_res),
                   ('run_desc', run_description),
                   ('method_type', 'PREDCORR_PARABOLIC'),
                   ('particle_shape', 'TSC'),
                   ('boundary_type', 'damped'),
                   ('particle_boundary', rstring),
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
                     drift_v     = drift_v,
                     nsp_ppc     = nsp_ppc,
                     density     = density,
                     N_species   = N_species,
                     Tpar        = Tpar,
                     Tper        = Tper,
                     Bc          = Bc)
    print('Particle data saved')
    return


def save_field_data(sim_time, dt, field_save_iter, qq, Ji, E, B, Ve, Te, dns, damping_array):
    d_path   = '%s/%s/run_%d/data/fields/' % (drive, save_path, run)
    r        = qq / field_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, E = E[:, 0:3], B = B[:, 0:3],   J = Ji[:, 0:3],
                       dns = dns,      Ve = Ve[:, 0:3], Te = Te, sim_time = sim_time,
                       damping_array = damping_array)
    print('Field data saved')
    
    
def save_particle_data(sim_time, dt, part_save_iter, qq, pos, vel, idx):
    d_path   = '%s/%s/run_%d/data/particles/' % (drive, save_path, run)
    r        = qq / part_save_iter

    d_fullpath = d_path + 'data%05d' % r
    
    np.savez(d_fullpath, pos = pos, vel = vel, idx=idx, sim_time = sim_time)
    print('Particle data saved')


#%% FIELDS
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

    B       -= 0.5 * DT * curlE                          # Advance using curl (apply retarding factor here?)
    
    for ii in nb.prange(1, B.shape[1]):                  # Apply damping, skipping x-axis
        B[:, ii] *= damping_array                        # Not sure if this needs to modified for half steps?
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
    physical space domain. Guard cells set to zero.
    '''
    temp     *= 0; grad_P *= 0
    nc        = qn.shape[0]
    grad_P[:] = qn * kB * te / q       # Store Pe in grad_P array for calculation
    LC        = NX + ND - 1

    # Central differencing, internal points
    for ii in nb.prange(ND + 1, LC - 1):
        temp[ii] = (grad_P[ii + 1] - grad_P[ii - 1])
    
    # Forwards/Backwards difference at physical boundaries
    temp[ND] = -3*grad_P[ND] + 4*grad_P[ND + 1] - grad_P[ND + 2]
    temp[LC] =  3*grad_P[LC] - 4*grad_P[LC - 1] + grad_P[LC - 2]
    temp    /= (2*dx)
    
    # Return value
    grad_P[:]    = temp[:nc]
    return


@nb.njit()
def calculate_E(B, Ji, q_dens, E, Ve, Te, temp3De, temp3Db, grad_P):
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
    
    To Do: In the interpolation function, add on B0 along with the 
    spline interpolation. No other part of the code requires B0 in the nodes.
    '''
    curl_B_term(B, temp3De)                                   # temp3De is now curl B term

    Ve[:, 0] = (Ji[:, 0] - temp3De[:, 0]) / q_dens
    Ve[:, 1] = (Ji[:, 1] - temp3De[:, 1]) / q_dens
    Ve[:, 2] = (Ji[:, 2] - temp3De[:, 2]) / q_dens

    get_electron_temp(q_dens, Te)

    get_grad_P(q_dens, Te, grad_P, temp3Db[:, 0])            # temp1D is now del_p term, temp3D2 slice used for computation
    interpolate_edges_to_center(B, temp3Db)              # temp3db is now B_center

    cross_product(Ve, temp3Db, temp3De)                  # temp3De is now Ve x B term

    E[:, 0]  = - temp3De[:, 0] - grad_P[:] / q_dens[:]
    E[:, 1]  = - temp3De[:, 1]
    E[:, 2]  = - temp3De[:, 2]
    
    # Diagnostic flag for testing
    if disable_waves == True:   
        E *= 0.
    return 


#%% SOURCE TERM COLLECTION
@nb.njit()
def deposit_moments_to_grid(vel, Ie, W_elec, idx, ni, nu):
    '''Collect number and velocity moments in each cell, weighted by their distance
    from cell nodes.

    INPUT:
        vel    -- Particle 3-velocities
        Ie     -- Particle leftmost to nearest E-node
        W_elec -- Particle TSC weighting across nearest, left, and right nodes
        idx    -- Particle species identifier

    OUTPUT:
        ni     -- Species number moment array(size, Nj)
        nui    -- Species velocity moment array (size, Nj)
        
    13/03/2020 :: Modified to ignore contributions from particles with negative
                    indices (i.e. "deactivated" particles)
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
def collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu, temp1D):
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

    # Mirror source term contributions at edge back into domain
    q_dens[ND]          += q_dens[ND - 1]
    q_dens[ND + NX - 1] += q_dens[ND + NX]

    # Set damping cell source values
    q_dens[:ND]    = q_dens[ND]
    q_dens[ND+NX:] = q_dens[ND+NX-1]
    
    for ii in range(3):
        Ji[ND, ii]          += Ji[ND - 1, ii]
        Ji[ND + NX - 1, ii] += Ji[ND + NX, ii]
    
        Ji[:ND, ii] = Ji[ND, ii]
        Ji[ND+NX:]  = Ji[ND+NX-1]
        
    # Set density minimum
    for ii in range(q_dens.shape[0]):
        if q_dens[ii] < min_dens * ne * q:
            q_dens[ii] = min_dens * ne * q
    return


#%% PARTICLES
@nb.njit()
def advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                  B, E, DT, q_dens_adv, Ji, ni, nu, temp1D, pc=0):
    '''
    Helper function to group the particle advance and moment collection functions
    '''
    velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, DT)
    position_update(pos, vel, idx, DT, Ie, W_elec)  
    
    if disable_waves == False:
        collect_moments(vel, Ie, W_elec, idx, q_dens_adv, Ji, ni, nu, temp1D)
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
    '''
    Np         = pos.shape[1]
    epsil      = 1e-15
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 0.0
    
    particle_transform = xmax + (ND - grid_offset)*dx  + epsil  # Offset to account for E/B grid and damping nodes
    
    for ii in np.arange(Np):
        xp          = (pos[0, ii] + particle_transform) / dx    # Shift particle position >= 0
        I[ii]       = int(round(xp) - 1.0)                      # Get leftmost to nearest node
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
            W[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))      # Get weighting factors
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
        
    Could totally vectorise this. Would have to change to give a particle_temp
    array for memory allocation or something
    '''
    rL     = np.sqrt(pos[1]**2 + pos[2]**2)
    
    B0_r   = - a * B_eq * pos[0] * rL
    Bp[0]  = eval_B0x(pos[0])   
    Bp[1] += B0_r * pos[1] / rL
    Bp[2] += B0_r * pos[2] / rL
    return


@nb.njit()
def velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, dt):
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
    
    4 temp arrays of length 3 used.
    '''
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)                     # Calculate magnetic node weights
    
    for ii in nb.prange(vel.shape[1]):  
        if idx[ii] >= 0:
            qmi = 0.5 * dt * qm_ratios[idx[ii]]                                 # Charge-to-mass ration for ion of species idx[ii]
    
            # These create two new length 3 arrays
            if disable_waves == False:
                Ep = E[Ie[ii]    , 0:3] * W_elec[0, ii]                             \
                   + E[Ie[ii] + 1, 0:3] * W_elec[1, ii]                             \
                   + E[Ie[ii] + 2, 0:3] * W_elec[2, ii]                             # Vector E-field at particle location
        
                Bp = B[Ib[ii]    , 0:3] * W_mag[0, ii]                              \
                   + B[Ib[ii] + 1, 0:3] * W_mag[1, ii]                              \
                   + B[Ib[ii] + 2, 0:3] * W_mag[2, ii]                              # b1 at particle location
            else:
                Ep = np.zeros(3); Bp = np.zeros(3)
                
            v_minus    = vel[:, ii] + qmi * Ep                                  # First E-field half-push
            
            eval_B0_particle(pos[:, ii], Bp)                                    # Add B0 at particle location
            
            T = qmi * Bp                                                        # Vector Boris variable
            S = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)                 # Vector Boris variable
            
            v_prime    = np.zeros(3)
            v_prime[0] = v_minus[0] + v_minus[1] * T[2] - v_minus[2] * T[1]     # Magnetic field rotation
            v_prime[1] = v_minus[1] + v_minus[2] * T[0] - v_minus[0] * T[2]
            v_prime[2] = v_minus[2] + v_minus[0] * T[1] - v_minus[1] * T[0]
                    
            v_plus     = np.zeros(3)
            v_plus[0]  = v_minus[0] + v_prime[1] * S[2] - v_prime[2] * S[1]
            v_plus[1]  = v_minus[1] + v_prime[2] * S[0] - v_prime[0] * S[2]
            v_plus[2]  = v_minus[2] + v_prime[0] * S[1] - v_prime[1] * S[0]
            
            vel[:, ii] = v_plus +  qmi * Ep                                     # Second E-field half-push
    return


@nb.njit()
def position_update(pos, vel, idx, dt, Ie, W_elec, diag=False):
    '''Updates the position of the particles using x = x0 + vt. 
    Also updates particle nearest node and weighting.

    INPUT:
        part   -- Particle array with positions to be updated
        dt     -- Time cadence of simulation

    OUTPUT:
        pos    -- Particle updated positions
        W_elec -- (0) Updated nearest E-field node value and (1-2) left/centre weights
        
    Reflective boundaries to simulate the "open ends" that would have flux coming in from the ionosphere side.
    
    These equations aren't quite right for xmax != xmin, but they'll do for now
    '''
    for ii in nb.prange(pos.shape[1]):
        # Only update particles that haven't been absorbed (positive species index)
        if idx[ii] >= 0:
            pos[0, ii] += vel[0, ii] * dt
            pos[1, ii] += vel[1, ii] * dt
            pos[2, ii] += vel[2, ii] * dt
    
            # Particle boundary conditions
            if (pos[0, ii] < xmin or pos[0, ii] > xmax):
                # Absorb particles
                vel[:, ii] *= 0       # Zero particle velocity
                idx[ii]    -= 128     # Fold index to negative values (preserves species ID)
                
# =============================================================================
#                 # Mario particles
#                 if pos[0, ii] > xmax:
#                     pos[0, ii] = pos[0, ii] - xmax + xmin
#                 elif pos[0, ii] < xmin:
#                     pos[0, ii] = pos[0, ii] + xmax - xmin
# =============================================================================
# =============================================================================
#                 # Reflect particles
#                 if pos[0, ii] > xmax:
#                     pos[0, ii] = 2*xmax - pos[0, ii]
#                 elif pos[0, ii] < xmin:
#                     pos[0, ii] = 2*xmin - pos[0, ii]
# 
#                 # 'Reflect' velocities as well. 
#                 # vel[0]   to make it travel in opposite directoin
#                 # vel[1:2] to keep it resonant with ions travelling in that direction
#                 vel[:, ii] *= -1.0
# =============================================================================

    assign_weighting_TSC(pos, Ie, W_elec)
    return


#%% AUXILLIARY FUNCTIONS
@nb.njit()
def cross_product(A, B, C):
    '''
    Vector (cross) product between two vectors, A and B of same dimensions.

    INPUT:
        A, B -- 3D vectors (ndarrays)

    OUTPUT:
        C -- The resultant cross product with same dimensions as input vectors
        
    Could be more memory efficient to "accumulate" operation, but would involve rewriting
    for each specific instance.
    '''
    for ii in nb.prange(A.shape[0]):
        C[ii, 0] = A[ii, 1] * B[ii, 2] - A[ii, 2] * B[ii, 1]
        C[ii, 1] = A[ii, 2] * B[ii, 0] - A[ii, 0] * B[ii, 2]
        C[ii, 2] = A[ii, 0] * B[ii, 1] - A[ii, 1] * B[ii, 0]
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
    
    interp[:ND,      0] = interp[ND,    0]
    interp[ND+NX+1:, 0] = interp[ND+NX, 0]
    return


@nb.njit()
def check_timestep(pos, vel, B, E, q_dens, Ie, W_elec, Ib, W_mag, B_center, \
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
        elecfreq        = qm_ratios.max()*(np.abs(E[:, 0] / vel.max()).max())               # Electron acceleration "frequency"
        Eacc_ts         = freq_res / elecfreq                            
    else:
        Eacc_ts = ion_ts
    
    if account_for_dispersion == True:
        B_tot           = np.sqrt(B_center[:, 0] ** 2 + B_center[:, 1] ** 2 + B_center[:, 2] ** 2)
    
        dispfreq        = ((np.pi / dx) ** 2) * (B_tot / (mu0 * q_dens)).max()           # Dispersion frequency

        disp_ts     = dispersion_allowance * freq_res / dispfreq     # Making this a little bigger so it doesn't wreck everything
    else:
        disp_ts     = ion_ts

    vel_ts          = 0.60 * dx / np.abs(vel[0, :]).max()                        # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than 'half' a cell in one time step
    DT_part         = min(Eacc_ts, vel_ts, ion_ts, disp_ts)                      # Smallest of the allowable timesteps
    
    if DT_part < 0.9*DT:

        velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, 0.5*DT)    # Re-sync vel/pos       

        DT         *= 0.5
        max_inc    *= 2
        qq         *= 2
        
        field_save_iter *= 2
        part_save_iter *= 2

        velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, -0.5*DT)   # De-sync vel/pos 
        print('Timestep halved. Syncing particle velocity...')
        set_damping_array(damping_array, DT)
            
# =============================================================================
#     elif DT_part >= 4.0*DT and qq%2 == 0 and part_save_iter%2 == 0 and field_save_iter%2 == 0 and max_inc%2 == 0:
#         particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, 0.5*DT)    # Re-sync vel/pos          
#         DT         *= 2.0
#         max_inc   //= 2
#         qq        //= 2
# 
#         field_save_iter //= 2
#         part_save_iter //= 2
#             
#         particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, -0.5*DT)   # De-sync vel/pos 
#         print('Timestep Doubled. Syncing particle velocity...')
#         init.set_damping_array(damping_array, DT)
# =============================================================================

    return qq, DT, max_inc, part_save_iter, field_save_iter, damping_array


@nb.njit()
def main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag,                      \
              B, E_int, E_half, q_dens, q_dens_adv, Ji, ni, nu,          \
              Ve, Te, temp3De, temp3Db, temp1D, old_particles, old_fields,\
              damping_array, qq, DT, max_inc, part_save_iter, field_save_iter):
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
    qq, DT, max_inc, part_save_iter, field_save_iter, damping_array \
    = check_timestep(pos, vel, B, E_int, q_dens, Ie, W_elec, Ib, W_mag, temp3De, \
                     qq, DT, max_inc, part_save_iter, field_save_iter, idx, damping_array)
    
    # Move particles, collect moments
    advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                            B, E_int, DT, q_dens_adv, Ji, ni, nu, temp1D)
    
    # Average N, N + 1 densities (q_dens at N + 1/2)
    q_dens *= 0.5
    q_dens += 0.5 * q_dens_adv
    
    # Push B from N to N + 1/2
    push_B(B, E_int, temp3Db, DT, qq, damping_array, half_flag=1)
    
    # Calculate E at N + 1/2
    calculate_E(B, Ji, q_dens, E_half, Ve, Te, temp3De, temp3Db, temp1D)
    
    if disable_waves == False:
        ###################################
        ### PREDICTOR CORRECTOR SECTION ###
        ###################################
    
        # Store old values
        old_particles[0:3 , :] = pos
        old_particles[3:6 , :] = vel
        old_particles[6   , :] = Ie
        old_particles[7:10, :] = W_elec
        
        old_fields[:,   0:3]  = B
        old_fields[:NC, 3:6]  = Ji
        old_fields[:NC, 6:9]  = Ve
        old_fields[:NC,   9]  = Te
        
        # Predict fields
        E_int *= -1.0
        E_int +=  2.0 * E_half
        
        push_B(B, E_int, temp3Db, DT, qq, damping_array, half_flag=0)
    
        # Advance particles to obtain source terms at N + 3/2
        advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, \
                                                B, E_int, DT, q_dens, Ji, ni, nu, temp1D, pc=1)
        
        q_dens *= 0.5;    q_dens += 0.5 * q_dens_adv
        
        # Compute predicted fields at N + 3/2
        push_B(B, E_int, temp3Db, DT, qq + 1, damping_array, half_flag=1)
        calculate_E(B, Ji, q_dens, E_int, Ve, Te, temp3De, temp3Db, temp1D)
        
        # Determine corrected fields at N + 1 
        E_int *= 0.5;    E_int += 0.5 * E_half
    
        # Restore old values: [:] allows reference to same memory (instead of creating new, local instance)
        pos[:]    = old_particles[0:3 , :]
        vel[:]    = old_particles[3:6 , :]
        Ie[:]     = old_particles[6   , :]
        W_elec[:] = old_particles[7:10, :]
        B[:]      = old_fields[:,   0:3]
        Ji[:]     = old_fields[:NC, 3:6]
        Ve[:]     = old_fields[:NC, 6:9]
        Te[:]     = old_fields[:NC,   9]
        
        push_B(B, E_int, temp3Db, DT, qq, damping_array, half_flag=0)   # Advance the original B
    
        q_dens[:] = q_dens_adv

    return qq, DT, max_inc, part_save_iter, field_save_iter


#%% _______________________________________
#%% INPUT PARAMETERS
if __name__ == '__main__':
    ### RUN DESCRIPTION ###
    run_description = '''SINGLEFILE Validation Test :: No wave fields :: Test to see if particles are trapped'''
    
    ### RUN PARAMETERS ###
    drive             = 'F:'                          # Drive letter or path for portable HDD e.g. 'E:/' or '/media/yoshi/UNI_HD/'
    save_path         = 'runs//validation_runs_v2'    # Series save dir   : Folder containing all runs of a series
    run               = 1                             # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)
    save_particles    = 1                             # Save data flag    : For later analysis
    save_fields       = 1                             # Save plot flag    : To ensure hybrid is solving correctly during run
    seed              = 3216587                       # RNG Seed          : Set to enable consistent results for parameter studies
    cpu_affin         = [(2*run)%8, (2*run + 1)%8]    # Set CPU affinity for run. Must be list. Auto-assign: None.
    
    ## DIAGNOSTIC FLAGS :: DOUBLE CHECK BEFORE EACH RUN! ##
    ## THESE FLAGS NO LONGER TRIGGER THE SETTING, ONLY THE SAVE PARAMETER.
    ## SETTING MUST BE CHANGED IN THE PARTICLES.PY FILE BY UNCOMMENTING THE APPROPRIATE CODE
    homogenous        = False                         # Set B0 to homogenous (as test to compare to parabolic)
    reflect           = False                         # Reflect particles when they hit boundary (Default: Absorb)
    periodic          = False                         # Set periodic boundary conditions for particles. Overrides reflection flag.
    
    # OTHER FLAGS
    disable_waves     = True                          # Disables solutions to wave fields. Only background magnetic field exists
    supress_text      = False                         # Supress initialization text
    
    ### PHYSICAL CONSTANTS ###
    q      = 1.602177e-19                       # Elementary charge (C)
    c      = 2.998925e+08                       # Speed of light (m/s)
    mp     = 1.672622e-27                       # Mass of proton (kg)
    me     = 9.109384e-31                       # Mass of electron (kg)
    kB     = 1.380649e-23                       # Boltzmann's Constant (J/K)
    e0     = 8.854188e-12                       # Epsilon naught - permittivity of free space
    mu0    = (4e-7) * np.pi                     # Magnetic Permeability of Free Space (SI units)
    RE     = 6.371e6                            # Earth radius in metres
    B_surf = 3.12e-5                            # Magnetic field strength at Earth surface
    
    
    ### SIMULATION PARAMETERS ###
    NX        = 1024                            # Number of cells - doesn't include ghost cells
    ND        = 256                             # Damping region length: Multiple of NX (on each side of simulation domain)
    max_rev   = 25000                           # Simulation runtime, in multiples of the ion gyroperiod (in seconds)
    dxm       = 1.0                             # Number of c/wpi per dx (Ion inertial length: anything less than 1 isn't "resolvable" by hybrid code, anything too much more than 1 does funky things to the waveform)
    L         = 5.35                            # Field line L shell
    
    ie        = 1                               # Adiabatic electrons. 0: off (constant), 1: on.
    B_eq      = None                          # Initial magnetic field at equator: None for L-determined value (in T)
    rc_hwidth = 0                               # Ring current half-width in number of cells (2*hwidth gives total cells with RC) 
      
    orbit_res = 0.02                            # Orbit resolution
    freq_res  = 0.02                            # Frequency resolution     : Fraction of angular frequency for multiple cyclical values
    part_res  = 0.25                            # Data capture resolution in gyroperiod fraction: Particle information
    field_res = 0.25                            # Data capture resolution in gyroperiod fraction: Field information
    
    
    ### PARTICLE PARAMETERS ###
    species_lbl= [r'$H^+$ cold', r'$H^+$ warm']                 # Species name/labels        : Used for plotting. Can use LaTeX math formatted strings
    temp_color = ['blue', 'red']
    temp_type  = np.array([0, 1])             	                # Particle temperature type  : Cold (0) or Hot (1) : Used for plotting
    dist_type  = np.array([0, 0])                               # Particle distribution type : Uniform (0) or sinusoidal/other (1) : Used for plotting (normalization)
    nsp_ppc    = np.array([200, 200])                          # Number of particles per cell, per species - i.e. each species has equal representation (or code this to be an array later?)
    
    mass       = np.array([1., 1.])    			                # Species ion mass (proton mass units)
    charge     = np.array([1., 1.])    			                # Species ion charge (elementary charge units)
    drift_v    = np.array([0., 0.])                             # Species parallel bulk velocity (alfven velocity units)
    density    = np.array([180., 20.]) * 1e6                    # Species density in /cc (cast to /m3)
    anisotropy = np.array([0.0, 5.0])                           # Particle anisotropy: A = T_per/T_par - 1
    
    # Particle energy: Choose one                                    
    E_per      = np.array([5.0, 50000.])                        # Perpendicular energy in eV
    beta_par   = np.array([1., 10.])                            # Overrides E_per if not None. Uses B_eq for conversion
    
    min_dens       = 0.05                                       # Allowable minimum charge density in a cell, as a fraction of ne*q
    E_e            = 10.0                                       # Electron energy (eV)
    
    # This will be fixed by subcycling later on, hopefully
    account_for_dispersion = False                              # Flag (True/False) whether or not to reduce timestep to prevent dispersion getting too high
    dispersion_allowance   = 1.                                 # Multiple of how much past frac*wD^-1 is allowed: Used to stop dispersion from slowing down sim too much  
    
    
    #%%### DERIVED SIMULATION PARAMETERS
    NC         = NX + 2*ND
    ne         = density.sum()
    E_par      = E_per / (anisotropy + 1)
    
    if B_eq is None:
        B_eq      = (B_surf / (L ** 3))                      # Magnetic field at equator, based on L value
        
    if beta_par is None:
        Te0        = E_e   * 11603.
        Tpar       = E_par * 11603.
        Tper       = E_per * 11603.
    else:
        beta_per   = beta_par * (anisotropy + 1)
        
        Tpar       = beta_par    * B_eq ** 2 / (2 * mu0 * ne * kB)
        Tper       = beta_per    * B_eq ** 2 / (2 * mu0 * ne * kB)
        Te0        = beta_par[0] * B_eq ** 2 / (2 * mu0 * ne * kB)
    
    wpi        = np.sqrt(ne * q ** 2 / (mp * e0))            # Proton   Plasma Frequency, wpi (rad/s)
    va         = B_eq / np.sqrt(mu0*ne*mp)                   # Alfven speed at equator: Assuming pure proton plasma
    
    dx         = dxm * c / wpi                               # Spatial cadence, based on ion inertial length
    xmax       = NX // 2 * dx                                # Maximum simulation length, +/-ve on each side
    xmin       =-NX // 2 * dx
    
    charge    *= q                                           # Cast species charge to Coulomb
    mass      *= mp                                          # Cast species mass to kg
    drift_v   *= va                                          # Cast species velocity to m/s
    
    Nj         = len(mass)                                   # Number of species
    n_contr    = density / nsp_ppc                           # Species density contribution: Each macroparticle contributes this density to a cell
    
    # Number of sim particles for each species, total
    N_species  = np.zeros(Nj, dtype=np.int64)
    for jj in range(Nj):
        # Cold species in every cell NX 
        if temp_type[jj] == 0:                               
            N_species[jj] = nsp_ppc[jj] * NX + 2   
            
        # Warm species only in simulation center, unless rc_hwidth = 0 (disabled)           
        elif temp_type[jj] == 1:
            if rc_hwidth == 0:
                N_species[jj] = nsp_ppc[jj] * NX + 2
            else:
                N_species[jj] = nsp_ppc[jj] * 2*rc_hwidth + 2    
    N = N_species.sum()
    
    idx_start  = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
    idx_end    = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order
    
    ############################
    ### MAGNETIC FIELD STUFF ###
    ############################
    B_nodes  = (np.arange(NC + 1) - NC // 2)       * dx      # B grid points position in space
    E_nodes  = (np.arange(NC)     - NC // 2 + 0.5) * dx      # E grid points position in space
    
    theta_xmax  = xmax/(L*RE)                                # Latitudinal extent of simulation , based on xmax
    r_xmax      = L * np.sin(np.pi / 2 - theta_xmax) ** 2    # Calculate radial distance of boundary in dipole and get field intensity
    cos_bit     = np.sqrt(3*np.cos(theta_xmax)**2 + 1)       # Intermediate variable (angular scale factor)
    B_xmax      = (B_surf / (r_xmax ** 3)) * cos_bit         # Magnetic field intensity at boundary
    a           = (B_xmax / B_eq - 1) / xmax ** 2            # Parabolic scale factor: Fitted to B_eq, B_xmax
    
    if homogenous == True:
        a      = 0
        B_xmax = B_eq
        
    Bc           = np.zeros((NC + 1, 3), dtype=np.float64)   # Constant components of magnetic field based on theta and B0
    Bc[:, 0]     = B_eq * (1 + a * B_nodes**2)               # Set constant Bx
    Bc[:ND]      = Bc[ND]                                    # Set B0 in damping cells (same as last spatial cell)
    Bc[ND+NX+1:] = Bc[ND+NX]
    
    # Freqs based on highest magnetic field value (at simulation boundaries)
    gyfreq     = q*B_xmax/ mp                                # Proton Gyrofrequency (rad/s) at boundary (highest)
    gyfreq_eq  = q*B_eq  / mp                                # Proton Gyrofrequency (rad/s) at equator (slowest)
    k_max      = np.pi / dx                                  # Maximum permissible wavenumber in system (SI???)
    qm_ratios  = np.divide(charge, mass)                     # q/m ratio for each species
    
    loss_cone  = np.arcsin(np.sqrt(B_eq / B_xmax))*180 / np.pi
    
    if rc_hwidth == 0:
        rc_print = NX
    else:
        rc_print = rc_hwidth*2
    
    
    #%%### INPUT TESTS AND CHECKS
    
    
    if supress_text == False:
        print('Run Started')
        print('Run Series         : {}'.format(save_path.split('//')[-1]))
        print('Run Number         : {}'.format(run))
        print('Field save flag    : {}'.format(save_fields))
        print('Particle save flag : {}\n'.format(save_particles))
        
        print('Sim domain length  : {:5.2f}R_E'.format(2 * xmax / RE))
        print('Density            : {:5.2f}cc'.format(ne / 1e6))
        print('Equatorial B-field : {:5.2f}nT'.format(B_eq*1e9))
        print('Maximum    B-field : {:5.2f}nT'.format(B_xmax*1e9))
        print('Loss cone          : {:<5.2f} degrees  '.format(loss_cone))
        print('Maximum MLAT (+/-) : {:<5.2f} degrees\n'.format(theta_xmax * 180. / np.pi))
        
        print('Equat. Gyroperiod: : {}s'.format(round(2. * np.pi / gyfreq, 3)))
        print('Inverse rad gyfreq : {}s'.format(round(1 / gyfreq, 3)))
        print('Maximum sim time   : {}s ({} gyroperiods)\n'.format(round(max_rev * 2. * np.pi / gyfreq_eq, 2), max_rev))
        
        print('{} spatial cells, {} with ring current, 2x{} damped cells'.format(NX, rc_print, ND))
        print('{} cells total'.format(NC))
        print('{} particles total\n'.format(N))
        
        if None not in cpu_affin:
            import psutil
            run_proc = psutil.Process()
            run_proc.cpu_affinity(cpu_affin)
            if len(cpu_affin) == 1:
                print('CPU affinity for run (PID {}) set to logical core {}'.format(run_proc.pid, run_proc.cpu_affinity()[0]))
            else:
                print('CPU affinity for run (PID {}) set to logical cores {}'.format(run_proc.pid, ', '.join(map(str, run_proc.cpu_affinity()))))
            
    print('Checking directories...')
    if (save_particles == 1 or save_fields == 1) == True:
        if os.path.exists('%s/%s' % (drive, save_path)) == False:
            os.makedirs('%s/%s' % (drive, save_path))                        # Create master test series directory
            print('Master directory created')

        path = ('%s/%s/run_%d' % (drive, save_path, run))          # Set root run path (for images)
        
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
                sys.exit('Program Terminated: Change run in simulation_parameters_1D')
            else:
                sys.exit('Unfamiliar input: Run terminated for safety')

    #%% Start simulation
    main()