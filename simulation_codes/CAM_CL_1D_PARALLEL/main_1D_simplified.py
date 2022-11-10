## PYTHON MODULES ##
from timeit import default_timer as timer
import numpy as np
import numba as nb
import sys, os, pdb

## PHYSICAL CONSTANTS ##
ECHARGE = 1.602177e-19                       # Elementary charge (C)
PMASS   = 1.672622e-27                       # Mass of proton (kg)
EMASS   = 9.109384e-31                       # Mass of electron (kg)
kB      = 1.380649e-23                       # Boltzmann's Constant (J/K)
e0      = 8.854188e-12                       # Epsilon naught - permittivity of free space
mu0     = (4e-7) * np.pi                     # Magnetic Permeability of Free Space (SI units)
RE      = 6.371e6                            # Earth radius in metres
B_surf  = 3.12e-5                            # Magnetic field strength at Earth surface (equatorial)

# A few internal flags
do_parallel         = True
print_timings       = False      # Diagnostic outputs timing each major segment (for efficiency examination)
print_runtime       = True       # Flag to print runtime every 50 iterations 

if not do_parallel:
    do_parallel = True
    nb.set_num_threads(1)          
#nb.set_num_threads(6)


#%% --- FUNCTIONS ---
#%% INITIALIZATION
@nb.njit()
def quiet_start_bimaxwellian():
    np.random.seed(seed)
    pos = np.zeros(N, dtype=np.float64)
    vel = np.zeros((3, N), dtype=np.float64)
    idx = np.ones(N,       dtype=np.int8) * Nj

    for jj in range(Nj):
        idx[idx_start[jj]: idx_end[jj]] = jj          # Set particle idx        
        half_n = nsp_ppc[jj] // 2                     # Half particles per cell - doubled later
        
        # Load particles in each applicable cell
        acc = 0
        for ii in range(NX):               
            # Particle index ranges
            st = idx_start[jj] + acc
            en = idx_start[jj] + acc + half_n
            
            # Set position for half: Analytically uniform
            for kk in range(half_n):
                pos[st + kk] = dx*(float(kk) / half_n + ii)
            
            # Turn [0, NC] distro into +/- NC/2 distro
            pos[st: en] -= 0.5*NX*dx              
            
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


def initialize_particles(B, E):
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
    rho_half = np.zeros(NC, dtype=np.float64)
    rho_int  = np.zeros(NC, dtype=np.float64)
    
    Ji       = np.zeros((NC, 3), dtype=np.float64)
    Ji_plus  = np.zeros((NC, 3), dtype=np.float64)
    Ji_minus = np.zeros((NC, 3), dtype=np.float64)
    L        = np.zeros( NC,     dtype=np.float64)
    G        = np.zeros((NC, 3), dtype=np.float64)
    return rho_half, rho_int, Ji, Ji_plus, Ji_minus, L, G



def set_timestep(vel):
    max_vx    = np.max(np.abs(vel[0, :]))         # Fastest particle velocity
    ion_ts    = orbit_res / gyfreq_eq             # Timestep to resolve gyromotion
    vel_ts    = 0.5*dx / max_vx                   # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step
    DT        = min(ion_ts, vel_ts)               # Set global timestep as smallest of these
    max_time  = max_wcinv / gyfreq_eq             # Total runtime in seconds
    subcycles = default_subcycles                 # Number of subcycles per particle step
    
    # Could put a check in here to make sure the interplay between
    # timestep and subcycle satisfies everything.
    
    while subcycles%4 != 0:                       # Force subcycle count to be a factor of 4
        subcycles += 1
    
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
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 0.0
    
    particle_transform = xmax + (ND - grid_offset)*dx  + 1e-15      # Offset to account for E/B grid and damping nodes
    
    for ii in nb.prange(pos.shape[0]):
        xp          = (pos[ii] + particle_transform) / dx       # Shift particle position >= 0
        I[ii]       = int(round(xp) - 1.0)                      # Get leftmost to nearest node (Vectorize?)
        delta_left  = I[ii] - xp                                # Distance from left node in grid units

        W[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))  # Get weighting factors
        W[1, ii] = 0.75 - np.square(delta_left + 1.)
        W[2, ii] = 1.0  - W[0, ii] - W[1, ii]
    return


@nb.njit(parallel=do_parallel)
def velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, 
                    dt):    
    for ii in nb.prange(pos.shape[0]):
        Ep = np.zeros(3, dtype=np.float64)  
        Bp = np.zeros(3, dtype=np.float64)
        for jj in nb.prange(3):
            for kk in nb.prange(3):
                Ep[kk] += E[Ie[ii] + jj, kk] * W_elec[jj, ii]   
                Bp[kk] += B[Ib[ii] + jj, kk] * W_mag[ jj, ii]   
        
        # q/m variable including dt
        qmi = 0.5 * dt * qm_ratios[idx[ii]]                             
        
        # vel -> v_minus
        vel[0, ii] += qmi * Ep[0]
        vel[1, ii] += qmi * Ep[1]
        vel[2, ii] += qmi * Ep[2]
        
        # Boris variables
        Bp[0]    += B_eq
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
def position_update(pos, vel, idx, Ie, W_elec, Ib, W_mag, dt):
    for ii in nb.prange(pos.shape[0]):
        pos[ii] += vel[0, ii] * dt
        
        # Check if particle has left simulation and apply boundary conditions
        if (pos[ii] < xmin or pos[ii] > xmax):
            idx[ii] += Nj                            
    
    periodic_BC(pos, idx)
    assign_weighting_TSC(pos, Ie, W_elec)
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)
    return


@nb.njit(parallel=do_parallel)
def periodic_BC(pos, idx):
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


#%% SOURCES
@nb.njit()
def push_current(J_in, J_out, E, B_center, L, G, dt):
    J_out    *= 0
    
    G_cross_B = np.zeros(E.shape, dtype=np.float64)
    for ii in np.arange(NC):
        G_cross_B[ii, 0] = G[ii, 1] * B_center[ii, 2] - G[ii, 2] * B_center[ii, 1]
        G_cross_B[ii, 1] = G[ii, 2] * B_center[ii, 0] - G[ii, 0] * B_center[ii, 2]
        G_cross_B[ii, 2] = G[ii, 0] * B_center[ii, 1] - G[ii, 1] * B_center[ii, 0]
    
    for ii in range(3):
        J_out[:, ii] = J_in[:, ii] + 0.5*dt * (L * E[:, ii] + G_cross_B[:, ii]) 
    
    # Copy periodic values
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
def deposit_moments(vel, Ie, W_elec, idx, ni, nu):
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


@nb.njit()
def manage_source_term_boundaries(arr):
    '''Will only ever get 1D arrays of length NC'''
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
    # TODO: Does this cause any issues if there are only exactly 2 ghost cells?
    arr[:lo2] = arr[lo2]
    arr[ro2:] = arr[ro2]
    return


@nb.njit()
def init_collect_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, rho_0, rho, J_init, J_plus,
                         L, G, dt):
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
                         
    deposit_moments(vel, Ie, W_elec, idx, ni_init, nu_init)
    position_update(pos, vel, idx, Ie, W_elec, Ib, W_mag, dt)
    deposit_moments(vel, Ie, W_elec, idx, ni, nu_plus)
    
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
    return


@nb.njit()
def advance_particles_and_collect_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E,
                    rho_int, rho_half, Ji, Ji_minus, Ji_plus, L, G, dt):
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
    deposit_moments(vel, Ie, W_elec, idx, ni, nu_minus)
    position_update(pos, vel, idx, Ie, W_elec, Ib, W_mag, dt)
    deposit_moments(vel, Ie, W_elec, idx, ni, nu_plus)
    
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
            
    rho_int += rho_half
    rho_int /= 2.0
    Ji[:]    = 0.5 * (Ji_plus  +  Ji_minus)
    return


#%% FIELDS
@nb.njit(parallel=False)
def get_curl_B(B):
    curlB = np.zeros((B.shape[0] - 1, 3), dtype=nb.float64)
    for ii in nb.prange(B.shape[0] - 1):
        curlB[ii, 1] = - (B[ii + 1, 2] - B[ii, 2])
        curlB[ii, 2] =    B[ii + 1, 1] - B[ii, 1]
    curlB /= dx
    return curlB


@nb.njit()
def get_curl_E(E, dE):
    dE *= 0.
    for ii in nb.prange(1, E.shape[0]):
        dE[ii, 1] = - (E[ii, 2] - E[ii - 1, 2])
        dE[ii, 2] =    E[ii, 1] - E[ii - 1, 1]

    dE /= dx
    return


@nb.njit(parallel=False)
def get_grad_P(qn, te):
    grad_pe       = np.zeros(NC    , dtype=np.float64)
    grad_pe_B     = np.zeros(NC + 1, dtype=np.float64)
    grad_pe[:]    = qn[:] * kB * te[:] / ECHARGE

    # Loop center points, set endpoints for no gradients (just to be safe)
    for ii in np.arange(1, qn.shape[0]):
        grad_pe_B[ii] = (grad_pe[ii] - grad_pe[ii - 1])/dx
    grad_pe_B[0]  = grad_pe_B[1]
    grad_pe_B[NC] = grad_pe_B[NC - 1]
    
    grad_pe = 0.5*(grad_pe_B[:-1] + grad_pe_B[1:])
    return grad_pe


#@nb.njit()
def cyclic_leapfrog(B1, B2, B_center, rho, Ji, E, Ve, Te, dt, subcycles,
                    sim_time, half_step):
    '''
    Subcycles are defined per particle step, so only half are used per
    field push (since they are only half timestep pushes)
    
    B2 is always the first pushed, and B1 is always the output
    (Subcycles will always land with B1 being the final updated)
    '''
    shalf = subcycles // 2
    H     = 0.5 * dt
    dh    = H / shalf
    curl  = np.zeros((NC + 1, 3), dtype=np.float64)
    
    ## MAIN SUBCYCLE LOOP ##
    for ii in range(shalf):
        if ii%2 == 0:
            calculate_E(B1, B_center, Ji, rho, E, Ve, Te, sim_time)
            get_curl_E(E, curl) 
            B2  -= (2 - half_step) * dh * curl
            get_B_cent(B2, B_center)
            print('Pushing B2, sum:', B2.sum())
        else:
            calculate_E(B2, B_center, Ji, rho, E, Ve, Te, sim_time)
            get_curl_E(E, curl) 
            B1  -= 2 * dh * curl
            get_B_cent(B1, B_center)
            print('Pushing B1, sum:', B1.sum())
            
        sim_time += dh

    ## ERROR CHECK ## 
    if True:
        print('Error checking...')
        E_temp = E.copy()
        calculate_E(B1, B_center, Ji, rho, E_temp, Ve, Te, sim_time)
        get_curl_E(E_temp, curl) 
        B_temp = B2 - dh * curl
        print(B_temp)
        error = (2.*np.abs(B_temp - B1) / (B_temp + B1)).sum()
        print('Error is', error)
        pdb.set_trace()
        if error > 1e-4:
            B1 += B_temp; B1 /= 2.0     # Average fields
            B2[:] = B1[:]               # Set them equal
            half_step = 1               # Flag for a desync at the next push 
            print('Averaged and flagged')
    
    # Calculate final values
    get_B_cent(B1, B_center)
    calculate_E(B1, B_center, Ji, rho, E, Ve, Te, sim_time)
    return sim_time, half_step


@nb.njit()
def calculate_E(B, B_center, Ji, qn, E, Ve, Te, sim_time):    
    curlB  = get_curl_B(B)
    curlB /= mu0
       
    Ve[:, 0] = (Ji[:, 0] - curlB[:, 0]) / qn
    Ve[:, 1] = (Ji[:, 1] - curlB[:, 1]) / qn
    Ve[:, 2] = (Ji[:, 2] - curlB[:, 2]) / qn
    
    Te
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
    _B_cent[:, 1] = 0.5*(B[:-1, 1] + B[1:, 1])
    _B_cent[:, 2] = 0.5*(B[:-1, 2] + B[1:, 2])
    return


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
                   ('method_type', 'CAM_CL_SIMPLE_PARALLEL'),
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
                       pos, vel, idx, Ji, E, B, Ve, Te, dns):
    d_path = '%s/%s/run_%d/data/particles/' % (drive, save_path, run_num)
    r      = qq / part_save_iter

    d_filename = 'data%05d' % r
    d_fullpath = os.path.join(d_path, d_filename)
    np.savez(d_fullpath, pos=pos, vel=vel, idx=idx, sim_time=sim_time,
             E=E, B=B, Ji=Ji, dns=dns, Ve=Ve, Te=Te)
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
                lo1, lo2, ro1, ro2, li1, li2, ri1, ri2, damping_multiplier
            
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
    return


def load_plasma_params():
    global species_lbl, temp_color, temp_type, dist_type, nsp_ppc, mass, charge, \
        drift_v, density, anisotropy, E_perp, E_e, beta_flag, L_val, B_eq, B_xmax_ovr,\
        qm_ratios, N, idx_start, idx_end, Nj, N_species, B_eq, ne, density, \
        E_par, Te0_scalar, vth_perp, vth_par, T_par, T_perp, \
        wpi, va, gyfreq_eq, dx, n_contr, xmax, xmin,\
            k_max
        
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
    gyfreq_eq  = ECHARGE*B_eq  / PMASS                       # Proton Gyrofrequency (rad/s) at equator (slowest)
    dx         = dxm * va / gyfreq_eq                        # Alternate method of calculating dx (better for multicomponent plasmas)
    n_contr    = density / nsp_ppc                           # Species density contribution: Each macroparticle contributes this SI density to a cell
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
## NOTE: Subcycles defined per particle step, but pushes are half-timestep
import argparse as ap
parser = ap.ArgumentParser()
parser.add_argument('-r', '--runfile'     , default='_run_params.run', type=str)
parser.add_argument('-p', '--plasmafile'  , default='_plasma_params.plasma', type=str)
parser.add_argument('-d', '--driverfile'  , default='_driver_params.txt'   , type=str)
parser.add_argument('-n', '--run_num'     , default=-1, type=int)
parser.add_argument('-s', '--subcycle'    , default=8, type=int)
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
    _L, _G               = initialize_source_arrays()
    _POS, _VEL, _IE, _W_ELEC, _IB, \
    _W_MAG, _IDX                   = initialize_particles(_B, _E)
    _DT, _MAX_INC, _PART_SAVE_ITER,\
    _FIELD_SAVE_ITER, _SUBCYCLES   = set_timestep(_VEL)    
        
    print('Loading initial state...')
    init_collect_moments(_POS, _VEL, _IE, _W_ELEC, _IB, _W_MAG, _IDX, _RHO_INT, _RHO_HALF,
                         _Ji, _Ji_PLUS, _L, _G, 0.5*_DT)
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

    _QQ = 1; _SIM_TIME = 0.0; _DESYNC_B = 1; _HALF=1
    print('Starting loop...')
    while _QQ < _MAX_INC:
        
        #######################
        ###### MAIN LOOP ######
        #######################
        
        # First field advance to N + 1/2
        _SIM_TIME, _HALF = cyclic_leapfrog(_B, _B2, _B_CENT, _RHO_INT, _Ji, _E, _VE, _TE,
                                    _DT, _SUBCYCLES, _SIM_TIME, _HALF)

        # CAM part
        push_current(_Ji_PLUS, _Ji, _E, _B_CENT, _L, _G, _DT)
        calculate_E(_B, _B_CENT, _Ji, _RHO_HALF,
                    _E, _VE, _TE, _SIM_TIME)
        
        # Particle advance, moment calculation
        advance_particles_and_collect_moments(_POS, _VEL, _IE, _W_ELEC, _IB, _W_MAG,
                                              _IDX, _B, _E, _RHO_INT, _RHO_HALF, _Ji,
                                              _Ji_MINUS, _Ji_PLUS, _L, _G, _DT)
        
        # Second field advance to N + 1
        _SIM_TIME, _HALF = cyclic_leapfrog(_B, _B2, _B_CENT, _RHO_INT, _Ji, _E, _VE, _TE,
                                    _DT, _SUBCYCLES, _SIM_TIME, _HALF)
        
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
        break
        
    runtime = round(timer() - start_time,2) 
    print('Run complete : {} s'.format(runtime))