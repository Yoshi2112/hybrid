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
#import diagnostics as diag

import particles_1D as particles
import fields_1D    as fields
import auxilliary_1D as aux

from simulation_parameters_1D import dx, NX, ND, NC, N, kB, Nj, nsp_ppc, B_eq, va, \
                                     idx_start, idx_end, seed, Tpar, Tper, mass, drift_v,  \
                                     Bc, qm_ratios, freq_res, rc_hwidth, temp_type, Te0

#@nb.njit()
def calc_losses(v_para, v_perp, st=0):
    '''
    For arrays of parallel and perpendicular velocities, finds the number and 
    indices of particles outside the loss cone.
    
    Calculation of in_loss_cone not compatible with njit(). Recode later if you want.
    '''
    alpha        = np.arctan(v_perp / v_para) * 180. / np.pi    # Calculate particle PA's
    in_loss_cone = (abs(alpha) < const.loss_cone)               # Determine if particle in loss cone
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
            B0x = fields.eval_B0x(pos[0, st: en])
            if const.homogenous == False:
                N_loss = const.N_species[jj]
                
                while N_loss > 0:
                    v_mag       = vel[0, st: en] ** 2 + vel[1, st: en] ** 2 + vel[2, st: en] ** 2
                    v_perp      = np.sqrt(vel[1, st: en] ** 2 + vel[2, st: en] ** 2)
                    v_perp_eq   = v_perp * np.sqrt(B_eq / B0x)
                    v_para_eq   = np.sqrt(v_mag - v_perp_eq ** 2)

                    N_loss, loss_idx = calc_losses(v_para_eq, v_perp_eq, st=st)
                    
                    # Catch for a particle on the boundary : Set 90 degree pitch angle (gyrophase shouldn't overly matter)
                    if N_loss == 1:
                        if abs(pos[0, loss_idx[0]]) == const.xmax:
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
    
    particles.assign_weighting_TSC(pos, Ie, W_elec)
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
    
    Iterative? Analytic?
    '''
    
    
    for ii in range(ND, ND + NX):
        pass
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
    ion_ts   = const.orbit_res / const.gyfreq         # Timestep to resolve gyromotion
    vel_ts   = 0.5 * const.dx / np.max(vel[0, :])     # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step 

    if E[:, 0].max() != 0:
        elecfreq        = qm_ratios.max()*(np.abs(E[:, 0] / vel.max()).max())               # Electron acceleration "frequency"
        Eacc_ts         = freq_res / elecfreq                            
    else:
        Eacc_ts = ion_ts

    gyperiod = 2 * np.pi / const.gyfreq
    DT       = min(ion_ts, vel_ts, Eacc_ts)
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
        save.store_run_parameters(DT, part_save_iter, field_save_iter)

    damping_array = np.ones(NC + 1)
    set_damping_array(damping_array, DT)

    print('Timestep: %.4fs, %d iterations total\n' % (DT, max_inc))
    return DT, max_inc, part_save_iter, field_save_iter, damping_array



if __name__ == '__main__':
    POS, VEL, IDX = uniform_gaussian_distribution_quiet()
    
    #VPERP_NEW = np.sqrt(VEL[1]     ** 2 + VEL[2]     ** 2)
    #VPERP_OLD = np.sqrt(OLD_VEL[1] ** 2 + OLD_VEL[2] ** 2)
    #rL        = np.sqrt(POS[1]     ** 2 + POS[2]     ** 2)
    
    import matplotlib.pyplot as plt
    
    #import diagnostics       as diag
    #diag.check_velocity_distribution(VEL)
    
    if True:
        for jj in range(const.Nj):
            plt.figure(jj)
            v_perp = np.sign(VEL[2, const.idx_start[jj]:idx_end[jj]]) * \
                     np.sqrt(VEL[1, const.idx_start[jj]:idx_end[jj]] ** 2 +
                             VEL[2, const.idx_start[jj]:idx_end[jj]] ** 2) / const.va
            
            v_para = VEL[0, const.idx_start[jj]:idx_end[jj]] / const.va
            
            plt.scatter(v_perp, v_para, c=const.temp_color[jj], s=1)
            plt.title(r'Total Velocity Distribution Functions (%s) :: $\alpha_L$ = %.1f$^\circ$' % (const.species_lbl[jj], const.loss_cone))
            plt.xlabel('$v_\perp / v_A$')
            plt.ylabel('$v_\parallel / v_A$')
            
            gradient = np.tan(np.pi/2 - const.loss_cone * np.pi / 180.)
            lmin, lmax = plt.gca().get_xlim()
            lcx  = np.linspace(lmin, lmax, 100, endpoint=True)
            lcy1 =  gradient * lcx
            lcy2 = -gradient * lcx
            
            plt.plot(lcx, lcy1, c='k', alpha=0.5, ls=':')
            plt.plot(lcx, lcy2, c='k', alpha=0.5, ls=':')
            
            plt.axvline(0, c='k')
            plt.axhline(0, c='k')
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
