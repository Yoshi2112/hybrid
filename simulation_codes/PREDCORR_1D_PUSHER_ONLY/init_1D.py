# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:27:33 2017

@author: iarey
"""
import numpy as np
import simulation_parameters_1D as const
import save_routines as save
import particles_1D as particles

from simulation_parameters_1D import dx, NX, N, kB, Nj, nsp_ppc, idx_start, idx_end, seed, Tpar, Tper, \
                                     mass, drift_v, qm_ratios, rc_hwidth, temp_type, B_xmax, ion_ts


def calc_losses(v_para, v_perp, B0x, st=0):
    '''
    For arrays of parallel and perpendicular velocities, finds the number and 
    indices of particles outside the loss cone.
    
    Calculation of in_loss_cone not compatible with njit(). Recode later if you want.
    '''
    alpha        = np.arctan(v_perp / v_para)                   # Calculate particle PA's
    loss_cone    = np.arcsin(np.sqrt(B0x / B_xmax))             # Loss cone per particle (based on B0 at particle)
    in_loss_cone = (abs(alpha) < loss_cone)                     # Determine if particle in loss cone
    N_loss       = in_loss_cone.sum()                           # Count number that are
    loss_idx     = np.where(in_loss_cone == True)[0]            # Find their indices
    loss_idx    += st                                           # Offset indices to account for position in master array
    return N_loss, loss_idx


def uniform_gaussian_distribution_quiet():
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
            B0x = particles.eval_B0x(pos[0, st: en])
            if const.homogenous == False:
                N_loss = const.N_species[jj]
                
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


def set_timestep(vel):
    vel_ts   = 0.5 * const.dx / np.max(np.abs(vel[0, :]))   # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step 
    DT       = min(ion_ts, vel_ts)
    
    gyperiod = 2 * np.pi / const.gyfreq
    
    max_time = const.max_rev * 2 * np.pi / const.gyfreq_eq  # Total runtime in seconds
    max_inc  = int(max_time / DT) + 1                       # Total number of time steps

    if const.part_res == 0:
        part_save_iter = 1
    else:
        part_save_iter = int(const.part_res*gyperiod / DT)

    if const.save_particles == 1:
        save.store_run_parameters(DT, part_save_iter)

    print('Timestep: %.4fs, %d iterations total\n' % (DT, max_inc))
    return DT, max_inc, part_save_iter