# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:27:33 2017

@author: iarey
"""
import numpy as np
from simulation_parameters_1D import dx, NX, N, kB, Nj, idx_start, idx_end,  \
                                     seed, Tpar, Tper, mass, drift_v, theta, \
                                     nsp_ppc, xmax

def uniform_distribution():
    '''Creates an analytically uniform distribution of N numbers within each cell boundary

    INPUT:
        ppc       -- Number of particles per cell, per species

    OUTPUT:
        dist -- numpy ndarray containing numerical distribution
    '''
    dist = np.zeros(N)
    idx  = np.zeros(N, dtype=np.uint8)

    for jj in range(Nj):                    # For each species
        acc = 0
        idx[idx_start[jj]: idx_end[jj]] = jj
        
        for ii in range(NX):                # For each cell
            n_particles = nsp_ppc[jj]

            for kk in range(n_particles):   # For each particle in that cell
                dist[idx_start[jj] + acc + kk] = dx*(float(kk) / n_particles + ii)
            acc += n_particles

    return dist, idx


def gaussian_distribution():
    '''Creates an N-sampled normal distribution across all particle species within each simulation cell

    INPUT:
        N   -- Number of particles to distribute
        idx -- Index identifier for particle type : Correlates to parameters in part_params.py

    OUTPUT:
        dist -- Output distribution: Maxwellian, as generated by numpy's random.normal in 3 dimensions
    '''
    np.random.seed(seed)         # Random seed
    dist = np.zeros((3, N))      # Initialize array

    for jj in range(Nj):
        acc = 0                  # Species accumulator
        for ii in range(NX):
            n_particles = nsp_ppc[jj]
            dist[0, (idx_start[jj] + acc): ( idx_start[jj] + acc + n_particles)] = np.random.normal(0, np.sqrt((kB *  Tpar[jj]) /  mass[jj]), n_particles) +  drift_v[jj]
            dist[1, (idx_start[jj] + acc): ( idx_start[jj] + acc + n_particles)] = np.random.normal(0, np.sqrt((kB *  Tper[jj]) /  mass[jj]), n_particles)
            dist[2, (idx_start[jj] + acc): ( idx_start[jj] + acc + n_particles)] = np.random.normal(0, np.sqrt((kB *  Tper[jj]) /  mass[jj]), n_particles)
            acc += n_particles
    
    # Rotate if theta != 0
    dist[0] = dist[0] * np.cos(np.pi * theta / 180.) - dist[2] * np.sin(np.pi * theta / 180.)
    dist[2] = dist[2] * np.cos(np.pi * theta / 180.) + dist[0] * np.sin(np.pi * theta / 180.)
    return dist


def quiet_start():
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
        
        sf_par = np.sqrt(kB *  Tpar[jj] /  mass[jj])  # Scale factors for velocity initialization
        sf_per = np.sqrt(kB *  Tper[jj] /  mass[jj])
        
        half_n = nsp_ppc[jj] // 2                     # Half particles per cell - doubled later

        # Load particles in each applicable cell
        acc = 0; offset  = 0
        for ii in range(NX):                
            # Particle index ranges
            st = idx_start[jj] + acc
            en = idx_start[jj] + acc + half_n
            
            # Set position for half: Analytically uniform
            for kk in range(half_n):
                pos[st + kk] = dx*(float(kk) / (half_n - offset) + ii)
           
            # Set velocity for half: Randomly Maxwellian
            vel[0, st: en] = np.random.normal(0, sf_par, half_n) +  drift_v[jj]
            vel[1, st: en] = np.random.normal(0, sf_per, half_n)
            vel[2, st: en] = np.random.normal(0, sf_per, half_n)
                                
            pos[en: en + half_n] = pos[st: en]                      # Other half, same position
            vel[0, en: en + half_n] = vel[0, st: en] *  1.0     # Set parallel
            vel[1, en: en + half_n] = vel[1, st: en] * -1.0         # Invert perp velocities (v2 = -v1)
            vel[2, en: en + half_n] = vel[2, st: en] * -1.0
            
            acc                    += half_n * 2
            
    # Rotate if theta != 0
    if theta != 0:
        vel[0] = vel[0] * np.cos(np.pi * theta / 180.) - vel[2] * np.sin(np.pi * theta / 180.)
        vel[2] = vel[2] * np.cos(np.pi * theta / 180.) + vel[0] * np.sin(np.pi * theta / 180.)
    return pos, vel, idx


if __name__ == '__main__':
    POS, VEL, IDX = quiet_start()