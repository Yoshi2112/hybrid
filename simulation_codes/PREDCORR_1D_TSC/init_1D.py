# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:27:33 2017

@author: iarey
"""
import numpy as np
from particles_1D             import assign_weighting_TSC
from simulation_parameters_1D import dx, NX, cellpart, N, kB, Bc, Nj, dist_type, sim_repr, idx_bounds,    \
                                     seed, Tpar, Tper, mass, velocity, theta

def particles_per_cell():
    '''
    Calculates how many particles per cell per specices to be placed in the simulation domain. Currently only does
    uniform, but useful shell function for later on.
    
    INPUT:
        <NONE>
        
    OUTPUT:
        ppc -- Number of particles per cell per species for each cell in simulation domain. NjxNX ndarray.
    '''
    ppc = np.zeros((Nj, NX), dtype=int)

    for ii in range(Nj):
        if dist_type[ii] == 0:
            ppc[ii, :] = cellpart * sim_repr[ii]
    return ppc


def uniform_distribution(ppc):
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
        idx[idx_bounds[jj, 0]: idx_bounds[jj, 1]] = jj
        
        for ii in range(NX):                # For each cell
            n_particles = ppc[jj, ii]

            for kk in range(n_particles):   # For each particle in that cell
                dist[idx_bounds[jj, 0] + acc + kk] = dx*(float(kk) / n_particles + ii)
            acc += n_particles

    return dist, idx


def gaussian_distribution(ppc):
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
            n_particles = ppc[jj, ii]
            dist[0, (idx_bounds[jj, 0] + acc): ( idx_bounds[jj, 0] + acc + n_particles)] = np.random.normal(0, np.sqrt((kB *  Tpar[jj]) /  mass[jj]), n_particles) +  velocity[jj]
            dist[1, (idx_bounds[jj, 0] + acc): ( idx_bounds[jj, 0] + acc + n_particles)] = np.random.normal(0, np.sqrt((kB *  Tper[jj]) /  mass[jj]), n_particles)
            dist[2, (idx_bounds[jj, 0] + acc): ( idx_bounds[jj, 0] + acc + n_particles)] = np.random.normal(0, np.sqrt((kB *  Tper[jj]) /  mass[jj]), n_particles)
            acc += n_particles
    
    # Rotate if theta != 0
    dist[0] = dist[0] * np.cos(np.pi * theta / 180.) - dist[2] * np.sin(np.pi * theta / 180.)
    dist[2] = dist[2] * np.cos(np.pi * theta / 180.) + dist[0] * np.sin(np.pi * theta / 180.)
    return dist


def initialize_particles():
    '''Initializes particle arrays.
    
    INPUT:
        <NONE>
        
    OUTPUT:
        pos    -- Particle position array (1, N)
        vel    -- Particle velocity array (3, N)
        W_elec -- Initial particle weights on E-grid
        idx    -- Particle type index
    '''
    ppc        = particles_per_cell()
    pos, idx   = uniform_distribution(ppc)
    vel        = gaussian_distribution(ppc)
    Ie, W_elec = assign_weighting_TSC(pos)
    return pos, vel, Ie, W_elec, idx


def initialize_fields():
    '''Initializes field ndarrays and sets initial values for fields based on parameters in config file.

    INPUT:
        <NONE>

    OUTPUT:
        B   -- Magnetic field array: (NX + 3) Node locations on cell edges/vertices (each cell +1 end boundary +2 guard cells)
        E   -- Electric field array: (NX + 3) Node locations in cell centres (each cell plus 1+2 guard cells)

    Note: Each field is initialized with one array value extra due to TSC trying to access it when a 
    particle is located exactly on 
    '''
    B = np.zeros((NX + 3, 3), dtype=float)
    E = np.zeros((NX + 3, 3), dtype=float)

    B[:, 0] = Bc[0]      # Set Bx initial
    B[:, 1] = Bc[1]      # Set By initial
    B[:, 2] = Bc[2]      # Set Bz initial
    return B, E