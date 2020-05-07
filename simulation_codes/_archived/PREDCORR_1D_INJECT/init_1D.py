# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:27:33 2017

@author: iarey
"""
import numba as nb
import numpy as np
import simulation_parameters_1D as const
import save_routines as save

from particles_1D             import assign_weighting_TSC
from simulation_parameters_1D import dx, NX, cellpart, N, kB, B0, Nj, dist_type, sim_repr, idx_bounds,    \
                                     seed, Tpar, Tper, mass, drift_v, theta
from fields_1D                import uniform_HM_field_value

@nb.njit()
def particles_per_cell():
    '''
    Calculates how many particles per cell per specices to be placed in the simulation domain. Currently only does
    uniform, but useful shell function for later on.
    
    INPUT:
        <NONE>
        
    OUTPUT:
        ppc -- Number of particles per cell per species for each cell in simulation domain. NjxNX ndarray.
    '''
    ppc = np.zeros((Nj, NX), dtype=nb.int32)

    for ii in range(Nj):
        if dist_type[ii] == 0:
            ppc[ii, :] = cellpart * sim_repr[ii]
    return ppc


@nb.njit()
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


@nb.njit()
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
            dist[0, (idx_bounds[jj, 0] + acc): ( idx_bounds[jj, 0] + acc + n_particles)] = np.random.normal(0, np.sqrt((kB *  Tpar[jj]) /  mass[jj]), n_particles) +  drift_v[jj]
            dist[1, (idx_bounds[jj, 0] + acc): ( idx_bounds[jj, 0] + acc + n_particles)] = np.random.normal(0, np.sqrt((kB *  Tper[jj]) /  mass[jj]), n_particles)
            dist[2, (idx_bounds[jj, 0] + acc): ( idx_bounds[jj, 0] + acc + n_particles)] = np.random.normal(0, np.sqrt((kB *  Tper[jj]) /  mass[jj]), n_particles)
            acc += n_particles
    
    # Rotate if theta != 0
    dist[0] = dist[0] * np.cos(np.pi * theta / 180.) - dist[2] * np.sin(np.pi * theta / 180.)
    dist[2] = dist[2] * np.cos(np.pi * theta / 180.) + dist[0] * np.sin(np.pi * theta / 180.)
    
    # DELET THIS
    dist[1:] *= 0.
   
    return dist


@nb.njit()
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
    ppc        = particles_per_cell()
    pos, idx   = uniform_distribution(ppc)
    vel        = gaussian_distribution(ppc)

    Ie         = np.zeros(N,      dtype=nb.uint16)
    Ib         = np.zeros(N,      dtype=nb.uint16)
    W_elec     = np.zeros((3, N), dtype=nb.float64)
    W_mag      = np.zeros((3, N), dtype=nb.float64)
    
    assign_weighting_TSC(pos, Ie, W_elec)
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)
    return pos, vel, Ie, W_elec, Ib, W_mag, idx


@nb.njit()
def initialize_fields():
    '''Initializes field ndarrays and sets initial values for fields based on
       parameters in config file.

    INPUT:
        <NONE>

    OUTPUT:
        B      -- Magnetic field array: (NX + 3) Node locations on cell edges/vertices (each cell +1 end boundary +2 guard cells)
        E_int  -- Electric field array: (NX + 3) Node locations in cell centres (each cell plus 1+2 guard cells)
        E_half -- Electric field array: (NX + 3) Node locations in cell centres (each cell plus 1+2 guard cells)
        Ve     -- Electron fluid velocity moment: Calculated as part of E-field update equation
        Te     -- Electron temperature          : Calculated as part of E-field update equation          
    Note: Each field is initialized with one array value extra due to TSC
    trying to access it when a particle is located exactly on xmax.
    '''
    Bc         = np.zeros(3)                                 # Constant components of magnetic field based on theta and B0
    Bc[0]      = B0 * np.cos(theta * np.pi / 180.)           # Constant x-component of magnetic field (theta in degrees)
    Bc[1]      = 0.                                          # Assume Bzc = 0, orthogonal to field line direction
    Bc[2]      = B0 * np.sin(theta * np.pi / 180.)           # Constant y-component of magnetic field (theta in degrees)
    
    B      = np.zeros((NX + 3, 3), dtype=nb.float64)
    E_int  = np.zeros((NX + 3, 3), dtype=nb.float64)
    E_half = np.zeros((NX + 3, 3), dtype=nb.float64)

    B[:, 0] = Bc[0]      # Set Bx initial
    B[:, 1] = Bc[1]      # Set By initial
    B[:, 2] = Bc[2]      # Set Bz initial
    
    B[:, 0]+= uniform_HM_field_value(0)             # Add initial HM field at t = 0
    
    Ve      = np.zeros((NX + 3, 3), dtype=nb.float64)
    Te      = np.zeros(NX + 3,      dtype=nb.float64)
    
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
    q_dens  = np.zeros(NX + 3,          dtype=nb.float64)    
    q_dens2 = np.zeros(NX + 3,          dtype=nb.float64) 
    Ji      = np.zeros((NX + 3, 3),     dtype=nb.float64)
    ni      = np.zeros((NX + 3, Nj),    dtype=nb.float64)
    nu      = np.zeros((NX + 3, Nj, 3), dtype=nb.float64)
    return q_dens, q_dens2, Ji, ni, nu


@nb.njit()
def initialize_tertiary_arrays():
    '''Initializes source term ndarrays. Each term is collected on the E-field grid.

    INPUT:
        <NONE>
        
    OUTPUT:
        temp3D        -- Swap-file vector array with grid dimensions
        temp3D2       -- Swap-file vector array with grid dimensions
        temp1D        -- Swap-file scalar array with grid dimensions
        old_particles -- Location to store old particle values (positions, velocities, weights)
                         as part of predictor-corrector routine
        old_fields   -- Location to store old B, Ji, Ve, Te field values for predictor-corrector routine
    '''
    temp3D        = np.zeros((NX + 3, 3), dtype=nb.float64)
    temp3D2       = np.zeros((NX + 3, 3), dtype=nb.float64)
    temp1D        = np.zeros(NX + 3,      dtype=nb.float64) 
    old_particles = np.zeros((8, N),      dtype=nb.float64)
    old_fields    = np.zeros((NX + 3, 10), dtype=nb.float64)
    return old_particles, old_fields, temp3D, temp3D2, temp1D


def set_timestep(vel):
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
    gyperiod = (2*np.pi) / const.gyfreq               # Gyroperiod within uniform field, initial B0 (s)         
    ion_ts   = const.orbit_res * gyperiod             # Timestep to resolve gyromotion
    vel_ts   = 0.5 * const.dx / np.max(vel[0, :])     # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step 

    DT       = min(ion_ts, vel_ts)
    max_time = const.max_rev * gyperiod               # Total runtime in seconds
    max_inc  = int(max_time / DT) + 1                 # Total number of time steps

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
    
    print('Timestep: %.4fs, %d iterations total\n' % (DT, max_inc))
    return DT, max_inc, part_save_iter, field_save_iter


