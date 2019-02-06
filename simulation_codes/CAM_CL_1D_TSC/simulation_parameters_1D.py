# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:00:58 2017

@author: iarey
"""
import numpy as np

### RUN DESCRIPTION ###                     # Saves within run for easy referencing
run_description = '''Winske anisotropy test'''

### RUN PARAMETERS ###
drive           = 'E:/'                     # Drive letter or path for portable HDD e.g. 'E:/'
save_path       = 'runs/CAM_CL_test2/'      # Series save dir   : Folder containing all runs of a series
run_num         = 0                         # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)
generate_data   = 1                         # Save data flag    : For later analysis
generate_plots  = 0                         # Save plot flag    : To ensure hybrid is solving correctly during run
seed            = 101                       # RNG Seed          : Set to enable consistent results for parameter studies

### PHYSICAL CONSTANTS ###
q   = 1.602177e-19                          # Elementary charge (C)
c   = 2.998925e+8                           # Speed of light (m/s)
mp  = 1.672622e-27                          # Mass of proton (kg)
me  = 9.109384e-31                          # Mass of electron (kg)
kB  = 1.380649e-23                          # Boltzmann's Constant (J/K)
e0  = 8.854188e-12                          # Epsilon naught - permittivity of free space
mu0 = (4e-7) * np.pi                        # Magnetic Permeability of Free Space (SI units)
RE  = 6.371e6                               # Earth radius in metres


### SIMULATION PARAMETERS ###
dxm      = 1                                # Number of c/wpi per dx (Ion inertial length: anything less than 1 isn't resolvable by hybrid code)
NX       = 128                              # Number of cells - doesn't include ghost cells
max_rev  = 16                               # Simulation runtime, in multiples of the gyroperiod

subcycles      = 12                         # Number of field subcycling steps for Cyclic Leapfrog
smooth_sources = 0                          # Flag for source smoothing: Gaussian

cellpart = 80                               # Number of Particles per cell. Ensure this number is divisible by macroparticle proportion
ie       = 0                                # Adiabatic electrons. 0: off (constant), 1: on.

theta    = 0                                # Angle of B0 to x axis (in xy plane in units of degrees)
B0       = 200e-9                           # Unform initial magnetic field value (in T)
ne       = 50e6                             # Electron density (in /m3, same as total ion density)

orbit_res= 0.01                             # Particle orbit resolution: fraction of gyroperiod (gyrofraction, lol)
data_res = 0                                # Data capture resolution in gyrofraction
plot_res = 0                                # Plot capture resolution in gyrofraction


### PARTICLE PARAMETERS ###
species    = [r'$H^+$ cold', r'$H^+$ hot']  # Species name/labels        : Used for plotting
temp_type  = np.asarray([0, 1])             # Particle temperature type  : Cold (0) or Hot (1)
dist_type  = np.asarray([0, 0])             # Particle distribution type : Uniform (0) or sinusoidal/other (1)

mass       = np.asarray([1.00 , 1.00])                      # Species ion mass (proton mass units)
charge     = np.asarray([1.00 , 1.00])                      # Species ion charge (elementary charge units)
velocity   = np.asarray([0.   , 1.00])                      # Species parallel bulk velocity (alfven velocity units)
density    = np.asarray([90.0 , 10.0])                      # Species density as percentage of total density, n_e
sim_repr   = np.asarray([50.0 , 50.0])                      # Macroparticle weighting: Percentage of macroparticles assigned to each species

beta_e     = 1.                                             # Electron beta
beta_par   = np.array([1., 10.])                            # Ion species parallel beta
beta_per   = np.array([1., 50.])                            # Ion species perpendicular beta

set_override = 1                                            # Flag to override magnetic field value for specific regime
wpiwci       = 1e4                                          # Desired plasma/cyclotron frequency ratio for override
    

#%%##################################                   ###############################################
### DERIVED SIMULATION PARAMETERS ###                   # Shouldn't need to touch anything below here #
#####################################                   ###############################################
if set_override == 1:
    B0   = c * (1. / wpiwci) * np.sqrt(mu0 * mp * ne)
    
Te0        = B0 ** 2 * beta_e   / (2 * mu0 * ne * kB)   # Temperatures of species in Kelvin
Tpar       = B0 ** 2 * beta_par / (2 * mu0 * ne * kB)
Tper       = B0 ** 2 * beta_per / (2 * mu0 * ne * kB)

wpi        = np.sqrt(ne * q ** 2 / (mp * e0))            # Proton   Plasma Frequency, wpi (rad/s)
wpe        = np.sqrt(ne * q ** 2 / (me * e0))            # Electron Plasma Frequency, wpe (rad/s)
gyfreq     = q*B0/mp                                     # Proton Gyrofrequency (rad/s) (since this will be the highest of all species)
va         = B0 / np.sqrt(mu0*ne*mp)                     # Alfven speed: Assuming pure proton plasma

velocity  *= va                                          # Cast velocity to m/s
mass      *= mp                                          # Cast mass to kg
charge    *= q                                           # Cast charge to Coulomb

dx         = dxm * c / wpi                               # Spatial cadence, based on ion inertial length
xmin       = 0                                           # Minimum simulation dimension
xmax       = NX * dx                                     # Maximum simulation dimension

N          = cellpart*NX                                 # Number of Particles to simulate: # cells x # particles per cell, excluding ghost cells
Bc         = np.zeros(3)                                 # Constant components of magnetic field based on theta and B0
Bc[0]      = B0 * np.sin((90 - theta) * np.pi / 180 )    # Constant x-component of magnetic field (theta in degrees)
Bc[1]      = B0 * np.cos((90 - theta) * np.pi / 180 )    # Constant y-component of magnetic field (theta in degrees)
Bc[2]      = 0                                           # Assume Bzc = 0, orthogonal to field line direction

N_real     = NX * dx * ne                                # Total number of real particles (charges? electrons)
N_species  = np.round(N * sim_repr * 0.01).astype(int)   # Number of sim particles for each species, total
Nj         = len(mass)                                   # Number of species (number of columns above)

n_contr    = (N_real * 0.01*density) / N_species         # Species density contribution: Real particles per sim particle
density    *= 0.01*ne                                    # Real density of each species (in /cc)

idx_start  = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
idx_end    = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order
idx_bounds = np.stack((idx_start, idx_end)).transpose()                              # idx_bounds[species, start/end]

freq_ratio = wpi / gyfreq
sped_ratio = c / va

print 'Frequency ratio: {}'.format(freq_ratio)
print 'Speed ratio: {}'.format(sped_ratio)
print 'Density: {}cc'.format(round(ne / 1e6, 2))
print 'Background magnetic field: {}nT'.format(round(B0*1e9, 1))
print 'Gyroperiod: {}s'.format(round(2. * np.pi / gyfreq, 2))
print 'Maximum simulation time: {}s ({} gyroperiods)'.format(round(max_rev * 2. * np.pi / gyfreq, 2), max_rev)