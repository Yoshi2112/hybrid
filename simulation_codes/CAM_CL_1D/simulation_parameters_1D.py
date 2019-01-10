# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:00:58 2017

@author: iarey
"""
import numpy as np
import pdb

### RUN DESCRIPTION ###                     # Saves within run for easy referencing
run_description = '''Winske 1D anisotropy simulation based on parameters from h1.f hybrid code (addendum to 1993 book)'''


### RUN PARAMETERS ###
drive           = 'E:/'                             # Drive letter or path for portable HDD e.g. 'E:/'
save_path       = 'runs/winske_anisotropy_test/'    # Series save dir   : Folder containing all runs of a series
run_num         = 1                                 # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)
generate_data   = 0                                 # Save data flag    : For later analysis
generate_plots  = 0                                 # Save plot flag    : To ensure hybrid is solving correctly during run
seed            = 21                                # RNG Seed          : Set to enable consistent results for parameter studies

### PHYSICAL CONSTANTS ###
q   = 1.602e-19                             # Elementary charge (C)
c   = 3.00e8                                # Speed of light (m/s)
mp  = 1.673e-27                             # Mass of proton (kg)
me  = 9.109e-31                             # Mass of electron (kg)
mu0 = (4e-7) * np.pi                        # Magnetic Permeability of Free Space (SI units)
kB  = 1.381e-23                             # Boltzmann's Constant (J/K)
e0  = 8.854e-12                             # Epsilon naught - permittivity of free space
RE  = 6.371e6                               # Earth radius in metres


### SIMULATION PARAMETERS ###
dxm      = 1                                # Number of c/wpi per dx (Ion inertial length: anything less than 1 isn't resolvable by hybrid code)
lam_res  = 0.05                             # Determines simulation DT by fraction of orbit per timestep
t_res    = 6                                # Time resolution. Determines how often data is captured. Every frame captured if '0'.
plot_res = 6                                # Determines how often a plot is generated (in seconds of simulation time). Every frame plotted if '0', or none if None (this is also controlled by the generate_plot flag)
NX       = 128                              # Number of cells - doesn't include ghost cells
max_sec  = 200                              # Simulation runtime, in seconds of simulated time
cellpart = 80                               # Number of Particles per cell. Ensure this number is divisible by macroparticle proportion
ie       = 0                                # Adiabatic electrons. 0: off (constant), 1: on.
B0       = 140e-9                           # Unform initial magnetic field value (in T)
theta    = 0                                # Angle of B0 to x axis (in xy plane in units of degrees)
ne       = 50e6                             # Electron density (used to assign portions of ion)

k        = 1                                # Sinusoidal Density Parameter - number of wavelengths in spatial domain
mhd_equil= 0                                # Temperature varied to give MHD Equilibrium condition?


### PARTICLE PARAMETERS ###
species    = [r'$H^+$ cold', r'$H^+$ hot']  # Species name/labels        : Used for plotting
temp_type  = np.asarray([0, 1, 0])             # Particle temperature type  : Cold (0) or Hot (1)
dist_type  = np.asarray([0, 0, 0])             # Particle distribution type : Uniform (0) or sinusoidal/other (1)

mass       = np.asarray([1.00 , 1.00 , 4.00])     # Species ion mass (amu to kg)
charge     = np.asarray([1.00 , 1.00 , 1.00])     # Species ion charge (elementary charge units to Coulombs)
velocity   = np.asarray([0.   , 0.   , 0.  ])     # Species parallel bulk velocity (in multiples of the alfven velocity)
density    = np.asarray([85.0 , 10.0 , 5.0 ])     # Species density as percentage of total density, n_e
sim_repr   = np.asarray([40.0 , 50.0 , 10.0])     # Macroparticle weighting: Percentage of macroparticles assigned to each species

Tpar       = np.array([487., 974. , 487.])*11603  # Parallel ion temperature (K)
Tper       = np.array([487., 4870., 487.])*11603  # Perpendicular ion temperature (K)
Te0        = 487.*11603                     # Electron temperature (K)


#%% Allows B/n ratio and temperatures to be altered to make comparison with other codes easier
set_override = 1

if set_override == 1:
    beta_e     = 1.                             # Parameters used to make intiial values more compatible with "normalized CGS" codes
    beta_par   = np.array([1., 2.,  1.])                           
    beta_per   = np.array([1., 10., 1.])            # Will overwrite temperature values
    wpiwci     = 1e4                            # Will overwrite density value
    
    B0   = c * (1. / wpiwci) * np.sqrt(mu0 * mp * ne)

    Te0  = B0 ** 2 * beta_e   / (2 * mu0 * ne * kB) # Should return temperatures as expressed in kB*T -> i.e. in eV
    Tpar = B0 ** 2 * beta_par / (2 * mu0 * ne * kB)
    Tper = B0 ** 2 * beta_per / (2 * mu0 * ne * kB)
    
#%%##################################                   ###############################################
### DERIVED SIMULATION PARAMETERS ###                   # Shouldn't need to touch anything below here #
#####################################                   ###############################################
wpi       = np.sqrt(ne * q ** 2 / (mp * e0))            # Proton   Plasma Frequency, wpi (rad/s)
wpe       = np.sqrt(ne * q ** 2 / (me * e0))            # Electron Plasma Frequency, wpe (rad/s)
gyfreq    = q*B0/mp                                     # Proton Gyrofrequency (rad/s) (since this will be the highest of all species)
va        = B0 / np.sqrt(mu0*ne*mp)                     # Alfven speed: Assuming proton plasma

velocity *= va                                          # Cast velocity to m/s
mass     *= mp                                          # Cast mass to kg
charge   *= q                                           # Cast charge to Coulomb

dx        = dxm * c / wpi                               # Spatial cadence, based on ion inertial length
xmin      = 0                                           # Minimum simulation dimension
xmax      = NX * dx                                     # Maximum simulation dimension

N         = cellpart*NX                                 # Number of Particles to simulate: # cells x # particles per cell, excluding ghost cells
Bc        = np.zeros(3)                                 # Constant components of magnetic field based on theta and B0
Bc[0]     = B0 * np.sin((90 - theta) * np.pi / 180 )    # Constant x-component of magnetic field (theta in degrees)
Bc[1]     = B0 * np.cos((90 - theta) * np.pi / 180 )    # Constant y-component of magnetic field (theta in degrees)
Bc[2]     = 0                                           # Assume Bzc = 0, orthogonal to field line direction

N_real    = NX * dx * ne                                # Total number of real particles (charges? electrons)
N_species = np.round(N * sim_repr * 0.01).astype(int)          # Number of sim particles for each species, total
Nj        = len(mass)                                   # Number of species (number of columns above)

n_contr   = (N_real * 0.01*density) / N_species         # Species density contribution: Real particles per sim particle
density   *= 0.01*ne                                    # Real density of each species (in /cc)

idx_start = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
idx_end   = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order
idx_bounds= np.stack((idx_start, idx_end)).transpose()  # Array index boundary values: idx_bounds[species, start/end]

freq_ratio = wpi / gyfreq
sped_ratio = c / va

print 'Frequency ratio: {}'.format(freq_ratio)
print 'Speed ratio: {}'.format(sped_ratio)
print 'Density: {}cc'.format(round(ne / 1e6, 2))
print 'Background magnetic field: {}nT\n'.format(round(B0*1e9, 1))
print 'Gyroperiod: {}s'.format(round(2. * np.pi / gyfreq, 2))