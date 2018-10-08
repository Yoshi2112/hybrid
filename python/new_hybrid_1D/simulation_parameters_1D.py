# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:00:58 2017

@author: iarey
"""
import numpy as np

### RUN DESCRIPTION ###                     # Saves within run for easy referencing
run_description = '''Stock standard run for control test'''


### RUN PARAMETERS ###
drive           = '/media/yoshi/UNI_HD/'    # Drive letter or path for portable HDD e.g. 'E:/'
save_path       = 'runs/check_1D_code/'     # Series save dir   : Folder containing all runs of a series
run_num         = 0                         # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)
generate_data   = 0                         # Save data flag    : For later analysis
generate_plots  = 1                         # Save plot flag    : To ensure hybrid is solving correctly during run
seed            = 21                        # RNG Seed          : Set to enable consistent results for parameter studies


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
dxm      = 2                                # Number of c/wpi per dx (Ion inertial length: anything less than 1 isn't resolvable by hybrid code)
t_res    = 0                                # Time resolution. Determines how often data is captured. Every frame captured if '0'.
NX       = 128                              # Number of cells - doesn't include ghost cells
max_sec  = 500                              # Simulation runtime, in seconds of simulated time
cellpart = 100                              # Number of Particles per cell. Ensure this number is divisible by macroparticle proportion
ie       = 0                                # Adiabatic electrons. 0: off (constant), 1: on.
B0       = 4e-9                             # Unform initial magnetic field value (in T)
theta    = 0                                # Angle of B0 to x axis (in xy plane in units of degrees)

ne       = 8.48e6                           # Electron density (used to assign portions of ion)
Te0      = 0.5 * 11603.                     # Electron temperature (eV to K)
k        = 1                                # Sinusoidal Density Parameter - number of wavelengths in spatial domain
mhd_equil= 0                                # Temperature varied to give MHD Equilibrium condition?



### PARTICLE PARAMETERS ###
species    = [r'$H^+$ cold', r'$H^+$ hot']  # Species name/labels        : Used for plotting
temp_type  = np.asarray([0, 1])             # Particle temperature type  : Cold (0) or Hot (1)
dist_type  = np.asarray([0, 0])             # Particle distribution type : Uniform (0) or sinusoidal/other (1)

mass       = np.asarray([1.00 , 1.00 ])     # Species ion mass (amu to kg)
charge     = np.asarray([1.00 , 1.00 ])     # Species ion charge (elementary charge units to Coulombs)
velocity   = np.asarray([-0.15, 10   ])     # Species bulk velocity (in multiples of the alfven velocity)
density    = np.asarray([98.5 , 1.5  ])     # Species density as percentage of total density, n_e
sim_repr   = np.asarray([50.0 , 50.0 ])     # Macroparticle weighting: Percentage of macroparticles assigned to each species

Tpar       = np.array([0.5, 0.5])           # Parallel ion temperature (eV)
Tper       = np.array([0.5, 0.5])           # Perpendicular ion temperature (eV)



#####################################                   ###############################################
### DERIVED SIMULATION PARAMETERS ###                   # Shouldn't need to touch anything below here #
#####################################                   ###############################################
wpi       = np.sqrt(ne * q ** 2 / (mp * e0))            # Proton   Plasma Frequency, wpi (rad/s)
wpe       = np.sqrt(ne * q ** 2 / (me * e0))            # Electron Plasma Frequency, wpe (rad/s)
gyfreq    = q*B0/mp                                     # Proton Gyrofrequency (rad/s) (since this will be the highest of all species)
va        = B0 / np.sqrt(mu0*ne*mp)                     # Alfven speed: Assuming proton plasma

velocity *= va                                          # Cast velocity to m/s
Tpar     *= 11603.                                      # Cast T_parallel to Kelvin
Tper     *= 11603.                                      # Cast T_perpendicular to Kelvin
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