# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:00:58 2017

@author: iarey
"""
import numpy as np
import sys
import platform

### RUN DESCRIPTION ###
run_description = '''Testing optimized CAM_CL version - this is the OLD code for comparison.'''


### RUN PARAMETERS ###
#drive           = 'G://MODEL_RUNS//Josh_Runs//' # Drive letter or path for portable HDD e.g. 'E:/'
#drive           = '/media/yoshi/UNI_HD/'
drive           = 'F:/'
save_path       = 'runs/CAM_CL_NEW_test/'         # Series save dir   : Folder containing all runs of a series 
run_num         = 1                           # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)
save_particles  = 1                           # Save data flag    : For later analysis
save_fields     = 1                           # Save plot flag    : To ensure hybrid is solving correctly during run
seed            = 101                         # RNG Seed          : Set to enable consistent results for parameter studies
cpu_affin       = [6, 7]                      # Set CPU affinity for run. Must be list. Auto-assign: None.



### PHYSICAL CONSTANTS ###
q   = 1.602177e-19                          # Elementary charge (C)
c   = 2.998925e+08                          # Speed of light (m/s)
mp  = 1.672622e-27                          # Mass of proton (kg)
me  = 9.109384e-31                          # Mass of electron (kg)
kB  = 1.380649e-23                          # Boltzmann's Constant (J/K)
e0  = 8.854188e-12                          # Epsilon naught - permittivity of free space
mu0 = (4e-7) * np.pi                        # Magnetic Permeability of Free Space (SI units)
RE  = 6.371e6                               # Earth radius in metres


### SIMULATION PARAMETERS ###
NX       = 256                              # Number of cells - doesn't include ghost cells
max_rev  = 150                              # Simulation runtime, in multiples of the gyroperiod

dxm         = 1.0                           # Number of c/wpi per dx (Ion inertial length: anything less than 1 isn't "resolvable" by hybrid code)
subcycles   = 12                            # Number of field subcycling steps for Cyclic Leapfrog
cellpart    = 512                           # Number of Particles per cell. Ensure this number is divisible by macroparticle proportion

ie       = 0                                # Adiabatic electrons. 0: off (constant), 1: on.
theta    = 0                                # Angle of B0 to x axis (in xy plane in units of degrees)
B0       = 200e-9                           # Unform initial magnetic field value (in T)
ne       = 200e6                            # Electron density (in /m3, same as total ion density (for singly charged ions))

orbit_res = 0.10                            # Particle orbit resolution: Fraction of gyroperiod in seconds
freq_res  = 0.05                            # Frequency resolution: Fraction of inverse radian frequencies
part_res  = 0.25                            # Data capture resolution in gyroperiod fraction: Particle information
field_res = 0.10                            # Data capture resolution in gyroperiod fraction: Field information


### PARTICLE PARAMETERS ###
species_lbl= [r'$H^+$ hot', r'$H^+$ cold']   # Species name/labels        : Used for plotting
temp_color = ['r', 'b']
temp_type  = np.asarray([1, 0])                          # Particle temperature type  : Cold (0) or Hot (1) : Used for plotting
dist_type  = np.asarray([0, 0])                          # Particle distribution type : Uniform (0) or sinusoidal/other (1) : Used for plotting (normalization)

mass       = np.asarray([1.000, 1.000])    	        # Species ion mass (proton mass units)
charge     = np.asarray([1.000, 1.000])       	    # Species ion charge (elementary charge units)
density    = np.asarray([0.100, 0.900])     			# Species charge density as normalized fraction (add to 1.0)
drift_v    = np.asarray([0.000, 0.000])     			# Species parallel bulk velocity (alfven velocity units)
sim_repr   = np.asarray([0.5  , 0.5])      		    # Macroparticle weighting: Percentage of macroparticles assigned to each species

beta       = True                                           # Flag: Specify temperatures by beta (True) or energy in eV (False)
E_e        = 0.1                                            # Electron beta
E_par      = np.array([10.0, 0.1, 0.1, 0.1])            			# Ion species parallel beta
E_per      = np.array([25.0, 0.1, 0.1, 0.1])            			# Ion species perpendicular beta

smooth_sources = 0                                          # Flag for source smoothing: Gaussian
min_dens       = 0.05                                       # Allowable minimum charge density in a cell, as a fraction of ne*q

adaptive_timestep   = True                                  # Flag (True/False) for adaptive timestep based on particle and field parameters
adaptive_subcycling = True                                  # Flag (True/False) to adaptively change number of subcycles during run to account for high-frequency dispersion
do_parallel         = False                                 # Flag (True/False) for auto-parallel using numba.njit()

ratio_override = 0                                          # Flag to override magnetic field value for specific regime
wpiwci         = 1e4                                        # Desired plasma/cyclotron frequency ratio for override
















#%%### DERIVED SIMULATION PARAMETERS

if ratio_override == 1:
    B0   = c * (1. / wpiwci) * np.sqrt(mu0 * mp * ne)
    print('\n')
    print('----------------------------------------------------------------')
    print('WARNING: RATIO OVERRIDE IN EFFECT - INPUT MAGNETIC FIELD IGNORED')
    print('----------------------------------------------------------------')

if beta == True:
    Te0        = B0 ** 2 * E_e   / (2 * mu0 * ne * kB)    # Temperatures of species in Kelvin (used for particle velocity initialization)
    Tpar       = B0 ** 2 * E_par / (2 * mu0 * ne * kB)
    Tper       = B0 ** 2 * E_per / (2 * mu0 * ne * kB)
else:
    Te0        = E_e   * 11603.
    Tpar       = E_par * 11603.
    Tper       = E_per * 11603.

wpi        = np.sqrt(ne * q ** 2 / (mp * e0))            # Proton   Plasma Frequency, wpi (rad/s)
wpe        = np.sqrt(ne * q ** 2 / (me * e0))            # Proton   Plasma Frequency, wpi (rad/s)
va         = B0 / np.sqrt(mu0*ne*mp)                     # Alfven speed: Assuming pure proton plasma

dx         = dxm * c / wpi                               # Spatial cadence, based on ion inertial length
xmin       = 0                                           # Minimum simulation dimension
xmax       = NX * dx                                     # Maximum simulation dimension

N          = cellpart*NX                                 # Number of Particles to simulate: # cells x # particles per cell, excluding ghost cells
Bc         = np.zeros(3)                                 # Constant components of magnetic field based on theta and B0
Bc[0]      = B0 * np.cos(theta * np.pi / 180.)           # Constant x-component of magnetic field (theta in degrees)
Bc[1]      = 0.                                          # Assume Bzc = 0, orthogonal to field line direction
Bc[2]      = B0 * np.sin(theta * np.pi / 180.)           # Constant y-component of magnetic field (theta in degrees)

density    = ne * (density / charge)                     # Density of each species per cell (in /m3)
charge    *= q                                           # Cast species charge to Coulomb
mass      *= mp                                          # Cast species mass to kg
drift_v   *= va                                          # Cast species velocity to m/s

Nj         = len(mass)                                   # Number of species
N_species  = np.round(N * sim_repr).astype(int)          # Number of sim particles for each species, total
n_contr    = density / (cellpart*sim_repr)               # Species density contribution: Each macroparticle contributes this density to a cell

idx_start  = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
idx_end    = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order
idx_bounds = np.stack((idx_start, idx_end)).transpose()                              # idx_bounds[species, start/end]

gyfreq     = q*B0/mp                                     # Proton   Gyrofrequency (rad/s) (since this will be the highest of all ion species)
e_gyfreq   = q*B0/me                                     # Electron Gyrofrequency (rad/s)
k_max      = np.pi / dx                                  # Maximum permissible wavenumber in system (SI???)


diag_file = drive + save_path + 'ne_{}_log.txt'.format(ne)

LH_frac  = 0.0                                           # Fraction of Lower Hybrid resonance: 
                                                         # Used to calculate electron resistivity by setting "anomalous"
                                                         # electron/ion collision as some multiple of the LHF. 0 disables e_resis.
LH_res_is  = 1. / (gyfreq * e_gyfreq) + 1. / wpi ** 2    # Lower Hybrid Resonance frequency, inverse squared
LH_res     = 1. / np.sqrt(LH_res_is)                     # Lower Hybrid Resonance frequency: DID I CHECK THIS???

e_resis    = (LH_frac * LH_res)  / (e0 * wpe ** 2)       # Electron resistivity (using intial conditions for wpi/wpe)























#%%
#%%
#%%
#%%
#%%
#%%
#%%### INPUT TESTS AND CHECKS
sped_ratio = c / va

print('Speed ratio: {}'.format(sped_ratio))
print('Density: {}cc'.format(round(ne / 1e6, 2)))
print('Background magnetic field: {}nT'.format(round(B0*1e9, 2)))
print('Gyroperiod: {}s'.format(round(2. * np.pi / gyfreq, 2)))
print('Inverse radian gyrofreqcy: {}s'.format(round(1 / gyfreq, 2)))
print('Maximum simulation time: {}s ({} gyroperiods)'.format(round(max_rev * 2. * np.pi / gyfreq, 2), max_rev))
print('\n{} particles per cell, {} cells'.format(cellpart, NX))
print('{} particles total\n'.format(cellpart * NX))

if None not in cpu_affin:
    import psutil
    run_proc = psutil.Process()
    run_proc.cpu_affinity(cpu_affin)
    if len(cpu_affin) == 1:
        print('CPU affinity for run (PID {}) set to logical core {}'.format(run_proc.pid, run_proc.cpu_affinity()[0]))
    else:
        print('CPU affinity for run (PID {}) set to logical cores {}'.format(run_proc.pid, ', '.join(map(str, run_proc.cpu_affinity()))))


python_version = sys.version.split()[0]
operating_sys  = platform.system()
if do_parallel == True and python_version[0] == '2' and operating_sys == 'Windows':
    do_parallel = False
    print('PYTHON VERSION {} DETECTED. PARALLEL PROCESSING ONLY WORKS IN PYTHON 3.x AND/OR LINUX')
    print('PARALLEL FLAG DISABLED')
    
density_normal_sum = (charge / q) * (density / ne)

if round(density_normal_sum.sum(), 5) != 1.0:
    print('-------------------------------------------------------------------------')
    print('WARNING: ION DENSITIES DO NOT SUM TO 1.0. SIMULATION WILL NOT BE ACCURATE')
    print('-------------------------------------------------------------------------')
    print('')
    print('ABORTING...')
    sys.exit()
    
    
if round(sim_repr.sum(), 5) != 1.0:
    print('-----------------------------------------------------------------------------------')
    print('WARNING: MACROPARTICLE DENSITIES DO NOT SUM TO 1.0. SIMULATION WILL NOT BE ACCURATE')
    print('-----------------------------------------------------------------------------------')
    print('sum_dens = {}'.format(sim_repr.sum()))
    print('ABORTING...')
    sys.exit()
    
simulated_density_per_cell = (n_contr * charge * cellpart * sim_repr).sum()
real_density_per_cell      = ne*q

if round(simulated_density_per_cell, 3) != round(real_density_per_cell, 3):
    print('--------------------------------------------------------------------------------')
    print('WARNING: DENSITY CALCULATION ISSUE: RECHECK HOW MACROPARTICLE CONTRIBUTIONS WORK')
    print('--------------------------------------------------------------------------------')
    print('{} != {}'.format(simulated_density_per_cell, real_density_per_cell))
    print('ABORTING...')
    sys.exit()

if subcycles == 0 or subcycles == 1:
    print('-----------------------------------------------------------------------------')
    print('Subcycling DISABLED: Magnetic field will advance only once per half-timestep.')
    print('-----------------------------------------------------------------------------')
    subcycles = 1
    
