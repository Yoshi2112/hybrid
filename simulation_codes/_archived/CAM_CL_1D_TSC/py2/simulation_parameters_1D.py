# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:00:58 2017

@author: iarey
"""
import numpy as np
import sys
import platform

### RUN DESCRIPTION ###
run_description = '''Event 2 with 2mil particles, low cold beta and realistic anisotropy. Variable ion percentage'''

### RUN PARAMETERS ###
drive           = 'F:'#'/media/yoshi/UNI_HD/' # Drive letter or path for portable HDD e.g. 'E:/'
save_path       = 'runs/ev2_lowbeta_varion/'  # Series save dir   : Folder containing all runs of a series
run_num         = 7                           # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)
generate_data   = 1                           # Save data flag    : For later analysis
generate_plots  = 0                           # Save plot flag    : To ensure hybrid is solving correctly during run
seed            = 101                         # RNG Seed          : Set to enable consistent results for parameter studies
cpu_affin       = [run_num]                   # Set CPU affinity for run. Must be list. Auto-assign: None.


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
NX       = 128                              # Number of cells - doesn't include ghost cells
max_rev  = 25                               # Simulation runtime, in multiples of the gyroperiod

dxm         = 1.0                           # Number of c/wpi per dx (Ion inertial length: anything less than 1 isn't "resolvable" by hybrid code)
subcycles   = 4                             # Number of field subcycling steps for Cyclic Leapfrog
cellpart    = 20000                         # Number of Particles per cell. Ensure this number is divisible by macroparticle proportion

ie       = 1                                # Adiabatic electrons. 0: off (constant), 1: on.
theta    = 0                                # Angle of B0 to x axis (in xy plane in units of degrees)
nb       = 0.10
B0       = 215e-9                           # Unform initial magnetic field value (in T)
ne       = 250e6                            # Electron density (in /m3, same as total ion density)

LH_frac  = 0.0                              # Fraction of Lower Hybrid resonance: 
                                            # Used to calculate electron resistivity by setting "anomalous"
                                            # electron/ion collision as some multiple of the LHF. 0 disables e_resis.

orbit_res= 0.1                              # Particle orbit resolution: Fraction of gyroperiod in seconds
freq_res = 0.05                             # Frequency resolution: Fraction of inverse radian frequencies
data_res = 0.25                             # Data capture resolution in gyroperiod fraction
plot_res = 1.0                              # Plot capture resolution in gyroperiod fraction


### PARTICLE PARAMETERS ###
species_lbl= [r'$H^+$ cold', r'$H^+$ hot']                  # Species name/labels        : Used for plotting
temp_color = ['b', 'r']
temp_type  = np.asarray([0, 1])                             # Particle temperature type  : Cold (0) or Hot (1) : Used for plotting
dist_type  = np.asarray([0, 0])                             # Particle distribution type : Uniform (0) or sinusoidal/other (1) : Used for plotting (normalization)

mass       = np.asarray([1.00, 1.00])                       # Species ion mass (proton mass units)
charge     = np.asarray([1.00, 1.00])                       # Species ion charge (elementary charge units)
density    = np.asarray([1.-nb, nb])                        # Species charge density as normalized fraction (add to 1.0)
velocity   = np.asarray([0.00, 0.00])                       # Species parallel bulk velocity (alfven velocity units)
sim_repr   = np.asarray([0.50, 0.50])                       # Macroparticle weighting: Percentage of macroparticles assigned to each species

beta_e     = 0.1                                            # Electron beta
beta_par   = np.array([0.1, 10.])                           # Ion species parallel beta
beta_per   = np.array([0.1, 20.])                           # Ion species perpendicular beta

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
    print '\n'
    print '----------------------------------------------------------------'
    print 'WARNING: RATIO OVERRIDE IN EFFECT - INPUT MAGNETIC FIELD IGNORED'
    print '----------------------------------------------------------------'
    
Te0        = B0 ** 2 * beta_e   / (2 * mu0 * ne * kB)    # Temperatures of species in Kelvin (used for particle velocity initialization)
Tpar       = B0 ** 2 * beta_par / (2 * mu0 * ne * kB)
Tper       = B0 ** 2 * beta_per / (2 * mu0 * ne * kB)

wpi        = np.sqrt(ne * q ** 2 / (mp * e0))            # Proton   Plasma Frequency, wpi (rad/s)
wpe        = np.sqrt(ne * q ** 2 / (me * e0))            # Proton   Plasma Frequency, wpi (rad/s)
va         = B0 / np.sqrt(mu0*ne*mp)                     # Alfven speed: Assuming pure proton plasma

dx         = dxm * c / wpi                               # Spatial cadence, based on ion inertial length
xmin       = 0                                           # Minimum simulation dimension
xmax       = NX * dx                                     # Maximum simulation dimension

N          = cellpart*NX                                 # Number of Particles to simulate: # cells x # particles per cell, excluding ghost cells
Bc         = np.zeros(3)                                 # Constant components of magnetic field based on theta and B0
Bc[0]      = B0 * np.sin((90 - theta) * np.pi / 180 )    # Constant x-component of magnetic field (theta in degrees)
Bc[1]      = B0 * np.cos((90 - theta) * np.pi / 180 )    # Constant y-component of magnetic field (theta in degrees)
Bc[2]      = 0                                           # Assume Bzc = 0, orthogonal to field line direction

density    = ne * (density / charge)                     # Density of each species per cell (in /m3)
charge    *= q                                           # Cast species charge to Coulomb
mass      *= mp                                          # Cast species mass to kg
velocity  *= va                                          # Cast species velocity to m/s

Nj         = len(mass)                                   # Number of species
N_species  = np.round(N * sim_repr).astype(int)          # Number of sim particles for each species, total
n_contr    = density / (cellpart*sim_repr)               # Species density contribution: Each macroparticle contributes this density to a cell

idx_start  = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
idx_end    = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order
idx_bounds = np.stack((idx_start, idx_end)).transpose()                              # idx_bounds[species, start/end]

gyfreq     = q*B0/mp                                     # Proton   Gyrofrequency (rad/s) (since this will be the highest of all ion species)
e_gyfreq   = q*B0/me                                     # Electron Gyrofrequency (rad/s)
k_max      = np.pi / dx                                  # Maximum permissible wavenumber in system (SI???)

LH_res_is  = 1. / (gyfreq * e_gyfreq) + 1. / wpi ** 2    # Lower Hybrid Resonance frequency, inverse squared
LH_res     = 1. / np.sqrt(LH_res_is)                     # Lower Hybrid Resonance frequency

e_resis    = (LH_frac * LH_res)  / (e0 * wpe ** 2)       # Electron resistivity (using intial conditions for wpi/wpe)

























#%%
#%%
#%%
#%%
#%%
#%%
#%%### INPUT TESTS AND CHECKS
sped_ratio = c / va

print 'Speed ratio: {}'.format(sped_ratio)
print 'Density: {}cc'.format(round(ne / 1e6, 2))
print 'Background magnetic field: {}nT'.format(round(B0*1e9, 1))
print 'Gyroperiod: {}s'.format(round(2. * np.pi / gyfreq, 2))
print 'Maximum simulation time: {}s ({} gyroperiods)'.format(round(max_rev * 2. * np.pi / gyfreq, 2), max_rev)
print '\n{} particles per cell, {} cells'.format(cellpart, NX)
print '{} particles total'.format(cellpart * NX)

if None not in cpu_affin:
    import psutil
    run_proc = psutil.Process()
    run_proc.cpu_affinity(cpu_affin)
    if len(cpu_affin) == 1:
        print '\nCPU affinity for run (PID {}) set to logical core {}'.format(run_proc.pid, run_proc.cpu_affinity()[0])
    else:
        print '\nCPU affinity for run (PID {}) set to logical cores {}'.format(run_proc.pid, ', '.join(map(str, run_proc.cpu_affinity())))


python_version = sys.version.split()[0]
operating_sys  = platform.system()
if do_parallel == True and python_version[0] == '2' and operating_sys == 'Windows':
    do_parallel = False
    print '\n'
    print 'PYTHON VERSION {} DETECTED. PARALLEL PROCESSING ONLY WORKS IN PYTHON 3.x AND/OR LINUX'
    print 'PARALLEL FLAG DISABLED'
    
density_normal_sum = (charge / q) * (density / ne)

if density_normal_sum.sum() != 1.0:
    print '-------------------------------------------------------------------------'
    print 'WARNING: ION DENSITIES DO NOT SUM TO 1.0. SIMULATION WILL NOT BE ACCURATE'
    print '-------------------------------------------------------------------------'
    print ''
    print 'ABORTING...'
    sys.exit()
    
    
if sim_repr.sum() != 1.0:
    print '-----------------------------------------------------------------------------------'
    print 'WARNING: MACROPARTICLE DENSITIES DO NOT SUM TO 1.0. SIMULATION WILL NOT BE ACCURATE'
    print '-----------------------------------------------------------------------------------'
    print ''
    print 'ABORTING...'
    sys.exit()
    
simulated_density_per_cell = (n_contr * charge * cellpart * sim_repr).sum()
real_density_per_cell      = ne*q

if False:#simulated_density_per_cell != real_density_per_cell:
    print '--------------------------------------------------------------------------------'
    print 'WARNING: DENSITY CALCULATION ISSUE: RECHECK HOW MACROPARTICLE CONTRIBUTIONS WORK'
    print '--------------------------------------------------------------------------------'
    print ''
    print 'ABORTING...'
    sys.exit()

if subcycles == 0 or subcycles == 1:
    print '-----------------------------------------------------------------------------'
    print 'Subcycling DISABLED: Magnetic field will advance only once per half-timestep.'
    print '-----------------------------------------------------------------------------'
    subcycles = 1
    
