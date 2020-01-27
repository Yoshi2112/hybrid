# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:00:58 2017

@author: iarey
"""
import numpy as np
import sys

### RUN DESCRIPTION ###
run_description = '''Testing to see if I can get this open boundary thing to work'''

### RUN PARAMETERS ###
drive           = 'F:'                          # Drive letter or path for portable HDD e.g. 'E:/' or '/media/yoshi/UNI_HD/'
save_path       = 'runs//open_boundary_test'    # Series save dir   : Folder containing all runs of a series
run_num         = 2                             # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)
save_particles  = 1                             # Save data flag    : For later analysis
save_fields     = 1                             # Save plot flag    : To ensure hybrid is solving correctly during run
seed            = 3216587                       # RNG Seed          : Set to enable consistent results for parameter studies
cpu_affin       = [(2*run_num)%8, (2*run_num + 1)%8]      # Set CPU affinity for run. Must be list. Auto-assign: None.


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
ND       = 32                               # Damping region length: Multiple of NX (on each side of simulation domain)
max_rev  = 100                              # Simulation runtime, in multiples of the ion gyroperiod (in seconds)
r_damp   = 0.0129                           # Damping strength
dxm      = 1.0                              # Number of c/wpi per dx (Ion inertial length: anything less than 1 isn't "resolvable" by hybrid code, anything too much more than 1 does funky things to the waveform)

ie       = 1                                # Adiabatic electrons. 0: off (constant), 1: on.
theta    = 0                                # Angle of B0 to x axis (in xy plane in units of degrees)
B_eq     = 200e-9                           # Unform initial magnetic field value (in T)

orbit_res = 0.10                            # Particle orbit resolution: Fraction of gyroperiod in seconds
freq_res  = 0.02                            # Frequency resolution     : Fraction of inverse radian cyclotron frequency
part_res  = 0.20                            # Data capture resolution in gyroperiod fraction: Particle information
field_res = 0.10                            # Data capture resolution in gyroperiod fraction: Field information


### PARTICLE PARAMETERS ###
species_lbl= [r'$H^+$ cold', r'$H^+$ warm']                 # Species name/labels        : Used for plotting. Can use LaTeX math formatted strings
temp_color = ['blue', 'red']
temp_type  = np.asarray([0,      1])             	        # Particle temperature type  : Cold (0) or Hot (1) : Used for plotting
dist_type  = np.asarray([0,      0])                        # Particle distribution type : Uniform (0) or sinusoidal/other (1) : Used for plotting (normalization)
nsp_ppc    = np.asarray([200, 2000])                        # Number of particles per cell, per species - i.e. each species has equal representation (or code this to be an array later?)

mass       = np.asarray([1., 1.])    			            # Species ion mass (proton mass units)
charge     = np.asarray([1., 1.])    			            # Species ion charge (elementary charge units)
drift_v    = np.asarray([0., 0.])                           # Species parallel bulk velocity (alfven velocity units)
density    = np.asarray([190., 10.]) * 1e6                  # Species density in /cc (cast to /m3)
E_per      = np.array([5.0, 30000.])
anisotropy = np.array([0.0, 4.0])

smooth_sources = 0                                          # Flag for source smoothing: Gaussian
min_dens       = 0.05                                       # Allowable minimum charge density in a cell, as a fraction of ne*q
E_e            = 200.0                                      # Electron energy (eV)

account_for_dispersion = False                              # Flag (True/False) whether or not to reduce timestep to prevent dispersion getting too high
dispersion_allowance   = 1.                                 # Multiple of how much past frac*wD^-1 is allowed: Used to stop dispersion from slowing down sim too much  
adaptive_timestep      = True                               # Flag (True/False) for adaptive timestep based on particle and field parameters

HM_amplitude   = 0e-9                                       # Driven wave amplitude in T
HM_frequency   = 0.02                                       # Driven wave in Hz





#%%### DERIVED SIMULATION PARAMETERS
NC         = NX + 2*ND
ne         = density.sum()
E_par      = E_per / (anisotropy + 1)
    
Te0        = E_e   * 11603.
Tpar       = E_par * 11603.
Tper       = E_per * 11603.

wpi        = np.sqrt(ne * q ** 2 / (mp * e0))            # Proton   Plasma Frequency, wpi (rad/s)
va         = B_eq / np.sqrt(mu0*ne*mp)                   # Alfven speed at equator: Assuming pure proton plasma

dx         =   dxm * c / wpi                             # Spatial cadence, based on ion inertial length
xmin       = - NX // 2 * dx                              # Minimum simulation dimension
xmax       =   NX // 2 * dx                              # Maximum simulation dimension

charge    *= q                                           # Cast species charge to Coulomb
mass      *= mp                                          # Cast species mass to kg
drift_v   *= va                                          # Cast species velocity to m/s

Nj         = len(mass)                                   # Number of species
cellpart   = nsp_ppc * Nj                                # Number of Particles per cell.
N          = cellpart*NX + 2*Nj                          # Number of Particles to simulate: # cells x # particles per cell, 
                                                         # plus an extra two per species for end "boundary"
n_contr    = density / nsp_ppc                           # Species density contribution: Each macroparticle contributes this density to a cell

N_species  = np.ones(Nj, dtype=int) * nsp_ppc * NX + 2   # Number of sim particles for each species, total
idx_start  = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
idx_end    = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order
idx_bounds = np.stack((idx_start, idx_end)).transpose()                              # idx_bounds[species, start/end]

gyfreq     = q*B_eq/mp                                   # Proton   Gyrofrequency (rad/s) (since this will be the highest of all ion species)
e_gyfreq   = q*B_eq/me                                   # Electron Gyrofrequency (rad/s)
k_max      = np.pi / dx                                  # Maximum permissible wavenumber in system (SI???)
qm_ratios  = np.divide(charge, mass)                     # q/m ratio for each species

Bc        = np.zeros((NC + 1, 3), dtype=np.float64)      # Constant components of magnetic field based on theta and B0

L = 4.3
a = 4.5 / (L * RE)
    
if False:
    Bc[:, 0] += B_eq * np.cos(theta * np.pi / 180.)      # Constant x-component of magnetic field (theta in degrees)
    Bc[:, 1] += 0.                                       # Assume Bzc = 0, orthogonal to field line direction
    Bc[:, 2] += B_eq * np.sin(theta * np.pi / 180.)      # Constant y-component of magnetic field (theta in degrees)
else:
    
    
    Bc[:, 0] += B_eq * np.cos(theta * np.pi / 180.)      # Constant x-component of magnetic field (theta in degrees)
    Bc[:, 1] += 0.                                       # Assume Bzc = 0, orthogonal to field line direction












#%%
#%%
#%%
#%%
#%%
#%%
#%%### INPUT TESTS AND CHECKS
print('Run Started')
print('Run Series         : {}'.format(save_path.split('//')[-1]))
print('Run Number         : {}'.format(run_num))
print('Field save flag    : {}'.format(save_fields))
print('Particle save flag : {}\n'.format(save_particles))

print('Density            : {}cc'.format(round(ne / 1e6, 2)))
print('Equatorial B-field : {}nT'.format(round(B_eq*1e9, 1)))
print('HM amplitude       : {}nT'.format(HM_amplitude*1e9))
print('HM frequency       : {}mHz\n'.format(HM_frequency*1e3))

print('Gyroperiod         : {}s'.format(round(2. * np.pi / gyfreq, 2)))
print('Inverse rad gyfreq : {}s'.format(round(1 / gyfreq, 2)))
print('Maximum sim time   : {}s ({} gyroperiods)'.format(round(max_rev * 2. * np.pi / gyfreq, 2), max_rev))

print('\n{} particles per cell'.format(cellpart))
print('{} spatial cells, 2x{} damped cells'.format(NX, ND))
print('{} cells total'.format(NC))
print('{} particles total\n'.format(cellpart * NX))

if None not in cpu_affin:
    import psutil
    run_proc = psutil.Process()
    run_proc.cpu_affinity(cpu_affin)
    if len(cpu_affin) == 1:
        print('CPU affinity for run (PID {}) set to logical core {}'.format(run_proc.pid, run_proc.cpu_affinity()[0]))
    else:
        print('CPU affinity for run (PID {}) set to logical cores {}'.format(run_proc.pid, ', '.join(map(str, run_proc.cpu_affinity()))))
    
density_normal_sum = (charge / q) * (density / ne)

if round(density_normal_sum.sum(), 5) != 1.0:
    print('-------------------------------------------------------------------------')
    print('WARNING: ION DENSITIES DO NOT SUM TO 1.0. SIMULATION WILL NOT BE ACCURATE')
    print('-------------------------------------------------------------------------')
    print('')
    print('ABORTING...')
    sys.exit()
    
simulated_density_per_cell = (n_contr * charge * nsp_ppc).sum()
real_density_per_cell      = ne*q

if abs(simulated_density_per_cell - real_density_per_cell) / real_density_per_cell > 1e-10:
    print('--------------------------------------------------------------------------------')
    print('WARNING: DENSITY CALCULATION ISSUE: RECHECK HOW MACROPARTICLE CONTRIBUTIONS WORK')
    print('--------------------------------------------------------------------------------')
    print('')
    print('ABORTING...')
    sys.exit()

# =============================================================================
# if beta == True:
#     Te0        = B0 ** 2 * beta_e   / (2 * mu0 * ne * kB)    # Temperatures of species in Kelvin (used for particle velocity initialization)
#     Tpar       = B0 ** 2 * beta_par / (2 * mu0 * ne * kB)
#     Tper       = B0 ** 2 * beta_per / (2 * mu0 * ne * kB)
# =============================================================================