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
save_path       = 'runs//non_uniform_B0_test'   # Series save dir   : Folder containing all runs of a series
run_num         = 0                             # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)
save_particles  = 1                             # Save data flag    : For later analysis
save_fields     = 1                             # Save plot flag    : To ensure hybrid is solving correctly during run
seed            = 3216587                       # RNG Seed          : Set to enable consistent results for parameter studies
cpu_affin       = [(2*run_num)%8, (2*run_num + 1)%8]      # Set CPU affinity for run. Must be list. Auto-assign: None.
supress_text    = False

### PHYSICAL CONSTANTS ###
q      = 1.602177e-19                       # Elementary charge (C)
c      = 2.998925e+08                       # Speed of light (m/s)
mp     = 1.672622e-27                       # Mass of proton (kg)
me     = 9.109384e-31                       # Mass of electron (kg)
kB     = 1.380649e-23                       # Boltzmann's Constant (J/K)
e0     = 8.854188e-12                       # Epsilon naught - permittivity of free space
mu0    = (4e-7) * np.pi                     # Magnetic Permeability of Free Space (SI units)
RE     = 6.371e6                            # Earth radius in metres
B_surf = 3.12e-5                            # Magnetic field strength at Earth surface


### SIMULATION PARAMETERS ###
NX        = 512                             # Number of cells - doesn't include ghost cells
ND        = 128                             # Damping region length: Multiple of NX (on each side of simulation domain)
max_rev   = 10000                           # Simulation runtime, in multiples of the ion gyroperiod (in seconds)
r_damp    = 0.0129                          # Damping strength
dxm       = 1.0                             # Number of c/wpi per dx (Ion inertial length: anything less than 1 isn't "resolvable" by hybrid code, anything too much more than 1 does funky things to the waveform)
L         = 5.35                            # Field line L shell

ie        = 1                               # Adiabatic electrons. 0: off (constant), 1: on.
B_eq      = 200e-9                          # Initial magnetic field at equator: None for L-determined value (in T)
rc_hwidth = 16                              # Ring current half-width in number of cells (2*hwidth gives equatorial extent of RC) 

orbit_res = 0.10                            # Particle orbit resolution: Fraction of gyroperiod in seconds
freq_res  = 0.02                            # Frequency resolution     : Fraction of inverse radian cyclotron frequency
part_res  = 0.20                            # Data capture resolution in gyroperiod fraction: Particle information
field_res = 0.10                            # Data capture resolution in gyroperiod fraction: Field information


### PARTICLE PARAMETERS ###
species_lbl= [r'$H^+$ cold', r'$H^+$ warm']                 # Species name/labels        : Used for plotting. Can use LaTeX math formatted strings
temp_color = ['blue', 'red']
temp_type  = np.array([0, 1])             	                # Particle temperature type  : Cold (0) or Hot (1) : Used for plotting
dist_type  = np.array([0, 0])                               # Particle distribution type : Uniform (0) or sinusoidal/other (1) : Used for plotting (normalization)
nsp_ppc    = np.array([1000, 6000])                         # Number of particles per cell, per species - i.e. each species has equal representation (or code this to be an array later?)

mass       = np.array([1., 1.])    			                # Species ion mass (proton mass units)
charge     = np.array([1., 1.])    			                # Species ion charge (elementary charge units)
drift_v    = np.array([0., 0.])                             # Species parallel bulk velocity (alfven velocity units)
density    = np.array([195., 5.]) * 1e6                     # Species density in /cc (cast to /m3)
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

if B_eq is None:
    B_eq      = (B_surf / (L ** 3))                      # Magnetic field at equator, based on L value

wpi        = np.sqrt(ne * q ** 2 / (mp * e0))            # Proton   Plasma Frequency, wpi (rad/s)
va         = B_eq / np.sqrt(mu0*ne*mp)                   # Alfven speed at equator: Assuming pure proton plasma

dx         = dxm * c / wpi                               # Spatial cadence, based on ion inertial length
xmax       = NX // 2 * dx                                # Maximum simulation length, +/-ve on each side
xmin       =-NX // 2 * dx

charge    *= q                                           # Cast species charge to Coulomb
mass      *= mp                                          # Cast species mass to kg
drift_v   *= va                                          # Cast species velocity to m/s

Nj         = len(mass)                                   # Number of species
n_contr    = density / nsp_ppc                           # Species density contribution: Each macroparticle contributes this density to a cell

# Number of sim particles for each species, total
N_species  = np.zeros(Nj, dtype=np.int64)
for jj in range(Nj):
    if temp_type[jj] == 0:                               # Cold species in every cell NX 
        N_species[jj] = nsp_ppc[jj] * NX + 2                 
    elif temp_type[jj] == 1:
        N_species[jj] = nsp_ppc[jj] * 2*rc_hwidth + 2    # Warm species only in simulation center
N = N_species.sum()

idx_start  = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
idx_end    = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order

############################
### MAGNETIC FIELD STUFF ###
############################
B_nodes  = (np.arange(NC + 1) - NC // 2)       * dx      # B grid points position in space
E_nodes  = (np.arange(NC)     - NC // 2 + 0.5) * dx      # E grid points position in space

theta_xmax  = xmax/(L*RE)                                # Latitudinal extent of simulation , based on xmax
r_xmax      = L * np.sin(np.pi / 2 - theta_xmax) ** 2    # Calculate radial distance of boundary in dipole and get field intensity
cos_bit     = np.sqrt(3*np.cos(theta_xmax)**2 + 1)       # Intermediate variable (angular scale factor)
B_xmax      = (B_surf / (r_xmax ** 3)) * cos_bit         # Magnetic field intensity at boundary
a           = (B_xmax / B_eq - 1) / xmax ** 2            # Parabolic scale factor

Bc          = np.zeros((NC + 1, 3), dtype=np.float64)    # Constant components of magnetic field based on theta and B0
Bc[:, 0]    = B_eq * (1 + a * B_nodes**2)                # Set constant Bx

# Set B0 in damping cells (same as last spatial cell)
Bc[:ND]      = Bc[ND]
Bc[ND+NX+1:] = Bc[ND+NX]

# Freqs based on highest magnetic field value (at simulation boundaries)
gyfreq     = q*B_xmax/mp                                 # Proton   Gyrofrequency (rad/s) (since this will be the highest of all ion species)
e_gyfreq   = q*B_xmax/me                                 # Electron Gyrofrequency (rad/s)
k_max      = np.pi / dx                                  # Maximum permissible wavenumber in system (SI???)
qm_ratios  = np.divide(charge, mass)                     # q/m ratio for each species



#%%
#%%
#%%
#%%
#%%
#%%
#%%### INPUT TESTS AND CHECKS
if supress_text == False:
    print('Run Started')
    print('Run Series         : {}'.format(save_path.split('//')[-1]))
    print('Run Number         : {}'.format(run_num))
    print('Field save flag    : {}'.format(save_fields))
    print('Particle save flag : {}\n'.format(save_particles))
    
    print('Sim domain length  : {:5.2f}R_E'.format(2 * xmax / RE))
    print('Density            : {:5.2f}cc'.format(ne / 1e6))
    print('Equatorial B-field : {:5.2f}nT'.format(B_eq*1e9))
    print('Maximum    B-field : {:5.2f}nT\n'.format(Bc.max()*1e9))
    
    print('Equat. Gyroperiod: : {}s'.format(round(2. * np.pi / gyfreq, 2)))
    print('Inverse rad gyfreq : {}s'.format(round(1 / gyfreq, 2)))
    print('Maximum sim time   : {}s ({} gyroperiods)\n'.format(round(max_rev * 2. * np.pi / gyfreq, 2), max_rev))
    
    print('{} spatial cells, {} ring current cells, 2x{} damped cells'.format(NX, rc_hwidth*2, ND))
    print('{} cells total'.format(NC))
    print('{} particles total\n'.format(N))
    
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