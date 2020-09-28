# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:00:58 2017

@author: iarey

Changing so that values are calculated here, but main inputs come from files
That way I'd only have to change the input file, rather than manually changing
every single value on each run. Makes doing pearl-studies much easier.

This script will just be about loading them in, doing checks, and initializing
derived values/casting to SI units (e.g. alfven velocity)
"""
import numpy as np
import sys
import os

event_inputs = True

# Hard-coded some plasma param files. Loads based on position in array and run number if event_inputs True
# Can update and change these later if desired. Or even use a string format to replace run series (e.g. H_ONLY)
plasma_list = ['/run_inputs/from_data/H_ONLY/plasma_params_20130725_213004105000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213050105000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213221605000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213248105000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213307605000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213406605000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213703105000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_213907605000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_214026105000_H_ONLY.txt',
               '/run_inputs/from_data/H_ONLY/plasma_params_20130725_214105605000_H_ONLY.txt']

# =============================================================================
# plasma_list = ['/run_inputs/variants/plasma_params_protons.txt',
#                '/run_inputs/variants/plasma_params_w_helium.txt',
#                '/run_inputs/variants/plasma_params_w_oxygen.txt',
#                '/run_inputs/variants/plasma_params_w_helium_and_oxygen.txt',
#                ]
# =============================================================================

## INPUT RUN/DRIVER FILE LOCATIONS ##
if os.name == 'posix':
    root_dir     = os.path.dirname(sys.path[0])
else:
    root_dir     = '..'

run_input    = root_dir +  '/run_inputs/run_params.txt'

## SIMULATION PARAMETERS ##
with open(run_input, 'r') as f:
    ### RUN PARAMETERS ###
    drive             = f.readline().split()[1]        # Drive letter or path for portable HDD e.g. 'E:/' or '/media/yoshi/UNI_HD/'
    save_path         = f.readline().split()[1]        # Series save dir   : Folder containing all runs of a series
    run               = f.readline().split()[1]        # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)

    save_particles    = int(f.readline().split()[1])   # Save data flag    : For later analysis
    save_fields       = int(f.readline().split()[1])   # Save plot flag    : To ensure hybrid is solving correctly during run
    seed              = int(f.readline().split()[1])   # RNG Seed          : Set to enable consistent results for parameter studies
    cpu_affin         = f.readline().split()[1]        # Set CPU affinity for run as list. Set as None to auto-assign. 

    ## FLAGS ##
    homogenous        = int(f.readline().split()[1])   # Set B0 to homogenous (as test to compare to parabolic)
    particle_periodic = int(f.readline().split()[1])   # Set particle boundary conditions to periodic
    particle_reflect  = int(f.readline().split()[1])   # Set particle boundary conditions to reflective
    particle_reinit   = int(f.readline().split()[1])   # Set particle boundary conditions to reinitialize
    field_periodic    = int(f.readline().split()[1])   # Set field boundary to periodic (False: Absorbtive Boundary Conditions)
    disable_waves     = int(f.readline().split()[1])   # Zeroes electric field solution at each timestep
    te0_equil         = int(f.readline().split()[1])   # Initialize te0 to be in equilibrium with density
    source_smoothing  = int(f.readline().split()[1])   # Smooth source terms with 3-point Gaussian filter
    E_damping         = int(f.readline().split()[1])   # Damp E in a manner similar to B for ABCs
    quiet_start       = int(f.readline().split()[1])   # Flag to use quiet start (False :: semi-quiet start)
    radix_loading     = int(f.readline().split()[1])   # Load particles with reverse-radix scrambling sets (not implemented in this version)
    damping_multiplier= float(f.readline().split()[1]) # Multiplies the r-factor to increase/decrease damping rate.

    ### SIMULATION PARAMETERS ###
    NX        = int(f.readline().split()[1])           # Number of cells - doesn't include ghost cells
    ND        = int(f.readline().split()[1])           # Damping region length: Multiple of NX (on each side of simulation domain)
    max_rev   = float(f.readline().split()[1])         # Simulation runtime, in multiples of the ion gyroperiod (in seconds)
    dxm       = float(f.readline().split()[1])         # Number of c/wpi per dx (Ion inertial length: anything less than 1 isn't "resolvable" by hybrid code, anything too much more than 1 does funky things to the waveform)
    r_A       = float(f.readline().split()[1])         # Ionospheric anchor point (loss zone/max mirror point) - "Below 100km" - Baumjohann, Basic Space Plasma Physics
    
    ie        = int(f.readline().split()[1])           # Adiabatic electrons. 0: off (constant), 1: on.
    min_dens  = float(f.readline().split()[1])         # Allowable minimum charge density in a cell, as a fraction of ne*q
    rc_hwidth = f.readline().split()[1]                # Ring current half-width in number of cells (2*hwidth gives total cells with RC) 
      
    orbit_res = float(f.readline().split()[1])         # Orbit resolution
    freq_res  = float(f.readline().split()[1])         # Frequency resolution     : Fraction of angular frequency for multiple cyclical values
    part_res  = float(f.readline().split()[1])         # Data capture resolution in gyroperiod fraction: Particle information
    field_res = float(f.readline().split()[1])         # Data capture resolution in gyroperiod fraction: Field information

    ### RUN DESCRIPTION ###
    run_description = f.readline()                     # Commentary to attach to runs, helpful to have a quick description

# Override because I keep forgetting to change this
if os.name == 'posix':
    drive = '/home/c3134027/'

# Load run num from file, autoset if necessary
if run == '-':
    if os.path.exists(drive + save_path) == False:
        run = 0
    else:
        run = len(os.listdir(drive + save_path))
    print('Run number AUTOSET to ', run)
else:
    run = int(run)


# Set plasma parameter file
if event_inputs == False:
    plasma_input = root_dir +  '/run_inputs/plasma_params.txt'
else:
    plasma_input = root_dir +  plasma_list[run]
print('LOADING PLASMA: {}'.format(plasma_input))


### PARTICLE/PLASMA PARAMETERS ###
with open(plasma_input, 'r') as f:
    species_lbl = np.array(f.readline().split()[1:])
    
    temp_color = np.array(f.readline().split()[1:])
    temp_type  = np.array(f.readline().split()[1:], dtype=int)
    dist_type  = np.array(f.readline().split()[1:], dtype=int)
    nsp_ppc    = np.array(f.readline().split()[1:], dtype=int)
    
    mass       = np.array(f.readline().split()[1:], dtype=float)
    charge     = np.array(f.readline().split()[1:], dtype=float)
    drift_v    = np.array(f.readline().split()[1:], dtype=float)
    density    = np.array(f.readline().split()[1:], dtype=float)*1e6
    anisotropy = np.array(f.readline().split()[1:], dtype=float)
    
    # Particle energy: If beta == 1, energies are in beta. If not, they are in eV                                    
    E_per      = np.array(f.readline().split()[1:], dtype=float)
    E_e        = float(f.readline().split()[1])
    beta_flag  = int(f.readline().split()[1])

    L         = float(f.readline().split()[1])         # Field line L shell
    B_eq      = f.readline().split()[1]                # Initial magnetic field at equator: None for L-determined value (in T) :: 'Exact' value in node ND + NX//2



#%%### DERIVED SIMULATION PARAMETERS
### PHYSICAL CONSTANTS ###
q      = 1.602177e-19                       # Elementary charge (C)
c      = 2.998925e+08                       # Speed of light (m/s)
mp     = 1.672622e-27                       # Mass of proton (kg)
me     = 9.109384e-31                       # Mass of electron (kg)
kB     = 1.380649e-23                       # Boltzmann's Constant (J/K)
e0     = 8.854188e-12                       # Epsilon naught - permittivity of free space
mu0    = (4e-7) * np.pi                     # Magnetic Permeability of Free Space (SI units)
RE     = 6.371e6                            # Earth radius in metres
B_surf = 3.12e-5                            # Magnetic field strength at Earth surface (equatorial)

NC          = NX + 2*ND                     # Total number of cells
ne          = density.sum()                 # Electron number density
E_par       = E_per / (anisotropy + 1)      # Parallel species energy
ND_src_type = 0                             # 0: Copy last, 1: Zero gradient, 2: Zero second derivative

particle_open = 0
if particle_reflect + particle_reinit + particle_periodic == 0:
    particle_open = 1
    
if B_eq == '-':
    B_eq      = (B_surf / (L ** 3))         # Magnetic field at equator, based on L value
else:
    B_eq = float(B_eq)
    
if rc_hwidth == '-':
    rc_hwidth = 0
    
if beta_flag == 0:
    # Input energies as (perpendicular) eV
    beta_per   = None
    Te0_scalar = E_e   * 11603.
    Tpar       = E_par * 11603.
    Tper       = E_per * 11603.
else:    
    # Input energies in terms of a (perpendicular) beta
    Tpar       = E_par * B_eq ** 2 / (2 * mu0 * ne * kB)
    Tper       = E_per * B_eq ** 2 / (2 * mu0 * ne * kB)
    Te0_scalar = E_e   * B_eq ** 2 / (2 * mu0 * ne * kB)

wpi        = np.sqrt(ne * q ** 2 / (mp * e0))            # Proton   Plasma Frequency, wpi (rad/s)
va         = B_eq / np.sqrt(mu0*ne*mp)                   # Alfven speed at equator: Assuming pure proton plasma

dx         = dxm * c / wpi                               # Spatial cadence, based on ion inertial length
xmax       = NX // 2 * dx                                # Maximum simulation length, +/-ve on each side
xmin       =-NX // 2 * dx

charge    *= q                                           # Cast species charge to Coulomb
mass      *= mp                                          # Cast species mass to kg
drift_v   *= va                                          # Cast species velocity to m/s

vth_perp   = np.sqrt(kB *  Tper /  mass)
vth_par    = np.sqrt(kB *  Tpar /  mass)

Nj         = len(mass)                                   # Number of species
n_contr    = density / nsp_ppc                           # Species density contribution: Each macroparticle contributes this density to a cell

# Number of sim particles for each species, total
N_species = nsp_ppc * NX + 2   

# Add number of spare particles proportional to # cells worth
if particle_open == 1:
    spare_ppc  = 5*nsp_ppc.copy()
else:
    spare_ppc  = np.zeros(Nj, dtype=int)
N = N_species.sum() + spare_ppc.sum()

idx_start  = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
idx_end    = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order
    
############################
### MAGNETIC FIELD STUFF ###
############################
B_nodes  = (np.arange(NC + 1) - NC // 2)       * dx      # B grid points position in space
E_nodes  = (np.arange(NC)     - NC // 2 + 0.5) * dx      # E grid points position in space

if homogenous == 1:
    a      = 0
    B_xmax = B_eq
    
    # Also need to set any numeric values
    B_A            = 0.0
    loss_cone_eq   = 0.0
    loss_cone_xmax = 0.0
    theta_xmax     = 0.0
    lambda_L       = 0.0
    lat_A          = 0.0
else:
    print('Calculating length of field line...')
    N_fl   = 1e5                                                                # Number of points to calculate field line length (higher is more accurate)
    lat0   = np.arccos(np.sqrt((RE + r_A)/(RE*L)))                              # Latitude for this L value (at ionosphere height)
    h      = 2.0*lat0/float(N_fl)                                               # Step size of lambda (latitude)
    f_len  = 0.0
    for ii in range(int(N_fl)):
        lda        = ii*h - lat0                                                # Lattitude for this step
        f_len     += L*RE*np.cos(lda)*np.sqrt(4.0 - 3.0*np.cos(lda) ** 2) * h   # Field line length accruance
    print('Field line length = {:.2f} RE'.format(f_len/RE))
    print('Simulation length = {:.2f} RE'.format(2*xmax/RE))
    
    if xmax > f_len / 2:
        sys.exit('Simulation length longer than field line. Aboring...')
        
    print('Finding simulation boundary MLAT...')
    dlam   = 1e-5                                            # Latitude increment in radians
    fx_len = 0.0; ii = 1                                     # Arclength/increment counters
    while fx_len < xmax:
        lam_i   = dlam * ii                                                             # Current latitude
        d_len   = L * RE * np.cos(lam_i) * np.sqrt(4.0 - 3.0*np.cos(lam_i) ** 2) * dlam     # Length increment
        fx_len += d_len                                                                 # Accrue arclength
        ii     += 1                                                                     # Increment counter
    
# =============================================================================
#         sys.stdout.write('\r{:.1f}% complete'.format(fx_len/xmax * 100.))
#         sys.stdout.flush()
#     print('\n')
# =============================================================================

    theta_xmax  = lam_i                                                                 # Latitude of simulation boundary
    r_xmax      = L * RE * np.cos(theta_xmax) ** 2                                      # Radial distance of simulation boundary
    B_xmax      = B_eq*np.sqrt(4 - 3*np.cos(theta_xmax)**2)/np.cos(theta_xmax)**6       # Magnetic field intensity at boundary
    a           = (B_xmax / B_eq - 1) / xmax ** 2                                       # Parabolic scale factor: Fitted to B_eq, B_xmax
    lambda_L    = np.arccos(np.sqrt(1.0 / L))                                           # Lattitude of Earth's surface at this L

    lat_A      = np.arccos(np.sqrt((RE + r_A)/(RE*L)))       # Anchor latitude in radians
    B_A        = B_eq * np.sqrt(4 - 3*np.cos(lat_A) ** 2)\
               / (np.cos(lat_A) ** 6)                        # Magnetic field at anchor point
    
    loss_cone_eq   = np.arcsin(np.sqrt(B_eq   / B_A))*180 / np.pi   # Equatorial loss cone in degrees
    loss_cone_xmax = np.arcsin(np.sqrt(B_xmax / B_A))               # Boundary loss cone in radians


# Freqs based on highest magnetic field value (at simulation boundaries)
gyfreq     = q*B_xmax/ mp                                # Proton Gyrofrequency (rad/s) at boundary (highest)
gyfreq_eq  = q*B_eq  / mp                                # Proton Gyrofrequency (rad/s) at equator (slowest)
k_max      = np.pi / dx                                  # Maximum permissible wavenumber in system (SI???)
qm_ratios  = np.divide(charge, mass)                     # q/m ratio for each species

# Calculate injection rate
if particle_open == 1:
    inject_rate = nsp_ppc * (vth_par / dx) / np.sqrt(2 * np.pi)
else:
    inject_rate = nsp_ppc * 0.0
    
species_plasfreq_sq   = (density * charge ** 2) / (mass * e0)
species_gyrofrequency = qm_ratios * B_eq


#%%### INPUT TESTS AND CHECKS
print('Run Started')
print('Run Series         : {}'.format(save_path.split('//')[-1]))
print('Run Number         : {}'.format(run))
print('Field save flag    : {}'.format(save_fields))
print('Particle save flag : {}\n'.format(save_particles))

print('Sim domain length  : {:5.2f}R_E'.format(2 * xmax / RE))
print('Density            : {:5.2f}cc'.format(ne / 1e6))
print('Equatorial B-field : {:5.2f}nT'.format(B_eq*1e9))
print('Maximum    B-field : {:5.2f}nT'.format(B_xmax*1e9))
print('Iono.      B-field : {:5.2f}mT'.format(B_A*1e6))
print('Equat. Loss cone   : {:<5.2f} degrees  '.format(loss_cone_eq))
print('Bound. Loss cone   : {:<5.2f} degrees  '.format(loss_cone_xmax * 180. / np.pi))
print('Maximum MLAT (+/-) : {:<5.2f} degrees  '.format(theta_xmax * 180. / np.pi))
print('Iono.   MLAT (+/-) : {:<5.2f} degrees\n'.format(lambda_L * 180. / np.pi))

print('Equat. Gyroperiod: : {}s'.format(round(2. * np.pi / gyfreq, 3)))
print('Inverse rad gyfreq : {}s'.format(round(1 / gyfreq, 3)))
print('Maximum sim time   : {}s ({} gyroperiods)\n'.format(round(max_rev * 2. * np.pi / gyfreq_eq, 2), max_rev))

print('{} spatial cells, 2x{} damped cells'.format(NX, ND))
print('{} cells total'.format(NC))
print('{} particles total\n'.format(N))


if cpu_affin != '-':
    if len(cpu_affin) == 1:
        cpu_affin = [int(cpu_affin)]        
    else:
        cpu_affin = list(map(int, cpu_affin.split(',')))
    
    import psutil
    run_proc = psutil.Process()
    run_proc.cpu_affinity(cpu_affin)
    print('CPU affinity for run (PID {}) set to :: {}'.format(run_proc.pid, ', '.join(map(str, run_proc.cpu_affinity()))))
else:
    print('CPU affinity not set.')

if theta_xmax > lambda_L:
    print('--------------------------------------------------')
    print('WARNING : SIMULATION DOMAIN LONGER THAN FIELD LINE')
    print('DO SOMETHING ABOUT IT')
    print('--------------------------------------------------')
    sys.exit()

if particle_periodic + particle_reflect + particle_reinit > 1:
    print('--------------------------------------------------')
    print('WARNING : ONLY ONE PARTICLE BOUNDARY CONDITION ALLOWED')
    print('DO SOMETHING ABOUT IT')
    print('--------------------------------------------------')
    
os.system("title Hybrid Simulation :: {} :: Run {}".format(save_path.split('//')[-1], run))
