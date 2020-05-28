# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:00:58 2017

@author: iarey
"""
import numpy as np
import sys
from os import system

### RUN DESCRIPTION ###
run_description = '''Marginal instability test for 25/07/2013 event -- Maximum growth paramters (growth expected)'''


### RUN PARAMETERS ###
drive             = 'F:'                          # Drive letter or path for portable HDD e.g. 'E:/' or '/media/yoshi/UNI_HD/'
save_path         = 'runs//july_25_lingrowth_PARABOLIC'  # Series save dir   : Folder containing all runs of a series
run               = 0                             # Series run number : For multiple runs (e.g. parameter studies) with same overall structure (i.e. test series)
save_particles    = 1                             # Save data flag    : For later analysis
save_fields       = 1                             # Save plot flag    : To ensure hybrid is solving correctly during run
seed              = 3216587                       # RNG Seed          : Set to enable consistent results for parameter studies
cpu_affin         = [(2*run)%8, (2*run + 1)%8]    # Set CPU affinity for run. Must be list. Auto-assign: None.


## DIAGNOSTIC FLAGS ##
supress_text      = False                         # Supress initialization text
homogenous        = False                         # Set B0 to homogenous (as test to compare to parabolic)
disable_waves     = False                         # Zeroes electric field solution at each timestep
shoji_approx      = False                         # Changes solution used for calculating particle B0r (1D vs. 3D)
te0_equil         = False                         # Initialize te0 to be in equilibrium with density
source_smoothing  = True                          # Smooth source terms with 3-point Gaussian filter
reflect           = False                         # THIS IS BROKEN!!! 'Reflects' particles at edges by randomizing their gyrophase


### SIMULATION PARAMETERS ###
NX        = 512                              # Number of cells - doesn't include ghost cells
ND        = 128                              # Damping region length: Multiple of NX (on each side of simulation domain)
max_rev   = 100                              # Simulation runtime, in multiples of the ion gyroperiod (in seconds)
dxm       = 1.0                              # Number of c/wpi per dx (Ion inertial length: anything less than 1 isn't "resolvable" by hybrid code, anything too much more than 1 does funky things to the waveform)
L         = 5.00                             # Field line L shell
r_A       = 100e3                           # Ionospheric anchor point (loss zone/max mirror point) - "Below 100km" - Baumjohann, Basic Space Plasma Physics

ie        = 1                               # Adiabatic electrons. 0: off (constant), 1: on.
B_eq      = None                            # Initial magnetic field at equator: None for L-determined value (in T)
rc_hwidth = 0                               # Ring current half-width in number of cells (2*hwidth gives total cells with RC) 

orbit_res = 0.10                            # Particle orbit resolution: Fraction of gyroperiod in seconds
freq_res  = 0.02                            # Frequency resolution     : Fraction of inverse radian cyclotron frequency
part_res  = 0.25                            # Data capture resolution in gyroperiod fraction: Particle information
field_res = 0.10                            # Data capture resolution in gyroperiod fraction: Field information


### PARTICLE PARAMETERS ###
species_lbl= [r'$H^+$ cold', r'$He^+$ cold', r'$O^+$ cold',
              r'$H^+$ warm', r'$He^+$ warm', r'$O^+$ warm',
              r'$H^+$ hot' , r'$He^+$ hot' , r'$O^+$ hot']  # Species name/labels        : Used for plotting. Can use LaTeX math formatted strings

temp_color = ['red'      , 'navy'    , 'purple',
              'firebrick', 'darkblue', 'violet',
              'rosybrown', 'blue'    , 'plum']

temp_type  = np.asarray([0, 0, 0,
                         1, 1, 1,
                         1, 1, 1])                   	    # Particle temperature type  : Cold (0) or Hot (1) : Used for plotting

dist_type  = np.asarray([0, 0, 0,
                         0, 0, 0,
                         0, 0, 0])                          # Particle distribution type : Uniform (0) or sinusoidal/other (1) : Used for plotting (normalization)

nsp_ppc    = np.array([200, 200, 200,
                       1000, 1000,1000,
                       1000, 1000,1000])                           # Number of particles per cell, per species

mass       = np.asarray([1., 4., 16.,
                         1., 4., 16.,
                         1., 4., 16.])    			        # Species ion mass (proton mass units)

charge     = np.asarray([1., 1., 1.,
                         1., 1., 1.,
                         1., 1., 1.])    			        # Species ion charge (elementary charge units)

drift_v    = np.asarray([0., 0., 0.,
                         0., 0., 0.,
                         0., 0., 0.])                       # Species parallel bulk velocity (alfven velocity units)

density    = np.asarray([166.469 , 47.562    , 23.781    ,
                         1.72644 , 1.3931    , 0.583791  ,
                         0.673249, 0.00739014, 0.00358241]) * 1e6

E_per      = np.array([5.0    , 5.0    , 5.0    ,
                       12785.2, 1178.23, 6124.12,
                       54714.7, 98254.9, 159204])

anisotropy = np.array([0.0      , 0.0    , 0.0     ,
                       0.2555503, 0.22624, 0.117326,
                       0.583481 , 1.27531, 0.686062])

min_dens       = 0.05                                       # Allowable minimum charge density in a cell, as a fraction of ne*q
E_e            = 200.0                                      # Electron energy (eV)

account_for_dispersion = False                              # Flag (True/False) whether or not to reduce timestep to prevent dispersion getting too high
dispersion_allowance   = 1.                                 # Multiple of how much past frac*wD^-1 is allowed: Used to stop dispersion from slowing down sim too much  



#%%### DERIVED SIMULATION PARAMETERS
### PHYSICAL CONSTANTS ###
q      = 1.602177e-19                          # Elementary charge (C)
c      = 2.998925e+08                          # Speed of light (m/s)
mp     = 1.672622e-27                          # Mass of proton (kg)
me     = 9.109384e-31                          # Mass of electron (kg)
kB     = 1.380649e-23                          # Boltzmann's Constant (J/K)
e0     = 8.854188e-12                          # Epsilon naught - permittivity of free space
mu0    = (4e-7) * np.pi                        # Magnetic Permeability of Free Space (SI units)
RE     = 6.371e6                               # Earth radius in metres
B_surf = 3.12e-5                            # Magnetic field strength at Earth surface

NC         = NX + 2*ND
ne         = density.sum()
E_par      = E_per / (anisotropy + 1)
    
if B_eq is None:
    B_eq      = (B_surf / (L ** 3))                      # Magnetic field at equator, based on L value

Te0_scalar = E_e   * 11603.
Tpar       = E_par * 11603.
Tper       = E_per * 11603.

wpi        = np.sqrt(ne * q ** 2 / (mp * e0))            # Proton   Plasma Frequency, wpi (rad/s)
va         = B_eq / np.sqrt(mu0*ne*mp)                   # Alfven speed: Assuming pure proton plasma

dx         = dxm * c / wpi                               # Spatial cadence, based on ion inertial length
xmin       =-NX // 2 * dx                                # Minimum simulation dimension
xmax       = NX // 2 * dx                                # Maximum simulation dimension
xmin       =-NX // 2 * dx                                # Maximum simulation dimension

charge    *= q                                           # Cast species charge to Coulomb
mass      *= mp                                          # Cast species mass to kg
drift_v   *= va                                          # Cast species velocity to m/s

Nj         = len(mass)                                   # Number of species
n_contr    = density / nsp_ppc                           # Species density contribution: Each macroparticle contributes this density to a cell

# Number of sim particles for each species, total
N_species  = np.zeros(Nj, dtype=np.int64)
for jj in range(Nj):
    # Cold species in every cell NX 
    if temp_type[jj] == 0:                               
        N_species[jj] = nsp_ppc[jj] * NX + 2   
        
    # Warm species only in simulation center, unless rc_hwidth = 0 (disabled)           
    elif temp_type[jj] == 1:
        if rc_hwidth == 0:
            N_species[jj] = nsp_ppc[jj] * NX + 2
        else:
            N_species[jj] = nsp_ppc[jj] * 2*rc_hwidth + 2    
N = N_species.sum()

idx_start  = np.asarray([np.sum(N_species[0:ii]    )     for ii in range(0, Nj)])    # Start index values for each species in order
idx_end    = np.asarray([np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)])    # End   index values for each species in order

############################
### MAGNETIC FIELD STUFF ###
############################
B_nodes  = (np.arange(NC + 1) - NC // 2)       * dx      # B grid points position in space
E_nodes  = (np.arange(NC)     - NC // 2 + 0.5) * dx      # E grid points position in space

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

    sys.stdout.write('\r{:.1f}% complete'.format(fx_len/xmax * 100.))
    sys.stdout.flush()
print('\n')

theta_xmax  = lam_i                                                                 # Latitude of simulation boundary
r_xmax      = L * RE * np.cos(theta_xmax) ** 2                                      # Radial distance of simulation boundary
B_xmax      = B_eq*np.sqrt(4 - 3*np.cos(theta_xmax)**2)/np.cos(theta_xmax)**6       # Magnetic field intensity at boundary
a           = (B_xmax / B_eq - 1) / xmax ** 2                                       # Parabolic scale factor: Fitted to B_eq, B_xmax
lambda_L    = np.arccos(np.sqrt(1.0 / L))                                           # Lattitude of Earth's surface at this L

if homogenous == True:
    a      = 0
    B_xmax = B_eq

Bc           = np.zeros((NC + 1, 3), dtype=np.float64)   # Constant components of magnetic field based on theta and B0
Bc[:, 0]     = B_eq * (1 + a * B_nodes**2)               # Set constant Bx
Bc[:ND]      = Bc[ND]                                    # Set B0 in damping cells (same as last spatial cell)
Bc[ND+NX+1:] = Bc[ND+NX]

# Freqs based on highest magnetic field value (at simulation boundaries)
gyfreq     = q*B_xmax/ mp                                # Proton Gyrofrequency (rad/s) at boundary (highest)
gyfreq_eq  = q*B_eq  / mp                                # Proton Gyrofrequency (rad/s) at equator (slowest)
k_max      = np.pi / dx                                  # Maximum permissible wavenumber in system (SI???)
qm_ratios  = np.divide(charge, mass)                     # q/m ratio for each species

lat_A      = np.arccos(np.sqrt((RE + r_A)/(RE*L)))       # Anchor latitude in radians
B_A        = B_eq * np.sqrt(4 - 3*np.cos(lat_A) ** 2)\
           / (np.cos(lat_A) ** 6)                        # Magnetic field at anchor point

loss_cone_eq   = np.arcsin(np.sqrt(B_eq   / B_A))*180 / np.pi   # Equatorial loss cone in degrees
loss_cone_xmax = np.arcsin(np.sqrt(B_xmax / B_A))               # Boundary loss cone in radians














#%%
#%%
#%%
#%%
#%%
#%%
#%%### INPUT TESTS AND CHECKS
if rc_hwidth == 0:
    rc_print = NX
else:
    rc_print = rc_hwidth*2

if supress_text == False:
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
    
    print('{} spatial cells, {} with ring current, 2x{} damped cells'.format(NX, rc_print, ND))
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

    
simulated_density_per_cell = (n_contr * charge * nsp_ppc).sum()
real_density_per_cell      = ne*q

if abs(simulated_density_per_cell - real_density_per_cell) / real_density_per_cell > 1e-10:
    print('--------------------------------------------------------------------------------')
    print('WARNING: DENSITY CALCULATION ISSUE: RECHECK HOW MACROPARTICLE CONTRIBUTIONS WORK')
    print('--------------------------------------------------------------------------------')
    print('')
    print('ABORTING...')
    sys.exit()

if theta_xmax > lambda_L:
    print('--------------------------------------------------')
    print('WARNING : SIMULATION DOMAIN LONGER THAN FIELD LINE')
    print('DO SOMETHING ABOUT IT')
    print('--------------------------------------------------')
    sys.exit()
    
system("title Hybrid Simulation :: {} :: Run {}".format(save_path.split('//')[-1], run))