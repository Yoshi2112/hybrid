# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:09:06 2016

@author: Joshua Williams
"""
from timeit import default_timer as timer
import numpy as np
from numpy import pi
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
import sys
from mpl_toolkits.mplot3d import Axes3D
import pdb
import matplotlib as mpl

# Metadata
seed           = 21                         # Some seeds are no good for the general resonance shape: Run simple test beforehand
drive          = 'H'                        # Drive letter for portable HDD (changes between computers)
save_path      = 'Runs\Smooth Two D'        # Save path on 'drive' HDD - each run then saved in numerically sequential subfolder with images and associated data
generate_plots = 1                          # Turn plot generation on (1) or off (0) during sim-time. Will still save data when '0'.
generate_data  = 1                          # As above, for data
start_time     = timer()                    # Start Timer
np.random.seed(seed)                        # Random seed


#%%        ----- SET VALUES / LAWS OF PHYSICS -----

# Constants
q   = 1.602e-19                             # Elementary charge (C)
c   = 3e8                                   # Speed of light (m/s)
me  = 9.11e-31                              # Mass of electron (kg)
mp  = 1.67e-27                              # Mass of proton (kg)
e   = -q                                    # Electron charge (C)
mu0 = (4e-7) * pi                           # Magnetic Permeability of Free Space (SI units)
kB  = 1.38065e-23                           # Boltzmann's Constant (J/K)
e0  = 8.854e-12                             # Epsilon naught - permittivity of free space

#%%

f   = 0.015                                 # Relative Density
V   = 10                                    # Streaming velocity as a multiple of the alfven speed 

# Species Characteristics - use column number as species identifier. Default - 0: Hot protons, 1: Cold protons, 2: Cold Helium2+
                   #H+ (H)      H+ (C)      He+(C)
partin = np.array([[1    ,       1     ],         #(0) Mass   (proton units)
                   [1    ,       1     ],         #(1) Charge (charge units)
                   [V    , -f*V/(1-f)  ],         #(2) Bulk Velocity (multiples of alfven speed)
                   [f    ,      1 - f  ],         #(3) Relative  (real)          Density (as a portion of 1)
                   [0.5  ,      0.5    ],         #(4) Simulated (superparticle) Density (as a portion of 1)
                   [1    ,       1     ],         #(5) Beta Parallel      (x)
                   [1    ,       1     ]])        #(6) Beta Perpendicular (y, z)
                   
part_type      = [r'$H^{+} beam$', r'$H^{+} core$']                   # Species labels
species_colors = ['red', 'cyan', 'magenta', 'lime']                   # Species colors (to use on plots to keep consistency)
Nj             = int(np.shape(partin)[1])                             # Number of species (number of columns above)

## User defined Model Variables ##
ts       = 0.05                             # Timestep as a fraction of a gyroperiod
dxm      = 2                                # Number of c/wpi per dx, dy (square space)
NX       = 128                              # Number of cells in x,y - dimension of array (not including ghost cells)
maxtime  = 4000                             # Number of iterations to progress through. Total time = maxtime * dt
cellpart = 16                               # Number of Particles per cell (make it an even number for 50/50 hot/cold)

n0    = 8.48e6                              # Total ion density - initial density per cell (in m-3 : First number representative of density in cm-3 )
B0    = 4e-9                                # Unform initial magnetic field value (in T) (must be parallel to an axis)
theta = 0                                   # Angle of B0 to x axis (in xy plane in units of degrees)
E0    = 0                                   # Uniform initial electric field (in V) (must be parallel to an axis)
    
ie            = 1                           # Adiabatic electrons. 0: off, 1: on.
electron_temp = 0                           # (Initial) Electron temperature. Set to 0 for isothermal approximation.
framegrab     = 1                           # Grab every x image

## Magnetic Field Stuff ##
Bxc   = B0 * np.sin((90 - theta) * pi / 180 )   # Constant x-component of magnetic field (theta in degrees)
Byc   = B0 * np.cos((90 - theta) * pi / 180 )   # Constant Y-component of magnetic field (theta in degrees)  
Bzc   = 0  

## Derived Values ##
size     = NX + 2                               # Includes ghost cells
av_rho   = np.sum([partin[0, nn] * mp * partin[3, nn] * n0 for nn in range(Nj)])    # Average mass density
alfie    = B0/np.sqrt(mu0 * av_rho)             # Alfven Velocity (m/s): Constant - Initialized at t = 0
gyfreq   = q*B0/mp                              # Proton Gyrofrequency (rad/s) (since this will be the highest of all species)
gyperiod = 2*pi / gyfreq                        # Gyroperiod in seconds
wpi      = np.sqrt((n0 * (q**2)) / (mp * e0 ))  # Plasma Frequency (rad/s)

DT       = ts / gyfreq                          # Time step as fraction of gyroperiod (T = 1 / f) - CHANGE TO BE FOR THE HEAVIEST ION?
dx       = dxm * c / wpi                        # Spacial step as function of plasma frequency (in metres)
dy       = dx
xmax     = NX * dx                              # Spatial length of simulation
ymax     = xmax

## Particle Values ##
psize         = 1                                                          # Size of particles on plots
N             = cellpart*(NX ** 2)                                         # Number of Particles to simulate: # cells x # particles per cell, excluding ghost cells
N_species     = np.round(N * partin[4, :]).astype(int)                     # Number of sim particles for each species, total
N_cell        = [float(N_species[ii]) / (NX * NX) for ii in range(Nj)]     # Number of sim particles per cell, per species

N_real        = (dx * dy * 1.) * n0 * NX *NX                               # Total number of real particles (rect prism with sides dx x 1 x 1 metres)

# Output Particle Values
partout = np.array([partin[0, :] * mp,                                     # (0) Actual Species Mass    (in kg)
                    partin[1, :] * q,                                      # (1) Actual Species Charge  (in coloumbs)
                    partin[2, :] * alfie,                                  # (2) Actual Species streaming velocity
                    (N_real * partin[3, :]) / N_species])                  # (3) Density contribution of each particle of species (real particles per sim particle)
                    
N_real_cell = [N_cell[ii] * partout[3, ii] for ii in range(Nj)] 
                  
Tpar = ((alfie ** 2) * partout[0, :] * partin[5, :]) / (2 * kB)        # Parallel Temperature
Tprp = ((alfie ** 2) * partout[0, :] * partin[6, :]) / (2 * kB)        # Perpendicular Temperature

if electron_temp == 0:                                      # Initial electron temp (const. if ie = 0)
    Te0 = Tprp[0]                                           # Isothermal with perpendicular temperature of first species
    
print 'speeds: %d \n freqs: %d' % (c / alfie, wpi / gyfreq)

#%%         ----- PARTICLES -----
# Particle Array: Initalization and Loading
part     = np.zeros((9, N),    dtype=float)         # Create array of zeroes N x 13 for pos, vel and F 3-vectors
old_part = np.zeros((9, N),    dtype=float)         # Place to store last particle states while using Predictor-Corrector method

idx_start = [np.sum(N_species[0:ii]    )     for ii in range(0, Nj)]                     # Start index values for each species in order
idx_end   = [np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)]                     # End   index values for each species in order
idx = 0                                                                                  # Index of particle based on position in partin

for ii in N_species:                             
    part[8, idx_start[idx]: idx_end[idx]] = idx                                     # Give index identifier to each particle
    part[0:2, idx_start[idx]: idx_end[idx]] = np.random.uniform(0, xmax, (2, ii))   # Ok, this will do for a random distribution

                                                 # Load into velocity space
    part[3, idx_start[idx]: idx_end[idx]] = partout[2, idx] + np.random.normal(0, np.sqrt((kB * Tpar[idx] / partout[0, idx])), ii)        # Bulk velocity + V_therm(x)
    part[4, idx_start[idx]: idx_end[idx]] =                   np.random.normal(0, np.sqrt((kB * Tprp[idx] / partout[0, idx])), ii)        # V_therm(y)
    part[5, idx_start[idx]: idx_end[idx]] =                   np.random.normal(0, np.sqrt((kB * Tprp[idx] / partout[0, idx])), ii)        # V_therm(z)
    
    idx += 1                                    # Move into next species

# Re-load stream particles just to make things easier
stream_width = 2*dy                             # Radius of stream in metres

part[1, idx_start[0]: idx_end[0]] = np.random.uniform(ymax / 2. - stream_width, ymax / 2. + stream_width, N_species[0])
part[6, :] = part[0, :] / dx + 0.5              # Initial leftmost node,    Ix
part[7, :] = part[1, :] / dx + 0.5              # Initial bottom-most node, Iy


#%%        ----- ARRAYS -----

# Initialize Field Arrays: (size, size, 3) for 2D, (size, size, size, 3) for 3D
B = np.zeros((size, size, 6), dtype=float)    # Magnetic Field Array of shape (size), each element holds 3-vector
    # Where:
    #       B[mm, nn, 0-2] represent the current field and
    #       B[mm, nn, 3-5] store the last state of the magnetic field previous to the Predictor-Corrector scheme
B[:, :, 0] = Bxc      # Set Bx initial
B[:, :, 1] = Byc      # Set By initial
B[:, :, 2] = Bzc      # Set Bz initial

E = np.zeros((size, size, 9), dtype=float)    # Electric Field Array
    # Where:
    #       E[mm, nn, 0-2] represent the current field and
    #       E[mm, nn, 3-5] store the last state of the electric field previous to the Predictor-Corrector scheme E (N + 0.5)
    #       E[mm, nn, 6-8] store two steps ago: E-field at E^N
E[:, :, 0] = 0         # Set Ex initial
E[:, :, 1] = 0         # Set Ey initial
E[:, :, 2] = 0         # Set Ez initial

Vi      = np.zeros((size, size, Nj, 3), dtype=float)          # Ion Flow (3 dimensions)
dns     = np.zeros((size, size, Nj),    dtype=float)          # Species number density in each cell (in /m3)
dns_old = np.zeros((size, size, Nj),    dtype=float)          # For PC method
W       = np.zeros((2, 2, N),           dtype=float)          # Weighting array (to make it easier to move around and assign)


#%%        ----- FUNCTIONS -----

def check_integrity(array, nn, labels):
    if np.sum(array[3, :]) != 1:
        sys.exit('Real particle density sum not equal to 1')
        
    if np.sum(array[4, :]) != 1:
        sys.exit('Simulated particle density sum not equal to 1')
    
    if len(labels) != nn:
        sys.exit('Too many labels for number of species')
    return

def check_particles(density_array):
    ''' Takes input density array (for all cells/species) and calculates
        the total number of particles in the spatial domain, as a percentage
        of all initialized, real particles. Density_array values are 
        in /m3, and divided between species.'''
    
    total = np.sum(dns[1:size-1, 1:size-1, :]) * dx * dx    # Exclude ghost cells
    percent = (total / N_real) * 100                        # Calculate as % of initial particles
    
    print 'Spatial domain currently contains %.1f%% of real particles' % percent
    return

def check_velocity(idj):
    rcParams.update({'text.color'   : 'k',
                'axes.labelcolor'   : 'k',
                'axes.edgecolor'    : 'k',
                'axes.facecolor'    : 'w',
                'mathtext.default'  : 'regular',
                'xtick.color'       : 'k',
                'ytick.color'       : 'k',
                'axes.labelsize'    : 24,
                })
        
    fig = plt.figure(figsize=(40,18))
    fig.patch.set_facecolor('w') 
    num_bins = 200
    
    ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
    ax_y = plt.subplot2grid((2, 3), (0,2))
    ax_z = plt.subplot2grid((2, 3), (1,2))
    
    xs, BinEdgesx = np.histogram(part[3, idx_start[idj]:idx_end[idj]] / alfie, bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$v_x$')
    
    ys, BinEdgesy = np.histogram(part[3, idx_start[idj]:idx_end[idj]] / alfie, bins=num_bins)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y$')
    
    zs, BinEdgesz = np.histogram(part[3, idx_start[idj]:idx_end[idj]] / alfie, bins=num_bins)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z$')
    
    plt.show()
    

def check_position():   # Hard-coded for 2 species
    rcParams.update({'text.color'   : 'k',
                'axes.labelcolor'   : 'k',
                'axes.edgecolor'    : 'k',
                'axes.facecolor'    : 'w',
                'mathtext.default'  : 'regular',
                'xtick.color'       : 'k',
                'ytick.color'       : 'k',
                'axes.labelsize'    : 30,
                'xtick.labelsize'   : 24,
                'ytick.labelsize'   : 24,
                })
    
    fig = plt.figure(figsize=(40,18))
    fig.patch.set_facecolor('w') 
    num_bins = NX
    
    ax0 = plt.subplot2grid((2, 4), (0,0), colspan=2, rowspan=2)
    ax1 = plt.subplot2grid((2, 4), (0,2), colspan=2, rowspan=2)
    
    xs, BinEdgesx = np.histogram(part[0, idx_start[0]:idx_end[0]] / 1000, bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax0.plot(bx, xs, '-', c='r', drawstyle='steps')
    ax0.set_xlabel(r'x (m)')
    ax0.get_xaxis().get_major_formatter().set_useOffset(False)
    ax0.set_title('Species 0')
    
    ys, BinEdgesy = np.histogram(part[0, idx_start[1]:idx_end[1]] / 1000, bins=num_bins)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax1.plot(by, ys, '-', c='c', drawstyle='steps')
    ax1.set_xlabel(r'x (m)')
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.set_title('Species 1')
    
    plt.tight_layout(pad=1.0, w_pad=1.8)
    plt.show()
    

def assign_weighting(part):
    print 'Weighting particles...'
    h_i = np.zeros((2,2))                                                 # Empty distance array
    W_o = np.zeros((2, 2, N))    
   
    # Collect number density of all particles: Create subfunction particle_count ?
    for ii in range(N):
        Ix  = int(part[6, ii])                                 # x-floor
        Iy  = int(part[7, ii])                                 # y-floor
        p  = 2
            
        for xx in range(2):
            for yy in range(2):                               # Calculate distance from each four surrounding nodes
                h_i[xx,yy] = np.sqrt((((Ix + xx - 0.5)*dx - part[0, ii]) ** 2) + (((Iy + yy - 0.5)*dy - part[1, ii]) ** 2))
    
        w_i         = (h_i ** (-p)) / np.sum(h_i ** (-p))           
        W_o[:, :, ii] = w_i
    print 'Max weighting function: %.3f\n' % np.max(W_o)

    return W_o
                
    
def velocity_update(part, B, E, dt, W_in):  # Based on Appendix A of Ch5 : Hybrid Codes by Winske & Omidi.
    print 'Updating velocity...'
    for n in range(N):
        
        vn = part[3:6, n]                   # Existing particle velocity
        
        # Weighted E & B fields at particle location (node) - Weighted average of two nodes on either side of particle
        Ix  = int(part[6, n])       # Nearest (leftmost) node
        Iy  = int(part[7, n])       # Nearest (bottom)   node
        idx = int(part[8, n])       # Species type
        
        E_p = np.zeros(3)     ;   B_p = np.zeros(3)     # Zero fields at new particle to load in weightings
        
        for xx in range(2):
            for yy in range(2):
                E_p += E[Ix + xx, Iy + yy, 0:3] * W_in[xx, yy, n]
                B_p += B[Ix + xx, Iy + yy, 0:3] * W_in[xx, yy, n]
                
        # Intermediate calculations
        h = (partout[1 , idx] * dt) / partout[0 , idx] 
        f = 1 - (h**2) / 2 * (B_p[0]**2 + B_p[1]**2 + B_p[2]**2 )
        g = h / 2 * (B_p[0]*vn[0] + B_p[1]*vn[1] + B_p[2]*vn[2])
        v0 = vn + (h/2)*E_p
    
        # Velocity push
        part[3,n] = f * vn[0] + h * ( E_p[0] + g * B_p[0] + (v0[1]*B_p[2] - v0[2]*B_p[1]) )
        part[4,n] = f * vn[1] + h * ( E_p[1] + g * B_p[1] - (v0[0]*B_p[2] - v0[2]*B_p[0]) )
        part[5,n] = f * vn[2] + h * ( E_p[2] + g * B_p[2] + (v0[0]*B_p[1] - v0[1]*B_p[0]) )
        
    print 'Max velocity: %.2f vA\n' % np.max(part[3:5, :] / alfie)
        
    return part        
    
    
def check_boundary(pos):
    if pos[0] < 0:
        pos[0] += xmax
        
    if pos[1] < 0:
        pos[1] += ymax
        
    if pos[0] > xmax:
        pos[0] -= xmax
    
    if pos[1] > ymax:
        pos[1] -= ymax
    return pos


def position_update(part):  # Basic Push (x, v) vectors and I, W update
    print 'Pushing particles...'
    
    part[0, :] += part[3, :] * DT                       # Update position vectors
    part[1, :] += part[4, :] * DT
        
    for ii in range(N):                                 # Check for particles outside grid
        part[0:2, ii] = check_boundary(part[0:2, ii])

    part[6, :] = part[0, :] / dx + 0.5                  # Leftmost node index, Ix
    part[7, :] = part[1, :] / dy + 0.5                  # Bottom-most node index, Iy

    W_out = assign_weighting(part)                      # Calculate IDW coefficients from 4 nearest nodes
    print 'Max position values: %.4f, %.4f\n' % (np.max(part[0, :]) / xmax, np.max(part[1, :]) / ymax)
    
    return part, W_out
    
    
def push_B(B, E, dt):   # Basically Faraday's Law. (B, E) vectors
    print 'Updating magnetic field...'    
    Bp = np.zeros((size, size, 3), dtype=float)
 
    Bp[:, :, 0] = B[:, :, 0] - Bxc      # Consider B1 only (perturbation)   
    Bp[:, :, 1] = B[:, :, 1] - Byc
    Bp[:, :, 2] = B[:, :, 2] - Bzc   

    for mm in range (1, size - 1):      # Faraday's law calculation
        for nn in range(1, size - 1):
            Bp[mm, nn, 0] = Bp[mm, nn, 0] - (dt / 2) * (  (E[mm, nn + 1, 2] - E[mm, nn - 1, 2]) / (2 * dy) )
            Bp[mm, nn, 1] = Bp[mm, nn, 1] - (dt / 2) * (  (E[mm - 1, nn, 2] - E[mm + 1, nn, 2]) / (2 * dx) )
            Bp[mm, nn, 2] = Bp[mm, nn, 2] - (dt / 2) * ( ((E[mm + 1, nn, 1] - E[mm - 1, nn, 1]) / (2 * dx) ) - ((E[mm, nn + 1, 0] - E[mm, nn - 1, 0]) / (2 * dy)) )

    B[:, :, 0] = Bp[:, :, 0] + Bxc      # Combine total B-field: B0 + B1
    B[:, :, 1] = Bp[:, :, 1] + Byc
    B[:, :, 2] = Bp[:, :, 2] + Bzc 

    Bp[1, :, :]        += Bp[size - 1, :, :]    # Move ghost cell contributions (sides)
    Bp[size - 2, :, :] += Bp[0, :, :]
    
    Bp[:, 1, :]        += Bp[:, size - 1, :]    # Move ghost cell contributions (top/bottom)
    Bp[:, size - 2, :] += Bp[:, 0, :]
    
    Bp[0, :, :]  = Bp[size - 2, :, :]    # Update ghost cells
    Bp[:, 0, :]  = Bp[:, size - 2, :]
    
    Bp[size - 1, :, :] = Bp[1, :, :]     # Update ghost cells
    Bp[:, size - 1, :] = Bp[:, 1, :]    
        
    print 'Max B-field value: %.2f B0\n' % (np.max(B) / B0)
    return B
    
def electron_temperature(e_flag, n_i):
    if e_flag == 1:         # Adiabatic Push
        gamma = 5./3.
        Te = Te0 * ((n_i / (q*n0)) ** (gamma - 1))  # RECONSIDER THIS
    elif e_flag == 2:       # Eqn of state push
        Te = 0
    else:                   # Const. Temp push
        Te = np.ones((size, size)) * Te0    
    return Te

def push_E(B, V_i, n_i, dt): # Based off big F(B, n, V) eqn on pg. 140 (eqn. 10)
    print 'Updating electric field...'
    E_out = np.zeros((size, size, 3))     # Output array - new electric field
    VxB   = np.zeros((size, size, 3))     # V cross B holder
    del_p = np.zeros((size, size, 3))     # Electron pressure tensor gradient array
    dB    = np.zeros((size, size, 3))     # Curl B (del B)
    BdB   = np.zeros((size, size, 3))     # B cross del cross B holder  
   
    e_flag = ie
    # Adiabatic Electron Temperature Calculation 
    Te = electron_temperature(e_flag, n_i)   
        
    # Calculate average/summations over species
    V_av = np.zeros((size, size, 3), dtype=float)
    qn   = np.zeros((size, size)   , dtype=float)
    
    for ii in range(3):        
        for jj in range(Nj):
            V_av[:, :, ii] += partin[3, jj] * V_i[:, :, jj, ii]  # Average flow velocity, weighted by relative densities sum(nj * Vj)
            
    for jj in range(Nj):
        qn += partout[1, jj] * n_i[:, :, jj]                     # Average charge density, sum(qj * nj)

    # V cross B
    VxB[:, :, 0] =    V_av[:, :, 1] * B[:, :, 2] - V_av[:, :, 2] * B[:, :, 1]
    VxB[:, :, 1] = - (V_av[:, :, 0] * B[:, :, 2] - V_av[:, :, 2] * B[:, :, 0])
    VxB[:, :, 2] =    V_av[:, :, 0] * B[:, :, 1] - V_av[:, :, 1] * B[:, :, 0]
    
    for mm in range(1, size - 1):
        for nn in range(1, size-1):
            # Curl B            
            dB[mm, nn, 0] =    (B[mm, nn + 1, 2] - B[mm, nn - 1, 2]) / (2 * dy)
            dB[mm, nn, 1] = - ((B[mm + 1, nn, 2] - B[mm - 1, nn, 2]) / (2 * dx))
            dB[mm, nn, 2] =   ((B[mm + 1, nn, 1] - B[mm - 1, nn, 1]) / (2 * dx)) - ((B[mm, nn + 1, 0] - B[mm, nn - 1, 0]) / (2 * dx))
            
            # del P
            del_p[mm, nn, 0] = ((qn[mm + 1, nn] - qn[mm - 1, nn]) / (2*dx*q)) * kB * Te[mm, nn]
            del_p[mm, nn, 1] = ((qn[mm, nn + 1] - qn[mm, nn - 1]) / (2*dy*q)) * kB * Te[mm, nn]
            del_p[mm, nn, 2] = 0

    # B cross del cross B:
    BdB[:, :, 0] =    B[:, :, 1] * dB[:, :, 2] - B[:, :, 2] * dB[:, :, 1]
    BdB[:, :, 1] = - (B[:, :, 0] * dB[:, :, 2] - B[:, :, 2] * dB[:, :, 0])
    BdB[:, :, 2] =    B[:, :, 0] * dB[:, :, 1] - B[:, :, 1] * dB[:, :, 0]
    
    # Final Calculation : Cumulative sum with each species
    E_out[:, :, 0] = - VxB[:, :, 0] - (del_p[:, :, 0] / qn) - (BdB[:, :, 0] / (mu0*qn))
    E_out[:, :, 1] = - VxB[:, :, 1] - (del_p[:, :, 1] / qn) - (BdB[:, :, 1] / (mu0*qn))
    E_out[:, :, 2] = - VxB[:, :, 2] - (del_p[:, :, 2] / qn) - (BdB[:, :, 2] / (mu0*qn))
    
    
    E_out[1, :, :]        += E_out[size - 1, :, :]    # Move ghost cell contributions (sides)
    E_out[size - 2, :, :] += E_out[0, :, :]
    
    E_out[:, 1, :]        += E_out[:, size - 1, :]    # Move ghost cell contributions (top/bottom)
    E_out[:, size - 2, :] += E_out[:, 0, :]
    
    # Update ghost cells
    E_out[0, :, :] = E_out[size - 2, :, :]
    E_out[:, 0, :] = E_out[:, size - 2, :]
    
    E_out[size - 1, :, :] = E_out[1, :, :]
    E_out[:, size - 1, :] = E_out[:, 1, :]
    
    for ii in range(3):
        E_out[:, :, ii] = smooth(E_out[:, :, ii])
    
    print 'Max E-field value: %.2f microvolts\n' % np.max(E_out * 1.0e6)       
    return E_out


def collect_density(Nx, Ny, Nidx, W_in): 
    '''Function to collect charge density in each cell in each cell
    at each timestep. These values are weighted by their distance
    from cell nodes on each side. Can send whole array or individual particles?
    How do I sum up the densities one at a time?'''
    print 'Collecting density...'
    n_i = np.zeros((size, size, Nj), float)
    
    # Collect number density of all particles: Create subfunction particle_count ?
    for ii in range(N):

        Ix  = int(  Nx[ii])                                 # x-floor
        Iy  = int(  Ny[ii])                                 # y-floor
        idx = int(Nidx[ii])
        
        for xx in range(2):
            for yy in range(2):                             # Actual density-counting part
                n_i[Ix + xx, Iy + yy, idx] += W_in[xx, yy, ii] * partout[3, idx]       
    
    # Divide by cell dimensions to give densities per cubic metre (assume square grid)
    n_i = n_i / float(dx * dy)

    # Move ghost cell contributions - Ghost cells at 0 and size - 2
    n_i[size - 2, :, :] += n_i[0, :, :]
    n_i[0, :, :]         = n_i[size - 2, :, :]              # Fill ghost cell  
    
    n_i[:, size - 2, :] += n_i[:, 0, :]
    n_i[:, 0, :]         = n_i[:, size - 2, :]              # Fill ghost cell  
    
    n_i[1, :, :]        += n_i[size - 1, :, :]
    n_i[size - 1, :, :]  = n_i[1, :, :]                      # Fill ghost cell
    
    n_i[:, 1, :]        += n_i[:, size - 1, :]
    n_i[:, size - 1, :]  = n_i[:, 1,  :]                     # Fill ghost cell
    
    for ii in range(Nj):
        n_i[:, :, ii] = smooth(n_i[:, :, ii])

    # Make sure no cell is ever zero - add 'ghost' 5% as minimum level
    for ii in range(1, size - 1):
        for jj in range(1, size - 1):
            if np.sum(n_i[ii, jj, :]) == 0:
                n_i[ii, jj, 1] += 0.05 * (partin[3, 1] * n0)      # 5% of initial cold H+ density (hard-coded for now)
                print 'Extra added at node (%d, %d)' % (ii, jj)

    return n_i
  
# Recieves particle array containing all info, plus 'n_i', which is the *simulated* particle density across all cells
def collect_flow(Nx, Ny, Nidx, ni, W_in):
    print 'Collecting flow velocities...'
    # Empty 3-vector for flow velocities at each node
    V_i = np.zeros((size, size, Nj, 3), float)    
    
    # Loop through all particles: sum velocities for each species. Alter for parallelization?
    for ii in range(N):
        Ix  = int(  Nx[ii])             # x-floor
        Iy  = int(  Ny[ii])             # y-floor
        idx = int(Nidx[ii])             # Particle species
        
        for xx in range(2):
            for yy in range(2):
                for vv in range(3):
                    V_i[Ix + xx, Iy + yy, idx, vv] += W_in[xx, yy, ii] * partout[3, idx] * part[3 + vv, ii]      
        
    # Move ghost cell contributions - Ghost cells at 0 and size - 2
    V_i[size - 2, :, :, :] += V_i[0, :, :, :]
    V_i[0, :, :, :]         = V_i[size - 2, :, :, :]              # Fill ghost cell  
    
    V_i[:, size - 2, :, :] += V_i[:, 0, :, :]
    V_i[:, 0, :, :]         = V_i[:, size - 2, :, :]              # Fill ghost cell  
    
    V_i[1, :, :, :]        += V_i[size - 1, :, :, :]
    V_i[size - 1, :, :, :]  = V_i[1, :, :, :]                      # Fill ghost cell
    
    V_i[:, 1, :, :]        += V_i[:, size - 1, :, :]
    V_i[:, size - 1, :, :]  = V_i[:, 1,  :, :]                     # Fill ghost cell
    
    for ii in range(Nj):
        for jj in range(3):
            V_i[:, :, ii, jj] = smooth(V_i[:, :, ii, jj])

    for mm in range(size):
        for nn in range(size):
            for ii in range(3):
                for jj in range(Nj):                        # Divide each dimension by density for averaging (ion flow velocity)
                    if ni[mm, nn, jj] == 0:                 # ni is in m3 - multiply by dx to get entire cell's density (for averaging purposes)                                          
                        V_i[mm, nn, jj, ii] = 0
                    else:
                        V_i[mm, nn, jj, ii] /= (ni[mm, nn, jj] * dx * dy)  
                        
    print 'Max ion flow velocity: %d km/s\n' % np.max(V_i / 1e3)     
    return V_i
    

# Simple quarter-half-quarter smoothing in 2D: Gaussian 3x3
def smooth(fn): 
    
    new_function = np.zeros((size, size))
    
    # Smooth: Assumes nothing in ghost cells
    for ii in range(1, size - 1):
        for jj in range(1, size - 1):
            new_function[ii, jj] = ( (4. / 16.) * (fn[ii, jj])
                                  +  (2. / 16.) * (fn[ii + 1, jj]     + fn[ii - 1, jj]     + fn[ii, jj + 1]     + fn[ii, jj - 1])
                                  +  (1. / 16.) * (fn[ii + 1, jj + 1] + fn[ii + 1, jj - 1] + fn[ii - 1, jj + 1] + fn[ii - 1, jj - 1]) )
        
    # Move Ghost Cell Contributions: Periodic Boundary Condition
    new_function[size - 2, :] += new_function[0, :]
    new_function[0, :]         = new_function[size - 2, :]              # Fill ghost cell  
    
    new_function[:, size - 2] += new_function[:, 0]
    new_function[:, 0]         = new_function[:, size - 2]              # Fill ghost cell  
    
    new_function[1, :]        += new_function[size - 1, :]
    new_function[size - 1, :]  = new_function[1, :]                      # Fill ghost cell
    
    new_function[:, 1]        += new_function[:, size - 1]
    new_function[:, size - 1]  = new_function[:, 1]                     # Fill ghost cell
    
    return new_function
    

# Remove ghost cells on either side of spatial array. Preserves ordering. Doesn't preseve anything in ghost cells.
def remove_ghost(function): 
    return function[1: size - 1]
    

#%% Main Program Script
if __name__ == '__main__':
        
    # Last minute stuff
    plt.ioff()                                                   # Supress figure drawing
    
    #%%        ----- MAIN PROGRAM -----
    for qq in range(maxtime):
        if qq == 0:
            print '\nLoop 0: Initializing----'
            check_integrity(partin, Nj, part_type)
            
            # Initalize fields and collect source terms, etc.            : Use dt = 0
            W           = assign_weighting(part)
            dns         = collect_density(part[6, :], part[7, :], part[8, :], W)                    # Input: node values I, weighting factors W, species index idx      Output: Ion number density, n_i 
            check_particles(dns)
            Vi          = collect_flow(part[6, :], part[7, :], part[8, :], dns, W)                  # Input: particle array, ion number density n_i                     Output: Ion flow (3-vector)
                    
            B[:, :, 0:3] = push_B(B[:, :, 0:3], E[:, :, 0:3], 0)                                    # Input: Current B, Current E, timestep dt                          Output: Updated B-field
            E[:, :, 0:3] = push_E(B[:, :, 0:3], Vi, dns, 0)                                         # Input: Current B, ion flow V_i, ion density n_i, timestep dt      Output: Updated E-field ,  ii = 0
                
            # Retard Velocity : Use dt = - 0.5
            part = velocity_update(part, B[:, :, 0:3], E[:, :, 0:3], -0.5*DT, W)                    # This stops the bad things from happening (e.g. Numerical instabilities)
            print '\n----------------------'
           
        else:
            print '\nLoop %d' % qq
            # N + 1/2
            part      = velocity_update(part, B[:, :, 0:3], E[:, :, 0:3], -0.5*DT, W)               # Advance Velocity to N + 1/2
            part, W   = position_update(part)                                                       # Advance Position to N + 1
            B[:, :, 0:3] = push_B(B[:, :, 0:3], E[:, :, 0:3], DT)                                   # Advance Magnetic Field to N + 1/2
            
            dns       = 0.5 * (dns + collect_density(part[6, :], part[7, :], part[8, :], W))        # Collect ion density at N + 1/2 : Collect N + 1 and average with N                                             
            Vi        = collect_flow(part[6, :], part[7, :], part[8, :], dns, W)                    # Collect ion flow at N + 1/2
            E[:, :, 6:9] = E[:, :, 0:3]                                                             # Store Electric Field at N because PC, yo
            E[:, :, 0:3] = push_E(B[:, :, 0:3], Vi, dns, DT)                                        # Advance Electric Field to N + 1/2   ii = even numbers
            
            
            # ----- Predictor-Corrector Method ----- #
        
            # Predict values of fields at N + 1 
            B[:, :, 3:6] = B[:, :, 0:3]                                                             # Store last "real" magnetic field (N + 1/2)
            E[:, :, 3:6] = E[:, :, 0:3]                                                             # Store last "real" electric field (N + 1/2)
            E[:, :, 0:3] = -E[:, :,6:9] + 2*E[:, :, 0:3]                                            # Predict Electric Field at N + 1
            B[:, :, 0:3] = push_B(B[:, :, 0:3], E[:, :, 0:3], DT)                                   # Predict Magnetic Field at N + 1 (Faraday, based on E(N + 1))
            
            # Extrapolate Source terms and fields at N + 3/2
            old_part = part                                                                         # Back up particle attributes at N + 1  
            dns_old = dns                                                                           # Store last "real" densities (in an E-field position, I know....)
            
            part      = velocity_update(part, B[:, :, 0:3], E[:, :, 0:3], -0.5*DT, W)               # Advance particle velocities to N + 3/2
            part, W   = position_update(part)                                                       # Push particles to positions at N + 2
           
            dns       = 0.5 * (dns + collect_density(part[6, :], part[7, :], part[8, :], W))        # Collect ion density as average of N + 1, N + 2
            Vi        = collect_flow(part[6, :], part[7, :], part[8, :], dns, W)                    # Collect ion flow at N + 3/2
            B[:, :, 0:3] = push_B(B[:, :, 0:3], E[:, :, 0:3], DT)                                   # Push Magnetic Field again to N + 3/2 (Use same E(N + 1)
            E[:, :, 0:3] = push_E(B[:, :, 0:3], Vi, dns, DT)                                        # Push Electric Field to N + 3/2   ii = odd numbers
            
            # Correct Fields
            E[:, :, 0:3] = 0.5 * (E[:, :, 3:6] + E[:, :, 0:3])                                      # Electric Field interpolation
            B[:, :, 0:3] = push_B(B[:, :, 3:6], E[:, :, 0:3], DT)                                   # Push B using new E and old B
            
            # Reset Particle Array to last real value
            part = old_part                                                         
            dns  = dns_old                                                                          # The stored densities at N + 1/2 before the PC method took place (previously held PC at N + 3/2)
       
#%%   ##### --------- END OF ACTUAL PROGRAM PART --------- ######
    
         # ----- Plot commands ----- #
        if generate_plots == 1:
            # Initialize Figure Space
            fig_size = 4, 7
            fig = plt.figure(figsize=(20,10))   
            fig.patch.set_facecolor('k')    
            
            # Set font things
            rcParams.update({'text.color'   : 'w',
                        'axes.labelcolor'   : 'w',
                        'axes.edgecolor'    : 'w',
                        'axes.facecolor'    : 'k',
                        'mathtext.default'  : 'regular',
                        'xtick.color'       : 'w',
                        'ytick.color'       : 'w',
                        'axes.labelsize'    : 16,
                        })
            
            # Set some axis parameters    
            sim_time = qq * 2*pi * DT
            
            x_pos = part[0, 0:N] / 1000                 # Particle x-positions (km) (For looking at particle characteristics)  
            x_cell = np.arange(0, xmax, dx) / 1000      # Cell x-positions (km) (For looking at cell characteristics)
            x_cell_num = np.arange(NX)                  # Numerical cell numbering: x-axis
            
            y_pos = part[1, 0:N] / 1000                 # Particle x-positions (km) (For looking at particle characteristics)  
            y_cell = np.arange(0, ymax, dx) / 1000      # Cell x-positions (km) (For looking at cell characteristics)
            y_cell_num = np.arange(NX)                  # Numerical cell numbering: x-axis
      
        #####       
            # Plot: Normalized vy - Assumes species 0 is hot hydrogen population
            ax_vy = plt.subplot2grid(fig_size, (0,0), projection='3d', rowspan=4, colspan=4)    

            # Normalized to Alfven velocity: For y-axis plot
            norm_yvel = part[4, 0:N] / alfie        # vy (vA / ms-1)

            # Plot 3D Scatterplot: All species
            for ii in range(1):
                ax_vy.scatter(x_pos[idx_start[ii]: idx_end[ii]],        # Plot particles
                              y_pos[idx_start[ii]: idx_end[ii]],
                              norm_yvel[idx_start[ii]: idx_end[ii]],
                              s=psize, lw=0, c=species_colors[ii])
         
            # Make it look pretty
            ax_vy.set_title(r'Normalized velocity $v_y$ vs. Position (x, y)')    
            
            ax_vy.set_xlim(0, xmax/1000)
            ax_vy.set_ylim(0, ymax/1000)
            ax_vy.set_zlim(-15, 15)
            
            ax_vy.set_xlabel('x position (km)', labelpad=10)
            ax_vy.set_ylabel('y position (km)', labelpad=10)
            ax_vy.set_zlabel(r'$\frac{v_y}{v_A}$', fontsize=24, rotation=0, labelpad=8) 
            
            ax_vy.view_init(elev=25., azim=300)
            
            
        #####
            # Plot Density
            dns_norm = dns[1:size-1, 1:size-1, :] / n0

            ax0 = plt.subplot2grid(fig_size, (0, 4), colspan=2, rowspan=2)
            ax0.contourf(x_cell_num, x_cell_num, dns_norm[:, :, 0], 100)
            
            ax1 = plt.subplot2grid(fig_size, (2, 4), colspan=2, rowspan=2)
            ax1.contourf(x_cell_num, x_cell_num, dns_norm[:, :, 1], 100)
            
            ax0.set_title('Densities of Hot / Cold Proton Species')
           
            
        #####
            # Last Minute plot adjustments
            plt.tight_layout(pad=1.0, w_pad=1.8)
            fig.subplots_adjust(hspace=0)    
            
            # Figure Text
            text1  = plt.figtext(0.84, 0.01, 'Simulation Time = %.2f s'     % (sim_time),           fontsize = 16, color='#ffff00')
            text2  = plt.figtext(0.01, 0.01, 'Ion Time: %.2f'               % (gyfreq * qq * DT),   fontsize = 16, color='#ffff00') 
            
            text3  = plt.figtext(0.86, 0.94, 'N  = %d'                      % N,                    fontsize = 18)
            text4  = plt.figtext(0.86, 0.91, r'$n_b$ = %.1f%%'              % (partin[3, 0] * 100), fontsize = 18)
            text5  = plt.figtext(0.86, 0.88, 'NX = %d'                      % NX,                   fontsize = 18)
            text6  = plt.figtext(0.86, 0.85, r'$\Delta t$  = %.4fs'         % DT,                   fontsize = 18)
            
            text7  = plt.figtext(0.86, 0.80, r'$\theta$  = %d$^{\circ}$'    % theta,                fontsize = 18)            
            text8  = plt.figtext(0.86, 0.77, r'$B_0$ = %.1f nT'             % (B0 * 1e9),           fontsize = 18)
            text9  = plt.figtext(0.86, 0.74, r'$n_0$ = %.2f $cm^{-3}$'      % (n0 / 1e6),           fontsize = 18)
            
            text10 = plt.figtext(0.86, 0.69, r'$\beta_{b\perp}$ = %.1f'     % partin[6, 0],         fontsize = 18)
            text11 = plt.figtext(0.86, 0.66, r'$\beta_{b\parallel}$ = %.1f' % partin[5, 0],         fontsize = 18)
            text12 = plt.figtext(0.86, 0.63, r'$\beta_{core}$ = %.1f'       % partin[5, 1],         fontsize = 18)
            
            text13 = plt.figtext(0.86, 0.58, r'$T_e$  = %dK'                % Te0,                  fontsize = 18)
            text14 = plt.figtext(0.86, 0.55, r'$T_{b\perp}$ = %dK'          % Tprp[0],              fontsize = 18)
            text15 = plt.figtext(0.86, 0.52, r'$T_{b\parallel}$ = %dK'      % Tpar[0],              fontsize = 18)
            
#%%
        r = qq / framegrab          # Capture number
       
        # Grab every mod-ii frame (For saving anything)
        if qq%framegrab == 0:            
            
            # Initialize directory
            if qq == 0:
                # Create main test directory
                if generate_plots + generate_data > 0:
                    if os.path.exists('%s:\%s' % (drive, save_path)) == False:
                        os.makedirs('%s:\%s' % (drive, save_path))
                                    
                    # Create and set paths for images and data for this run
                    run_num = len(os.listdir('%s:\%s' % (drive, save_path)))
                    path    = ('%s:\%s\Run %d' % (drive, save_path, run_num))
                    d_path  = ('%s:\%s\Run %d\Data' % (drive, save_path, run_num))
                    
                    if os.path.exists(path) == False:
                        os.makedirs(path)
                    
            if generate_plots == 1:
                # Save Plot
                filename = 'anim%05d.png' % r
                fullpath = os.path.join(path, filename)
                plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
                print 'Plot %d produced. Elapsed time: %.1fs' % (r, timer() - start_time)
                plt.close('all')
            
            # Save Data
            if generate_data == 1:
                if os.path.exists(d_path) == False:
                    os.makedirs(d_path)
                
                if qq == 0:
                    # Save Header File: Important variables for Data Analysis
                    params = dict([('f', f),
                                   ('V', V),
                                   ('Nj', Nj),
                                   ('ts', ts),
                                   ('dxm', dxm),
                                   ('NX', NX),
                                   ('maxtime', maxtime), 
                                   ('cellpart', cellpart),
                                   ('n0', n0),
                                   ('B0', B0),
                                   ('E0', E0),
                                   ('Te0', Te0),
                                   ('ie', ie),
                                   ('theta', theta),
                                   ('seed', seed),
                                   ('framegrab', framegrab)])
                    h_name = os.path.join(d_path, 'Header.pckl')                                # Data file containing variables used in run
                    
                    with open(h_name, 'wb') as f:
                        pickle.dump(params, f)
                        f.close() 
                        print 'Header file saved'
                    
                    p_file = os.path.join(d_path, 'p_data')
                    np.savez(p_file, partin=partin, partout=partout, part_type=part_type)       # Data file containing particle information
                    print 'Particle data saved'

                d_filename = 'data%05d' % r
                d_fullpath = os.path.join(d_path, d_filename)
                np.savez(d_fullpath, part=part, Vi=Vi, dns=dns, E = E[:, :, 0:3], B = B[:, :, 0:3])   # Data file for each iteration
                print 'Data %d saved. Elapsed time: %.1fs' % (r, timer() - start_time)
    
    #%%        ----- PRINT RUNTIME -----
    elapsed = timer() - start_time
    print "Time to execute program: {0:.2f} seconds".format(round(elapsed,2))
