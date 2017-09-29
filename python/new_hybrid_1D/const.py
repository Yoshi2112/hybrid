# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:00:58 2017

@author: iarey
"""
import numpy as np

## RUN PARAMETERS ##
drive          = 'D:/'                         # Drive letter or path for portable HDD e.g. 'E:/'
save_path      = 'runs/func_test'              # Save path on 'drive' HDD - each run then saved in numerically sequential subfolder with images and associated data
run_num        = 0                             # Default value. Will be modified at runtime
generate_data  = 0                             # Save data flag
generate_plots = 0                             # Save plot flag
run_desc = '''None'''                          # Run description as string


## PHYSICAL CONSTANTS ##
q   = 1.602e-19                             # Elementary charge (C)
c   = 3e8                                   # Speed of light (m/s)
mp  = 1.67e-27                              # Mass of proton (kg)
me  = 9.11e-31                              # Mass of electron (kg)
mu0 = (4e-7) * np.pi                        # Magnetic Permeability of Free Space (SI units)
kB  = 1.38065e-23                           # Boltzmann's Constant (J/K)
e0  = 8.854e-12                             # Epsilon naught - permittivity of free space
RE  = 6371000.                              # Earth radius in metres


## SIMULATION CONSTANTS ##
dxm      = 2                                # Number of c/wpi per dx
t_res    = 0                                # Time resolution of data in seconds (default 1s). Determines how often data is captured. Every frame captured if '0'.
NX       = 1024                             # Number of cells - dimension of array (not including ghost cells)
max_sec  = 1800                             # Number of (real) seconds to run program for   
cellpart = 1000                             # Number of Particles per cell (make it an even number for 50/50 hot/cold)
ie       = 0                                # Adiabatic electrons. 0: off (constant), 1: on.    
B0       = 4e-9                             # Unform initial magnetic field value (in T) (must be parallel to an axis)
theta    = 0                                # Angle of B0 to x axis (in xy plane in units of degrees)

ne       = 8.48e6                           # Electron density (used to assign portions of ion)
Te0      = 0.5 * 11603.                     # Electron temperature (eV to K)
k        = 1                                # Sinusoidal Density Parameter - number of wavelengths in spatial domain
mhd_equil= 1                                # Temperature varied to give MHD Equilibrium condition?


## DERIVED PLASMA PARAMETERS ## 
wpi = np.sqrt(ne * q ** 2 / (mp * e0))      # Proton   Plasma Frequency, wpi (rad/s)
wpe = np.sqrt(ne * q ** 2 / (me * e0))      # Electron Plasma Frequency, wpe (rad/s)
gyfreq   = q*B0/mp        # Proton Gyrofrequency (rad/s) (since this will be the highest of all species)
va  = B0 / np.sqrt(mu0*ne*mp)               # Alfven speed: Assuming proton plasma

dx   = dxm * c / wpi                        # Spatial cadence, based on characteristic frequency of plasma 
xmin = 0                                    # Minimum simulation dimension
xmax = NX * dx                              # Maximum simulation dimension

## DERIVED SIMULATION CONSTANTS ##
N        = cellpart*NX                                # Number of Particles to simulate: # cells x # particles per cell, excluding ghost cells
Bc       = np.zeros(3)                                # Constant components of magnetic field based on theta and B0
Bc[0]    = B0 * np.sin((90 - theta) * np.pi / 180 )   # Constant x-component of magnetic field (theta in degrees)
Bc[1]    = B0 * np.cos((90 - theta) * np.pi / 180 )   # Constant Y-component of magnetic field (theta in degrees)  
Bc[2]    = 0                                          # Assume Bzc = 0, orthogonal to field line direction


