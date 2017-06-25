# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:56:34 2016

@author: c3134027
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from numpy import pi
import pickle
import itertools as it
from mpl_toolkits.mplot3d import Axes3D

#%% Just to stop those annoying warnings
part, Vi, dns, B, E, partin, Nj, DT, NX, NY, dx, dy, xmax, ymax, k, ne, size, cellpart, B0, Te0, ie, seed, theta, framegrab, ts_history, run_desc = [None] * 26

#%% ---- RUN PARAMETERS ----
drive_letter = '/media/yoshi/VERBATIM HD/'   # Master directory (or drive) on which run is stored
sub_path = 'two_d_draft'                     # Location of 'Run' folder, with series name + optional subfolders
run = 1                                      # Number of run within series
plot_skip = 1                                # Skip every x files from input directory (Default: 1)
plot_style = 99

#%% ---- CONSTANTS ----
q   = 1.602e-19               # Elementary charge (C)
c   = 3e8                     # Speed of light (m/s)
me  = 9.11e-31                # Mass of electron (kg)
mp  = 1.67e-27                # Mass of proton (kg)
e   = -q                      # Electron charge (C)
mu0 = (4e-7) * pi             # Magnetic Permeability of Free Space (SI units)
kB  = 1.38065e-23             # Boltzmann's Constant (J/K)
e0  = 8.854e-12               # Epsilon naught - permittivity of free space
framegrab = 1                 # Just in case it's not there


#%% ---- SET DIRECTORY PARAMETERS ----
load_dir = '%s/runs/%s/run_%d/data' % (drive_letter, sub_path, run)                         # Load run data from this dir    INPUT
save_dir = '%s/runs/%s/run_%d/analysis/plot_%d' % (drive_letter, sub_path, run, plot_style) # Save analysis data here        OUTPUT
data_dir = '%s/runs/%s/run_%d/data/temp' % (drive_letter, sub_path, run)                    # Save Matrices here             OUTPUT

if os.path.exists(save_dir) == False:                # Make Output Folder if it doesn't exist
    os.makedirs(save_dir)

if os.path.exists(data_dir) == False:                # Make Output Folder if it doesn't exist
    os.makedirs(data_dir)  

#%% ---- LOAD HEADER ----
h_name = os.path.join(load_dir, 'header.pckl')       # Load Header File
f = open(h_name)
obj = pickle.load(f)
f.close()

for name in obj.keys():                              # Assign Header variables to namespace (dicts)
    globals()[name] = obj[name]

np.random.seed(seed)
num_files = len(os.listdir(load_dir)) - 2            # Minus two for header and particle files     
#num_files = 700                                     # Manually change this to read custom length data sets

print 'Header file loaded.'

#%% ---- LOAD PARTICLE PARAMETERS ----
p_path = os.path.join(load_dir, 'p_data.npz')     # File location
p_data = np.load(p_path)                          # Load file

array_names = p_data.files                        # Create list of stored variable names
num_index   = np.arange(len(array_names))         # Number of stored variables

# Load E, B, SRC and part arrays
for var_name, index_position in zip(array_names, num_index):                # Magic subroutine to attach variable names to their stored arrays
    globals()[var_name] = p_data[array_names[index_position]]               # Generally contains 'partin', 'partout' and 'part_type' arrays from model run

print 'Particle parameters loaded'

''' 
partin:      (0) Mass   (proton units)
             (1) Charge (elementary units)
             (2) Bulk Velocity (m/s)
             (3) Real density (portion of ne)
             (4) Simulated density
             (5) Distribution Type
             (6) Parallel Temperature (eV, x)
             (7) Perpendicular Temperature (eV, y, z)
             (8) Hot (0) or cold (1) species

part_type:   String identifiers (labels) for the species
'''

#%% ---- DERIVED VALUES ----

## Derived Values ##
av_rho   = np.sum([partin[0, nn] * mp * partin[3, nn] * ne for nn in range(Nj)])    # Average mass density
alfie    = B0/np.sqrt(mu0 * av_rho)             # Alfven Velocity (m/s): Constant - Initialized at t = 0
gyfreq   = q*B0/mp                              # Proton Gyrofrequency (rad/s) (since this will be the highest of all species)
gyperiod = 2*pi / gyfreq                        # Gyroperiod in seconds
wpi      = np.sqrt((ne * (q**2)) / (mp * e0 ))  # Plasma Frequency (rad/s)

## Particle Values ##
psize         = 1                                                  # Size of particles on plots
N             = cellpart*NX*NY                                     # Number of Particles to simulate: # cells x # particles per cell, excluding ghost cells
sample_percent= 0.001                                              # Number of sample particles in each population
N_species     = np.round(N * partin[4, :]).astype(int)             # Number of sim particles for each species, total
N_cell        = [float(N_species[ii]) / (NX*NY) for ii in range(Nj)]    # Number of sim particles per cell, per species
N_real        = (dx * 1. * 1.) * ne * NX * NY                           # Total number of real particles (rect prism with sides dx x 1 x 1 metres)

Tpar = partin[6, :] * 11603.         # Parallel Temperatures (K)
Tper = partin[7, :] * 11603.         # Perpendicular Temperatures (K)

idx_start = [np.sum(N_species[0:ii]    )     for ii in range(0, Nj)]                     # Start index values for each species in order
idx_end   = [np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)]  
    
plt.ioff()

#%% ---- PLOT AXES ---- 

# Spatial Arrays
x_cell = np.arange(0, xmax, dx) / 1000      # Cell x-positions (km) (For looking at cell characteristics)
x_cell_num = np.arange(NX)                  # Numerical cell numbering: x-axis    

# Time arrays (different flavours)
sim_time = np.array([DT * tt for tt in range(num_files)]) * framegrab   # "Real" time - Will have to modify this for fractional DT
inc_time = np.arange(num_files) * framegrab                             # Timesteps

# K-space specs : Operate in metres, but have scaling factors for comparison
k_max = 1. / dx
k_nyq = 0.5 * k_max
dk = 1./((NX) * dx)       
k_array = [ii * dk * 2*pi * 1e6 for ii in range(NX)]

# Frequency Space specs 
nDT = DT * framegrab                # The real dt - accounting for skipped frames        
max_f = 1. / nDT
f_nyq = 0.5 * max_f
df = 1 / (num_files * nDT)
f = np.asarray([ii * df * 1000 for ii in range(num_files / 2)])   # frequency in mHz         
            
#%% ---- LOAD FILES AND PLOT -----
# Read files and assign variables. Do the other things as well
for ii in np.arange(1):
    if ii%plot_skip == 0:
        d_file = 'data%05d.npz' % ii                    # Define target file
        input_path = os.path.join(load_dir, d_file)     # File location
        data = np.load(input_path)                      # Load file
        
        array_names = data.files                         # Create list of stored variable names
        num_index = np.arange(len(array_names))          # Number of stored variables
        
        # Load E, B, SRC and part arrays
        for var_name, index_position in zip(array_names, num_index):                # Magic subroutine to attach variable names to their stored arrays
            globals()[var_name] = data[array_names[index_position]]                 # Manual would be something like part = data['part']
        
        print 'Data file %d loaded' % ii
        
        if plot_style != 0:
            fig_size = 4, 7
            fig = plt.figure(figsize=(20,10))    
                
        #rcParams['mathtext.default'] = 'it'
        rcParams.update({'text.color'   : 'w',
                    'axes.labelcolor'   : 'w',
                    'axes.edgecolor'    : 'w',
                    'axes.facecolor'    : 'k',
                    'mathtext.default'  : 'regular',
                    'xtick.color'       : 'w',
                    'ytick.color'       : 'w',
                    'figure.facecolor'  : 'k',
                    'axes.labelsize'    : 24,
                    })
                    
        species_colors = ['cyan', 'red']
                    
        # Set positional arrays (different flavours) 
        x_pos = part[0, :] / 1000                 # Particle x-positions (km) (For looking at particle characteristics) 
        y_pos = part[1, :] / 1000

#%%     Plot Style 0
        if plot_style == 0: # Use this to collect time-varying data (points, etc.)

            # Collate all By information           
            if os.path.isfile(os.path.join(data_dir, 'By_array.npz')) == False:
                By_all[ii, :] = B[1:size-1, 1] / B0
                print 'File %d of %d read' % ((ii + 1), num_files)
            else:
                ii = num_files - 2
                print 'Loading saved array.... Please Wait....'
                By_all = np.load(os.path.join(data_dir, 'By_array.npz'))['By_array']
                print 'Saved array loaded'
            
#%%         ---- FINAL PLOTS ----
            
            if ii == (num_files - 2): # For when all values are read - generate plot
                
            # Misc. Plot stuff - Formatting
                plt.tight_layout(pad=1, w_pad=0.8)
                fig.subplots_adjust(hspace=0)

                # Save (single) Plot
                filename = 'omega on k scaled.png' #% len(os.listdir(save_dir))
                fullpath = os.path.join(save_dir, filename)    
                plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')#, bbox_inches='tight')
                plt.close()
                print 'Plot saved'
                
                # Save Data Array
                d_filename = 'By_array.npz'
                d_fullpath = os.path.join(data_dir, d_filename)
                if os.path.isfile(d_fullpath) == False:
                    np.savez(d_fullpath, By_array=By_all)
                    print 'Array saved'
                break
      
#%%         ---- SAVE OUTPUT -----
       
        #if plot_style != 0:

            # Figure Text
            #text1  = plt.figtext(0.84, 0.01, 'Simulation Time = %.2f s'     % (sim_time[ii]),       fontsize = 16, color='#ffff00')
            #text2  = plt.figtext(0.01, 0.01, 'Ion Time: %.2f'               % (gyfreq * ii * DT),   fontsize = 16, color='#ffff00') 
            
            #text3  = plt.figtext(0.86, 0.94, 'N  = %d'                      % N,                    fontsize = 18)
            #text4  = plt.figtext(0.86, 0.91, r'$n_b$ = %.1f%%'              % (partin[3, 0] * 100), fontsize = 18)
            #text5  = plt.figtext(0.86, 0.88, 'NX = %d'                      % NX,                   fontsize = 18)
            #text6  = plt.figtext(0.86, 0.85, r'$\Delta t$  = %.4fs'         % DT,                   fontsize = 18)
            
            #text7  = plt.figtext(0.86, 0.80, r'$\theta$  = %d$^{\circ}$'    % theta,                fontsize = 18)            
            #text8  = plt.figtext(0.86, 0.77, r'$B_0$ = %.1f nT'             % (B0 * 1e9),           fontsize = 18)
            #text9  = plt.figtext(0.86, 0.74, r'$n_0$ = %.2f $cm^{-3}$'      % (n0 / 1e6),           fontsize = 18)
            
            #text10 = plt.figtext(0.86, 0.69, r'$\beta_{b\perp}$ = %.1f'     % partin[6, 0],         fontsize = 18)
            #text11 = plt.figtext(0.86, 0.66, r'$\beta_{b\parallel}$ = %.1f' % partin[5, 0],         fontsize = 18)
            #text12 = plt.figtext(0.86, 0.63, r'$\beta_{core}$ = %.1f'       % partin[5, 1],         fontsize = 18)
            
            #text13 = plt.figtext(0.86, 0.58, r'$T_e$  = %dK'                % Te0,                  fontsize = 18)
            #text14 = plt.figtext(0.86, 0.55, r'$T_{b\perp}$ = %dK'          % Tprp[0],              fontsize = 18)
            #text15 = plt.figtext(0.86, 0.52, r'$T_{b\parallel}$ = %dK'      % Tpar[0],              fontsize = 18)
            
            # Misc. Plot stuff
            #plt.tight_layout(pad=2, w_pad=0.8)
            #fig.subplots_adjust(hspace=0)
            
            # Save Plot
            #filename = 'anim%05d.png' % ii
            #fullpath = os.path.join(save_dir, filename)    
            #plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')#, bbox_inches='tight')
            #plt.close()
        
            #print 'Plot %d of %d created' % (ii + 1, num_files)
