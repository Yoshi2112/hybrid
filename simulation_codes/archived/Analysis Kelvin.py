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
from mpl_toolkits.mplot3d import Axes3D

#%% ---- USER INPUT ----
# Plot Style: Pre-made ones using 1,2,3... etc. as options in an if statement. Produces plots varying with time
# 0: Collect time-variant info to generate one plot
# 1: big vy (both species) - den, Ez, By, |B| on side
# 2: big vx (both species) - den, phi, By, |B| on side
# 3: big vy and vx (both species)
# 4: 3D Velocity phase space plot (vx-vy-vz) 
# 5: 3D Magnetic Field
# 
# 3D magnetic and electric fields
# 0: Stuff that requires all times at once (for single plot stuff) XXXXXX
#       A: Spatial FFT with time (k vs. t ,contour plot)
#       B: Dispersion plot (w vs. k, contour plot)
#       C: Waterfall Plot
#       D: Energy Histories
#       E: 4-velocity thing
#       F: 4-field thing
#       G: 4-density thing
#       H: 3D Phase Plots
#       I: 4-Polarization


#%% Just to stop those annoying warnings
part, B, E, dns, partin, partout, B0, n0, dxm, size, cellpart, theta, Te0, seed, part_type, DT = [None] * 16

#%% ---- RUN PARAMETERS ----
drive_letter = 'C'            # Drive on which run is stored
sub_path = 'Multi Species'  # Location of 'Run' folder, with series name + optional subfolders
run = 1                       # Number of run within series
plot_style = 0                # Type of plot to produce as described above
subplot_style = 'A'           # Type of final plot to produce if option '0' is selected
point = 60                    # Location at which the time-series is sampled (cell number)
plot_skip = 1                 # Skip every x files from input directory (Default: 1)
side_info = 1                 # Plot side info or not (For final plots, I may not want them)

field = 0                     # Field used for k, w, or any other field study (0: Magnetic, 1: Electric)
component = 1                 # Component of field under study (x: 0, y: 1, z: 2)
start_file = 0                # First file to scan - For starting mid-way    (Default 0: Read all files in directory)
end_file   = 0                # Last file to scan  - To read in partial sets (Default 0: Read all files in directory)


#%% ---- CONSTANTS ----
q = 1.602e-19                 # Elementary charge (C)
c = 3e8                       # Speed of light (m/s)
me = 9.11e-31                 # Mass of electron (kg)
mp = 1.67e-27                 # Mass of proton (kg)
e = -q                        # Electron charge (C)
mu0 = (4e-7) * pi             # Magnetic Permeability of Free Space (SI units)
kB = 1.38065e-23              # Boltzmann's Constant (J/K)
e0 = 8.854e-12                # Epsilon naught - permittivity of free space
framegrab = 1                 # Just in case it's not there

grab = [0, 10, 20, 30]
grab_count = 0


#%% ---- LOAD FILE PARAMETERS ----

load_dir = '%s:\Runs\%s\Run %d\Data'          % (drive_letter, sub_path, run)                 # Load run data from this dir    INPUT
save_dir = '%s:\Runs\%s\Run %d\Plot %d'       % (drive_letter, sub_path, run, plot_style)     # Save analysis data here        OUTPUT
data_dir = '%s:\Runs\%s\Run %d\Data Analysis' % (drive_letter, sub_path, run)                 # Save Matrices here             OUTPUT

if os.path.exists(save_dir) == False:                                       # Make Output Folder if it doesn't exist
    os.makedirs(save_dir)

if os.path.exists(data_dir) == False:                                       # Make Output Folder if it doesn't exist
    os.makedirs(data_dir)      

#%% ---- LOAD HEADER ----

h_name = os.path.join(load_dir, 'Header.pckl')                              # Load header file
f = open(h_name)                                                            # Open header file 
obj = pickle.load(f)                                                        # Load variables from header file into python object
f.close()                                                                   # Close header file
        
for name in list(obj.keys()):                                                     # Assign Header variables to namespace (dicts)
    globals()[name] = obj[name]                                             # Magic variable creation function

np.random.seed(seed)    
                    

if end_file == 0:
    num_files = len(os.listdir(load_dir)) - 2
else:
    num_files = end_file                            

print('Header file loaded.')

#%% ---- LOAD PARTICLE PARAMETERS ----

p_path = os.path.join(load_dir, 'p_data.npz')                               # File location
p_data = np.load(p_path)                                                    # Load file

array_names = p_data.files                                                  # Create list of stored variable names
num_index = np.arange(len(array_names))                                     # Number of stored variables

# Load E, B, SRC and part arrays
for var_name, index_position in zip(array_names, num_index):                # Magic subroutine to attach variable names to their stored arrays
    globals()[var_name] = p_data[array_names[index_position]]               # Generally contains 'partin', 'partout' and 'part_type' arrays from model run

print('Particle parameters loaded')

''' 
partin:      (0) Mass (proton units)
             (1) Charge (multiples of e)
             (2) Bulk Velocity (multiples of vA)
             (3) Relative (real) density
             (4) Simulated density
             (5) Density type (Uniform / Sinusoidal, etc.)
             (6) Temp Parallel (eV)
             (7) Temp Perpendicular (eV)

partout:     (0) Actual species mass (in kg)
             (1) Actual species charge (in C)
             (2) Actual species bulk velocity (in m/s)
             (3) Density contribution of each superparticle (particles per sim particle)
'''

#%% ---- DERIVED VALUES ----

## Derived Values ##
Nj       = int(np.shape(partin)[1])   
av_rho   = np.sum([partin[0, nn] * mp * partin[3, nn] * n0 for nn in range(Nj)])    # Average mass density
alfie    = B0/np.sqrt(mu0 * av_rho)                                         # Alfven Velocity (m/s): Constant - Initialized at t = 0
gyfreq   = q*B0/mp                                                          # Proton Gyrofrequency (rad/s) (since this will be the highest of all species)
gyperiod = 2*pi / gyfreq                                                    # Gyroperiod in seconds
wpi      = np.sqrt((n0 * (q**2)) / (mp * e0 ))                              # Plasma Frequency (rad/s)

NX       = size - 2                                                         # Number of spatial cells (in simulation domain - excludes ghost cells)
dx       = dxm * c / wpi                                                    # Spacial step as function of plasma frequency (in metres)
xmax     = NX * dx                                                          # Spatial size of simulation

## Particle Values ##
psize         = 1                                                           # Size of particles on plots
N             = cellpart*(size - 2)                                         # Number of Particles to simulate: # cells x # particles per cell, excluding ghost cells
sample_number = 20                                                          # Number of sample particles in each population
N_species     = np.round(N * partin[4, :]).astype(int)                      # Number of sim particles for each species, total
N_cell        = [float(N_species[ii]) / NX for ii in range(Nj)]             # Number of sim particles per cell, per species
N_real        = (dx * 1. * 1.) * n0 * NX                                    # Total number of real particles (rect prism with sides dx x 1 x 1 metres)

# Output Particle Values
partout = np.array([partin[0, :] * mp,                                      # (0) Actual Species Mass    (in kg)
                    partin[1, :] * q,                                       # (1) Actual Species Charge  (in coloumbs)
                    partin[2, :] * alfie,                                   # (2) Actual Species streaming velocity
                    (N_real * partin[3, :]) / N_species])                   # (3) Density contribution of each particle of species (real particles per sim particle)
                    
N_real_cell = [N_cell[ii] * partout[3, ii] for ii in range(Nj)] 
n_rel       = partin[3, :] / n0

Tpar = partin[6, :] * 11600                                                 # Parallel Temperatures
Tprp = partin[7, :] * 11600                                                 # Perpendicular Temperatures

idx_start = [np.sum(N_species[0:ii]    )     for ii in range(0, Nj)]        # Start index values for each species in order
idx_end   = [np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)]  
    
plt.ioff()

#%% ---- PLOT AXES ---- 

# Spatial Arrays
x_cell = np.arange(0, xmax, dx) / 1000                                      # Cell x-positions (km) (For looking at cell characteristics)
x_cell_num = np.arange(NX)                                                  # Numerical cell numbering: x-axis    
x_cwpi = x_cell_num * dxm                                                   # Spatial dimensions in units of c/wpi

# Time arrays (different flavours)
inc_time = np.arange(num_files) * framegrab                                        # Timesteps
sim_time = inc_time * DT                                                           # Real time
ion_time = inc_time * (2*pi / gyperiod)                                            # Time in radians (as per Winske)

# K-space specs : Operate in metres, but have scaling factors for comparison
k_max   = 1. / dx
k_nyq   = 0.5 * k_max
dk      = 1./(NX * dx)       
k_array = [ii * dk * 2*pi * 1e6 for ii in range(NX)]

# Frequency Space specs 
nDT   = DT * framegrab                                                # Actual capture rate of data (# Iterations x Simulation Timestep)
max_f = 1. / nDT
f_nyq = 0.5 * max_f
df    = 1 / (num_files * nDT)
f     = np.asarray([ii * df * 1000 for ii in range(num_files / 2)])   # frequency in mHz         

# Create random index arrays for each species
samples = np.zeros((Nj, sample_number))                                                         # Store addresses of 20 sample particles per population

for ii in range(Nj):
    samples[ii, :] = np.random.choice(np.arange(idx_start[ii], idx_end[ii]), sample_number)


#%% ---- LOAD FILES AND PLOT -----
# Read files and assign variables. Do the other things as well
ii = start_file

#for ii in grab:
while ii < num_files:
    if ii%plot_skip == 0:
        d_file = 'data%05d.npz' % ii                    # Define target file
        input_path = os.path.join(load_dir, d_file)     # File location
        data = np.load(input_path)                      # Load file
        
        array_names = data.files                         # Create list of stored variable names
        num_index = np.arange(len(array_names))          # Number of stored variables
        
        # Load E, B, SRC and part arrays
        for var_name, index_position in zip(array_names, num_index):                # Magic subroutine to attach variable names to their stored arrays
            globals()[var_name] = data[array_names[index_position]]                 # Manual would be something like part = data['part']
  
#%% ---- FONT ----
        # Set font things
        #rcParams['mathtext.default'] = 'it'
        rcParams.update({'text.color'   : 'k',
                    'axes.labelcolor'   : 'k',
                    'axes.edgecolor'    : 'k',
                    'axes.facecolor'    : 'w',
                    'mathtext.default'  : 'regular',
                    'xtick.color'       : 'k',
                    'ytick.color'       : 'k',
                    'figure.facecolor'  : 'w',
                    'axes.labelsize'    : 24,
                    })
                    
        pad = 13
                    
        species_colors = ['red', 'red', 'red', 'orange', 'yellow', 'blue', 'magenta']
                    
        # Load Particle Positions
        x_pos = part[0, 0:N] / 1000                 # Particle x-positions (km)  
        
        if plot_style != 0:
            
        # Initialize Figure Space
            if side_info == 1:
                fig_size = 4, 7
            else:
                fig_size = 4, 6
                
            fig = plt.figure(figsize=(20, 10))  
            
#%%     Plot Style 1
        if plot_style == 1:
            
            # ---- Side Bits (den, phi, By, |B|) ----#
            
        # Density
            dns_norm = np.zeros((NX, Nj), dtype=float)            
            for jj in range(Nj):
                dns_norm[:, jj] = dns[1: size-1, jj] / (N_real_cell[jj] / dx)
                
            den_pos  = 0, 3  
            ax_den  = plt.subplot2grid(fig_size, den_pos,  colspan = 3)

            for jj in range(Nj):
                ax_den.plot(x_cell_num, dns_norm[:, jj], color='green')
                
            ax_den.set_ylabel('Normalized Density', fontsize=16)  
            ax_den.set_ylim(0, 2)
            ax_den.set_title('Various Spatial Parameters vs. x (cell cdts) incl. Ghost Cells') 
            
        # Electric Field: Check this!
            Ez = E[1:size-1, 2]
            
            Ez_pos   = 1, 3
            Ez_ax = plt.subplot2grid(fig_size, Ez_pos, colspan=3, sharex=ax_den)
            Ez_ax.plot(x_cell_num, Ez, color='#800080')
            Ez_ax.set_ylim(-10e-3, 10e-3)
            Ez_ax.set_yticks(np.arange(-10e-3, 10.01e-3, 2e-3))
            Ez_ax.set_yticklabels(np.arange(-0.150, 0.201, 0.50)) 
            Ez_ax.set_ylabel(r'$E_z$ ($\mu$V$m^{-1}$)', labelpad=25, rotation=0, fontsize=14)
            
        # Magnetic Field: y-component
            By = B[1:size-1, 1] / B0            
            
            By_pos   = 2, 3
            By_ax   = plt.subplot2grid(fig_size, By_pos,   colspan = 3, sharex=ax_den)
            By_ax.plot(x_cell_num, By, color='#33cc33')
            By_ax.set_ylabel(r'$\frac{B_y}{B_0}$', rotation=0, labelpad=15)
            By_ax.set_ylim(-1, 1)

        # Magnetic Field: Magnitude
            Bmag = np.sqrt(B[1:size-1, 0] ** 2 + B[1:size-1, 1] ** 2 + B[1:size-1, 2] ** 2) / B0            
            
            Bmag_pos = 3, 3
            Bmag_ax = plt.subplot2grid(fig_size, Bmag_pos, colspan = 3, sharex=ax_den)
            Bmag_ax.plot(x_cell_num, Bmag, color='#33cc33')
            Bmag_ax.set_ylabel(r'$\frac{|B|}{B_0}$', rotation=0, labelpad=20)
            Bmag_ax.set_ylim(0, 2)
            
           
        # All
            for ax in [By_ax, Bmag_ax, ax_den, Ez_ax]:
                ax.set_xlim(0, size)
            
            # Not Bottom
            for ax in [By_ax, Ez_ax, ax_den]:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.set_yticks(ax.get_yticks()[1:])
                
            # Bottom
            for ax in [Bmag_ax]:
                ax.set_xlabel('Cell Number (x)')
           
           
            # ---- Big Bit (vy) ---- #       
           
            vy_pos_hot =  0, 0
            vy_pos_cold = 2, 0
            
            ax_vy_hot  = plt.subplot2grid(fig_size, vy_pos_hot,  rowspan=2, colspan=3)    
            ax_vy_core = plt.subplot2grid(fig_size, vy_pos_cold, rowspan=2, colspan=3)
            
            # Normalized to Alfven velocity: For y-axis plot
            norm_yvel0 = part[4, :] / alfie        # vy (vA / ms-1)
               
            # Plot Scatter
            ax_vy_hot.scatter(x_pos[idx_start[0]: idx_end[0]], norm_yvel0[idx_start[0]: idx_end[0]], s=psize, c=species_colors[0], lw=0)        # Hot population
                      
            for jj in range(1, Nj):
                ax_vy_core.scatter(x_pos[idx_start[jj]: idx_end[jj]], norm_yvel0[idx_start[jj]: idx_end[jj]], s=psize, lw=0, color='g')  
 
            # Make it look pretty
            ax_vy_hot.set_title(r'Normalized velocity $v_y$ vs. Position (x)')    
            
            ax_vy_hot.set_xlim(0, xmax/1000)
            ax_vy_hot.set_ylim(-15, 15)
            ax_vy_hot.set_xlabel('Position (km)', labelpad=10)
            ax_vy_hot.set_ylabel(r'$\frac{v_y}{v_A}$', fontsize=24, rotation=0, labelpad=8) 
        
            ax_vy_core.set_xlim(0, xmax/1000)
            ax_vy_core.set_ylim(-15, 15)
            ax_vy_core.set_xlabel('Position (km)', labelpad=10)
            ax_vy_hot.set_ylabel(r'$\frac{v_y}{v_A}$', fontsize=24, rotation=0, labelpad=8) 
            
            plt.setp(ax_vy_hot.get_xticklabels(), visible=False)
            ax_vy_hot.set_yticks(ax_vy_hot.get_yticks()[1:])  


#%%     Plot Style 2
        if plot_style == 2:
            
            # ---- Side Bits (den, phi, By, |B|) ----#

        # Density
            dns_norm = np.zeros((NX, Nj), dtype=float)            
            for jj in range(Nj):
                dns_norm[:, jj] = dns[1: size-1, jj] / (N_real_cell[jj] / dx)
                
            den_pos  = 0, 3  
            ax_den   = plt.subplot2grid(fig_size, den_pos,  colspan = 3)

            for jj in range(Nj):
                ax_den.plot(x_cell_num, dns_norm[:, jj], color=species_colors[jj])
                
            ax_den.set_ylabel('Normalized Density', fontsize=16)  
            ax_den.set_ylim(0, 3)
            ax_den.set_title('Various Spatial Parameters vs. x (cell cdts) incl. Ghost Cells') 
            
        # Magnetic Field: y-component
            By = B[1:size-1, 1] / B0            
            
            By_pos   = 2, 3
            By_ax   = plt.subplot2grid(fig_size, By_pos,   colspan = 3, sharex=ax_den)
            By_ax.plot(x_cell_num, By, color='#33cc33')
            By_ax.set_ylabel(r'$\frac{B_y}{B_0}$', rotation=0, labelpad=15)
            By_ax.set_ylim(-2, 2)
            
            
        # Magnetic Field: Magnitude
            Bmag = np.sqrt(B[1:size-1, 0] ** 2 + B[1:size-1, 1] ** 2 + B[1:size-1, 2] ** 2) / B0            
            
            Bmag_pos = 3, 3
            Bmag_ax = plt.subplot2grid(fig_size, Bmag_pos, colspan = 3, sharex=ax_den)
            Bmag_ax.plot(x_cell_num, Bmag, color='#33cc33')
            Bmag_ax.set_ylabel(r'$\frac{|B|}{B_0}$', rotation=0, labelpad=20)
            Bmag_ax.set_ylim(0, 4)
            
        # Phi angle
            phi_pos = 1, 3
            phi_ax = plt.subplot2grid(fig_size, phi_pos, colspan=3, sharex=ax_den)
            
            B_limit = 0.01 * B0
            phi = np.zeros(NX)
            
            for jj in np.arange(1, size-1, 1):       
                if (abs(B[jj, 2]) > B_limit) and (abs(B[jj, 1]) > B_limit):
                    phi[jj - 1] = np.arctan2(B[jj, 2], B[jj, 1]) + pi
                else:
                    phi[jj - 1] = pi
                    
            phi_ax.plot(x_cell_num, phi, color='#ff9933')
            phi_ax.set_ylabel(r'$\phi$', rotation=0, labelpad=10)
            phi_ax.set_ylim(0, 2*pi)
            phi_ax.set_yticks(np.arange(0, 2*pi+0.001, 0.5*pi))
            phi_ax.set_yticklabels([r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

            
        ### Set Bulk Plot Things
            
            # All
            for ax in [By_ax, Bmag_ax, ax_den, phi_ax]:
                ax.set_xlim(0, NX)
            
            # Not Bottom
            for ax in [By_ax, phi_ax, ax_den]:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.set_yticks(ax.get_yticks()[1:])
                
            # Bottom
            for ax in [Bmag_ax]:
                ax.set_xlabel('Cell Number (x)')
           
            # ---- Big Bit (vx) ---- #       
           
            # Plot: x-component Velocity vx/vA (1 plot: Hot/Cold)
            vx_pos =  0, 0
            ax_vx = plt.subplot2grid((fig_size), (vx_pos), rowspan=4, colspan=3)    
    
             # Normalized to Alfven velocity: For y-axis plot
            norm_xvel0 = part[3, 0:N] / alfie        # vy (vA / ms-1)
                      
            for jj in range(0, Nj):
                ax_vx.scatter(x_pos[idx_start[jj]: idx_end[jj]], norm_xvel0[idx_start[jj]: idx_end[jj]], s=psize, lw=0, color=species_colors[jj]) 
                
            #    for jj in it.chain(species_sample0, species_sample1)                
                
                
            # Make it look pretty
            ax_vx.set_title(r'Normalized velocity $v_x$ vs. Position (x)')    
            ax_vx.set_xlim(0, xmax/1000)
            ax_vx.set_ylim(-5, 15)
            ax_vx.set_xlabel('Position (km)', labelpad=10)
            ax_vx.set_ylabel(r'$\frac{v_x}{v_A}$', fontsize=24, rotation=0, labelpad=8) 
            
        
#%%     Plot Style 3
        elif plot_style == 3:
            
            # ---- Left Bit (vx) ---- #       
           
            # Plot: x-component Velocity vx/vA (1 plot: Hot/Cold)
            vx_pos =  0, 0
            ax_vx = plt.subplot2grid((fig_size), (vx_pos), rowspan=4, colspan=3)    
    
             # Normalized to Alfven velocity: For y-axis plot
            norm_xvel0 = part[3, 0:N] / alfie        # vy (vA / ms-1)
        
            for jj in range(0, Nj):
                ax_vx.scatter(x_pos[idx_start[jj]: idx_end[jj]], norm_xvel0[idx_start[jj]: idx_end[jj]], s=psize, lw=0, color=species_colors[jj]) 

            # Make it look pretty
            ax_vx.set_title(r'Normalized velocity $v_x$ vs. Position (x)')    
            ax_vx.set_xlim(0, xmax/1000)
            ax_vx.set_ylim(-5, 15)
            ax_vx.set_xlabel('Position (km)', labelpad=10)
            ax_vx.set_ylabel(r'$\frac{v_x}{v_A}$', fontsize=24, rotation=0, labelpad=8) 
            

            # ---- Right Bit (vy) ----#
         
            # Plot: y-component Velocity vy/vA (2 plots: Hot/Cold)
            vy_pos_hot =  0, 3
            vy_pos_cold = 2, 3
            
            ax_vy_hot = plt.subplot2grid((fig_size), (vy_pos_hot), rowspan=2, colspan=3)    
            ax_vy_core = plt.subplot2grid(fig_size, vy_pos_cold, rowspan=2, colspan=3)
            
            # Normalized to Alfven velocity: For y-axis plot
            norm_yvel0 = part[4, 0:N] / alfie        # vy (vA / ms-1)
        
            # Plot Scatter
            ax_vy_hot.scatter(x_pos[idx_start[0]: idx_end[0]], norm_yvel0[idx_start[0]: idx_end[0]], s=psize, c=species_colors[0], lw=0)        # Hot population
                      
            for jj in range(1, Nj):
                ax_vy_core.scatter(x_pos[idx_start[jj]: idx_end[jj]], norm_yvel0[idx_start[jj]: idx_end[jj]], s=psize, lw=0, color=species_colors[jj])  

            # Make it look pretty
            ax_vy_hot.set_title(r'Normalized velocity $v_y$ vs. Position (x)')    
            
            ax_vy_hot.set_xlim(0, xmax/1000)
            ax_vy_hot.set_ylim(-15, 15)
            ax_vy_hot.set_xlabel('Position (km)', labelpad=10)
            ax_vy_hot.set_ylabel(r'$\frac{v_y}{v_A}$', fontsize=24, rotation=0, labelpad=8) 
        
            ax_vy_core.set_xlim(0, xmax/1000)
            ax_vy_core.set_ylim(-15, 15)
            ax_vy_core.set_xlabel('Position (km)', labelpad=10)
            ax_vy_hot.set_ylabel(r'$\frac{v_y}{v_A}$', fontsize=24, rotation=0, labelpad=8) 
            
            plt.setp(ax_vy_hot.get_xticklabels(), visible=False)
            ax_vy_hot.set_yticks(ax_vy_hot.get_yticks()[1:])  

        
#%%     Plot Style 4
        elif plot_style ==  4:
            
            # Phase plot (vx/vy)
            phase_pos = 0,0        
            ax_phase = plt.subplot2grid(fig_size, phase_pos, projection='3d', colspan=3, rowspan=4)
            
            norm_xvel = part[3, :] / alfie        # vx (vA / ms-1)
            norm_yvel = part[4, :] / alfie        # vy (vA / ms-1)
            norm_zvel = part[5, :] / alfie        # vz (vA / ms-1)
            
            for jj in range(Nj):
                ax_phase.scatter(norm_xvel[idx_start[jj]: idx_end[jj]],
                                 norm_yvel[idx_start[jj]: idx_end[jj]],
                                 norm_zvel[idx_start[jj]: idx_end[jj]],
                                 s=psize, lw=0, c=species_colors[jj]) 

          
            ax_phase.set_xlabel(r'$\frac{v_x}{v_A}$')        
            ax_phase.set_ylabel(r'$\frac{v_y}{v_A}$', rotation=0)
            ax_phase.set_zlabel(r'$\frac{v_z}{v_A}$', rotation=90)
            
            ax_phase.set_xlim(-12, 12)
            ax_phase.set_ylim(-12, 12)
            ax_phase.set_zlim(-12, 12)
            
            ax_phase.xaxis.labelpad=pad
            ax_phase.yaxis.labelpad=pad
            ax_phase.zaxis.labelpad=pad
            
            ax_phase.zaxis.label.set_rotation(0)
            ax_phase.set_title(r'Velocity Phase Space (Normalized to $v_A$)')
            
            # Beam velocity distribution plots
            ax_distx = plt.subplot2grid(fig_size, (0, 3), rowspan=2, colspan=3)
            ax_disty = plt.subplot2grid(fig_size, (2, 3), rowspan=2, colspan=3, sharex=ax_distx)

            yx, BinEdgesx = np.histogram(norm_xvel[0:N], bins=1000)
            bcx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
            ax_distx.plot(bcx, yx, '-', c='c')
            ax_distx.set_ylabel('$v_x$', rotation=0, labelpad=15)
            ax_distx.set_title('Particle Velocity Distributions')            
            
            yy, BinEdgesy = np.histogram(norm_yvel[0:N], bins=1000)
            bcy = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
            ax_disty.plot(bcy, yy, '-', c='c')
            ax_disty.set_xlabel('$v_A$')
            ax_disty.set_ylabel('$v_y$', rotation=0, labelpad=15)
            
            for ax in [ax_distx, ax_disty]:
                ax.set_xlim(-10, 15)
                ax.set_ylim(0, 2000)
                
            plt.setp(ax_distx.get_xticklabels(), visible=False)
            ax_distx.set_yticks(ax_distx.get_yticks()[1:])  
        
#%%     Plot Style 5
        elif plot_style == 5:
          
          # Book scaling factors
            By_FFT = np.fft.fft(B[1:size - 1, 1] / B0)             # Neglect ghost cells + Normalize in terms of B0
            By_power = np.log10(abs(By_FFT[0:size/2]) ** 2)        # Take FFT power (Fourier Spectrum) - cutting off ghost cells twice?

            ax_ksp = plt.subplot2grid(fig_size, (0,0), colspan=6, rowspan=4)
            ax_ksp.plot(k_array, By_power, c='c', drawstyle='steps')
            
            ax_ksp.set_xlabel(r'$\kappa$')
            ax_ksp.set_ylabel(r'$B_y$ Power (dB)')
            
            ax_ksp.set_xlim(0, 0.2)
            ax_ksp.set_ylim(-4, 4)

        
#%%     ----- Initialize Empty Arrays -----
        elif plot_style == 0: # Use this to collect time-varying data (points, etc.)
            
            if (subplot_style == 'A' or subplot_style == 'B' or subplot_style == 'C') and ii == 0:
                By_all      = np.zeros((num_files, NX), dtype='float64')
                By_k        = np.zeros((num_files, NX), dtype='complex128')
                By_kf       = np.zeros((num_files, NX), dtype='complex128')
                By_kf_pwr   = np.zeros((num_files, NX), dtype='complex128')   
                
            if subplot_style == 'D' and ii == 0:
                wb = np.zeros(num_files)
                we = np.zeros(num_files)
                
                wx = np.zeros((num_files, Nj))
                wy = np.zeros((num_files, Nj))
                wz = np.zeros((num_files, Nj))
                
            if ((subplot_style == 'E') or (subplot_style == 'F') or (subplot_style == 'G') or (subplot_style == 'H') or (subplot_style == 'I')) and ii == 0:
                fourB     = np.zeros((4, 3, NX))
                fourE     = np.zeros((4, 3, NX))
                fourden   = np.zeros((4, Nj, NX))
                fourpos   = np.zeros((4, N))
                fourvel   = np.zeros((4, 3, N))
                fourB2   = np.zeros((4, 2, NX))


#%%     ----- COLLATE DATA (Plot 0) -----
            if (subplot_style == 'A') or (subplot_style == 'B') or (subplot_style == 'C'):
                # Collate all By (field) information
                if os.path.isfile(os.path.join(data_dir, 'By_array.npz')) == False:
                    By_all[ii, :] = B[1:size-1, 1] / B0                                 ## This bit determines which field!!!
                    print('File %d of %d read' % ((ii + 1), num_files))
                    ii += 1
                else:
                    ii = num_files - 2
                    print('Loading saved array....\n Please Wait....')
                    By_all = np.load(os.path.join(data_dir, 'By_array.npz'))['By_array']
                    print('Saved array loaded')
                    ii = num_files

            if subplot_style == 'D':                    # Collect data if no saved array
                if os.path.isfile(os.path.join(data_dir, 'W_array.npz')) == False:
                    
                    wb[ii] = ((0.5 / mu0) * (np.sum([((B[xx, 0] ** 2) + (B[xx, 1] ** 2) + (B[xx, 2] ** 2)) for xx in range(1, size - 1)]) / NX))     # 'Average' field over NX
                    we[ii] = ((0.5 * e0 ) * (np.sum([((E[xx, 0] ** 2) + (E[xx, 1] ** 2) + (E[xx, 2] ** 2)) for xx in range(1, size - 1)]) / NX))
                    
                    # Calculate average velocities
                    for jj in range(Nj):
                        x_av = np.sum(part[3, idx_start[jj]: idx_end[jj]] ** 2) / N_species[jj]
                        y_av = np.sum(part[4, idx_start[jj]: idx_end[jj]] ** 2) / N_species[jj]
                        z_av = np.sum(part[5, idx_start[jj]: idx_end[jj]] ** 2) / N_species[jj]
    
                        wx[ii, jj] = n_rel[jj] * 0.5 * partin[0, jj] * (x_av)
                        wy[ii, jj] = n_rel[jj] * 0.5 * partin[0, jj] * (y_av)
                        wz[ii, jj] = n_rel[jj] * 0.5 * partin[0, jj] * (z_av)
                    
                    print('File %d of %d read' % ((ii + 1), num_files))
                    W     = np.asarray([wx, wy, wz])
                    F     = np.asarray([wb, we])
                    ii += 1
                  
                else:               # Load array if it exists
                    print('Loading energy arrays....\n Please Wait....')
                    W = np.load(os.path.join(data_dir, 'W_array.npz'))['W_array']
                    F = np.load(os.path.join(data_dir, 'F_array.npz'))['F_array']
                    print('Energy arrays loaded')  
                    ii = num_files
                    
            if (subplot_style == 'E') or (subplot_style == 'F') or (subplot_style == 'G') or (subplot_style == 'H') or (subplot_style == 'I'):
                fourB[grab_count, :, :] = np.transpose(B[1: NX+1, :])
                fourE[grab_count, :, :] = np.transpose(E[1: NX+1, :])
                fourden[grab_count, :, :] = np.transpose(dns[1: NX+1, :])
                
                fourB2[grab_count, :, :] = np.transpose(B[1: NX+1, 1:3])
                
                fourpos[grab_count, :] = x_pos                
                fourvel[grab_count, :, :] = part[3:6, :]                
                grab_count += 1

    
#%%         ---- SAVE OUTPUT -----
       
        if plot_style != 0:

            text1  = plt.figtext(0.84, 0.01, 'Real Time = %.2f s'           % sim_time[ii],         fontsize = 16)#, color='#ffff00')
            text2  = plt.figtext(0.01, 0.01, 'Simulation Time: %.2f rads'              % ion_time[ii],         fontsize = 16)#, color='#ffff00')             
            
            if side_info == 1:
                # Figure Text
                text3  = plt.figtext(0.86, 0.94, 'N  = %d'                      % N,                    fontsize = 18)
                text4  = plt.figtext(0.86, 0.91, r'$n_b$ = %.1f%%'              % (partin[3, 0] * 100), fontsize = 18)
                text5  = plt.figtext(0.86, 0.88, 'NX = %d'                      % NX,                   fontsize = 18)
                text6  = plt.figtext(0.86, 0.85, r'$\Delta t_{sim}$  = %.4fs'   % DT,                   fontsize = 18)
                
                text7  = plt.figtext(0.86, 0.80, r'$\theta$  = %d$^{\circ}$'    % theta,                fontsize = 18)            
                text8  = plt.figtext(0.86, 0.77, r'$B_0$ = %.1f nT'             % (B0 * 1e9),           fontsize = 18)
                text9  = plt.figtext(0.86, 0.74, r'$n_0$ = %.2f $cm^{-3}$'      % (n0 / 1e6),           fontsize = 18)
                
                text10 = plt.figtext(0.86, 0.69, r'$\beta_{b\perp}$ = %.1f'     % partin[7, 0],         fontsize = 18)
                text11 = plt.figtext(0.86, 0.66, r'$\beta_{b\parallel}$ = %.1f' % partin[6, 0],         fontsize = 18)
                text12 = plt.figtext(0.86, 0.63, r'$\beta_{core}$ = %.1f'       % partin[6, 1],         fontsize = 18)
                
                text13 = plt.figtext(0.86, 0.58, r'$T_e$  = %dK'                % Te0,                  fontsize = 18)
                text14 = plt.figtext(0.86, 0.55, r'$T_{b\perp}$ = %dK'          % Tprp[0],              fontsize = 18)
                text15 = plt.figtext(0.86, 0.52, r'$T_{b\parallel}$ = %dK'      % Tpar[0],              fontsize = 18)
            
            # Misc. Plot stuff
            plt.tight_layout(pad=2, w_pad=0.8)
            fig.subplots_adjust(hspace=0)
            
            # Save Plot
            filename = 'anim%05d.png' % ii
            fullpath = os.path.join(save_dir, filename)    
            plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')#, bbox_inches='tight')
            plt.close()
        
            print('Plot %d of %d created' % (ii + 1, num_files))
            ii += 1
            
#%%         ---- DO PLOTS ----

if plot_style == 0:
    # Turn plot space on
    fig_size = 4, 7
    fig = plt.figure(figsize=(20,10))    
    
#%%             ----- SPATIAL FFT vs. TIME PLOT -----
    if subplot_style == 'A':
    
        for kk in range(num_files):
            By_k[kk, :] = np.fft.fft(By_all[kk, :])                     # Spatial FFT at each time
            
        levels = np.linspace(0, 50, 500)
        
        plt.contourf(k_array[1: NX/2], sim_time, (abs(By_k[:, 1: NX/2])), levels=levels, extend='both')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True
        plt.ylabel(r'Time ($\Omega^{-1}$)')
        plt.xlabel(r'Spatial Frequency ($10^{-6}m^{-1})$')
        #plt.xlim(0, 10)
        plt.ylim(0, sim_time[num_files - 1])
        plt.colorbar()
        plt.grid()
    
        plot_name = 'k vs t.png'     
        print('Spatial plot created')
        
#%%             ----- TEMPORAL FFT vs. TIME PLOT (A2) -----
    if subplot_style == 'A2':
    
        for mm in range(num_files):
            By_k[kk, :] = np.fft.fft(By_all[kk, :])                     # Spatial FFT at each time
            
        levels = np.linspace(0, 50, 500)
        
        plt.contourf(k_array[1: NX/2], sim_time, (abs(By_k[:, 1: NX/2])), levels=levels, extend='both')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True
        plt.ylabel(r'Time ($\Omega^{-1}$)')
        plt.xlabel(r'Spatial Frequency ($10^{-6}m^{-1})$')
        #plt.xlim(0, 10)
        plt.ylim(0, sim_time[num_files - 1])
        plt.colorbar()
        plt.grid()
    
        plot_name = 'k vs t.png'     
        print('Spatial plot created')
    
#%%             ----- OMEGA vs. K DISPERSION PLOT -----                
    if subplot_style == 'B':
              
        for kk in range(num_files):
            By_all[kk, :]-= np.mean(By_all[kk, :])
            By_k[kk, :]   = np.fft.fft(By_all[kk, :])                # Spatial FFT at each time
        
        for kk in range(NX):
            By_k[0:num_files, kk] -= np.mean(By_k[0:num_files, kk])
            By_kf[0:num_files, kk] = np.fft.fft(By_k[0:num_files, kk])                 # Temporal FFT at each k
            
        By_wk = By_kf[0: num_files/2, 0: NX/2]                     # Cut original FFT array into 1/4 (aliased in 2D)
        By_kf_pwr = (By_wk * np.conj(By_wk))                       # Take power spectrum of this
        By_masked = np.ma.masked_where(By_kf_pwr < 1e5, By_kf_pwr)
       
        # w/k plot - Power
        f_minus = 0       
       
        #v = np.linspace(9e6, 4e7, 500)
        plt.contourf(k_array[0: NX/2], f[f_minus: len(f)], (By_kf_pwr[f_minus:len(f), 0: NX/2]), 500, cmap='jet', extend='both')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True
        #plt.contourf(k_array[1: NX/2], f[f_minus: len(f)], (By_masked[f_minus:len(f), 1: NX/2]), 500, cmap='jet', extend='both')        
        
        plt.title(r'Dispersion Plot: $\omega/k$', fontsize=22)
        plt.ylabel('Temporal Frequency (mHz)')
        plt.xlabel(r'Spatial Frequency ($10^{-6}m^{-1})$')
        
        # major ticks every 20, minor ticks every 5   
        #minor_yticks = np.arange(0, 26, 2)                                              
        #minor_xticks = np.arange(0, 2.1, 0.1)         
        
        plt.xlim(0, 50)
        plt.ylim(0, 100)                
    
        #plt.xticks(minor_xticks)
        #plt.yticks(minor_yticks)
        # plt.clim(0.0001*high, 0.01*high)                    
        plt.colorbar()
        #plt.grid()
        
        plot_name = 'w vs k.png'
        print('Dispersion plot created')
        
#%%             ----- WATERFALL PLOT -----                
    if subplot_style == 'C':
        sep = 1        # Separation between waterfall plots
        skip = 30
        start = 0
        end = len(By_all)     # Last time to plot                
        ii = 0
        
        for kk in np.arange(start, end, skip):
            plt.plot(x_cell_num, (By_all[kk]*120 + sep*kk), color='b')
        
        plt.ylim(start, end)
        plt.xlim(0, NX)
        
        plot_name = 'Waterfall.png'
    
    
#%%             ----- ENERGY HISTORIES -----
    
    if subplot_style == 'D':

        wb = F[0, :]    ;   we = F[1, :]    ;   wf = wb + we    # Field total energies
        b0 = wb[0]                                              # Initial magnetic field energy
        wb /= b0        ;   we /= b0        ;   wf /= b0        # Normalize to initial magnetic field energy

        # Particle total energies
        w_tot = (W[0, :, :] + W[1, :, :] + W[2, :, :])
        w_par = (W[0, :, :])
        w_per = (W[1, :, :] + W[2, :, :])
        
        # Normalize to initial streaming energy
        wb0 = w_tot[0, 0]           # Initial beaming KE
        
        for xx in range(Nj):
            w_tot[:, xx] /= wb0
            w_par[:, xx] /= wb0
            w_per[:, xx] /= wb0
            
        # Generate Axes
        fig_size = (2, 4)
        
#==============================================================================
#         # Fields

#==============================================================================
        
#==============================================================================
#         # Particles
#         w_ax = plt.subplot2grid(fig_size, (0, 0), colspan=1)
#         
#         for xx in range(Nj):
#             w_ax.plot(ion_time, w_tot[:, xx], label = part_type[xx], color=species_colors[xx])
#             
#         w_ax.plot(ion_time, wf, label="Field Energy", color='g')
#         w_ax.legend()
#         w_ax.set_title('Normalized Field and Particle Energies')
#         w_ax.set_xlabel('Ion gyroperiods')
#         w_ax.set_ylabel('Magnitude')
#==============================================================================
        
        b_ax  = plt.subplot2grid(fig_size, (0,0), colspan=2)
        b_ax.plot(sim_time, w_par[:, 0], color='g', label=r"Beam $\parallel$")
        b_ax.plot(sim_time, w_per[:, 0], color='y', label=r"Beam $\perp$")
        b_ax.legend()

        c_ax  = plt.subplot2grid(fig_size, (1, 0), colspan=2)
        c_ax.plot(sim_time, w_par[:, 1], color='g', label=r"Core $\parallel$")
        c_ax.plot(sim_time, w_per[:, 1], color='y', label=r"Core $\perp$")
        c_ax.set_xlabel('Ion Gyroperiods')
        c_ax.legend()        
        
        bc_ax = plt.subplot2grid(fig_size, (0, 2), colspan=2)
        bc_ax.plot(sim_time, w_tot[:, 0], color='r', label="Beam Energy")
        bc_ax.plot(sim_time, w_tot[:, 1], color='b', label="Core Energy")
        bc_ax.legend()
        
        f_ax  = plt.subplot2grid(fig_size, (1, 2), colspan=2)
        f_ax.plot(sim_time, wf, label="Field Energy")
        f_ax.set_xlabel('Ion Gyroperiods')
        f_ax.legend()
        
        plot_name = 'Four Energies.png'    

#%%                  -----  4 Velocity ------
    if subplot_style == 'E':
        
        fig_size = (2, 8)
        
        times = [sim_time[jj] for jj in grab]
        
        ax0 = plt.subplot2grid((fig_size), (0, 0), colspan=2)
        ax1 = plt.subplot2grid((fig_size), (0, 2), colspan=2, sharey=ax0)
        ax2 = plt.subplot2grid((fig_size), (0, 4), colspan=2, sharey=ax0)
        ax3 = plt.subplot2grid((fig_size), (0, 6), colspan=2, sharey=ax0)
        
        ax4 = plt.subplot2grid((fig_size), (1, 0), colspan=2, sharex=ax0)
        ax5 = plt.subplot2grid((fig_size), (1, 2), colspan=2, sharex=ax1, sharey=ax4)
        ax6 = plt.subplot2grid((fig_size), (1, 4), colspan=2, sharex=ax2, sharey=ax4)
        ax7 = plt.subplot2grid((fig_size), (1, 6), colspan=2, sharex=ax3, sharey=ax4)
             
        # pdb.set_trace()
        for ax, jj in zip([ax0, ax1, ax2, ax3], list(range(4))):

            ax.set_title(r'$\Omega_i t$ = %.1f' % times[jj])
            
            for kk in range(Nj):
                ax.scatter(fourpos[jj, idx_start[kk]: idx_end[kk]], fourvel[jj, 0, idx_start[kk]: idx_end[kk]] / alfie, s=psize, lw=0, c=species_colors[kk])
                ax.set_xlim(0, xmax / 1000)
                ax.set_ylim(-15, 15)
                
                
        for ax, jj in zip([ax4, ax5, ax6, ax7], list(range(4))):
            for kk in range(Nj):
                ax.scatter(fourpos[jj, idx_start[kk]: idx_end[kk]], fourvel[jj, 1, idx_start[kk]: idx_end[kk]] / alfie, s=psize, lw=0, c=species_colors[kk])
                ax.set_xlim(0, xmax / 1000)
                ax.set_ylim(-15, 15)
                
        for ax in [ax1, ax2, ax3, ax5, ax6, ax7]:
            plt.setp(ax.get_yticklabels(), visible=False)
            
        for ax in [ax0, ax1, ax2, ax3]:
            plt.setp(ax.get_xticklabels(), visible=False)
            
        for ax in [ax0]:
            ax.set_yticks(ax.get_yticks()[1:])
            
        for ax in [ax4, ax5, ax6, ax7]:
            ax.set_xticks(ax.get_xticks()[:-1])
            ax.set_xlabel('x (km)')
            
        for ax in [ax0, ax4]:
            ax.set_ylabel(r'$\frac{v_x}{v_A}$', rotation=0)
        
        #fig.suptitle('Normalized x, y velocity evolution')
        fig.subplots_adjust(hspace=0, wspace=0)
        
        plot_name = 'velocities.png'

#%%                  -----  4 Fields ------
    if subplot_style == 'F':
        
        fig_size = (2, 8)
        
        times = [sim_time[jj] for jj in grab]
        
        ax0 = plt.subplot2grid((fig_size), (0, 0), colspan=2)
        ax1 = plt.subplot2grid((fig_size), (0, 2), colspan=2)
        ax2 = plt.subplot2grid((fig_size), (0, 4), colspan=2)
        ax3 = plt.subplot2grid((fig_size), (0, 6), colspan=2)
        
        for ax, jj in zip([ax0, ax1, ax2, ax3], list(range(4))):

            ax.set_title('t = %.1f s' % times[jj])
        
            ax.plot(x_cell_num, fourB[jj, 1, :] / B0, color='green')
            ax.set_xlim(0, NX)
            ax.set_ylim(-1, 1)
                
        for ax in [ax1, ax2, ax3]:
            plt.setp(ax.get_yticklabels(), visible=False)
            
        for ax in [ax0, ax1, ax2, ax3]:
            ax.set_xticks(ax.get_xticks()[:-1])
            ax.set_xlabel('x (cell)')
            
        for ax in [ax0]:
            ax.set_ylabel(r'$\frac{B_y}{B_0}$', rotation=0)
        
        if run == 0:
            fig.suptitle('Run %d: Single Hot Proton Species' % (run + 1), fontsize=20)
        else:
            fig.suptitle('Run %d: Multiple Hot Proton Species' % (run + 1), fontsize=20)
            
        #fig.suptitle('Normalized x, y velocity evolution')
        fig.subplots_adjust(wspace=0)
        
        plot_name = 'By.png'


#%%         ----- 4 Density ----

    if subplot_style == 'G':
        
        fig_size = (2, 8)
        
        times = [sim_time[jj] for jj in grab]
        
        ax0 = plt.subplot2grid((fig_size), (0, 0), colspan=2)
        ax1 = plt.subplot2grid((fig_size), (0, 2), colspan=2)
        ax2 = plt.subplot2grid((fig_size), (0, 4), colspan=2)
        ax3 = plt.subplot2grid((fig_size), (0, 6), colspan=2)
        
        for ax, jj in zip([ax0, ax1, ax2, ax3], list(range(4))):

            ax.set_title(r'$\Omega_i t$ = %.1f' % times[jj])
            
            for kk in range(Nj):
                ax.plot(x_cell_num, fourden[jj, kk, :] / fourden[0, kk, 0], color=species_colors[kk])
                ax.set_xlim(0, NX)
                ax.set_ylim(0, 3)
                
        for ax in [ax1, ax2, ax3]:
            plt.setp(ax.get_yticklabels(), visible=False)
            
        for ax in [ax0, ax1, ax2, ax3]:
            ax.set_xticks(ax.get_xticks()[:-1])
            ax.set_xlabel('x (cell)')
            
        for ax in [ax0]:
            ax.set_ylabel(r'$\frac{n_j}{n_0}$', rotation=0)
        
        #fig.suptitle('Normalized x, y velocity evolution')
        fig.subplots_adjust(wspace=0)
        
        plot_name = 'Densities.png'  
        
        
#%%         ---- 4 Phase ----
    if subplot_style == 'H':
        
        fig_size = (6, 6)
        
        times = [sim_time[jj] for jj in grab]
        
        ax0 = plt.subplot2grid(fig_size, (0, 0), projection='3d', colspan=2, rowspan=3)
        ax1 = plt.subplot2grid(fig_size, (0, 2), projection='3d', colspan=2, rowspan=3)
        ax2 = plt.subplot2grid(fig_size, (3, 0), projection='3d', colspan=2, rowspan=3)
        ax3 = plt.subplot2grid(fig_size, (3, 2), projection='3d', colspan=2, rowspan=3)
        
        fourvel /= alfie  
        fourpos /= 1000
        
        pad = 50           
        
        for ax, jj in zip([ax0, ax1, ax2, ax3], list(range(4))):
            ax.zaxis.set_rotate_label(False)
            ax.set_title(r'$\Omega_i t$ = %.1f' % times[jj])
            
            for kk in range(Nj):
                ax.scatter(fourvel[jj, 0, idx_start[kk]: idx_end[kk]],
                 fourvel[jj, 1, idx_start[kk]: idx_end[kk]],
                 fourvel[jj, 2, idx_start[kk]: idx_end[kk]],
                 s=psize, lw=0, c=species_colors[kk], alpha=0.1) 
                
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.set_zlim(-12, 12)
            
            ax.set_xlabel(r'$v_x$')
            ax.set_ylabel(r'$v_y$')
            ax.set_zlabel(r'$v_z$', rotation=0)
            
            ax.xaxis._axinfo['label']['space_factor'] = 2.1
            ax.yaxis._axinfo['label']['space_factor'] = 2.1
            ax.zaxis._axinfo['label']['space_factor'] = 2.1
           
            ax.azim = -60
            ax.elev = 30            
            
            #ax.zaxis.label.set_rotation(90)
        
        #fig.suptitle('Normalized x, y velocity evolution')
        #fig.subplots_adjust(wspace=0)
        
        plot_name = 'Phase_space.png'       
        
#%%         ----- 4 Polarization ----

    if subplot_style == 'I':
        
        fig_size = (2, 8)
        
        times = [sim_time[jj] for jj in grab]
        
        ax0 = plt.subplot2grid((fig_size), (0, 0), colspan=4)
        ax1 = plt.subplot2grid((fig_size), (0, 4), colspan=4)
        ax2 = plt.subplot2grid((fig_size), (1, 0), colspan=4)
        ax3 = plt.subplot2grid((fig_size), (1, 4), colspan=4)
        
        B_limit = 0.01 * B0             # Power limit
        
        for ax, jj in zip([ax0, ax1, ax2, ax3], list(range(4))):

            ax.set_title(r't = %.1fs' % times[jj], fontsize=18)
            phi = np.zeros(NX)
            
            for kk in range(NX):       
                if (abs(fourB2[jj, 0, kk]) > B_limit) and (abs(fourB2[jj, 1, kk]) > B_limit):
                    phi[kk] = np.arctan2(fourB2[jj, 1, kk], fourB2[jj, 0, kk]) + pi
                else:
                    if jj != 0:
                        phi[kk] = phi[kk - 1]
                    else:
                        phi[kk] = pi
                    
            ax.plot(x_cell, phi, color='#ff9933')

        for ax in [ax1, ax2, ax3]:
            plt.setp(ax.get_yticklabels(), visible=False)
            
        for ax in [ax0, ax1, ax2, ax3]:
            ax.set_xticks(ax.get_xticks()[:-1])
            ax.set_xlabel('x (km)', fontsize=18)
            ax.set_xlim(0, xmax/1000)
            
            ax.set_ylim(0, 2*pi)
            ax.set_yticks(np.arange(0, 2*pi+0.001, 0.5*pi))
            
        for ax in [ax0]:
            ax.set_ylabel(r'$\phi$', rotation=0, labelpad=10)
            ax.set_yticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']) 
        
        fig.suptitle('Wave polarization')
        #fig.subplots_adjust(wspace=0)
        #fig.subplots_adjust(hspace=0)
        
        plot_name = 'phis.png'  
    
                    
#%%             ----- PLOT 0 SAVE -----
    # Misc. Plot stuff - Formatting
    #plt.tight_layout(pad=1, w_pad=0.8)
    
    # Save (single) Plot
    fullpath = os.path.join(save_dir, plot_name)    
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')#, bbox_inches='tight')
    plt.close()
    print('Plot saved')
    
    # Save Data Array
    By_filename = 'By_array.npz'    # Magnetic Field history: y-component
    W_filename  = 'W_array.npz'     # Particle Kinetic Energies
    F_filename  = 'F_array.npz'     # Field Energies
    
    if (subplot_style == 'A' or subplot_style == 'B' or subplot_style == 'C'):
        By_fullpath = os.path.join(data_dir, By_filename)
        
        if os.path.isfile(By_fullpath) == False:
            np.savez(By_fullpath, By_array=By_all)
            print('By array saved')
            
    elif subplot_style == 'D':
        W_fullpath = os.path.join(data_dir, W_filename)
        
        if os.path.isfile(W_fullpath) == False:
            np.savez(W_fullpath, W_array=W)
            print('W array saved')
            
        F_fullpath = os.path.join(data_dir, F_filename)
        
        if os.path.isfile(F_fullpath) == False:
            np.savez(F_fullpath, F_array=F)
            print('F array saved')
