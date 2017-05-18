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
#import scipy.stats as stats

#%% ---- USER INPUT ----
# Plot Style: Pre-made ones using 1,2,3... etc. as options in an if statement. Produces plots varying with time
# 0: Collect time-variant info to generate one plot
# 1: 3D vy plots - two species
# 2: big vx (both species) - den, phi, By, |B| on side
# 3: Two-Field Plots : E and B
# 4: 2D Pure Phase space plot (vx vs. vy) 
# 5: Fourier spectrum in k-space                                XXXXXXX
# 6: Temperatures and Energies
# 7: Troubleshoot
# 
# 3D magnetic and electric fields
# 0: Stuff that requires all times at once (for single plot stuff) XXXXXX
#       - Spatial FFT with time (contour plot)
#       - Temps with time XXXXXXXXXXX
#       - Energy Histories
#      

#%% Just to stop those annoying warnings
part, B, E, dns, partin, partout, B0, ts, n0, dxm, size, cellpart, theta, Te0, seed, NX = [None] * 16

#%% ---- RUN PARAMETERS ----
drive_letter = 'F'            # Drive on which run is stored
sub_path = 'Smooth Two D'     # Location of 'Run' folder, with series name + optional subfolders
run = 0                       # Number of run within series
plot_style = 1                # Type of plot to produce as described above
y_slice = 64
plot_skip = 1                 # Skip every x files from input directory (Default: 1)


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


#%% ---- LOAD FILE PARAMETERS ----

load_dir = '%s:\Runs\%s\Run %d\Data' % (drive_letter, sub_path, run)                        # Load run data from this dir    INPUT
save_dir = '%s:\Analysis\%s\Run %d\Plot %d' % (drive_letter, sub_path, run, plot_style)     # Save analysis data here        OUTPUT
data_dir = '%s:\Analysis\%s\Run %d\Data' % (drive_letter, sub_path, run)                    # Save Matrices here             OUTPUT

if os.path.exists(save_dir) == False:                # Make Output Folder if it doesn't exist
    os.makedirs(save_dir)

if os.path.exists(data_dir) == False:                # Make Output Folder if it doesn't exist
    os.makedirs(data_dir)  
    

#%% ---- LOAD HEADER ----

h_name = os.path.join(load_dir, 'Header.pckl')       # Load Header File
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
num_index = np.arange(len(array_names))           # Number of stored variables

# Load E, B, SRC and part arrays
for var_name, index_position in zip(array_names, num_index):                # Magic subroutine to attach variable names to their stored arrays
    globals()[var_name] = p_data[array_names[index_position]]               # Generally contains 'partin', 'partout' and 'part_type' arrays from model run

print 'Particle parameters loaded'

''' 
partin:      (0) Mass (proton units)
             (1) Charge (multiples of e)
             (2) Bulk Velocity (multiples of vA)
             (3) Relative (real) density
             (4) Simulated density
             (5) Beta Parallel
             (6) Beta Perpendicular

partout:     (0) Actual species mass (in kg)
             (1) Actual species charge (in C)
             (2) Actual species bulk velocity (in m/s)
             (3) Density contribution of each superparticle (particles per sim particle)
'''

#%% ---- DERIVED VALUES ----

## Derived Values ##
Nj       = int(np.shape(partin)[1])   
av_rho   = np.sum([partin[0, nn] * mp * partin[3, nn] * n0 for nn in range(Nj)])    # Average mass density
alfie    = B0/np.sqrt(mu0 * av_rho)             # Alfven Velocity (m/s): Constant - Initialized at t = 0
gyfreq   = q*B0/mp                              # Proton Gyrofrequency (rad/s) (since this will be the highest of all species)
gyperiod = 2*pi / gyfreq                        # Gyroperiod in seconds
wpi      = np.sqrt((n0 * (q**2)) / (mp * e0 ))  # Plasma Frequency (rad/s)

DT       = ts / gyfreq                          # Time step as fraction of gyroperiod (T = 1 / f) - CHANGE TO BE FOR THE HEAVIEST ION?
dx       = dxm * c / wpi                        # Spacial step as function of plasma frequency (in metres)
xmax     = NX * dx                              # Spatial size of simulation
ymax     = NX * dx

## Particle Values ##
psize         = 1                                                  # Size of particles on plots
N             = cellpart*NX*NX                                     # Number of Particles to simulate: # cells x # particles per cell, excluding ghost cells
sample_percent= 0.001                                              # Number of sample particles in each population
N_species     = np.round(N * partin[4, :]).astype(int)             # Number of sim particles for each species, total
N_cell        = [float(N_species[ii]) / (NX*NX) for ii in range(Nj)]    # Number of sim particles per cell, per species
N_real        = (dx * 1. * 1.) * n0 * NX * NX                           # Total number of real particles (rect prism with sides dx x 1 x 1 metres)

# Output Particle Values
partout = np.array([partin[0, :] * mp,                       # (0) Actual Species Mass    (in kg)
                    partin[1, :] * q,                        # (1) Actual Species Charge  (in coloumbs)
                    partin[2, :] * alfie,                    # (2) Actual Species streaming velocity
                    (N_real * partin[3, :]) / N_species])    # (3) Density contribution of each particle of species (real particles per sim particle)
                    
N_real_cell = [N_cell[ii] * partout[3, ii] for ii in range(Nj)] 

Tpar = ((alfie ** 2) * partout[0, :] * partin[5, :]) / (2 * kB)         # Parallel Temperatures
Tprp = ((alfie ** 2) * partout[0, :] * partin[6, :]) / (2 * kB)         # Perpendicular Temperatures

idx_start = [np.sum(N_species[0:ii]    )     for ii in range(0, Nj)]                     # Start index values for each species in order
idx_end   = [np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)]  
    
plt.ioff()

#%% ---- PLOT AXES ---- 

# Spatial Arrays
x_cell = np.arange(0, xmax, dx) / 1000      # Cell x-positions (km) (For looking at cell characteristics)
x_cell_num = np.arange(NX)                  # Numerical cell numbering: x-axis    
x_cwpi = x_cell_num * dxm                   # Spatial dimensions in units of c/wpi

# Time arrays (different flavours)
sim_time = np.array([tt * ts * gyperiod for tt in range(num_files)]) * framegrab   # "Real" time
inc_time = np.arange(num_files) * framegrab                                        # Timesteps
ion_time = np.arange(num_files) * framegrab * ts                                   # Time in ion cycles

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
            

#%% ---- PLOT 0 Initializations ----
if plot_style == 0:
    plt.ioff()
    B_point = np.zeros(num_files)    # Place to store B time-series
    
    # Initialize temperature arrays - b = beam (hot), c = core (cold)
    temps = np.zeros((num_files, 4))  # b para, perp + c para, perp   [:, x] 
    vyb = np.zeros(num_files)
    vzb = np.zeros(num_files)
    
    # Initialize FFT arrays
    By_all      = np.zeros((num_files, NX), dtype='float64')
    By_k        = np.zeros((num_files, NX), dtype='complex128')     # "Size" includes ghost cells, the -2 gets rid of them
    By_kf       = np.zeros((num_files, NX), dtype='complex128')
    By_kf_pwr   = np.zeros((num_files, NX), dtype='complex128')

#%% ---- LOAD FILES AND PLOT -----
# Read files and assign variables. Do the other things as well
for ii in np.arange(num_files):
    if ii%plot_skip == 0:
        d_file = 'data%05d.npz' % ii                    # Define target file
        input_path = os.path.join(load_dir, d_file)     # File location
        data = np.load(input_path)                      # Load file
        
        array_names = data.files                         # Create list of stored variable names
        num_index = np.arange(len(array_names))          # Number of stored variables
        
        # Load E, B, SRC and part arrays
        for var_name, index_position in zip(array_names, num_index):                # Magic subroutine to attach variable names to their stored arrays
            globals()[var_name] = data[array_names[index_position]]                 # Manual would be something like part = data['part']
                
        if plot_style != 0:
        # Initialize Figure Space
            fig_size = 4, 7
            fig = plt.figure(figsize=(20,10))    
                
#%% ---- FONT ----
        # Set font things
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
                    
        species_colors = ['red', 'cyan', 'magenta', 'lime']
                    
        # Set positional arrays (different flavours) 
        x_pos = part[0, :] / 1000                 # Particle x-positions (km) (For looking at particle characteristics) 
        y_pos = part[1, :] / 1000

#%%     Plot Style 1
        if plot_style == 1:
            
            # Plot 3D Scatterplot: All species
            norm_yvel = part[4, :] / alfie        # vy (vA / ms-1)
            
            ax_vy_hot  = plt.subplot2grid(fig_size, (0, 0), projection='3d', rowspan=4, colspan=3)   
            ax_vy_core = plt.subplot2grid(fig_size, (0, 3), projection='3d', rowspan=4, colspan=3)
            
            for ax, jj in zip([ax_vy_hot, ax_vy_core], [0, 1]):
                
                ax.scatter(x_pos[idx_start[jj]: idx_end[jj]],        # Plot particles
                           y_pos[idx_start[jj]: idx_end[jj]],
                           norm_yvel[idx_start[jj]: idx_end[jj]],
                           s=psize, lw=0, c=species_colors[jj])

                # Make it look pretty
                ax.set_title(r'Normalized velocity $v_y$ vs. Position (x, y)')    
                
                ax.set_xlim(0, xmax/1000)
                ax.set_ylim(0, ymax/1000)
                ax.set_zlim(-15, 15)
                
                ax.set_xlabel('x position (km)', labelpad=10)
                ax.set_ylabel('y position (km)', labelpad=10)
                ax.set_zlabel(r'$\frac{v_y}{v_A}$', fontsize=24, rotation=0, labelpad=8) 
                
                ax.view_init(elev=25., azim=300)          
 


#%%     Plot Style 2
        if plot_style == 2:
            
            # ---- Side Bits (den, phi, By, |B|) ----#

        # Density
            dns_norm = np.zeros((NX, Nj), dtype=float)            
            
            for jj in range(Nj):
                dns_norm[:, jj] = dns[1: NX + 1, y_slice, jj] / (N_real_cell[jj] / dx)
                
            den_pos  = 0, 3  
            ax_den   = plt.subplot2grid(fig_size, den_pos,  colspan = 3)

            for jj in range(Nj):
                ax_den.plot(x_cell_num, dns_norm[:, jj], color=species_colors[jj])
                
            ax_den.set_ylabel('Normalized Density', fontsize=16)  
            ax_den.set_ylim(0, 3)
            
            ax_den.set_title('Various Spatial Parameters vs. x (cell cdts) incl. Ghost Cells')
            
        # Magnetic Field: y-component
            By = B[1:NX + 1, y_slice, 1] / B0            
            
            By_pos   = 2, 3
            By_ax   = plt.subplot2grid(fig_size, By_pos,   colspan = 3, sharex=ax_den)
            By_ax.plot(x_cell_num, By, color='#33cc33')
            By_ax.set_ylabel(r'$\frac{B_y}{B_0}$', rotation=0, labelpad=15)
            By_ax.set_ylim(-2, 2)

            
        # Magnetic Field: Magnitude
            Bmag = np.sqrt(B[1:NX + 1, y_slice, 0] ** 2 + B[1:NX + 1, y_slice, 1] ** 2 + B[1:NX + 1, y_slice, 2] ** 2) / B0            
            
            Bmag_pos = 3, 3
            Bmag_ax = plt.subplot2grid(fig_size, Bmag_pos, colspan = 3, sharex=ax_den)
            Bmag_ax.plot(x_cell_num, Bmag, color='#33cc33')
            Bmag_ax.set_ylabel(r'$\frac{|B|}{B_0}$', rotation=0, labelpad=20)
            Bmag_ax.set_ylim(0, 4)
            
#==============================================================================
#         # Phi angle
#             phi_pos = 1, 3
#             phi_ax = plt.subplot2grid(fig_size, phi_pos, colspan=3, sharex=ax_den)
#             phi = np.arctan2(B[1:NX + 1, y_slice, 2], B[1:NX + 1, y_slice, 1]) + pi
#             phi_ax.plot(x_cell_num, phi, color='#ff9933')
#             phi_ax.set_ylabel(r'$\phi$', rotation=0, labelpad=10)
#             phi_ax.set_ylim(0, 2*pi)
#             phi_ax.set_yticks(np.arange(0, 2*pi+0.001, 0.5*pi))
#             phi_ax.set_yticklabels([r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
#         
#==============================================================================
        # Ez
            Ez = E[1:NX + 1, y_slice, 2] / 1e6           
            
            Ez_pos   = 1, 3
            Ez_ax   = plt.subplot2grid(fig_size, Ez_pos,   colspan = 3, sharex=ax_den)
            Ez_ax.plot(x_cell_num, Ez, color='#ffff00')
            Ez_ax.set_ylabel(r'$E_z$ ($\mu$V)', rotation=0, labelpad=20)
            Ez_ax.set_ylim(-50, 50)
             
            
        ### Set Bulk Plot Things
            
            # All
            for ax in [By_ax, Bmag_ax, ax_den, Ez_ax]:
                ax.set_xlim(0, NX)
            
            # Not Bottom
            for ax in [By_ax, Ez_ax, ax_den]:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.set_yticks(ax.get_yticks()[1:])
                
            # Bottom
            for ax in [Bmag_ax]:
                ax.set_xlabel('Cell Number (x)')
           
            # ---- Big Bit (vx) ---- #       
           
            # Plot 3D Scatterplot: All species
            norm_xvel = part[4, :] / alfie        # vy (vA / ms-1)
            ax_vx  = plt.subplot2grid(fig_size, (0, 0), projection='3d', rowspan=4, colspan=3)   

            for jj in [0, 1]:
                ax_vx.scatter(x_pos[idx_start[jj]: idx_end[jj]],        # Plot particles
                           y_pos[idx_start[jj]: idx_end[jj]],
                           norm_xvel[idx_start[jj]: idx_end[jj]],
                           s=psize, lw=0, c=species_colors[jj])

            # Make it look pretty
            ax_vx.set_title(r'Normalized velocity $v_y$ vs. Position (x, y)')    
            
            ax_vx.set_xlim(0, xmax/1000)
            ax_vx.set_ylim(0, ymax/1000)
            ax_vx.set_zlim(-15, 15)
            
            ax_vx.set_xlabel('x position (km)', labelpad=10)
            ax_vx.set_ylabel('y position (km)', labelpad=10)
            ax_vx.set_zlabel(r'$\frac{v_y}{v_A}$', fontsize=24, rotation=0, labelpad=8) 
            
            ax_vx.view_init(elev=25., azim=300)     
            
        
#%%     Plot Style 3
        elif plot_style == 3:

            incl_ghost = np.arange(NX + 2)     
            
            # Plot 3D Surface: Electric Field
            ax_Ez = plt.subplot2grid(fig_size, (0, 0), projection='3d', rowspan=4, colspan=3)
            ax_Ez.set_title(r'Electric Field Strength ($\mu V$)')
            X, Y = np.meshgrid(incl_ghost, incl_ghost)
            
            #ax_Ez.plot_surface(X, Y, (E[1:NX+1, 1:NX+1, 2]*1e6))
            ax_Ez.plot_surface(X, Y, (E[:, :, 2]*1e6))
            ax_Ez.set_xlim(0, NX+2)
            ax_Ez.set_ylim(0, NX+2)
            ax_Ez.set_zlim(-1000, 1000)
            
            #ax_Ez.contourf(x_cell_num, x_cell_num, (E[1:NX+1, 1:NX+1, 2]*1e6), np.linspace(-500, 500, 300), zdir='z', offset=-1000)
            
            ax_Ez.view_init(elev=21., azim=300.)

            # Plot 3D Surface: Electric Field
            ax_By = plt.subplot2grid(fig_size, (0, 3), projection='3d', rowspan=4, colspan=3)
            ax_By.set_title(r'Magnetic Field Strength ($B_0nT$)')
            
            #ax_By.plot_surface(X, Y, (B[1:NX+1, 1:NX+1, 1] / B0) )
            ax_By.plot_surface(X, Y, (B[:, :, 1] / B0) )
            ax_By.set_xlim(0, NX+2)
            ax_By.set_ylim(0, NX+2)
            ax_By.set_zlim(-3, 3)
            
            #ax_By.contourf(x_cell_num, x_cell_num, (B[1:NX+1, 1:NX+1, 1] / B0), np.linspace(-3, 3, 200), zdir='z', offset=-3)
            
            ax_By.view_init(elev=21., azim=300.)


        
#%%     Plot Style 4
        elif plot_style ==  4:
            
            # Phase plot (vx/vy)
            phase_pos = 0,0        
            ax_phase = plt.subplot2grid(fig_size, phase_pos, colspan=3, rowspan=4)
            
            norm_xvel = part[3, 0:N] / alfie        # vx (vA / ms-1)
            norm_yvel = part[4, 0:N] / alfie        # vy (vA / ms-1)
            
            for jj in range(0, Nj):
                ax_phase.scatter(norm_xvel0[idx_start[jj]: idx_end[jj]], norm_yvel0[idx_start[jj]: idx_end[jj]], s=psize, lw=0, color=species_colors[jj]) 
            
            ax_phase.set_xlabel(r'$\frac{v_x}{v_A}$')        
            ax_phase.set_xlim(-14, 14)
            ax_phase.set_ylabel(r'$\frac{v_y}{v_A}$', rotation=0)
            ax_phase.set_ylim(-12, 12)
            ax_phase.set_title('Velocity Phase Space (Normalized)')
            
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
            
            
#%%     Plot Style 6
#==============================================================================
#         elif plot_style == 6:
#             var_bpar  = stats.moment(part[3, 0:N_cold], moment = 2)
#             var_bperp = stats.moment(np.sqrt((part[4, 0:N_cold] ** 2) + (part[4, 0:N_cold] ** 2)), moment = 2)
#             
#             var_cpar  = stats.moment(part[3, N_cold:N], moment = 2)
#             var_cperp = stats.moment(np.sqrt((part[4, N_cold:N] ** 2) + (part[4, N_cold:N] ** 2)), moment = 2)
#             
#             for var, jj in zip([var_bpar, var_cperp], [0, 1]):
#                 temps[ii, jj] = (var * mp) / kB
#                 
#             for var, jj in zip([var_cpar, var_cperp], [2, 3]): # Is this normal or maxwellian?
#                 temps[ii, jj] = 
#==============================================================================
            
        elif plot_style == 7:
            print np.max()

        
#%%     Plot Style 0
        elif plot_style == 0: # Use this to collect time-varying data (points, etc.)

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
                
                # Turn plot space on
                fig_size = 4, 7
                fig = plt.figure(figsize=(20,10))    
                
#==============================================================================
#                 # Temperature Plots                
#                 b_para = (mi * (vxb ** 2) / kB) / T_para
#                 c_para = (mi * (vxc ** 2) / kB) / T_para
#                 
#                 b_perp = ((0.5 * mi * ((vyb ** 2) + (vzb ** 2))) / kB) / T_perp
#                 c_perp = ((0.5 * mi * ((vyc ** 2) + (vzc ** 2))) / kB) / T_perp
# 
#                 ax_bx = plt.subplot2grid(fig_size, (0,0), colspan=3, rowspan=2)         # Beam Perpendicular (crossed)
#                 ax_cx = plt.subplot2grid(fig_size, (0,3), colspan=3, rowspan=2)         # Core Perpendicular (crossed)
#                 ax_bp = plt.subplot2grid(fig_size, (2,0), colspan=3, rowspan=2)         # Beam Parallel
#                 ax_cp = plt.subplot2grid(fig_size, (2,3), colspan=3, rowspan=2)         # Core Parallel
#                                 
#                 timestep = np.arange(0, num_files)                
#                 t = np.asarray([ii * DT for ii in range(len(timestep))]) * gyfreq       # Normalized in terms of gyrofrequency
#                 
#                 ax_bp.plot(t, b_para)
#                 ax_bp.set_ylabel('$T_{b\parallel}$\n (K)', rotation=0, labelpad=25)     
#                 ax_bp.set_xlabel('$\Omega_it$')
#                 
#                 ax_cp.plot(t, c_para)
#                 ax_cp.set_ylabel('$T_{c\parallel}$\n (K)', rotation=0, labelpad=25)  
#                 ax_cp.set_xlabel('$\Omega_it$')
#                 
#                 ax_bx.plot(t, b_perp)
#                 ax_bx.set_ylabel('$T_{b\perp}$\n (K)', rotation=0, labelpad=25)  
#                 
#                 ax_cx.plot(t, c_perp)
#                 ax_cx.set_ylabel('$T_{c\perp}$\n (K)', rotation=0, labelpad=25)  
#==============================================================================
                
#==============================================================================
#                 # Waterfall plot
#                 sep = 1        # Separation between waterfall plots
#                 skip = 30
#                 start = 0
#                 end = 8000      # Last time to plot                
#                 ii = 0
#                 
#                 for ii in np.arange(start, end, skip):
#                     plt.plot(x_cell_num, (By_all[ii]*120 + sep*ii), color='b')
#                 
#                 plt.ylim(start, end)
#                 plt.xlim(0, size-2)
#==============================================================================
                
                # Spatial FFT at each time
                for kk in range(num_files):
                    By_k[kk, :] = np.fft.fft(By_all[kk, :])                  # Spatial FFT at each time
                                    
                # Temporal FFT at each k
                for kk in range(size-2):
                    By_kf[:, kk] = np.fft.fft(By_k[:, kk])                  # Temporal FFT at each k
                    
                By_wk = By_kf[0:num_files/2, 0:(size-2)/2]
                #By_kf_pwr = By_kf[0:num_files/2, 0:(size-2)/2].imag
                By_kf_pwr = By_wk * np.conj(By_wk)
                high = np.max(By_kf_pwr)

                # w/k plot - Power
                plt.contourf(k_array[1:((size-2)/2)], f, By_kf_pwr[:, 1:((size-2)/2)], 400)      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k]
                plt.clim(0.0001*high, 0.01*high)
                plt.ylabel('Temporal Frequency (mHz)')
                plt.xlabel(r'Spatial Frequency ($10^{-6}m^{-1})$')
                plt.ylim(0, 100)                
                
                plt.colorbar()
                                
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
       
        if plot_style != 0:

            # Figure Text
            text1  = plt.figtext(0.84, 0.01, 'Simulation Time = %.2f s'     % (sim_time[ii]),       fontsize = 16, color='#ffff00')
            text2  = plt.figtext(0.01, 0.01, 'Ion Time: %.2f'               % (gyfreq * ii * DT),   fontsize = 16, color='#ffff00') 
            
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
            
#==============================================================================
#             # Misc. Plot stuff
#             plt.tight_layout(pad=2, w_pad=0.8)
#             fig.subplots_adjust(hspace=0)
#==============================================================================
            
            # Save Plot
            filename = 'anim%05d.png' % ii
            fullpath = os.path.join(save_dir, filename)    
            plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')#, bbox_inches='tight')
            plt.close()
        
            print 'Plot %d of %d created' % (ii + 1, num_files)
