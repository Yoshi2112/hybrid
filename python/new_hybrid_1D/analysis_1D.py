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
import pdb

def save_array_file(arr, saveas, overwrite=False):
    if os.path.isfile(temp_dir + saveas) == False:
        print 'Saving array file as {}'.format(temp_dir + saveas)
        np.save(temp_dir + saveas, arr)
    else:
        if overwrite == False:
            print 'Array already exists as {}, skipping...'.format(saveas)
        else:
            print 'Array already exists as{}, overwriting...'.format(saveas)
    return


def plot_dispersion(arr, saveas):
    plt.ioff()
    df = 1. / (num_files * dt)
    f  = np.arange(0, 1. / (2*dt), df) * 1000.

    dk = 1. / (NX * dx)
    k  = np.arange(0, 1. / (2*dx), dk) * 1e6

    fft_matrix  = np.zeros(arr.shape, dtype='complex128')
    fft_matrix2 = np.zeros(arr.shape, dtype='complex128')

    for ii in range(arr.shape[0]): # Take spatial FFT at each time
        fft_matrix[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())

    for ii in range(arr.shape[1]):
        fft_matrix2[:, ii] = np.fft.fft(fft_matrix[:, ii] - fft_matrix[:, ii].mean())

    dispersion_plot = fft_matrix2[:f.shape[0], :k.shape[0]] * np.conj(fft_matrix2[:f.shape[0], :k.shape[0]])

    fig, ax = plt.subplots()

    ax.contourf(k[1:], f[1:], np.log10(dispersion_plot[1:, 1:].real), 500, cmap='jet', extend='both')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True

    ax.set_title(r'Dispersion Plot: $\omega/k$', fontsize=22)
    ax.set_ylabel('Temporal Frequency (mHz)')
    ax.set_xlabel(r'Spatial Frequency ($10^{-6}m^{-1})$')

    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 40)

    fullpath = anal_dir + saveas + '.png'
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print 'Dispersion Plot saved'

    return

start_file = 0                # First file to scan - For starting mid-way    (Default 0: Read all files in directory)
end_file   = 0                # Last file to scan  - To read in partial sets (Default 0: Read all files in directory)

#%% ---- CONSTANTS ----
q   = 1.602e-19               # Elementary charge (C)
c   = 3e8                     # Speed of light (m/s)
me  = 9.11e-31                # Mass of electron (kg)
mp  = 1.67e-27                # Mass of proton (kg)
e   = -q                      # Electron charge (C)
mu0 = (4e-7) * pi             # Magnetic Permeability of Free Space (SI units)
kB  = 1.38065e-23             # Boltzmann's Constant (J/K)
e0  = 8.854e-12               # Epsilon naught - permittivity of free space

#%% ---- LOAD FILE PARAMETERS ----
run_dir  = 'E:/runs/winske_anisotropy_test/run_1/'  # Main run directory
data_dir = run_dir + 'data/'                                        # Directory containing .npz output files for the simulation run
anal_dir = run_dir + 'analysis/'                                    # Output directory for all this analysis (each will probably have a subfolder)
temp_dir = run_dir + 'temp/'                                        # Saving things like matrices so we only have to do them once

for this_dir in [anal_dir, temp_dir]:
    if os.path.exists(this_dir) == False:                           # Make Output folder if they don't exist
        os.makedirs(this_dir)

#%% ---- LOAD HEADER ----

h_name = os.path.join(data_dir, 'Header.pckl')                      # Load header file
f      = open(h_name)                                               # Open header file
obj    = pickle.load(f)                                             # Load variables from header file into python object
f.close()                                                           # Close header file

for name in obj.keys():                                             # Assign Header variables to namespace (dicts)
    globals()[name] = obj[name]                                     # Magic variable creation function

seed = 21
np.random.seed(seed)

if end_file == 0:
    num_files = len(os.listdir(data_dir)) - 2
else:
    num_files = end_file

print 'Header file loaded.'

#%% ---- LOAD PARTICLE PARAMETERS ----

p_path = os.path.join(data_dir, 'p_data.npz')                               # File location
p_data = np.load(p_path)                                                    # Load file

array_names = p_data.files                                                  # Create list of stored variable names
num_index   = np.arange(len(array_names))                                   # Number of stored variables

# Load E, B, SRC and part arrays
for var_name, index_position in zip(array_names, num_index):                # Magic subroutine to attach variable names to their stored arrays
    globals()[var_name] = p_data[array_names[index_position]]               # Generally contains 'partin', 'partout' and 'part_type' arrays from model run

print 'Particle parameters loaded'

#%% ---- LOAD FILES AND PLOT -----
# Read files and assign variables. Do the other things as well
ii        = start_file
plot_skip = 1

#all_B = np.zeros((num_files, NX + 2, 3))
#all_E = np.zeros((num_files, NX))

#for ii in grab:
max_v     = np.zeros(num_files)
spc_av_By = np.zeros(num_files)
spc_av_Ex = np.zeros(num_files)
 
print 'dt={}'.format(dt)
for ii in range(num_files):
    if ii%plot_skip == 0:
        print 'Loading file {} of {}'.format(ii, num_files)
        d_file     = 'data%05d.npz' % ii                # Define target file
        input_path = data_dir + d_file                  # File location
        data       = np.load(input_path)                # Load file

        max_v[ii]       = abs(data['part'][3]).max()
        spc_av_By[ii]   = abs(data['B'][1: -1, 1]).max()
        spc_av_Ex[ii]   = abs(data['E'][1: -1, 0]).max()
        #all_B[ii, :, :] = data['B']

t = np.arange(num_files) * dt * data_dump_iter     
#%%
fig = plt.figure(figsize=(18,10))
ax  = fig.add_subplot(111)


ax.plot(t, max_v / max_v.max(), label='velocity')
ax.plot(t, spc_av_By / spc_av_By.max(), label='B')
ax.plot(t, spc_av_Ex / spc_av_Ex.max(), label='E')
ax.legend()
plt.show()
#save_array_file(all_B, 'all_B')
#plot_dispersion(all_B[:, :, 1], 'wk_By')
        #E    = data['E']
        #part = data['part']
        #Ji   = data['Ji']
        #dns  = data['dns']



