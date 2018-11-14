# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:56:34 2016

@author: c3134027
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import pi
import pickle
import matplotlib.gridspec as gs
import pdb

def load_constants():
    q   = 1.602e-19               # Elementary charge (C)
    c   = 3e8                     # Speed of light (m/s)
    me  = 9.11e-31                # Mass of electron (kg)
    mp  = 1.67e-27                # Mass of proton (kg)
    e   = -q                      # Electron charge (C)
    mu0 = (4e-7) * pi             # Magnetic Permeability of Free Space (SI units)
    kB  = 1.38065e-23             # Boltzmann's Constant (J/K)
    e0  = 8.854e-12               # Epsilon naught - permittivity of free space
    return q, c, me, mp, e, mu0, kB, e0


def load_particles():
    p_path = os.path.join(data_dir, 'p_data.npz')                               # File location
    p_data = np.load(p_path)                                                    # Load file

    density    = p_data['density'] 
    dist_type  = p_data['dist_type']
    idx_bounds = p_data['idx_bounds']
    charge     = p_data['charge']
    mass       = p_data['mass']
    Tper       = p_data['Tper']
    sim_repr   = p_data['sim_repr']
    temp_type  = p_data['temp_type']
    velocity   = p_data['velocity']
    Tpar       = p_data['Tpar']
    species    = p_data['species']   
     
    print 'Particle parameters loaded'
    return density, dist_type, idx_bounds, charge, mass, Tper, sim_repr, temp_type, velocity, Tpar, species


def load_header():
    h_name = os.path.join(data_dir, 'Header.pckl')                      # Load header file
    f      = open(h_name)                                               # Open header file
    obj    = pickle.load(f)                                             # Load variables from header file into python object
    f.close()                                                           # Close header file
    
    Nj              = obj['Nj']
    cellpart        = obj['cellpart']
    data_dump_iter  = obj['data_dump_iter']
    ne              = obj['ne']
    NX              = obj['NX']
    dxm             = obj['dxm']
    seed            = obj['seed']
    B0              = obj['B0']
    dx              = obj['dx']
    Te0             = obj['Te0']
    theta           = obj['theta']
    dt              = obj['dt']
    max_sec         = obj['max_sec']
    ie              = obj['ie']
    run_desc        = obj['run_desc']
    seed            = obj['seed']
    
    print 'Header file loaded.'
    print 'dt = {}s\n'.format(dt)
    return Nj, cellpart, data_dump_iter, ne, NX, dxm, seed, B0, dx, Te0, theta, dt, max_sec, ie, run_desc, seed

def load_timestep(ii):
    print 'Loading file {} of {}'.format(ii+1, num_files)
    d_file     = 'data%05d.npz' % ii                # Define target file
    input_path = data_dir + d_file                  # File location
    data       = np.load(input_path)                # Load file

    tposition        = data['part'][0]
    tvelocity        = data['part'][3:6]
    tB               = data['B']
    tE               = data['E']
    tdns             = data['dns']
    tJ               = data['J']
    
    return tposition, tvelocity, tB, tE, tdns, tJ

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


def plot_k_time(arr, saveas):
    
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


def winske_stackplot(qq):
#----- Prepare some values for plotting
    x_cell_num  = np.arange(NX)                                         # Numerical cell numbering: x-axis
    
    norm_xvel   = 1e3*velocity[0, :] / c                                # Normalize to speed of light
    norm_yvel   = 1e3*velocity[1, :] / c
    By          = B[  1: NX + 1, 1] / B0                                # Normalize to background field
    dnb         = dns[1: NX + 1, 1] / density[1]                        # Normalize beam density to initial value
    rat         = B[1: NX + 1, 2] / B[1: NX + 1, 1]                     # Bz/By ratio
    phi         = np.arctan2(rat) + pi                                  # Wave magnetic phase angle
    
#----- Create plots
    plt.ioff()
    fig    = plt.figure(1, figsize=(16,9))                              # Initialize figure
    grids  = gs.GridSpec(5, 1)                                          # Create gridspace
    fig.patch.set_facecolor('w')                                        # Set figure face color

    ax_vx   = fig.add_subplot(grids[0, 0]) 
    ax_vy   = fig.add_subplot(grids[1, 0]) 
    ax_den  = fig.add_subplot(grids[2, 0])                              # Initialize axes
    ax_by   = fig.add_subplot(grids[3, 0]) 
    ax_phi  = fig.add_subplot(grids[4, 0]) 

    ax_vx.scatter(position[idx_bounds[1, 0]: idx_bounds[1, 1]], norm_xvel[idx_bounds[1, 0]: idx_bounds[1, 1]], s=1, c='k', lw=0)        # Hot population
    ax_vy.scatter(position[idx_bounds[1, 0]: idx_bounds[1, 1]], norm_yvel[idx_bounds[1, 0]: idx_bounds[1, 1]], s=1, c='k', lw=0)        # 'Other' population
    
    ax_den.plot(x_cell_num, dnb, c='k')                                 # Create overlayed plots for densities of each species
    ax_by.plot(x_cell_num, By, c='k')
    ax_phi.plot(x_cell_num, phi, c='k')

    ax_vx.set_ylabel(r'$v_{b, x} (\times 10^{-3})$', rotation=90)
    ax_vy.set_ylabel(r'$v_{b, y} (\times 10^{-3})$', rotation=90)
    ax_by.set_ylabel(r'$\frac{B_y}{B_0}$', rotation=0, labelpad=10, fontsize=14)

    plt.setp(ax_vx.get_xticklabels(), visible=False)
    #ax_vx.set_yticks(ax_vx.get_yticks()[1:])

    for ax in [ax_vx, ax_vy, ax_den, ax_by]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_yticks(ax.get_yticks()[1:])

#----- Plot adjustments
    plt.tight_layout(pad=1.0, w_pad=1.8)
    fig.subplots_adjust(hspace=0)

#----- Save plots
    filename = 'stackplot%05d.png' % qq
    path     = anal_dir + '/stackplot/'
    
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
        
    fullpath = path + filename
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close('all')
    return



if __name__ == '__main__':
    series   = 'winske_anisotropy_test'
    run_num  = 0
    
    run_dir  = 'E:/runs/{}/run_{}/'.format(series, run_num)             # Main run directory
    data_dir = run_dir + 'data/'                                        # Directory containing .npz output files for the simulation run
    anal_dir = run_dir + 'analysis/'                                    # Output directory for all this analysis (each will probably have a subfolder)
    temp_dir = run_dir + 'temp/'                                        # Saving things like matrices so we only have to do them once
    
    for this_dir in [anal_dir, temp_dir]:
        if os.path.exists(this_dir) == False:                           # Make Output folder if they don't exist
            os.makedirs(this_dir)
    
    num_files = len(os.listdir(data_dir)) - 2
    
    q, c, me, mp, e, mu0, kB, e0  = load_constants()  
    density, dist_type, idx_bounds, charge, mass, Tper, sim_repr, temp_type, velocity, Tpar, species = load_particles()
    Nj, cellpart, data_dump_iter, ne, NX, dxm, seed, B0, dx, Te0, theta, dt, max_sec, ie, run_desc, seed = load_header()
    
    np.random.seed(seed)
    
    for ii in range(num_files):
        position, velocity, B, E, dns, J = load_timestep(ii)
    