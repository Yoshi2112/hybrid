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
import numba as nb
from collections import OrderedDict
from matplotlib.lines import Line2D

def manage_dirs():
    global run_dir, data_dir, anal_dir, temp_dir, base_dir
    
    base_dir = '{}/runs/{}/'.format(drive, series)                      # Main series directory, containing runs
    run_dir  = '{}/runs/{}/run_{}/'.format(drive, series, run_num)      # Main run directory
    data_dir = run_dir + 'data/'                                        # Directory containing .npz output files for the simulation run
    anal_dir = run_dir + 'analysis/'                                    # Output directory for all this analysis (each will probably have a subfolder)
    temp_dir = run_dir + 'temp/'                                        # Saving things like matrices so we only have to do them once
    
    for this_dir in [anal_dir, temp_dir]:
        if os.path.exists(this_dir) == False:                           # Make Output folder if they don't exist
            os.makedirs(this_dir)        
    return


def load_constants():
    global q, c, me, mp, e, mu0, kB, e0
    q   = 1.602e-19               # Elementary charge (C)
    c   = 3e8                     # Speed of light (m/s)
    me  = 9.11e-31                # Mass of electron (kg)
    mp  = 1.67e-27                # Mass of proton (kg)
    e   = -q                      # Electron charge (C)
    mu0 = (4e-7) * pi             # Magnetic Permeability of Free Space (SI units)
    kB  = 1.38065e-23             # Boltzmann's Constant (J/K)
    e0  = 8.854e-12               # Epsilon naught - permittivity of free space
    return


def load_particles():
    global density, dist_type, idx_bounds, charge, mass, Tper, sim_repr, temp_type, temp_color, velocity, Tpar, species_lbl, n_contr
    
    p_path = os.path.join(data_dir, 'p_data.npz')                               # File location
    p_data = np.load(p_path)                                                    # Load file

    density    = p_data['density'] 
    idx_bounds = p_data['idx_bounds']
    charge     = p_data['charge']
    mass       = p_data['mass']
    Tper       = p_data['Tper']
    sim_repr   = p_data['sim_repr']
    temp_type  = p_data['temp_type']
    temp_color = p_data['temp_color']
    dist_type  = p_data['dist_type']
    velocity   = p_data['velocity']
    Tpar       = p_data['Tpar']
    species_lbl= p_data['species_lbl']   
    
    n_contr    = density / (cellpart*sim_repr)                        # Species density contribution: Each macroparticle contributes this density to a cell

    print 'Particle parameters loaded'
    return


def load_header():
    global Nj, cellpart, data_dump_iter, ne, NX, dxm, seed, B0, dx, Te0, theta, dt, max_rev,\
           ie, run_desc, seed, subcycles, LH_frac, orbit_res, freq_res, particle_shape, method_type
    
    h_name = os.path.join(data_dir, 'Header.pckl')                      # Load header file
    f      = open(h_name)                                               # Open header file
    obj    = pickle.load(f)                                             # Load variables from header file into python object
    f.close()                                                           # Close header file
    
    Nj              = obj['Nj']
    cellpart        = obj['cellpart']
    data_dump_iter  = obj['data_dump_iter']
    subcycles       = obj['subcycles']
    ne              = obj['ne']
    NX              = obj['NX']
    dxm             = obj['dxm']
    seed            = obj['seed']
    B0              = obj['B0']
    dx              = obj['dx']
    Te0             = obj['Te0']
    theta           = obj['theta']
    dt              = obj['dt']
    max_rev         = obj['max_rev']
    ie              = obj['ie']
    LH_frac         = obj['LH_frac']
    orbit_res       = obj['orbit_res']
    freq_res        = obj['freq_res'] 
    run_desc        = obj['run_desc']
    particle_shape  = obj['particle_shape']
    method_type     = obj['method_type']

    print 'Header file loaded.'
    print 'dt = {}s\n'.format(dt)
    return 


def load_timestep(ii):
    print 'Loading file {} of {}'.format(ii+1, num_files)
    d_file     = 'data%05d.npz' % ii                # Define target file
    input_path = data_dir + d_file                  # File location
    data       = np.load(input_path)                # Load file

    tB               = data['B']
    tE               = data['E']
    tVe              = data['Ve']
    tTe              = data['Te']
    tJ               = data['J']
    tpos             = data['pos']
    tdns             = data['dns']
    tvel             = data['vel']

    return tB, tE, tVe, tTe, tJ, tpos, tdns, tvel


@nb.njit()
def collect_number_density(pos):
    '''Collect number and velocity density in each cell at each timestep, weighted by their distance
    from cell nodes.

    INPUT:
        pos    -- position of each particle
    '''
    size      = NX + 2

    n_i       = np.zeros((size, Nj))
    
    node      = pos / dx + 0.5 
    weight    = (pos / dx) - node + 0.5
    n_contr    = density / (cellpart*sim_repr)      # Density: initial /m3 of species in cell, divide this by number of particles in the cell
                                                    # Each macroparticle contributes this amount to the cell's density in /m3
    for jj in range(Nj):
        for ii in range(idx_bounds[jj, 0], idx_bounds[jj, 1]):
            I   = int(node[ii])
            W   = weight[ii]

            n_i[I,     jj] += (1 - W) * n_contr[jj]
            n_i[I + 1, jj] +=      W  * n_contr[jj]

    n_i[1]  += n_i[-1]
    n_i[-1] += n_i[0]
    
    n_i[0]   = 0.
    n_i[-1]  = 0
    return n_i


def get_array(component, tmin, tmax):
    if component[-1].lower() == 'x':
        comp_idx = 0
    elif component[-1].lower() == 'y':
        comp_idx = 1
    elif component[-1].lower() == 'z':
        comp_idx = 2
    
    if tmax == None:
        tmax = num_files
    
    num_iter = tmax - tmin
    arr      = np.zeros((num_iter, NX))
    
    if tmin == 0 and tmax == num_files:
        check_path = temp_dir + component.lower() + '_array' + '.npy'
    else:
        check_path = temp_dir + component.lower() + '_array' + '_{}'.format(tmin) + '_{}'.format(tmax) + '.npy'

    if os.path.isfile(check_path) == True:
        print 'Array file for {} loaded from memory...'.format(component.upper())
        arr = np.load(check_path)   
    else:
        for ii in range(tmin, tmax):
            B, E, Ve, Te, J, position, q_dns, velocity = load_timestep(ii)
            
            if component[0].upper() == 'B':
                arr[ii] = B[0:-1, comp_idx]
            elif component[0].upper() == 'E':
                arr[ii] = E[1:-1, comp_idx]
                
        print 'Saving array file as {}'.format(check_path)
        np.save(check_path, arr)
    return arr




def get_run_energies():
    '''
    Scans through the files of a run and calculates the energy for each simulation component at each
    timestep. Returns a single array with columns as the components (and time) and rows for each time slice.
    
    INPUT:
        None (Uses global variables defined in __main__)
        
    OUTPUT:
        energy_array -- Output array with time and energies of each simulation component
        
    The layout of the array is as follows:
        0: Time slices
        1: Total simulation energy
        2: Magnetic field
        3: Electrons
        4+: Ion species (may be several)
    '''
    
    energy_output    = np.zeros((4 + Nj, num_files))
    time_gperiods    = np.arange(0, num_files * data_ts, data_ts)  / gyperiod
    energy_output[0] = time_gperiods        
    
    for ii in range(num_files):
        B, E, Ve, Te, J, position, q_dns, velocity = load_timestep(ii)
            
        energy_output[2, ii] = (0.5 / mu0) * np.square(B[1:-2]).sum() * NX * dx    # Magnetic potential energy 
        energy_output[3, ii] = 1.5 * (kB * Te * q_dns / q).sum()      * NX * dx    # Electron pressure energy
    
        for jj in range(Nj):
            vp2 = velocity[0, idx_bounds[jj, 0]:idx_bounds[jj, 1]] ** 2 \
                + velocity[1, idx_bounds[jj, 0]:idx_bounds[jj, 1]] ** 2 \
                + velocity[2, idx_bounds[jj, 0]:idx_bounds[jj, 1]] ** 2           # Total real particle kinetic energy
            
            energy_output[4+jj, ii] = 0.5 * mass[jj] * vp2.sum() * n_contr[jj] * NX * dx 
    
    energy_output[1] = energy_output[2:].sum(axis=0)
    return energy_output


def create_legend(fn_ax):
    legend_elements = []
    for label, style in zip(run_labels, run_styles):
        legend_elements.append(Line2D([0], [0], color='k', lw=1, label=label, linestyle=style))
        
    new_legend = fn_ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.6))
    return new_legend, fn_ax


def plot_energies(energy, ax, normalize=True):

    if normalize == True:
        for kk in range(1, energy.shape[0]):
            energy[kk, :] /= energy[kk, 0]
    
    ax.plot(energy[0], energy[1], label = r'$Total$', c='k', linestyle=run_styles[ii])
    ax.plot(energy[0], energy[2], label = r'$U_B$', c='green', linestyle=run_styles[ii])
    ax.plot(energy[0], energy[3], label = r'$U_e$', c='orange', linestyle=run_styles[ii])
    
    for jj in range(Nj):
        ax.plot(energy[0], energy[4 + jj], label='$K_E$ {}'.format(species_lbl[jj]), 
                                 c=temp_color[jj], linestyle=run_styles[ii])

    ax.set_xlabel('Time (Gyroperiods)')
    
    if normalize == True:
        ax.set_ylabel('Normalized Energy', rotation=90)
        fig.subplots_adjust(bottom=0.07, top=0.96, left=0.04)
    else:
        ax.set_ylabel('Energy (Joules)', rotation=90)
        fig.subplots_adjust(bottom=0.07, top=0.96, left=0.055)
        
    ax.set_title('Energy Distribution in Simulation Space')
    return


if __name__ == '__main__':   
    plt.ioff()
    
    drive           = 'F:'
    series          = 'CAM_CL_velocity_test'                    # Run identifier string 
    
    runs_to_analyse = [0                ,   1 ]                 # Run number
    run_styles      = ['-'              , '--']
    run_labels      = ['Two-step Leapfrog', 'Boris Algorithm']
    
    fig  = plt.figure(figsize=(15, 7))
    ax   = plt.subplot2grid((7, 7), (0, 0), colspan=6, rowspan=7)
    run_legend, ax  = create_legend(ax)
        
    for run_num, ii in zip(runs_to_analyse, range(len(runs_to_analyse))):
        manage_dirs()                                           # Initialize directories
        load_constants()                                        # Load SI constants
        load_header()                                           # Load simulation parameters
        load_particles()                                        # Load particle parameters
        num_files = len(os.listdir(data_dir)) - 2               # Number of timesteps to load
        
        wpi       = np.sqrt(ne * q ** 2 / (mp * e0))            # Ion plasma frequency
        gyfreq    = q * B0 / mp                                 # Proton gyrofrequency (rad/s)
        gyperiod  = (mp * 2 * np.pi) / (q * B0)                 # Proton gyroperiod (s)
        data_ts   = data_dump_iter * dt                         # Timestep between data records (seconds)
               
        print 'Analysing run {}'.format(run_num)
        energies = get_run_energies()

        plot_energies(energies, ax)
    
    fig.tight_layout()
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.8))

    ax.add_artist(run_legend)
    
    fullpath = base_dir + 'energy_plot'
    ax.set_ylim(0.85, 1.4)
    ax.set_xlim(0, 60)
    fig.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close('all')
    