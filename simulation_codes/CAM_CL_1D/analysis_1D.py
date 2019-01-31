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
import numba as nb
import pdb


def manage_dirs():
    global run_dir, data_dir, anal_dir, temp_dir
    
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
    global density, dist_type, idx_bounds, charge, mass, Tper, sim_repr, temp_type, velocity, Tpar, species
    
    p_path = os.path.join(data_dir, 'p_data.npz')                               # File location
    p_data = np.load(p_path)                                                    # Load file

    density    = p_data['density'] 
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
    return


def load_header():
    global Nj, cellpart, data_dump_iter, ne, NX, dxm, seed, B0, dx, Te0, theta, dt, max_rev, ie, run_desc, seed, subcycles
    
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
    run_desc        = obj['run_desc']
    
    print 'Header file loaded.'
    print 'dt = {}s\n'.format(dt)
    return 


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


@nb.njit()
def collect_density(pos):
    '''Collect number and velocity density in each cell at each timestep, weighted by their distance
    from cell nodes.

    INPUT:
        pos    -- position of each particle
    '''
    size      = NX + 2

    n_i       = np.zeros((size, Nj))
    
    node      = pos / dx + 0.5 
    weight    = (pos / dx) - node + 0.5
    n_contr   = density / sim_repr                  # Density: initial /m3 of species in cell, divide this by number of particles in the cell
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
            position, velocity, B, E, dns, J = load_timestep(ii)
            
            if component[0].upper() == 'B':
                arr[ii] = B[0:-1, comp_idx]
            elif component[0].upper() == 'E':
                arr[ii] = E[1:-1, comp_idx]
                
        print 'Saving array file as {}'.format(check_path)
        np.save(check_path, arr)
    return arr


def generate_wk_plot(component='By', plot=True, tmin=0, tmax=None, normalize=False):
    ''' Create w/k dispersion plot for times between tmin and tmax. STILL ISN'T WORKING GREAT....
    
    INPUT:
        component -- field component to analyse. Loads from array file or data files if array file doesn't exist
        plot      -- Boolean, create plot or only load/save field components
        tmin      -- First iteration to load
        tmax      -- Last  iteration to load
    OUTPUT:
        None
        
    Note -- component keywork is not case sensitive, and should be one of Ex, Ey, Ez, Bx, By or Bz
    '''
    arr = get_array(component, tmin, tmax)
        
    if plot == True:
        plot_wk(arr, normalize, saveas='dispersion_relation_{}'.format(component.lower()))
    return


def generate_kt_plot(component='By', tmin=0, tmax=None, plot=True, normalize=False):
    ''' Create spatial frequency (Fourier mode) vs. time plot for times between tmin and tmax.
    
    INPUT:
        component -- field component to analyse. Loads from array file or data files if array file doesn't exist
        plot      -- Boolean, create plot or only load/save field components
        tmin      -- First iteration to load
        tmax      -- Last  iteration to load
    OUTPUT:
        None
        
    Note -- component keyword is not case sensitive, and should be one of Ex, Ey, Ez, Bx, By or Bz
    '''
    arr = get_array(component, tmin, tmax)
    
    if plot == True:
        plot_kt(arr, normalize, saveas='kt_plot_{}'.format(component.lower()))
    return


def plot_kt(arr, norm, saveas='kt_plot'):
    plt.ioff()
    k  = np.arange(NX)
    
# =============================================================================
#     if norm == True:
#         k    = np.arange(0, 1. / (2*dx), dk) * c / wpi 
#         xlab = r'$kc/\omega_i$'
#     else:
#         k    = np.arange(0, 1. / (2*dx), dk) * 1e6
#         xlab = r'$k (m^{-1})$'
# =============================================================================

    fft_matrix  = np.zeros(arr.shape, dtype='complex128')
    for ii in range(arr.shape[0]): # Take spatial FFT at each time
        fft_matrix[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())

    kt = (fft_matrix[:, :arr.shape[1] / 2] * np.conj(fft_matrix[:, :arr.shape[1] / 2])).real

    fig = plt.figure(1, figsize=(12, 8))
    ax  = fig.add_subplot(111)
    
    ax.pcolormesh(k[:arr.shape[1] / 2], time_radperiods, kt, cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True

    ax.set_title(r'k-t Plot (CAM-CL)', fontsize=14)
    ax.set_ylabel(r'$\Omega_i t$', rotation=0)
    ax.set_xlabel('k (m-number)')
    
    plt.xlim(None, 32)
    fullpath = anal_dir + saveas + '.png'
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print 'K-T Plot saved'
    return


def plot_wk(arr, norm, saveas='dispersion_relation'):
    print 'Plotting dispersion relation...'
    plt.ioff()
    num_times = arr.shape[0]
    
    df = 1. / (num_times * data_ts)
    dk = 1. / (NX * dx)

    if norm == True:     
        f  = np.arange(0, 1. / (2*dt), df) / gyfreq
        k  = np.arange(0, 1. / (2*dx), dk) * c / wpi 
        
        xlab = r'$kc/\omega_i$'
        ylab = r'$\omega / \Omega_i$'
    else:
        f  = np.arange(0, 1. / (2*dt), df)
        k  = np.arange(0, 1. / (2*dx), dk) * 1e6
        
        xlab = r'$k (\times 10^{-6}m^{-1})$'
        ylab = r'f (Hz)'

    fft_matrix  = np.zeros(arr.shape, dtype='complex128')
    fft_matrix2 = np.zeros(arr.shape, dtype='complex128')

    for ii in range(arr.shape[0]): # Take spatial FFT at each time
        fft_matrix[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())

    for ii in range(arr.shape[1]):
        fft_matrix2[:, ii] = np.fft.fft(fft_matrix[:, ii] - fft_matrix[:, ii].mean())

    dispersion_plot = fft_matrix2[:f.shape[0], :k.shape[0]] * np.conj(fft_matrix2[:f.shape[0], :k.shape[0]])

    fig = plt.figure(1, figsize=(12, 8))
    ax  = fig.add_subplot(111)
    
    ax.pcolormesh(k[1:], f[1:], np.log10(dispersion_plot[1:, 1:].real), cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k]
    #ax.pcolormesh(np.log10(dispersion_plot[1:, 1:].real), cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k]

    ax.set_title(r'$\omega/k$ plot (Predictor-Corrector)', fontsize=14)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)

    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.4)

    fullpath = anal_dir + saveas + '.png'
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print 'Dispersion Plot saved'
    return


def waterfall_plot(field):
    plt.ioff()
    
    arr = get_array()
    amp   = 100.                 # Amplitude multiplier of waves: 
    
    cells  = np.arange(NX)
    
    for (ii, t) in zip(np.arange(num_files), np.arange(0, num_files*data_ts, data_ts)):
        plt.plot(cells, amp*(arr[ii] / arr.max()) + ii, c='k', alpha=0.05)
        
    plt.xlim(0, NX)
    plt.show()
    return


def winske_stackplot(qq, title=None):
#----- Prepare some values for plotting
    x_cell_num  = np.arange(NX)                                         # Numerical cell numbering: x-axis
    
    pos         = position / dx
    norm_xvel   = 1e3*velocity[0, :] / c                                # Normalize to speed of light
    norm_yvel   = 1e3*velocity[1, :] / c
    By          = B[  1: NX + 1, 1] / B0                                # Normalize to background field
    Bz          = B[  1: NX + 1, 2] / B0
    dnb         = dns[1: NX + 1, 1] / density[1]                        # Normalize beam density to initial value
    phi         = np.arctan2(Bz, By) + pi                               # Wave magnetic phase angle
    
#----- Create plots
    plt.ioff()
    fig    = plt.figure(1, figsize=(8.27, 11.69))                       # Initialize figure
    grids  = gs.GridSpec(5, 1)                                          # Create gridspace
    fig.patch.set_facecolor('w')                                        # Set figure face color

    ax_vx   = fig.add_subplot(grids[0, 0]) 
    ax_vy   = fig.add_subplot(grids[1, 0]) 
    ax_den  = fig.add_subplot(grids[2, 0])                              # Initialize axes
    ax_by   = fig.add_subplot(grids[3, 0]) 
    ax_phi  = fig.add_subplot(grids[4, 0]) 

    if title != None and type(title) == str:
        ax_vx.set_title(title)

    ax_vx.scatter(pos[idx_bounds[1, 0]: idx_bounds[1, 1]], norm_xvel[idx_bounds[1, 0]: idx_bounds[1, 1]], s=1, c='k', lw=0)        # Hot population
    ax_vy.scatter(pos[idx_bounds[1, 0]: idx_bounds[1, 1]], norm_yvel[idx_bounds[1, 0]: idx_bounds[1, 1]], s=1, c='k', lw=0)        # 'Other' population
    
    ax_den.plot(x_cell_num, dnb, c='k')                                 # Create overlayed plots for densities of each species
    ax_by.plot(x_cell_num, 10*By, c='k')
    ax_phi.plot(x_cell_num, phi, c='k')

    ax_vx.set_ylim(- 1.41 , 1.41) 
    ax_vy.set_ylim(- 1.41 , 1.41) 
    ax_den.set_ylim( 0.71 , 1.39)                                         # Initialize axes
    ax_by.set_ylim(- 5.99 , 4.55) 
    ax_phi.set_ylim( 0.01 , 6.24) 

    ax_vx.set_yticks( [-1.41, -0.71, 0.00, 0.71, 1.41])
    ax_vy.set_yticks( [-1.41, -0.71, 0.00, 0.71, 1.41])
    ax_den.set_yticks([0.71, 0.88, 1.05, 1.22, 1.39])
    ax_by.set_yticks( [-5.99, -3.36, -0.72, 1.91, 4.55])
    ax_phi.set_yticks([0.01, 1.57, 3.13, 4.68, 6.24])

    ax_vx.set_ylabel('VX ($x 10^{-3}$)', rotation=90)
    ax_vy.set_ylabel('VY ($x 10^{-3}$)', rotation=90)
    ax_den.set_ylabel('DNB', rotation=90)
    ax_by.set_ylabel('BY ($x 10^{-1}$)', rotation=90)
    ax_phi.set_ylabel('PHI', rotation=90)

    ax_phi.set_xlim(0, 128)
    ax_phi.set_xlabel('X (CELL)')

    plt.setp(ax_vx.get_xticklabels(), visible=False)
    #ax_vx.set_yticks(ax_vx.get_yticks()[1:])

    for ax in [ax_vx, ax_vy, ax_den, ax_by]:
        plt.setp(ax.get_xticklabels(), visible=False)
        #ax.set_yticks(ax.get_yticks()[1:])
        ax.set_xlim(0, 128)

#----- Plot adjustments
    fig.text(0.42, 0.045, 'IT = {}'.format(data_dump_iter * qq), fontsize=13)    
    fig.text(0.58, 0.045, 'T = %.2f' % (data_dump_iter * qq * dt * gyfreq), fontsize=13)
    
    #plt.tight_layout(pad=1.0, w_pad=1.8)
    fig.subplots_adjust(hspace=0.1)

#----- Save plots
    filename = 'stackplot%05d.png' % qq
    path     = anal_dir + '/stackplot/'
    
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
        
    fullpath = path + filename
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close('all')
    return


def get_gyrophase(vel):
    gyro = np.arctan2(vel[:, 1], vel[:, 2])
    return gyro


def plot_energies(ii):
    
    U_B = 0.5 * (1 / mu0) * np.square(B[1:-1]).sum()     # Magnetic potential energy 
    
    for jj in range(Nj):
        vp2 = velocity[0, idx_bounds[jj, 0]:idx_bounds[jj, 1]] ** 2 \
            + velocity[1, idx_bounds[jj, 0]:idx_bounds[jj, 1]] ** 2 \
            + velocity[2, idx_bounds[jj, 0]:idx_bounds[jj, 1]] ** 2
            
        K_E = 0.5 * mass[jj] * vp2.sum()                 # Particle total kinetic energy 
        particle_energy[ii, jj] = K_E
               
        
    mag_energy[ii]      = U_B
    
    if ii == num_files - 1:
        fig  = plt.figure()
        
        plt.plot(time_radperiods, mag_energy / mag_energy.max(), label = r'$U_B$')
        
        for jj in range(Nj):
            plt.plot(time_radperiods, particle_energy[:, jj] / particle_energy[:, jj].max(), label='$K_E$ {}'.format(species[jj]))
        
        plt.legend()
        plt.title('Energy Distribution in Simulation')
        plt.xlabel('Time ($\Omega t$)')
        plt.xlim(0, 100)
        plt.ylabel('Normalized Energy', rotation=90)
        fullpath = anal_dir + 'energy_plot'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')
        
        print 'Energy plot saved'
    return


if __name__ == '__main__':   
    drive    = 'E:'
    series   = 'CAM_CL_test2'                               # Run identifier string 
    run_num  = 0                                            # Run number

    manage_dirs()                                           # Initialize directories
    load_constants()                                        # Load SI constants
    load_particles()                                        # Load particle parameters
    load_header()                                           # Load simulation parameters
    
    num_files = len(os.listdir(data_dir)) - 2               # Number of timesteps to load
    wpi       = np.sqrt(ne * q ** 2 / (mp * e0))            # Ion plasma frequency
    gyfreq    = q * B0 / mp                                 # Proton gyrofrequency (rad/s)
    gyperiod  = (mp * 2 * np.pi) / (q * B0)                 # Proton gyroperiod (s)
    data_ts   = data_dump_iter * dt                         # Timestep between data records (seconds)
    
    time_seconds    = np.arange(0, num_files * data_ts, data_ts)
    time_gperiods   = time_seconds / gyperiod
    time_radperiods = time_seconds * gyfreq 
    
    np.random.seed(seed)                                    # Initialize random seed
    
    mag_energy      = np.zeros(num_files)
    particle_energy = np.zeros((num_files, Nj))

    #generate_kt_plot(normalize=True)
        
    for ii in range(num_files):
        position, velocity, B, E, q_dns, J = load_timestep(ii)
        dns                                = collect_density(position)
        
        plot_energies(ii)
        
        #winske_stackplot(ii, title=r'Winske test check: $\Delta t$ = 0.02$\omega^{-1}$, Smoothing ON')

    