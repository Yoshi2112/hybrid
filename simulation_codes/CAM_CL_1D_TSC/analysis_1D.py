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
           ie, run_desc, seed, subcycles, LH_frac, orbit_res, freq_res
    
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


def plot_energies(ii, normalize=True):
    
    mag_energy[ii]      = (0.5 / mu0) * np.square(B[1:-2]).sum() * NX * dx    # Magnetic potential energy 
    electron_energy[ii] = 1.5 * (kB * Te * q_dns / q).sum() * NX * dx         # Electron pressure energy
    
    for jj in range(Nj):
        vp2 = velocity[0, idx_bounds[jj, 0]:idx_bounds[jj, 1]] ** 2 \
            + velocity[1, idx_bounds[jj, 0]:idx_bounds[jj, 1]] ** 2 \
            + velocity[2, idx_bounds[jj, 0]:idx_bounds[jj, 1]] ** 2           # Total real particle kinetic energy
            
        particle_energy[ii, jj] = 0.5 * mass[jj] * vp2.sum() * n_contr[jj] * NX * dx 
               
    if ii == num_files - 1:
        total_energy = mag_energy + particle_energy.sum(axis=1) + electron_energy
                
        fig  = plt.figure(figsize=(15, 7))
        ax   = plt.subplot2grid((7, 7), (0, 0), colspan=6, rowspan=7)
        
        if normalize == True:
            ax.plot(time_radperiods, mag_energy      / mag_energy[0],      label = r'$U_B$', c='green')
            ax.plot(time_radperiods, electron_energy / electron_energy[0], label = r'$U_e$', c='orange')
            ax.plot(time_radperiods, total_energy    / total_energy[0],    label = r'$Total$', c='k')
            
            for jj in range(Nj):
                ax.plot(time_radperiods, particle_energy[:, jj] / particle_energy[0, jj],
                         label='$K_E$ {}'.format(species_lbl[jj]), c=temp_color[jj])
        else:
            ax.plot(time_radperiods, mag_energy     , label = r'$U_B$', c='green')
            ax.plot(time_radperiods, electron_energy, label = r'$U_e$', c='orange')
            ax.plot(time_radperiods, total_energy   , label = r'$Total$', c='k')
            
            for jj in range(Nj):
                ax.plot(time_gperiods, particle_energy[:, jj],
                        label='$K_E$ {}'.format(species_lbl[jj]), c=temp_color[jj])
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))
        fig.tight_layout()
        
        
        percent_ion = np.zeros(Nj)
        for jj in range(Nj):
            percent_ion[jj] = round(100.*(particle_energy[-1, jj] - particle_energy[0, jj]) / particle_energy[0, jj], 2)
        
        percent_elec  = round(100.*(electron_energy[-1] - electron_energy[0]) / electron_energy[0], 2)
        percent_mag   = round(100.*(mag_energy[-1]      - mag_energy[0])      / mag_energy[0], 2)
        percent_total = round(100.*(total_energy[-1]    - total_energy[0])    / total_energy[0], 2)
            
        fsize = 14; fname='monospace'
        plt.figtext(0.85, 0.92, r'$\Delta E$ OVER RUNTIME',            fontsize=fsize+2, fontname=fname)
        plt.figtext(0.85, 0.92, '________________________',            fontsize=fsize+2, fontname=fname)
        plt.figtext(0.85, 0.88, 'TOTAL   : {:>7}%'.format(percent_total),  fontsize=fsize,  fontname=fname)
        plt.figtext(0.85, 0.84, 'MAGNETIC: {:>7}%'.format(percent_mag),    fontsize=fsize,  fontname=fname)
        plt.figtext(0.85, 0.80, 'ELECTRON: {:>7}%'.format(percent_elec),   fontsize=fsize,  fontname=fname)
        
        for jj in range(Nj):
            plt.figtext(0.85, 0.76-jj*0.04, 'ION{}    : {:>7}%'.format(jj, percent_ion[jj]), fontsize=fsize,  fontname=fname)
        
        ax.set_xlabel('Time (Gyroperiods)')
        ax.set_xlim(0, time_gperiods[-1])
        
        if normalize == True:
            ax.set_title('Normalized Energy Distribution in Simulation Space')
            ax.set_ylabel('Normalized Energy', rotation=90)
            fullpath = anal_dir + 'norm_energy_plot'
            fig.subplots_adjust(bottom=0.07, top=0.96, left=0.04)
        else:
            ax.set_title('Energy Distribution in Simulation Space')
            ax.set_ylabel('Energy (Joules)', rotation=90)
            fullpath = anal_dir + 'energy_plot'
            fig.subplots_adjust(bottom=0.07, top=0.96, left=0.055)
        
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')
        
        print 'Energy plot saved'
    return


if __name__ == '__main__':   
    drive    = 'F:'
    series   = 'CAM_CL_TSC_winske'                          # Run identifier string 
    run_num  = 3                                            # Run number

    manage_dirs()                                           # Initialize directories
    load_constants()                                        # Load SI constants
    load_header()                                           # Load simulation parameters
    load_particles()                                        # Load particle parameters
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
    electron_energy = np.zeros(num_files)

    #generate_kt_plot(normalize=True)
        
    for ii in range(num_files):
        B, E, Ve, Te, J, position, q_dns, velocity = load_timestep(ii)
        dns                                        = collect_number_density(position)
        #winske_stackplot(ii, title=r'CAM_CL_TSC /w Winske Parameters')
        plot_energies(ii, normalize=False)
        
        

    