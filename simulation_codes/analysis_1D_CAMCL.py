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

    print('Particle parameters loaded')
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

    print('Header file loaded.')
    print('dt = {}s\n'.format(dt))
    return 


def load_timestep(ii):
    print('Loading file {} of {}'.format(ii+1, num_files))
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
def create_idx():
    N_part = cellpart * NX
    idx    = np.zeros(N_part, dtype=nb.int32)
    
    for jj in range(Nj):
        idx[idx_bounds[jj, 0]: idx_bounds[jj, 1]] = jj
    return idx


@nb.njit()
def manage_ghost_cells(arr):
    '''Deals with ghost cells: Moves their contributions and mirrors their counterparts.
       Works like a charm if spatial dimensions always come first in an array.'''

    arr[NX]     += arr[0]                 # Move contribution: Start to end
    arr[1]      += arr[NX + 1]            # Move contribution: End to start

    arr[NX + 1]  = arr[1]                 # Fill ghost cell: End
    arr[0]       = arr[NX]                # Fill ghost cell: Start
    
    arr[NX + 2]  = arr[2]                 # This one doesn't get used, but prevents nasty nan's from being in array.
    return arr


@nb.njit()
def assign_weighting_TSC(pos, E_nodes=True):
    '''Triangular-Shaped Cloud (TSC) weighting scheme used to distribute particle densities to
    nodes and interpolate field values to particle positions.

    INPUT:
        pos  -- particle positions (x)
        BE   -- Flag: Weighting factor for Magnetic (0) or Electric (1) field node
        
    OUTPUT:
        weights -- 3xN array consisting of leftmost (to the nearest) node, and weights for -1, 0 TSC nodes
    '''
    Np         = pos.shape[0]
    
    left_node  = np.zeros(Np,      dtype=np.uint16)
    weights    = np.zeros((3, Np), dtype=np.float64)
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 1.0
    
    for ii in nb.prange(Np):
        left_node[ii]  = int(round(pos[ii] / dx + grid_offset) - 1.0)
        delta_left     = left_node[ii] - pos[ii] / dx - grid_offset
    
        weights[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))
        weights[1, ii] = 0.75 - np.square(delta_left + 1.)
        weights[2, ii] = 1.0  - weights[0, ii] - weights[1, ii]
    return left_node, weights

@nb.njit()
def collect_moments(Ie, W_elec, idx):
    n_contr   = density / (cellpart*sim_repr)
    size      = NX + 3
    n_i       = np.zeros((size, Nj))
    
    for ii in nb.prange(Ie.shape[0]):
        I   = Ie[ ii]
        sp  = idx[ii]
        
        n_i[I,     sp] += W_elec[0, ii]
        n_i[I + 1, sp] += W_elec[1, ii]
        n_i[I + 2, sp] += W_elec[2, ii]
        
    for jj in range(Nj):
        n_i[:, jj] *= n_contr[jj]

    n_i   = manage_ghost_cells(n_i)
    return n_i
    
@nb.njit()
def collect_number_density(pos):
    '''Collect number and velocity density in each cell at each timestep, weighted by their distance
    from cell nodes.

    INPUT:
        pos    -- position of each particle
    '''


    left_node, weights  = assign_weighting_TSC(pos, E_nodes=True) 
    idx                 = create_idx()
    den                 = collect_moments(left_node, weights, idx)   
    return den


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
        print('Array file for {} loaded from memory...'.format(component.upper()))
        arr = np.load(check_path)   
    else:
        for ii in range(tmin, tmax):
            B, E, Ve, Te, J, position, q_dns, velocity = load_timestep(ii)
            
            if component[0].upper() == 'B':
                arr[ii] = B[0:-1, comp_idx]
            elif component[0].upper() == 'E':
                arr[ii] = E[1:-1, comp_idx]
                
        print('Saving array file as {}'.format(check_path))
        np.save(check_path, arr)
    return arr


def extract_all_arrays():
    for comp in ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']:
        get_array(comp)
    return


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
    
    ax.pcolormesh(k[:arr.shape[1] / 2], time_gperiods, kt, cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True

    ax.set_title(r'k-t Plot (CAM-CL)', fontsize=14)
    ax.set_ylabel(r'$\Omega_i t$', rotation=0)
    ax.set_xlabel('k (m-number)')
    
    plt.xlim(None, 32)
    fullpath = anal_dir + saveas + '.png'
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print('K-T Plot saved')
    return


def plot_wk(arr, norm, saveas='dispersion_relation'):
    print('Plotting dispersion relation...')
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
    print('Dispersion Plot saved')
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


def diagnostic_multiplot(qq):
    plt.ioff()

    fig_size = 4, 7                                                             # Set figure grid dimensions
    fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
    fig.patch.set_facecolor('w')                                                # Set figure face color

    va        = B0 / np.sqrt(mu0*ne*mp)                                         # Alfven speed: Assuming pure proton plasma

    pos       = position / dx                                                   # Cell particle position
    vel       = velocity / va                                                   # Normalized velocity

    den_norm  = dns / density                                                   # Normalize density for each species to initial values
    qdens_norm= q_dns / (density*charge).sum()                                  # Normalized change density
     
#----- Velocity (x, y) Plots: Hot Species
    ax_vx   = plt.subplot2grid(fig_size, (0, 0), rowspan=2, colspan=3)
    ax_vy   = plt.subplot2grid(fig_size, (2, 0), rowspan=2, colspan=3)

    for jj in range(Nj):
        ax_vx.scatter(pos[idx_bounds[jj, 0]: idx_bounds[jj, 1]], vel[0, idx_bounds[jj, 0]: idx_bounds[jj, 1]], s=3, c=temp_color[jj], lw=0, label=species_lbl[jj])
        ax_vy.scatter(pos[idx_bounds[jj, 0]: idx_bounds[jj, 1]], vel[1, idx_bounds[jj, 0]: idx_bounds[jj, 1]], s=3, c=temp_color[jj], lw=0)

    ax_vx.legend()
    ax_vx.set_title(r'Particle velocities vs. Position (x)')
    ax_vy.set_xlabel(r'Cell', labelpad=10)

    ax_vx.set_ylabel(r'$\frac{v_x}{c}$', rotation=90)
    ax_vy.set_ylabel(r'$\frac{v_y}{c}$', rotation=90)

    plt.setp(ax_vx.get_xticklabels(), visible=False)
    ax_vx.set_yticks(ax_vx.get_yticks()[1:])

    for ax in [ax_vy, ax_vx]:
        ax.set_xlim(0, NX)
        ax.set_ylim(-10, 10)

#----- Density Plot
    ax_den = plt.subplot2grid((fig_size), (0, 3), colspan=3)                     # Initialize axes
    
    ax_den.plot(qdens_norm, color='green')                                       # Create overlayed plots for densities of each species

    for jj in range(Nj):
        ax_den.plot(den_norm, color=temp_color[jj])
        
    ax_den.set_title('Normalized Densities and Fields')                          # Axes title (For all, since density plot is on top
    ax_den.set_ylabel(r'$\frac{n_i}{n_0}$', fontsize=14, rotation=0, labelpad=5) # Axis (y) label for this specific axes
    ax_den.set_ylim(0, 2)
    
#----- E-field (Ex) Plot
    ax_Ex = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_den)

    ax_Ex.plot(E[:, 0], color='red', label=r'$E_x$')
    ax_Ex.plot(E[:, 1], color='cyan', label=r'$E_x$')
    ax_Ex.plot(E[:, 2], color='black', label=r'$E_x$')

    ax_Ex.set_xlim(0, NX)

    #ax_Jx.set_yticks(np.arange(-200e-5, 201e-5, 50e-5))
    #ax_Jx.set_yticklabels(np.arange(-150, 201, 50))
    ax_Ex.set_ylabel(r'$E$', labelpad=25, rotation=0, fontsize=14)

#----- Magnetic Field (By) and Magnitude (|B|) Plots
    ax_By = plt.subplot2grid((fig_size), (2, 3), colspan=3, sharex=ax_den)
    ax_B  = plt.subplot2grid((fig_size), (3, 3), colspan=3, sharex=ax_den)

    mag_B  = (np.sqrt(B[:-1, 0] ** 2 + B[:-1, 1] ** 2 + B[:-1, 2] ** 2)) / B0
    B_norm = B[:-1, :] / B0                                                           

    ax_B.plot(mag_B, color='g')                                                        # Create axes plots
    ax_By.plot(B_norm[:, 1], color='g') 
    ax_By.plot(B_norm[:, 2], color='b') 

    ax_B.set_xlim(0,  NX)                                                               # Set x limit
    ax_By.set_xlim(0, NX)

    ax_B.set_ylim(0, 2)                                                                 # Set y limit
    ax_By.set_ylim(-1, 1)

    ax_B.set_ylabel( r'$|B|$', rotation=0, labelpad=20, fontsize=14)                    # Set labels
    ax_By.set_ylabel(r'$\frac{B_{y,z}}{B_0}$', rotation=0, labelpad=10, fontsize=14)
    ax_B.set_xlabel('Cell Number')                                                      # Set x-axis label for group (since |B| is on bottom)

    for ax in [ax_den, ax_Ex, ax_By]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_yticks(ax.get_yticks()[1:])

    for ax in [ax_den, ax_Ex, ax_By, ax_B]:
        qrt = NX / (4.)
        ax.set_xticks(np.arange(0, NX + qrt, qrt))
        ax.grid()

#----- Plot Adjustments
    plt.tight_layout(pad=1.0, w_pad=1.8)
    fig.subplots_adjust(hspace=0)

    filename = 'diag%05d.png' % ii
    path     = anal_dir + '/diagnostic_plot/'
    
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
        
    fullpath = path + filename
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
    print('Plot saved'.format(ii))
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
            ax.plot(time_gperiods, mag_energy      / mag_energy[0],      label = r'$U_B$', c='green')
            ax.plot(time_gperiods, electron_energy / electron_energy[0], label = r'$U_e$', c='orange')
            ax.plot(time_gperiods, total_energy    / total_energy[0],    label = r'$Total$', c='k')
            
            for jj in range(Nj):
                ax.plot(time_gperiods, particle_energy[:, jj] / particle_energy[0, jj],
                         label='$K_E$ {}'.format(species_lbl[jj]), c=temp_color[jj])
        else:
            ax.plot(time_gperiods, mag_energy     , label = r'$U_B$', c='green')
            ax.plot(time_gperiods, electron_energy, label = r'$U_e$', c='orange')
            ax.plot(time_gperiods, total_energy   , label = r'$Total$', c='k')
            
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
        
        print('Energy plot saved')
    return


if __name__ == '__main__':   
    drive      = 'G://MODEL_RUNS//Josh_Runs//'
    series     = 'ev1_lowbeta'                                  # Run identifier string 
    series_dir = '{}/runs//{}//'.format(drive, series)
    num_runs   = len([name for name in os.listdir(series_dir) if 'run_' in name])
    #run_num    = 0 

    for run_num in range(num_runs):
        manage_dirs()                                           # Initialize directories
        load_constants()                                        # Load SI constants
        load_header()                                           # Load simulation parameters
        load_particles()                                        # Load particle parameters
        
        num_files = len(os.listdir(data_dir)) - 2               # Number of timesteps to load
    
        wpi       = np.sqrt(ne * q ** 2 / (mp * e0))            # Ion plasma frequency
        gyfreq    = q * B0 / mp                                 # Proton gyrofrequency (rad/s)
        gyperiod  = (mp * 2 * np.pi) / (q * B0)                 # Proton gyroperiod (s)
        data_ts   = data_dump_iter * dt                         # Timestep between data records (seconds)
        
        time_seconds    = np.array([ii * data_ts for ii in range(num_files)])
        time_gperiods   = time_seconds / gyperiod
        #time_radperiods = time_seconds * gyfreq 
        
        np.random.seed(seed)                                    # Initialize random seed
        
        mag_energy      = np.zeros(num_files)
        particle_energy = np.zeros((num_files, Nj))
        electron_energy = np.zeros(num_files)
    
        generate_kt_plot(normalize=True)
            
        if os.path.exists(anal_dir + 'norm_energy_plot.png') == False:
            for ii in range(num_files):
                B, E, Ve, Te, J, position, q_dns, velocity = load_timestep(ii)
                #dns                                        = collect_number_density(position)
                plot_energies(ii, normalize=True)
        
        

    