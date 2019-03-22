# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:56:34 2016

@author: c3134027
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from numpy import pi
import pickle
import numba as nb
from collections import OrderedDict
from matplotlib.lines import Line2D
import tabulate
import pdb


def manage_dirs(create_new=True):
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
    return


def load_header():
    global Nj, cellpart, data_dump_iter, ne, NX, dxm, seed, B0, dx, Te0, theta, dt, max_rev,\
           ie, run_desc, seed, subcycles, LH_frac, orbit_res, freq_res, particle_shape, method_type, method_type
    
    h_name = os.path.join(data_dir, 'Header.pckl')                      # Load header file
    f      = open(h_name)                                               # Open header file
    obj    = pickle.load(f)                                             # Load variables from header file into python object
    f.close()                                                           # Close header file
    
    #pdb.set_trace()
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
    max_rev         = obj['max_rev']
    ie              = obj['ie']
    orbit_res       = obj['orbit_res']
    freq_res        = obj['freq_res'] 
    run_desc        = obj['run_desc']
    particle_shape  = obj['particle_shape']
    method_type     = obj['method_type']
    
    if method_type == 'CAM_CL':
        LH_frac         = obj['LH_frac']
        subcycles       = obj['subcycles']
    elif method_type == 'PREDCORR':
        pass
    return 


def create_idx():
    N_part = cellpart * NX
    idx    = np.zeros(N_part, dtype=int)
    
    for jj in range(Nj):
        idx[idx_bounds[jj, 0]: idx_bounds[jj, 1]] = jj
    return idx

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
    tidx             = create_idx()
    return tB, tE, tVe, tTe, tJ, tpos, tdns, tvel, tidx


@nb.njit()
def collect_number_density(pos):
    '''Collect number and velocity density in each cell at each timestep, weighted by their distance
    from cell nodes.

    INPUT:
        pos    -- position of each particle
    '''
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
    def collect_moments(vel, Ie, W_elec, idx):
        n_contr   = density / (cellpart*sim_repr)
        size      = NX + 3
        n_i       = np.zeros((size, Nj))
        
        for ii in nb.prange(vel.shape[1]):
            I   = Ie[ ii]
            sp  = idx[ii]
            
            n_i[I,     sp] += W_elec[0, ii]
            n_i[I + 1, sp] += W_elec[1, ii]
            n_i[I + 2, sp] += W_elec[2, ii]
            
        for jj in range(Nj):
            n_i[:, jj] *= n_contr[jj]
    
        n_i   = manage_ghost_cells(n_i)
        return n_i

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
            B, E, Ve, Te, J, position, q_dns, velocity, idx = load_timestep(ii)
            
            if component[0].upper() == 'B':
                arr[ii] = B[:-1, comp_idx]
            elif component[0].upper() == 'E':
                arr[ii] = E[:, comp_idx]
                
        print('Saving array file as {}'.format(check_path))
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
    time_gperiods    = np.array([ii * data_ts for ii in range(num_files)])  / gyperiod
    energy_output[0] = time_gperiods        
    
    for ii in range(num_files):
        B, E, Ve, Te, J, position, q_dns, velocity, idx = load_timestep(ii)
            
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


def plot_growth_grid(plot_ratio=True, norm=False):
    '''
    Density on x axis
    Field (B0) on y axis
    Growth (max b1) coded in color
    '''
    if norm == False:
        add  = '(nT)'
    else:
        add  = r'/$B_0$'
        
    # Turn field values into colors to plot
    cmap      = matplotlib.cm.get_cmap('nipy_spectral')
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=max_field[2].max())
    colors    = [cmap(normalize(value)) for value in max_field[2]]
    
    # Plot model values
    fig, ax = plt.subplots(figsize=(15,10))
    ax.scatter(max_field[0], max_field[1], color=colors)
    
    if plot_ratio == True:
        npts = 100
        rat  = np.sqrt(e0 / mp)
        
        B_min  = 0e-9
        B_max  = 300e-9
        B_axis = np.linspace(B_min, B_max, npts)
        
        powers = list(range(1, 5))
        for pwr in powers:
            #sqrt_n = (rat * (10 ** pwr) * B_axis) / 1e3
            n      = (rat * (10 ** pwr) * B_axis) ** 2 / 1e6
            plt.plot(n, B_axis*1e9, label='$cv_A^{-1} = 10^%d$' % pwr)
    
    ## PLOT DATA POINTS ##
    n_data = np.array([38, 160, 38, 160])
    B_data = np.array([158, 158, 134, 134])
    plt.scatter(n_data, B_data, label='Event Data Limits', marker='x')
    plt.legend()
    
    ## LABELS AND LIMITS ##
    plt.xlim(0, 500)
    plt.ylim(B_min*1e9, B_max*1e9)
    
    plt.xlabel(r'$n_0 ({cm^{-3}})$')
    plt.ylabel(r'$B_0 (nT)$')
    

        
    plt.title('Event 1: Max |By|{} for varying B0, ne'.format(add))
    
    # Optionally add a colorbar
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar   = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
    cbar.set_label('|By|{}'.format(add), rotation=0)
    
    plt.show()
    return


def examine_run_parameters():
    '''
    Diagnostic information to compare runs at a glance. Values include
    
    N
    Nj
    NX
    B0
    ne
    
    number of files
    '''
    global run_num, num_files
    
    run_params = ['cellpart', 'Nj', 'B0', 'ne', 'NX', 'num_files', 'Te0', 'max_rev', 'ie', 'theta', 'dxm']
    run_head   = []
    run_dict = {}
    for param in run_params:
        run_dict[param] = []
    
    for run_num in range(num_runs):
        manage_dirs(create_new=False)
        load_header()                                           # Load simulation parameters
        load_particles()                                        # Load particle parameters
        num_files = len(os.listdir(data_dir)) - 2               # Number of timesteps to load
        run_head.append('Run {}'.format(run_num))
        
        for param in run_params:
            run_dict[param].append(globals()[param])
        
        #table = [["Sun",696000,1989100000],["Earth",6371,5973.6], ["Moon",1737,73.5],["Mars",3390,641.85]]
    
    print('\n')
    print('Simulation parameters for runs in series \'{}\':'.format(series))
    print('\n')
    print((tabulate.tabulate(run_dict, headers="keys")))
    return

if __name__ == '__main__':   
    plt.ioff()
    
    plot_energy_comparison = False
    
    drive      = 'F:'#'/media/yoshi/UNI_HD/'
    series     = 'compare_two'                                  # Run identifier string 
    series_dir = '{}/runs//{}//'.format(drive, series)
    num_runs   = len([name for name in os.listdir(series_dir) if 'run_' in name])
    
    examine_run_parameters()
    
    for run_num in range(num_runs):
        manage_dirs()                                           # Initialize directories
        load_constants()                                        # Load SI constants
        load_header()                                           # Load simulation parameters
        load_particles()                                        # Load particle parameters
        num_files = len(os.listdir(data_dir)) - 2               # Number of timesteps to load
        max_field  = np.zeros((3, num_runs))                        # B0, n0, max_By

        BY = get_array('By', 0, None)
        
        max_field[0, run_num] = ne / 1e6                        # Density in cc
        max_field[1, run_num] = B0 * 1e9                        # B0 in nT
        max_field[2, run_num] = abs(BY).max() * 1e9             # Max abs(By) in nT
        
    plot_growth_grid(norm=True)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    if plot_energy_comparison == True:
        runs_to_analyse = [4, 5, 6, 7]                        # Run number
        run_styles      = ['-' , '--', ':', '-.']
        
        run_labels      = ['1k',
                           '5k',
                           '10k',
                           '20k']
        
        energy_suffix = '_CAM_CL'
        
        total_energies  = np.zeros(len(runs_to_analyse))
        
        fig  = plt.figure(figsize=(15, 7))
        ax   = plt.subplot2grid((7, 7), (0, 0), colspan=6, rowspan=7)
        run_legend, ax  = create_legend(ax)
            
        fsize = 10; fname='monospace'; left = 0.86
        plt.figtext(left, 0.50, r'TOTAL ENERGY CONSERVATION',    fontsize=fsize, fontname=fname)
        plt.figtext(left, 0.50,  '_________________________',     fontsize=fsize, fontname=fname)
    
        for run_num, ii in zip(runs_to_analyse, list(range(len(runs_to_analyse)))):
            manage_dirs()                                           # Initialize directories
            load_constants()                                        # Load SI constants
            load_header()                                           # Load simulation parameters
            load_particles()                                        # Load particle parameters
            num_files = len(os.listdir(data_dir)) - 2               # Number of timesteps to load
            
            wpi       = np.sqrt(ne * q ** 2 / (mp * e0))            # Ion plasma frequency
            gyfreq    = q * B0 / mp                                 # Proton gyrofrequency (rad/s)
            gyperiod  = (mp * 2 * np.pi) / (q * B0)                 # Proton gyroperiod (s)
            data_ts   = data_dump_iter * dt                         # Timestep between data records (seconds)
                   
            print('Analysing run {}'.format(run_num))
            energies = get_run_energies()
    
            plot_energies(energies, ax)
            
            total_energy_change = 100.*(energies[1, -1] - energies[1, -0]) / energies[1, 0]
            plt.figtext(left, 0.475-ii*0.02, '{:>8} : {:>7}%'.format(method_type, round(total_energy_change, 2)),  fontsize=fsize,  fontname=fname)
    
        fig.tight_layout()
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(list(zip(labels, handles)))
        ax.legend(list(by_label.values()), list(by_label.keys()), loc='center left', bbox_to_anchor=(1, 0.8))
    
        ax.add_artist(run_legend)
    
        fullpath = base_dir + 'energy_plot' + energy_suffix
        ax.set_ylim(0.85, 1.4)
        ax.set_xlim(0, 15)
        fig.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')
        
