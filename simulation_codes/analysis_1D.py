# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:56:34 2016

@author: c3134027
"""
import sys
data_scripts_dir = 'C://Users//iarey//Documents//GitHub//hybrid//linear_theory//'
sys.path.append(data_scripts_dir)

from convective_growth_rate import calculate_growth_rate
from chen_warm_dispersion   import get_dispersion_relation
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from numpy import pi
import pickle
import numba as nb
import pdb
import lmfit as lmf
import tabulate


def manage_dirs(create_new=True):
    global run_dir, data_dir, anal_dir, temp_dir, base_dir
    
    base_dir = '{}/runs/{}/'.format(drive, series)                      # Main series directory, containing runs
    run_dir  = '{}/runs/{}/run_{}/'.format(drive, series, run_num)      # Main run directory
    data_dir = run_dir + 'data/'                                        # Directory containing .npz output files for the simulation run
    anal_dir = run_dir + 'analysis/'                                    # Output directory for all this analysis (each will probably have a subfolder)
    temp_dir = run_dir + 'temp/'                                        # Saving things like matrices so we only have to do them once

   # Make Output folder if they don't exist
    for this_dir in [anal_dir, temp_dir]:
        if os.path.exists(run_dir) == True:
            if os.path.exists(this_dir) == False:
                os.makedirs(this_dir)
        else:
            raise IOError('Run {} does not exist for series {}. Check range argument.'.format(run_num, series))
    return


# =============================================================================
# def ax_add_run_params(ax):
#     font    = 'monospace'
#     top     = 1.07
#     left    = 0.78
#     h_space = 0.04
#     
#     ## Simulation Parameters ##
#     
#     ## Particle Parameters ##
#     ax1.text(0.00, top - 0.02, '$B_0 = $variable'     , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(0.00, top - 0.05, '$n_0 = $variable' % n0, transform=ax1.transAxes, fontsize=10, fontname=font)
#     
#     ax1.text(left + 0.06,  top, 'Cold'    , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 0.099, top, 'Warm'    , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 0.143, top, '$A_i$'   , transform=ax1.transAxes, fontsize=10, fontname=font)
#     
#     ax1.text(left + 0.192, top, r'$\beta_{\parallel}$'                    , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 4.2*h_space, top - 0.02, '{:>7.2f}'.format(betapar[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 4.2*h_space, top - 0.04, '{:>7.2f}'.format(betapar[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 4.2*h_space, top - 0.06, '{:>7.2f}'.format(betapar[2]), transform=ax1.transAxes, fontsize=10, fontname=font)
# 
#     ax1.text(left + 0.49*h_space, top - 0.02, ' H+:'                    , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 1*h_space, top - 0.02, '{:>7.3f}'.format(H_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 2*h_space, top - 0.02, '{:>7.3f}'.format(H_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 3*h_space, top - 0.02, '{:>7.2f}'.format(A[0]),      transform=ax1.transAxes, fontsize=10, fontname=font)
#     
#     ax1.text(left + 0.49*h_space, top - 0.04, 'He+:'                     , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 1*h_space, top - 0.04, '{:>7.3f}'.format(He_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 2*h_space, top - 0.04, '{:>7.3f}'.format(He_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 3*h_space, top - 0.04, '{:>7.2f}'.format(A[1])      , transform=ax1.transAxes, fontsize=10, fontname=font)
#     
#     ax1.text(left + 0.49*h_space, top - 0.06, ' O+:'                    , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 1*h_space, top - 0.06, '{:>7.3f}'.format(O_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 2*h_space, top - 0.06, '{:>7.3f}'.format(O_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 3*h_space, top - 0.06, '{:>7.2f}'.format(A[2])     , transform=ax1.transAxes, fontsize=10, fontname=font)
#     return
# =============================================================================

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
    global species_present, density, dist_type, idx_bounds, charge, mass, Tper, sim_repr, temp_type, temp_color, velocity, Tpar, species_lbl, n_contr

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

    n_contr    = density / (cellpart*sim_repr)                                  # Species density contribution: Each macroparticle contributes this density to a cell
    species_present = [False, False, False]                                     # Test for the presence of singly charged H, He, O
        
    for ii in range(Nj):
        if 'H^+' in species_lbl[ii]:
            species_present[0] = True
        elif 'He^+' in species_lbl[ii]:
            species_present[1] = True
        elif 'O^+'  in species_lbl[ii]:
            species_present[2] = True

    print('Particle parameters loaded')
    return


def load_header():
    global Nj, cellpart, data_dump_iter, ne, NX, dxm, seed, B0, dx, Te0, theta, dt_sim, max_rev,\
           ie, run_desc, seed, subcycles, LH_frac, orbit_res, freq_res, method_type, particle_shape, dt_slice

    h_name = os.path.join(data_dir, 'Header.pckl')                      # Load header file
    f      = open(h_name, 'rb')                                         # Open header file
    obj    = pickle.load(f)                                             # Load variables from header file into python object
    f.close()                                                           # Close header file
    
    seed            = obj['seed']
    Nj              = obj['Nj']
    dt_sim          = obj['dt']                                         # Simulation timestep (seconds)
    NX              = obj['NX']
    dxm             = obj['dxm']
    dx              = obj['dx']
    cellpart        = obj['cellpart']
    subcycles       = obj['subcycles']
    B0              = obj['B0']
    ne              = obj['ne']
    Te0             = obj['Te0']
    ie              = obj['ie']
    theta           = obj['theta']
    data_dump_iter  = obj['data_dump_iter']
    max_rev         = obj['max_rev']
    LH_frac         = obj['LH_frac']
    orbit_res       = obj['orbit_res']
    freq_res        = obj['freq_res']
    run_desc        = obj['run_desc']
    method_type     = obj['method_type'] 
    particle_shape  = obj['particle_shape'] 
    dt_slice        = dt_sim * data_dump_iter                           # Time between data slices (seconds)

    print('Simulation parameters loaded.')
    return 


def initialize_simulation_variables():
    global wpi, gyfreq, gyperiod, time_seconds, time_gperiods, time_radperiods
    wpi       = np.sqrt(ne * q ** 2 / (mp * e0))            # Ion plasma frequency
    gyfreq    = q * B0 / mp                                 # Proton gyrofrequency (rad/s)
    gyperiod  = (mp * 2 * np.pi) / (q * B0)                 # Proton gyroperiod (s)
    
    time_seconds    = np.array([ii * dt_slice for ii in range(num_files)])
    time_gperiods   = time_seconds / gyperiod
    time_radperiods = time_seconds * gyfreq 
    
    extract_all_arrays()
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

    try:
        global real_time
        real_time = data['real_time'][0]
    except:
        pass
    
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


def get_array(component):
    check_path = temp_dir + component.lower() + '_array' + '.npy'

    if os.path.isfile(check_path) == True:
        print('Array file for {} loaded from memory...'.format(component.upper()))
        arr = np.load(check_path)   
    else:
        extract_all_arrays()
        arr = np.load(check_path) 
    return arr


def extract_all_arrays():
    '''
    Extracts and saves all arrays separate from the timestep slice files for easy
    access. Note that magnetic field arrays exclude the last value due to periodic
    boundary conditions. This may be changed later.
    '''
    bx_arr   = np.zeros((num_files, NX)); ex_arr  = np.zeros((num_files, NX))
    by_arr   = np.zeros((num_files, NX)); ey_arr  = np.zeros((num_files, NX))
    bz_arr   = np.zeros((num_files, NX)); ez_arr  = np.zeros((num_files, NX))
    
    # Check that all components are extracted
    comps_missing = 0
    for component in ['bx', 'by', 'bz', 'ex', 'ey', 'ez']:
        check_path = temp_dir + component + '_array.npy'
        if os.path.isfile(check_path) == False:
            comps_missing += 1
    
    if comps_missing == 0:
        print('Field component arrays already extracted.')
        return
    else:
        for ii in range(num_files):
            B, E, Ve, Te, J, position, q_dns, velocity = load_timestep(ii)

            bx_arr[ii, :] = B[:-1, 0]; ex_arr[ii, :] = E[:, 0]
            by_arr[ii, :] = B[:-1, 1]; ey_arr[ii, :] = E[:, 1]
            bz_arr[ii, :] = B[:-1, 2]; ez_arr[ii, :] = E[:, 2]

        np.save(temp_dir + 'bx' +'_array.npy', bx_arr)
        np.save(temp_dir + 'by' +'_array.npy', by_arr)
        np.save(temp_dir + 'bz' +'_array.npy', bz_arr)
        
        np.save(temp_dir + 'ex' +'_array.npy', ex_arr)
        np.save(temp_dir + 'ey' +'_array.npy', ey_arr)
        np.save(temp_dir + 'ez' +'_array.npy', ez_arr)
        print('Field component arrays saved in {}'.format(temp_dir))
    return



def plot_wx(component='By', linear_overlay=False):
    plt.ioff()
    arr = get_array(component)
    
    x  = np.arange(NX)
    f  = np.fft.fftfreq(time_seconds.shape[0], d=dt_slice)
    
    proton_gyrofrequency = 1 / gyperiod
    helium_gyrofrequency = 0.25  * proton_gyrofrequency
    oxygen_gyrofrequency = 0.125 * proton_gyrofrequency
    
    ## CALCULATE IT
    fft_matrix  = np.zeros(arr.shape, dtype='complex128')
    for ii in range(arr.shape[1]):
        fft_matrix[:, ii] = np.fft.fft(arr[:, ii] - arr[:, ii].mean())
    
    wx = (fft_matrix[1:arr.shape[0] // 2, :] * np.conj(fft_matrix[1:arr.shape[0] // 2, :])).real
    
    ## PLOT IT
    fig = plt.figure(1, figsize=(15, 10))
    ax  = fig.add_subplot(111)
    
    ax.pcolormesh(x, f[1:arr.shape[0] // 2], wx, cmap='nipy_spectral')      # Remove f[0] since FFT[0] >> FFT[1, 2, ... , k]

    plt.axhline(proton_gyrofrequency, c='white')
    plt.axhline(helium_gyrofrequency, c='yellow')
    plt.axhline(oxygen_gyrofrequency, c='red')
    
    if linear_overlay == True:
        freqs, cgr, stop = get_cgr_from_sim()
        max_idx          = np.where(cgr == cgr.max())
        max_lin_freq     = freqs[max_idx]
        plt.axhline(max_lin_freq, c='green', linestyle='--')

    ax.set_title(r'w-x Plot', fontsize=14)
    ax.set_ylabel(r'f (Hz)', rotation=0, labelpad=15)
    ax.set_xlabel('x (cell)')

    plt.xlim(None, 32)
    fullpath = anal_dir + 'wx_plot_{}'.format(component.lower()) + '.png'
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print('w-x Plot saved')
    return


def plot_kt(component='By', saveas='kt_plot'):
    arr = get_array(component)
    
    plt.ioff()
    
    k  = np.arange(NX)

    k         = np.fft.fftfreq(NX, dx)
    k         = k[k>=0] * 1e6

    fft_matrix  = np.zeros(arr.shape, dtype='complex128')
    for ii in range(arr.shape[0]): # Take spatial FFT at each time
        fft_matrix[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())

    kt = (fft_matrix[:, :k.shape[0]] * np.conj(fft_matrix[:, :k.shape[0]])).real

    fig = plt.figure(1, figsize=(12, 8))
    ax  = fig.add_subplot(111)
    
    im1 = ax.pcolormesh(k[:k.shape[0]], time_gperiods, kt, cmap='jet')      # Remove k[0] since FFT[0] >> FFT[1, 2, ... , k] antialiased=True
    fig.colorbar(im1)
    ax.set_title(r'k-t Plot', fontsize=14)
    ax.set_ylabel(r'$\Omega_i t$', rotation=0)
    ax.set_xlabel(r'$k (m^{-1}) \times 10^6$')
    #ax.set_ylim(0, 15)
    
    fullpath = anal_dir + saveas + '.png'
    plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print('K-T Plot saved')
    return


def get_dispersion_from_sim(k=None, plot=False, save=False):
    '''
    Still not sure how this will work for a H+, O+ mix, but H+-He+ should be fine
    '''
    if k is None:
        k         = np.fft.fftfreq(NX, dx)
        k         = k[k>=0]
    
    N_present    = species_present.count(True)
    cold_density = np.zeros(N_present)
    warm_density = np.zeros(N_present)
    cgr_ani      = np.zeros(N_present)
    tempperp     = np.zeros(N_present)
    anisotropies = Tper / Tpar - 1
    
    for ii in range(Nj):
        if temp_type[ii] == 0:
            if 'H^+'    in species_lbl[ii]:
                cold_density[0] = density[ii]
            elif 'He^+' in species_lbl[ii]:
                cold_density[1] = density[ii]
            elif 'O^+'  in species_lbl[ii]:
                cold_density[2] = density[ii]
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')
                
        if temp_type[ii] == 1:
            if 'H^+'    in species_lbl[ii]:
                warm_density[0] = density[ii]
                cgr_ani[0]      = anisotropies[ii]
                tempperp[0]     = Tper[ii] / 11603.
            elif 'He^+' in species_lbl[ii]:
                warm_density[1] = density[ii]
                cgr_ani[1]      = anisotropies[ii]
                tempperp[1]     = Tper[ii] / 11603
            elif 'O^+'  in species_lbl[ii]:
                warm_density[2] = density[ii]
                cgr_ani[2]      = anisotropies[ii]
                tempperp[2]     = Tper[ii] / 11603
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')

    if save == True:
        savepath = anal_dir
    else:
        savepath = None
    
    k_vals, CPDR_solns, warm_solns = get_dispersion_relation(B0, cold_density, warm_density, cgr_ani, tempperp,
               norm_k=False, norm_w=False, kmin=k[0], kmax=k[-1], k_input_norm=0, plot=plot, save=save, savepath=savepath)

    return k_vals, CPDR_solns, warm_solns


def plot_wk(component='By', dispersion_overlay=False, plot=False, save=False):
    arr = get_array(component)
    
    print('Plotting dispersion relation...')
    plt.ioff()
    num_times = arr.shape[0]

    df = 1. / (num_times * dt_slice)
    dk = 1. / (NX * dx)

    f  = np.arange(0, 1. / (2*dt_slice), df)
    k  = np.arange(0, 1. / (2*dx), dk)

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

    ax.set_title(r'$\omega/k$ Dispersion Plot for {}'.format(component), fontsize=14)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)

    M    = np.array([1., 4., 16.])
    cyc  = q * B0 / (2 * np.pi * mp * M)
    for ii in range(3):
        if species_present[ii] == True:
            ax.axhline(cyc[ii], linestyle='--', c='k')
    
    if dispersion_overlay == True:
        '''
        Some weird factor of about 2pi inaccuracy? Is this inherent to the sim? Or a problem
        with linear theory? Or problem with the analysis?
        '''
        k_vals, CPDR_solns, warm_solns = get_dispersion_from_sim(k)
        for ii in range(CPDR_solns.shape[1]):
            ax.plot(k_vals, CPDR_solns[:, ii]*2*np.pi,      c='k', linestyle='--')
            ax.plot(k_vals, warm_solns[:, ii].real*2*np.pi, c='k', linestyle='-')
    
    if plot == True:
        plt.show()
    
    if save == True:
        filename ='{}_dispersion_relation'.format(component.upper())
        fullpath = anal_dir + filename + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('Dispersion Plot saved')
        plt.close(fig)
    return


def waterfall_plot(field):
    plt.ioff()

    arr = get_array()
    amp   = 100.                 # Amplitude multiplier of waves:

    cells  = np.arange(NX)

    for (ii, t) in zip(np.arange(num_files), np.arange(0, num_files*dt_slice, dt_slice)):
        plt.plot(cells, amp*(arr[ii] / arr.max()) + ii, c='k', alpha=0.05)

    plt.xlim(0, NX)
    plt.show()
    return


# =============================================================================
# def diagnostic_multiplot(qq):
#     plt.ioff()
# 
#     fig_size = 4, 7                                                             # Set figure grid dimensions
#     fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
#     fig.patch.set_facecolor('w')                                                # Set figure face color
# 
#     va        = B0 / np.sqrt(mu0*ne*mp)                                         # Alfven speed: Assuming pure proton plasma
# 
#     pos       = position / dx                                                   # Cell particle position
#     vel       = velocity / va                                                   # Normalized velocity
# 
#     den_norm  = dns / density                                                   # Normalize density for each species to initial values
#     qdens_norm= q_dns / (density*charge).sum()                                  # Normalized change density
#      
# #----- Velocity (x, y) Plots: Hot Species
#     ax_vx   = plt.subplot2grid(fig_size, (0, 0), rowspan=2, colspan=3)
#     ax_vy   = plt.subplot2grid(fig_size, (2, 0), rowspan=2, colspan=3)
# 
#     for jj in range(Nj):
#         ax_vx.scatter(pos[idx_bounds[jj, 0]: idx_bounds[jj, 1]], vel[0, idx_bounds[jj, 0]: idx_bounds[jj, 1]], s=3, c=temp_color[jj], lw=0, label=species_lbl[jj])
#         ax_vy.scatter(pos[idx_bounds[jj, 0]: idx_bounds[jj, 1]], vel[1, idx_bounds[jj, 0]: idx_bounds[jj, 1]], s=3, c=temp_color[jj], lw=0)
# 
#     ax_vx.legend()
#     ax_vx.set_title(r'Particle velocities vs. Position (x)')
#     ax_vy.set_xlabel(r'Cell', labelpad=10)
# 
#     ax_vx.set_ylabel(r'$\frac{v_x}{c}$', rotation=90)
#     ax_vy.set_ylabel(r'$\frac{v_y}{c}$', rotation=90)
# 
#     plt.setp(ax_vx.get_xticklabels(), visible=False)
#     ax_vx.set_yticks(ax_vx.get_yticks()[1:])
# 
#     for ax in [ax_vy, ax_vx]:
#         ax.set_xlim(0, NX)
#         ax.set_ylim(-10, 10)
# 
# #----- Density Plot
#     ax_den = plt.subplot2grid((fig_size), (0, 3), colspan=3)                     # Initialize axes
#     
#     ax_den.plot(qdens_norm, color='green')                                       # Create overlayed plots for densities of each species
# 
#     for jj in range(Nj):
#         ax_den.plot(den_norm, color=temp_color[jj])
#         
#     ax_den.set_title('Normalized Densities and Fields')                          # Axes title (For all, since density plot is on top
#     ax_den.set_ylabel(r'$\frac{n_i}{n_0}$', fontsize=14, rotation=0, labelpad=5) # Axis (y) label for this specific axes
#     ax_den.set_ylim(0, 2)
#     
# #----- E-field (Ex) Plot
#     ax_Ex = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_den)
# 
#     ax_Ex.plot(E[:, 0], color='red', label=r'$E_x$')
#     ax_Ex.plot(E[:, 1], color='cyan', label=r'$E_x$')
#     ax_Ex.plot(E[:, 2], color='black', label=r'$E_x$')
# 
#     ax_Ex.set_xlim(0, NX)
# 
#     #ax_Jx.set_yticks(np.arange(-200e-5, 201e-5, 50e-5))
#     #ax_Jx.set_yticklabels(np.arange(-150, 201, 50))
#     ax_Ex.set_ylabel(r'$E$', labelpad=25, rotation=0, fontsize=14)
# 
# #----- Magnetic Field (By) and Magnitude (|B|) Plots
#     ax_By = plt.subplot2grid((fig_size), (2, 3), colspan=3, sharex=ax_den)
#     ax_B  = plt.subplot2grid((fig_size), (3, 3), colspan=3, sharex=ax_den)
# 
#     mag_B  = (np.sqrt(B[:-1, 0] ** 2 + B[:-1, 1] ** 2 + B[:-1, 2] ** 2)) / B0
#     B_norm = B[:-1, :] / B0                                                           
# 
#     ax_B.plot(mag_B, color='g')                                                        # Create axes plots
#     ax_By.plot(B_norm[:, 1], color='g') 
#     ax_By.plot(B_norm[:, 2], color='b') 
# 
#     ax_B.set_xlim(0,  NX)                                                               # Set x limit
#     ax_By.set_xlim(0, NX)
# 
#     ax_B.set_ylim(0, 2)                                                                 # Set y limit
#     ax_By.set_ylim(-1, 1)
# 
#     ax_B.set_ylabel( r'$|B|$', rotation=0, labelpad=20, fontsize=14)                    # Set labels
#     ax_By.set_ylabel(r'$\frac{B_{y,z}}{B_0}$', rotation=0, labelpad=10, fontsize=14)
#     ax_B.set_xlabel('Cell Number')                                                      # Set x-axis label for group (since |B| is on bottom)
# 
#     for ax in [ax_den, ax_Ex, ax_By]:
#         plt.setp(ax.get_xticklabels(), visible=False)
#         ax.set_yticks(ax.get_yticks()[1:])
# 
#     for ax in [ax_den, ax_Ex, ax_By, ax_B]:
#         qrt = NX / (4.)
#         ax.set_xticks(np.arange(0, NX + qrt, qrt))
#         ax.grid()
# 
# #----- Plot Adjustments
#     plt.tight_layout(pad=1.0, w_pad=1.8)
#     fig.subplots_adjust(hspace=0)
# 
#     filename = 'diag%05d.png' % ii
#     path     = anal_dir + '/diagnostic_plot/'
#     
#     if os.path.exists(path) == False:                                   # Create data directory
#         os.makedirs(path)
# 
#     fullpath = path + filename
#     plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
#     print('Plot saved'.format(ii))
#     plt.close('all')
#     return
# =============================================================================


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


def get_cgr_from_sim(norm_flag=0):
    cold_density = np.zeros(3)
    warm_density = np.zeros(3)
    cgr_ani      = np.zeros(3)
    tempperp     = np.zeros(3)
    anisotropies = Tper / Tpar - 1
    
    for ii in range(Nj):
        if temp_type[ii] == 0:
            if 'H^+'    in species_lbl[ii]:
                cold_density[0] = density[ii] / 1e6
            elif 'He^+' in species_lbl[ii]:
                cold_density[1] = density[ii] / 1e6
            elif 'O^+'  in species_lbl[ii]:
                cold_density[2] = density[ii] / 1e6
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')
                
        if temp_type[ii] == 1:
            if 'H^+'    in species_lbl[ii]:
                warm_density[0] = density[ii] / 1e6
                cgr_ani[0]      = anisotropies[ii]
                tempperp[0]     = Tper[ii] / 11603.
            elif 'He^+' in species_lbl[ii]:
                warm_density[1] = density[ii] / 1e6
                cgr_ani[1]      = anisotropies[ii]
                tempperp[1]     = Tper[ii] / 11603.
            elif 'O^+'  in species_lbl[ii]:
                warm_density[2] = density[ii] / 1e6
                cgr_ani[2]      = anisotropies[ii]
                tempperp[2]     = Tper[ii] / 11603.
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')
    
    freqs, cgr, stop = calculate_growth_rate(B0*1e9, cold_density, warm_density, cgr_ani, temperp=tempperp, norm_freq=norm_flag)
    return freqs, cgr, stop


def get_derivative(arr):
    ''' Caculates centered derivative for values in 'arr', with forward and backward differences applied
    for boundary points'''
    
    deriv = np.zeros(arr.shape[0])
    
    deriv[0 ] = (-3*arr[ 0] + 4*arr[ 1] - arr[ 2]) / (2 * dt_slice)
    deriv[-1] = ( 3*arr[-1] - 4*arr[-2] + arr[-3]) / (2 * dt_slice)
    
    for ii in np.arange(1, arr.shape[0] - 1):
        deriv[ii] = (arr[ii + 1] - arr[ii - 1]) / (2 * dt_slice)
    return deriv


def get_max_frequency(arr, plot=False):
    '''
    Calculates strongest frequency within a given field component across
    the simulation space. Returns frequency and power axes and index of 
    maximum frequency in axis.
    '''
    npts      = arr.shape[0]
    fft_freqs = np.fft.fftfreq(npts, d=dt_slice)
    fft_freqs = fft_freqs[fft_freqs >= 0]
    
    # For each gridpoint, take temporal FFT
    fft_matrix  = np.zeros((npts, NX), dtype='complex128')
    for ii in range(NX):
        fft_matrix[:, ii] = np.fft.fft(arr[:, ii] - arr[:, ii].mean())

            
    # Convert FFT output to power and normalize
    fft_pwr   = (fft_matrix[:fft_freqs.shape[0], :] * np.conj(fft_matrix[:fft_freqs.shape[0], :])).real
    fft_pwr  *= 4. / (npts ** 2)
    fft_pwr   = fft_pwr.sum(axis=1)

    max_idx = np.where(fft_pwr == fft_pwr.max())[0][0]
    print('Maximum frequency at {}Hz\n'.format(fft_freqs[max_idx]))
    
    if plot == True:
        plt.figure()
        plt.plot(fft_freqs, fft_pwr)
        plt.scatter(fft_freqs[max_idx], fft_pwr[max_idx], c='r')
        plt.title('Frequencies across simulation domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (nT^2 / Hz)')
        plt.legend()
        plt.show()
    return fft_freqs, fft_pwr, max_idx


def growing_sine(pars, t, data=None):
    vals   = pars.valuesdict()
    amp    = vals['amp']
    freq   = vals['freq']
    growth = vals['growth']

    model = amp * np.exp(1j*2*np.pi*freq*t).imag * np.exp(growth*t)
    
    if data is None:
        return model
    else:
        return model - data
    




def fit_fied_component(arr, fi, component, cut_idx=None, plot=False, plot_cell=64):
    '''
    Calculates and returns parameters for growing sine wave function for each
    gridpoint up to the linear cutoff time.
    '''
    print('Fitting field component')
    time_fit  = time_seconds[:cut_idx]
    gyfreq_hz = gyfreq/(2*np.pi)    
    
    growth_rates = np.zeros(NX)
    frequencies  = np.zeros(NX)
    amplitudes   = np.zeros(NX)
    
    fit_params = lmf.Parameters()
    fit_params.add('amp'   , value=1.0         , vary=True, min=-0.5*B0*1e9 , max=0.5*B0*1e9)
    fit_params.add('freq'  , value=fi          , vary=True, min=-gyfreq_hz     , max=gyfreq_hz)
    fit_params.add('growth', value=0.001*gyfreq, vary=True, min=0.0         , max=0.1*gyfreq_hz)
    
    for cell_num in range(NX):
        data_to_fit  = arr[:cut_idx, cell_num]
        
        fit_output      = lmf.minimize(growing_sine, fit_params, args=(time_fit,), kws={'data': data_to_fit},
                                   method='leastsq')
        
        fit_function    = growing_sine(fit_output.params, time_fit)
    
        fit_dict        = fit_output.params.valuesdict()
        
        growth_rates[cell_num] = fit_dict['growth']
        frequencies[ cell_num] = fit_dict['freq']
        amplitudes[  cell_num] = fit_dict['amp']
    
        if plot != None and cell_num == plot_cell:
            plt.figure()
            plt.plot(time_fit, data_to_fit,  label='Magnetic field')
            plt.plot(time_fit, fit_function, label='Fit')
            plt.figtext(0.135, 0.73, r'$f = %.3fHz$' % (frequencies[cell_num] / (2 * np.pi)))
            plt.figtext(0.135, 0.69, r'$\gamma = %.3fs^{-1}$' % (growth_rates[cell_num] / (2 * np.pi)))
            plt.figtext(0.135, 0.65, r'$A_0 = %.3fnT$' % (amplitudes[cell_num] ))
            plt.title('{} cell {}'.format(component, plot_cell))
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (nT)')
            plt.legend()
            print(lmf.fit_report(fit_output))
            
            if plot == 'save':
                save_path = anal_dir + '{}_envfit_{}.png'.format(component, plot_cell)
                plt.savefig(save_path)
                plt.close('all')
            elif plot == 'show':
                plt.show()
            else:
                pass
            
    return amplitudes, frequencies, growth_rates


def residual_exp(pars, t, data=None):
    vals   = pars.valuesdict()
    amp    = vals['amp']
    growth = vals['growth']

    model  = amp * np.exp(growth*t)
    
    if data is None:
        return model
    else:
        return model - data
    

def fit_magnetic_energy(by, bz, plot=False):
    '''
    Calculates an exponential growth rate based on transverse magnetic field
    energy.
    '''
    print('Fitting magnetic energy')
    bt  = np.sqrt(by ** 2 + bz ** 2) * 1e-9
    U_B = 0.5 * np.square(bt).sum(axis=1) * NX * dx / mu0
    dU  = get_derivative(U_B)
    
    linear_cutoff = np.where(dU == dU.max())[0][0]
    
    time_fit = time_seconds[:linear_cutoff]

    fit_params = lmf.Parameters()
    fit_params.add('amp'   , value=1.0         , min=None , max=None)
    fit_params.add('growth', value=0.001*gyfreq, min=0.0  , max=None)
    
    fit_output      = lmf.minimize(residual_exp, fit_params, args=(time_fit,), kws={'data': U_B[:linear_cutoff]},
                               method='leastsq')
    fit_function    = residual_exp(fit_output.params, time_fit)

    fit_dict        = fit_output.params.valuesdict()

    if plot != None:
        plt.ioff()
        plt.figure()
        plt.plot(time_seconds[:linear_cutoff], U_B[:linear_cutoff], color='green', marker='o', label='Energy')
        plt.plot(time_seconds[:linear_cutoff], fit_function, color='b', label='Exp. fit')
        plt.figtext(0.135, 0.725, r'$\gamma = %.3fs^{-1}$' % (fit_dict['growth'] / (2 * np.pi)))
        plt.title('Transverse magnetic field energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.legend()
        
# =============================================================================
#         plt.figure()
#         plt.plot(time_seconds[:linear_cutoff], dU[:linear_cutoff])
# =============================================================================
        
        if plot == 'save':
            save_path = anal_dir + 'magnetic_energy_expfit.png'
            plt.savefig(save_path)
            plt.close('all')
        elif plot == 'show':
            plt.show()
        else:
            pass
        
    return linear_cutoff, fit_dict['growth']


def exponential_sine(t, amp, freq, growth, phase):
    return amp * np.sin(2*np.pi*freq*t + phase) * np.exp(growth*t)


def growth_rate_kt(arr, cut_idx, fi, saveas='kt_growth'):
    plt.ioff()

    time_fit  = time_seconds[:cut_idx]
    k         = np.fft.fftfreq(NX, dx)
    k         = k[k>=0]

    # Take spatial FFT at each time
    mode_matrix  = np.zeros(arr.shape, dtype='complex128')
    for ii in range(arr.shape[0]):
        mode_matrix[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())

    # Cut off imaginary bits
    mode_matrix = 2*mode_matrix[:, :k.shape[0]]
    
    gmodel = lmf.Model(exponential_sine, nan_policy='propagate')
    
    gmodel.set_param_hint('amp',    value=1.0, min=0.0,     max=abs(mode_matrix).max())
    gmodel.set_param_hint('freq',   value=fi, min=-2*fi,    max=2*fi)
    gmodel.set_param_hint('growth', value=0.05, min=0.0,    max=0.5*fi)
    gmodel.set_param_hint('phase',  value=0.0, vary=False)
    
    for mode_num in [1]:#range(1, k.shape[0]):
        data_to_fit = mode_matrix[:cut_idx, mode_num].real
    
        result      = gmodel.fit(data_to_fit, t=time_fit, method='leastsq')

        plt.plot(time_fit, data_to_fit, 'ko', label='data')
        plt.plot(time_fit, result.best_fit, 'r-', label='lmfit')

        popt, pcov = curve_fit(exponential_sine, time_fit, data_to_fit, maxfev=1000000000)
        plt.plot(time_fit, exponential_sine(time_fit, *popt), label='curve_fit')
        plt.legend()
        print(popt)
# =============================================================================
#         fit_output      = minimize(exponential_sine, fit_params, args=(time_fit,), kws={'data': data_to_fit},
#                                    method='leastsq')
#         
#         fit_function    = exponential_sine(fit_output.params, time_fit)
# 
#         fit_dict        = fit_output.params.valuesdict()
#         
#         growth_rates[mode_num] = fit_dict['growth']
#         frequencies[ mode_num] = fit_dict['freq']
#         amplitudes[  mode_num] = fit_dict['amp']
#     
#         plt.plot(time_fit, data_to_fit)
#         plt.plot(time_fit, fit_function)
# =============================================================================

    plt.show()

    return


def get_growth_rates(do_plot=None):
    '''
    Extract the magnetic linear wave growth rate from:
        -- Fitting an exponential to the magnetic energy
        -- Fitting a growing sine wave to the field components at each cell
    
    The linear regime is calculated as all times before the maximum energy derivative,
    i.e. the growth is assumed exponential until the rate of energy transfer decreases.
    
    One could also take the min/max (i.e. abs) of the field through time and 
    fit an exponential to that, but it should be roughly equivalent to the energy fit.
    
    INPUT:
        -- do_plot : 'show', 'save' or 'None'. 'save' will also output growth rates to a text file.
    '''
    by  = get_array('By') * 1e9
    bz  = get_array('Bz') * 1e9
    
    linear_cutoff, gr_rate_energy   = fit_magnetic_energy(by, bz, plot=do_plot)
    freqs, power, max_idx           = get_max_frequency(by,       plot=do_plot)
    
    growth_rate_kt(by, linear_cutoff, freqs[max_idx])
    
# =============================================================================
#     
#     
#     by_wamps, by_wfreqs, by_gr_rate = fit_fied_component(by, freqs[max_idx], 'By', linear_cutoff, plot=do_plot)
#     bz_wamps, bz_wfreqs, bz_gr_rate = fit_fied_component(bz, freqs[max_idx], 'Bz', linear_cutoff, plot=do_plot)
#     
#     
#     if do_plot == 'save':
#         txt_path  = anal_dir + 'growth_rates.txt'
#         text_file = open(txt_path, 'w')
#     else:
#         text_file = None
#     
#     print('Energy growth rate: {}'.format(gr_rate_energy), file=text_file)
#     print('By av. growth rate: {}'.format(by_gr_rate.mean()), file=text_file)
#     print('Bz av. growth rate: {}'.format(bz_gr_rate.mean()), file=text_file)
#     print('By min growth rate: {}'.format(by_gr_rate.min()), file=text_file)
#     print('Bz min growth rate: {}'.format(bz_gr_rate.min()), file=text_file)
#     print('By max growth rate: {}'.format(by_gr_rate.max()), file=text_file)
#     print('Bz max growth rate: {}'.format(bz_gr_rate.max()), file=text_file)
# =============================================================================
    return


def examine_run_parameters(to_file=False):
    '''
    Diagnostic information to compare runs at a glance. Values include
    
    cellpart, Nj, B0, ne, NX, num_files, Te0, max_rev, ie, theta, dxm
    
    number of files
    '''
    global run_num, num_files
    
    run_params = ['cellpart', 'Nj', 'B0', 'ne', 'NX', 'num_files', 'Te0', 'max_rev', 'ie', 'theta', 'dxm']
    run_dict = {'run_num' : []}
    for param in run_params:
        run_dict[param] = []
    
    for run_num in range(num_runs):
        manage_dirs(create_new=False)
        load_header()                                           # Load simulation parameters
        load_particles()                                        # Load particle parameters
        num_files = len(os.listdir(data_dir)) - 2               # Number of timesteps to load
        run_dict['run_num'].append(run_num)
        for param in run_params:
            run_dict[param].append(globals()[param])
        
    if to_file == True:
        txt_path  = base_dir + 'growth_rates.txt'
        run_file = open(txt_path, 'w')
    else:
        run_file = None
        
    print('\n')
    print('Simulation parameters for runs in series \'{}\':'.format(series), file=run_file)
    print('\n')
    print((tabulate.tabulate(run_dict, headers="keys")), file=run_file)
    return


if __name__ == '__main__':   
    drive      = 'E://MODEL_RUNS//Josh_Runs//'
    series     = 'varying_density_better'
    series_dir = '{}/runs//{}//'.format(drive, series)
    num_runs   = len([name for name in os.listdir(series_dir) if 'run_' in name])
    examine_run_parameters(to_file=True)

    for run_num in range(num_runs):
        print('Run {}'.format(run_num))
        manage_dirs()                                           # Initialize directories
        load_constants()                                        # Load SI constants
        load_header()                                           # Load simulation parameters
        load_particles()                                        # Load particle parameters
        
        num_files   = len(os.listdir(data_dir)) - 2             # Number of timesteps to load
    
        initialize_simulation_variables()

        mag_energy      = np.zeros(num_files)                   # Magnetic potential energy
        particle_energy = np.zeros((num_files, Nj))             # Particle kinetic energy
        electron_energy = np.zeros(num_files)                   # Electron pressure energy
                
        #get_growth_rates()
        plot_wk(dispersion_overlay=True, save=True)
        #get_dispersion_from_sim(save=True)
        #plot_kt()
        #plot_wx(linear_overlay=True)
        
        
        
        if False:
            if os.path.exists(anal_dir + 'norm_energy_plot.png') == False:
                for ii in range(num_files):
                    B, E, Ve, Te, J, position, q_dns, velocity = load_timestep(ii)
                    #dns                                        = collect_number_density(position)
                    plot_energies(ii, normalize=True)
        
        

    
