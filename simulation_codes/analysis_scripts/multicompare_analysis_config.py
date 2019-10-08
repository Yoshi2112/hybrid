import os
import numpy as np
import numba as nb
import pickle
import pdb
'''
Used to initialise values for a run
e.g. directories, simulation/particle parameters, derived quantities, etc.

These are called in the main script by import, and will change each time
load_run() is called

The global calls allow variables to be accessed in the main script without
clogging up its namespace - i.e. run-specific parameters are called by
using e.g. cf.B0
'''
def load_run(drive, series, run_num, lmissing_t0_offset=0, extract_arrays=True):
    global missing_t0_offset
    missing_t0_offset = lmissing_t0_offset   # Flag for when I thought having a t=0 save file wasn't needed. I was wrong.
    load_simulation_params()
    load_species_params()
    initialize_simulation_variables()
    return


def load_species_params(_data_dir):
    global species_present, density, dist_type, idx_bounds, charge, mass, Tper, \
           sim_repr, temp_type, temp_color, velocity, Tpar, species_lbl, n_contr, drift_v

    p_path = os.path.join(_data_dir, 'particle_parameters.npz')                 # File location
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
    Tpar       = p_data['Tpar']
    species_lbl= p_data['species_lbl']
    
    # Changed it without checking, either should work now (but velocity is a bad name)
    try:
        velocity   = p_data['velocity']
    except:
        drift_v    = p_data['drift_v']
        
    n_contr    = density / (cellpart*sim_repr)                                  # Species density contribution: Each macroparticle contributes this density to a cell
    species_present = [False, False, False]                                     # Test for the presence of singly charged H, He, O
        
    for ii in range(Nj):
        if 'H^+' in species_lbl[ii]:
            species_present[0] = True
        elif 'He^+' in species_lbl[ii]:
            species_present[1] = True
        elif 'O^+'  in species_lbl[ii]:
            species_present[2] = True
    return

def load_simulation_params(_data_dir):
    global Nj, cellpart, ne, NX, dxm, seed, B0, dx, Te0, theta, dt_sim, max_rev,\
           ie, run_desc, seed, subcycles, LH_frac, orbit_res, freq_res, method_type,\
           particle_shape, part_save_iter, field_save_iter, dt_field, dt_particle, \
           HM_amplitude, HM_frequency 

    h_name = os.path.join(_data_dir, 'simulation_parameters.pckl')      # Load header file
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
    B0              = obj['B0']
    ne              = obj['ne']
    Te0             = obj['Te0']
    ie              = obj['ie']
    theta           = obj['theta']
    max_rev         = obj['max_rev']
    orbit_res       = obj['orbit_res']
    freq_res        = obj['freq_res']
    run_desc        = obj['run_desc']
    method_type     = obj['method_type'] 
    particle_shape  = obj['particle_shape'] 
    
    part_save_iter  = obj['part_save_iter']
    field_save_iter = obj['field_save_iter']
    
    if method_type == 'CAM_CL':
        subcycles   = obj['subcycles']
        LH_frac     = obj['LH_frac']
    else:
        pass
    
    try:
        HM_amplitude = obj['HM_amplitude']
        HM_frequency = obj['HM_frequency']
    except:
        HM_amplitude = 0
        HM_frequency = 0
    
    dt_field        = dt_sim * field_save_iter                         # Time between data slices (seconds)
    dt_particle     = dt_sim * part_save_iter
    return 

def initialize_simulation_variables(_data_dir):
    global wpi, gyfreq, gyperiod, time_seconds_field, time_seconds_particle, \
           time_gperiods_field, time_gperiods_particle, time_radperiods_field, time_radperiods_particle, va
    missing_t0_offset = 0
    load_simulation_params(_data_dir)
    
    num_field_steps    = len(os.listdir(_data_dir + '/fields/'))
    num_particle_steps = len(os.listdir(_data_dir + '/particles/'))
    
    q   = 1.602e-19               # Elementary charge (C)
    mp  = 1.67e-27                # Mass of proton (kg)
    e0  = 8.854e-12               # Epsilon naught - permittivity of free space
    mu0 = (4e-7) * np.pi          # Magnetic Permeability of Free Space (SI units)
    
    wpi       = np.sqrt(ne * q ** 2 / (mp * e0))            # Ion plasma frequency
    gyfreq    = q * B0 / mp                                 # Proton gyrofrequency (rad/s)
    gyperiod  = (mp * 2 * np.pi) / (q * B0)                 # Proton gyroperiod (s)
    va         = B0 / np.sqrt(mu0*ne*mp)                    # Alfven speed: Assuming pure proton plasma
    
    time_seconds_field    = np.array([ii * dt_field    for ii in range(missing_t0_offset, num_field_steps + missing_t0_offset)])
    time_seconds_particle = np.array([ii * dt_particle for ii in range(missing_t0_offset, num_particle_steps + missing_t0_offset)])
    
    time_gperiods_field   = time_seconds_field    / gyperiod
    time_gperiods_particle= time_seconds_particle / gyperiod
    
    time_radperiods_field    = time_seconds_field    * gyfreq 
    time_radperiods_particle = time_seconds_particle * gyfreq
    return


def load_particles(particle_dir, ii):    
    part_file  = 'data%05d.npz' % ii             # Define target file
    input_path = particle_dir + part_file        # File location
    data       = np.load(input_path)             # Load file
    
    try:
        tx               = data['pos']
        tv               = data['vel']
    except:
        tx               = data['position']
        tv               = data['velocity']
    return tx, tv


def get_array(temp_dir, component='by', get_all=False):
    '''
    Input  : Array Component
    Output : Array as np.ndarray
    
    Components:
        3D (x, y, z) -- B, E, J, Ve
        1D           -- qdens, Te
    '''
    if get_all == False:
        arr_path = temp_dir + component.lower() + '_array' + '.npy'
        arr      = np.load(arr_path)
        return arr
    else:
        bx = np.load(temp_dir + 'bx' +'_array.npy')
        by = np.load(temp_dir + 'by' +'_array.npy')
        bz = np.load(temp_dir + 'bz' +'_array.npy')
        
        ex = np.load(temp_dir + 'ex' +'_array.npy')
        ey = np.load(temp_dir + 'ey' +'_array.npy')
        ez = np.load(temp_dir + 'ez' +'_array.npy')
        
        jx = np.load(temp_dir + 'jx' +'_array.npy')
        jy = np.load(temp_dir + 'jy' +'_array.npy')
        jz = np.load(temp_dir + 'jz' +'_array.npy')
        
        vex = np.load(temp_dir + 'vex' +'_array.npy')
        vey = np.load(temp_dir + 'vey' +'_array.npy')
        vez = np.load(temp_dir + 'vez' +'_array.npy')
        
        te    = np.load(temp_dir + 'te' +'_array.npy')
        qdens = np.load(temp_dir + 'qdens' +'_array.npy')
        return bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens


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


