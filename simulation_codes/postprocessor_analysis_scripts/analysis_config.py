import os
import numpy as np
import numba as nb
import pickle
'''
Used to initialise values for a run
e.g. directories, simulation/particle parameters, derived quantities, etc.

These are called in the main script by import, and will change each time
load_run() is called

The global calls allow variables to be accessed in the main script without
clogging up its namespace - i.e. run-specific parameters are called by
using e.g. cf.B0
'''
def load_run(drive, series, run_num):
    manage_dirs(drive, series, run_num)
    load_simulation_params()
    load_species_params()
    initialize_simulation_variables()
    return

def manage_dirs(drive, series, run_num, create_new=True):
    global run_dir, data_dir, anal_dir, temp_dir, base_dir, field_dir, particle_dir, num_field_steps, num_particle_steps
    
    base_dir = '{}/runs/{}/'.format(drive, series)                      # Main series directory, containing runs
    run_dir  = '{}/runs/{}/run_{}/'.format(drive, series, run_num)      # Main run directory
    data_dir = run_dir + 'data/'                                        # Directory containing .npz output files for the simulation run
    anal_dir = run_dir + 'analysis/'                                    # Output directory for all this analysis (each will probably have a subfolder)
    temp_dir = run_dir + 'extracted/'                                   # Saving things like matrices so we only have to do them once

    field_dir    = data_dir + '/fields/'
    particle_dir = data_dir + '/particles/'
    
    num_field_steps    = len(os.listdir(field_dir))                    # Number of field    time slices
    num_particle_steps = len(os.listdir(particle_dir))                 # Number of particle time slices
    
   # Make Output folders if they don't exist
    for this_dir in [anal_dir, temp_dir]:
        if os.path.exists(run_dir) == True:
            if os.path.exists(this_dir) == False:
                os.makedirs(this_dir)
        else:
            raise IOError('Run {} does not exist for series {}. Check range argument.'.format(run_num, series))
    return

def load_species_params():
    global species_present, density, dist_type, idx_bounds, charge, mass, Tper, \
           sim_repr, temp_type, temp_color, velocity, Tpar, species_lbl, n_contr, drift_v

    p_path = os.path.join(data_dir, 'particle_parameters.npz')                  # File location
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

def load_simulation_params():
    global Nj, cellpart, ne, NX, dxm, seed, B0, dx, Te0, theta, dt_sim, max_rev,\
           ie, run_desc, seed, subcycles, LH_frac, orbit_res, freq_res, method_type,\
           particle_shape, part_save_iter, field_save_iter, dt_field, dt_particle

    h_name = os.path.join(data_dir, 'simulation_parameters.pckl')       # Load header file
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
    max_rev         = obj['max_rev']
    LH_frac         = obj['LH_frac']
    orbit_res       = obj['orbit_res']
    freq_res        = obj['freq_res']
    run_desc        = obj['run_desc']
    method_type     = obj['method_type'] 
    particle_shape  = obj['particle_shape'] 
    
    part_save_iter  = obj['part_save_iter']
    field_save_iter = obj['field_save_iter']
    
    dt_field        = dt_sim * field_save_iter                         # Time between data slices (seconds)
    dt_particle     = dt_sim * part_save_iter
    return 

def initialize_simulation_variables():
    global wpi, gyfreq, gyperiod, time_gperiods, time_radperiods, time_seconds_field, time_seconds_particle
    q   = 1.602e-19               # Elementary charge (C)
    mp  = 1.67e-27                # Mass of proton (kg)
    e0  = 8.854e-12               # Epsilon naught - permittivity of free space
    
    extract_all_arrays()
    
    wpi       = np.sqrt(ne * q ** 2 / (mp * e0))            # Ion plasma frequency
    gyfreq    = q * B0 / mp                                 # Proton gyrofrequency (rad/s)
    gyperiod  = (mp * 2 * np.pi) / (q * B0)                 # Proton gyroperiod (s)
    
    time_seconds_field    = np.array([ii * dt_field for ii in range(num_field_steps)])
    time_seconds_particle = np.array([ii * dt_field for ii in range(num_particle_steps)])
    
    time_gperiods   = time_seconds_field / gyperiod
    time_radperiods = time_seconds_field * gyfreq 
    return

def load_fields(ii):    
    field_file = 'data%05d.npz' % ii             # Define target file
    input_path = field_dir + field_file          # File location
    data       = np.load(input_path)             # Load file

    tB               = data['B']
    tE               = data['E']
    tVe              = data['Ve']
    tTe              = data['Te']
    tJ               = data['J']
    tdns             = data['dns']
    return tB, tE, tVe, tTe, tJ, tdns

def load_particles(ii):    
    part_file = 'data%05d.npz' % ii              # Define target file
    input_path = particle_dir + part_file        # File location
    data       = np.load(input_path)             # Load file

    tx               = data['position']
    tv               = data['velocity']
    return tx, tv

def extract_all_arrays():
    '''
    Extracts and saves all field arrays separate from the timestep slice files for easy
    access. Note that magnetic field arrays exclude the last value due to periodic
    boundary conditions. This may be changed later.
    '''
    bx_arr   = np.zeros((num_field_steps, NX)); ex_arr  = np.zeros((num_field_steps, NX))
    by_arr   = np.zeros((num_field_steps, NX)); ey_arr  = np.zeros((num_field_steps, NX))
    bz_arr   = np.zeros((num_field_steps, NX)); ez_arr  = np.zeros((num_field_steps, NX))
    
    # Check that all components are extracted
    comps_missing = 0
    for component in ['bx', 'by', 'bz', 'ex', 'ey', 'ez']:
        check_path = temp_dir + component + '_array.npy'
        if os.path.isfile(check_path) == False:
            comps_missing += 1
    
    if comps_missing == 0:
        return
    else:
        for ii in range(num_field_steps):
            B, E, Ve, Te, J, q_dns = load_fields(ii)
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

def get_array(component):
    arr_path = temp_dir + component.lower() + '_array' + '.npy'
    arr      = np.load(arr_path) 
    return arr



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


