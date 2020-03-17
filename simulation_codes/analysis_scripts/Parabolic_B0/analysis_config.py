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
def load_run(drive, series, run_num, extract_arrays=True):
    manage_dirs(drive, series, run_num)
    load_simulation_params()
    load_species_params()
    initialize_simulation_variables()
    
    if extract_arrays == True:
        extract_all_arrays()
    return


def manage_dirs(drive, series, run_num):
    global run_dir, data_dir, anal_dir, temp_dir, base_dir, field_dir, particle_dir, num_field_steps, num_particle_steps
    
    base_dir = '{}/runs/{}/'.format(drive, series)                      # Main series directory, containing runs
    run_dir  = '{}/runs/{}/run_{}/'.format(drive, series, run_num)      # Main run directory
    data_dir = run_dir + 'data/'                                        # Directory containing .npz output files for the simulation run
    anal_dir = run_dir + 'analysis/'                                    # Output directory for all this analysis (each will probably have a subfolder)
    temp_dir = run_dir + 'extracted/'                                   # Saving things like matrices so we only have to do them once

    field_dir    = data_dir + '/fields/'
    particle_dir = data_dir + '/particles/'

    # Make Output folders if they don't exist
    for this_dir in [anal_dir, temp_dir]:
        if os.path.exists(run_dir) == True:
            if os.path.exists(this_dir) == False:
                os.makedirs(this_dir)
        else:
            raise IOError('Run {} does not exist for series {}. Check range argument.'.format(run_num, series))
    return


def load_species_params():
    global species_present, density, dist_type, charge, mass, Tper,      \
           sim_repr, temp_type, temp_color, Tpar, species_lbl, n_contr,  \
           drift_v, idx_start, idx_end, N_species, Bc, nsp_ppc

    p_path = os.path.join(data_dir, 'particle_parameters.npz')                  # File location
    p_data = np.load(p_path)                                                    # Load file

    idx_start  = p_data['idx_start']
    idx_end    = p_data['idx_end']
    species_lbl= p_data['species_lbl']
    temp_color = p_data['temp_color']
    temp_type  = p_data['temp_type']
    dist_type  = p_data['dist_type']
    
    mass       = p_data['mass']
    charge     = p_data['charge']
    drift_v    = p_data['drift_v']
    nsp_ppc    = p_data['nsp_ppc']
    density    = p_data['density']
    N_species  = p_data['N_species']
    nsp_ppc    = p_data['nsp_ppc']
    Tpar       = p_data['Tpar']
    Tper       = p_data['Tper']
    Bc         = p_data['Bc']

    n_contr    = density / nsp_ppc  
    species_present = [False, False, False]                # Test for the presence of singly charged H, He, O
        
    for ii in range(Nj):
        if 'H^+' in species_lbl[ii]:
            species_present[0] = True
        elif 'He^+' in species_lbl[ii]:
            species_present[1] = True
        elif 'O^+'  in species_lbl[ii]:
            species_present[2] = True
    return


def load_simulation_params():
    global Nj, cellpart, ne, NX, dxm, seed, B0, dx, Te0, dt_sim, max_rev,           \
           ie, run_desc, seed, subcycles, LH_frac, orbit_res, freq_res, method_type,\
           particle_shape, part_save_iter, field_save_iter, dt_field, dt_particle,  \
           ND, NC, N, r_damp, xmax, B_xmax, B_eq, theta_xmax, a, boundary_type,     \
           particle_boundary, rc_hwidth, L, B_nodes, E_nodes

    h_name = os.path.join(data_dir, 'simulation_parameters.pckl')       # Load header file
    f      = open(h_name, 'rb')                                         # Open header file
    obj    = pickle.load(f)                                             # Load variables from header file into python object
    f.close()                                                           # Close header file
    
    seed              = obj['seed']
    Nj                = obj['Nj']
    dt_sim            = obj['dt']
    NX                = obj['NX']
    ND                = obj['ND']
    NC                = obj['NC']
    N                 = obj['N']
    dxm               = obj['dxm']
    dx                = obj['dx']
    ne                = obj['ne']
    Te0               = obj['Te0']
    ie                = obj['ie']
    r_damp            = obj['r_damp']
    xmax              = obj['xmax']
    B_xmax            = obj['B_xmax']
    B_eq              = obj['B_eq']
    a                 = obj['a']
    L                 = obj['L']
    rc_hwidth         = obj['rc_hwidth']
    theta_xmax        = obj['theta_xmax']
    max_rev           = obj['max_rev']
    orbit_res         = obj['orbit_res']
    freq_res          = obj['freq_res']
    run_desc          = obj['run_desc']
    method_type       = obj['method_type'] 
    particle_shape    = obj['particle_shape']
    boundary_type     = obj['boundary_type']
    particle_boundary = obj['particle_boundary']
    
    part_save_iter    = obj['part_save_iter']
    field_save_iter   = obj['field_save_iter']
    
    dt_field        = dt_sim * field_save_iter                         # Time between data slices (seconds)
    dt_particle     = dt_sim * part_save_iter
    
    B_nodes  = (np.arange(NC + 1) - NC // 2)       * dx      # B grid points position in space
    E_nodes  = (np.arange(NC)     - NC // 2 + 0.5) * dx      # E grid points position in space
    
    if rc_hwidth == 0: 
        rc_hwidth = NX
    return 


def initialize_simulation_variables():
    global wpi, gyfreq, gyperiod, va
    q   = 1.602e-19               # Elementary charge (C)
    mp  = 1.67e-27                # Mass of proton (kg)
    e0  = 8.854e-12               # Epsilon naught - permittivity of free space
    mu0 = (4e-7) * np.pi          # Magnetic Permeability of Free Space (SI units)
    
    wpi        = np.sqrt(ne * q ** 2 / (mp * e0))            # Ion plasma frequency
    gyfreq     = q * B_eq / mp                               # Proton gyrofrequency (rad/s)
    gyperiod   = (mp * 2 * np.pi) / (q * B_eq)               # Proton gyroperiod (s)
    va         = B_eq / np.sqrt(mu0*ne*mp)                   # Alfven speed: Assuming pure proton plasma
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
    tsim_time        = data['sim_time']
    tdamping_array   = data['damping_array']

    return tB, tE, tVe, tTe, tJ, tdns, tsim_time, tdamping_array


def load_particles(ii):    
    part_file  = 'data%05d.npz' % ii             # Define target file
    input_path = particle_dir + part_file        # File location
    data       = np.load(input_path)             # Load file
    
    tx               = data['pos']
    tv               = data['vel']
    tsim_time        = data['sim_time']
        
    return tx, tv, tsim_time


def extract_all_arrays():
    '''
    Extracts and saves all field arrays separate from the timestep slice files for easy
    access. Note that magnetic field arrays exclude the last value due to periodic
    boundary conditions. This may be changed later.
    '''
    num_field_steps    = len(os.listdir(field_dir)) 
    
    bx_arr, by_arr, bz_arr, damping_array = [np.zeros((num_field_steps, NC + 1)) for _ in range(4)]
    
    ex_arr,ey_arr,ez_arr,vex_arr,jx_arr,vey_arr,jy_arr,vez_arr,jz_arr,te_arr,qdns_arr\
    = [np.zeros((num_field_steps, NC)) for _ in range(11)]

    field_sim_time = np.zeros(num_field_steps)
    
    # Check that all components are extracted
    comps_missing = 0
    for component in ['bx', 'by', 'bz', 'ex', 'ey', 'ez']:
        check_path = temp_dir + component + '_array.npy'
        if os.path.isfile(check_path) == False:
            comps_missing += 1
        
    if comps_missing == 0:
        print('Field components already extracted.')
        return
    else:
        print('Extracting fields...')
        for ii in range(num_field_steps):
            print('Extracting field timestep {}'.format(ii))
            
            B, E, Ve, Te, J, q_dns, sim_time, damp = load_fields(ii)

            bx_arr[ii, :] = B[:, 0]
            by_arr[ii, :] = B[:, 1]
            bz_arr[ii, :] = B[:, 2]
            
            ex_arr[ii, :] = E[:, 0]
            ey_arr[ii, :] = E[:, 1]
            ez_arr[ii, :] = E[:, 2]

            jx_arr[ii, :] = J[:, 0]
            jy_arr[ii, :] = J[:, 1]
            jz_arr[ii, :] = J[:, 2]
            
            vex_arr[ii, :] = Ve[:, 0]
            vey_arr[ii, :] = Ve[:, 1]
            vez_arr[ii, :] = Ve[:, 2]
            
            te_arr[  ii, :]      = Te
            qdns_arr[ii, :]      = q_dns
            field_sim_time[ii]   = sim_time
            damping_array[ii, :] = damp
        
        np.save(temp_dir + 'bx' +'_array.npy', bx_arr)
        np.save(temp_dir + 'by' +'_array.npy', by_arr)
        np.save(temp_dir + 'bz' +'_array.npy', bz_arr)
        
        np.save(temp_dir + 'ex' +'_array.npy', ex_arr)
        np.save(temp_dir + 'ey' +'_array.npy', ey_arr)
        np.save(temp_dir + 'ez' +'_array.npy', ez_arr)
        
        np.save(temp_dir + 'jx' +'_array.npy', jx_arr)
        np.save(temp_dir + 'jy' +'_array.npy', jy_arr)
        np.save(temp_dir + 'jz' +'_array.npy', jz_arr)
        
        np.save(temp_dir + 'vex' +'_array.npy', vex_arr)
        np.save(temp_dir + 'vey' +'_array.npy', vey_arr)
        np.save(temp_dir + 'vez' +'_array.npy', vez_arr)
        
        np.save(temp_dir + 'te'    +'_array.npy', te_arr)
        np.save(temp_dir + 'qdens' +'_array.npy', qdns_arr)
        np.save(temp_dir + 'damping_array' +'_array.npy', damping_array)
        np.save(temp_dir + 'field_sim_time' +'_array.npy', field_sim_time)
        
        print('Field component arrays saved in {}'.format(temp_dir))
    return


def get_array(component='by', get_all=False, timebase=None):
    '''
    Input  : Array Component
    Output : Array as np.ndarray
    
    Components:
        3D (x, y, z) -- B, E, J, Ve
        1D           -- qdens, Te, damping_array
    
    kwargs:
        get_all  :: Flag to retrieve all recorded fields (Default False)
        timebase :: Flag to change timebase to 'gyperiod' or 'radperiod' (Default seconds)
    '''
    if get_all == False:
        arr_path   = temp_dir + component.lower() + '_array' + '.npy'
        arr        = np.load(arr_path)
        ftime_sec  = dt_field * np.arange(arr.shape[0])
        
        if timebase == 'gyperiod':
            ftime  = ftime_sec / gyperiod
        elif timebase == 'radperiod':
            ftime = ftime_sec * gyfreq 
        else:
            ftime = ftime_sec
        
        return ftime, arr
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
        damping_array = np.load(temp_dir + 'damping_array' +'_array.npy')
        field_sim_time = np.load(temp_dir + 'field_sim_time' +'_array.npy')
        
        ftime_sec = dt_field * np.arange(bx.shape[0])

        if timebase == 'gyperiod':
            ftime = ftime_sec / gyperiod
        elif timebase == 'radperiod':
            ftime = ftime_sec * gyfreq 
        else:
            ftime = ftime_sec

        return ftime, bx, by, bz, ex, ey, ez, vex, vey, vez, te, jx, jy, jz, qdens, field_sim_time, damping_array


def interpolate_fields_to_particle_time(num_particle_steps, timebase=None):
    '''
    For each particle timestep, interpolate field values
    '''
    ftime, bx, by, bz, ex, ey, ez, vex, vey, vez,\
    te, jx, jy, jz, qdens, fsim_time, damping_array = get_array(get_all=True)
    
    ptime_sec = np.arange(num_particle_steps) * dt_particle
    
    pbx, pby, pbz = [np.zeros((num_particle_steps, NC + 1)) for _ in range(3)]
    
    for ii in range(NC + 1):
        pbx[:, ii] = np.interp(ptime_sec, ftime, bx[:, ii])
        pby[:, ii] = np.interp(ptime_sec, ftime, by[:, ii])
        pbz[:, ii] = np.interp(ptime_sec, ftime, bz[:, ii])
    
    pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens = [np.zeros((num_particle_steps, NC)) for _ in range(11)]
    
    for ii in range(NC):
        pex[:, ii]    = np.interp(ptime_sec, ftime, ex[:, ii])
        pey[:, ii]    = np.interp(ptime_sec, ftime, ey[:, ii])
        pez[:, ii]    = np.interp(ptime_sec, ftime, ez[:, ii])
        pvex[:, ii]   = np.interp(ptime_sec, ftime, vex[:, ii])
        pvey[:, ii]   = np.interp(ptime_sec, ftime, vey[:, ii])
        pvez[:, ii]   = np.interp(ptime_sec, ftime, vez[:, ii])
        pte[:, ii]    = np.interp(ptime_sec, ftime, te[:, ii])
        pjx[:, ii]    = np.interp(ptime_sec, ftime, jx[:, ii])
        pjy[:, ii]    = np.interp(ptime_sec, ftime, jy[:, ii])
        pjz[:, ii]    = np.interp(ptime_sec, ftime, jz[:, ii])
        pqdens[:, ii] = np.interp(ptime_sec, ftime, qdens[:, ii])

    return ptime_sec, pbx, pby, pbz, pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens


@nb.njit()
def create_idx():
    N_part = cellpart * NX
    idx    = np.zeros(N_part, dtype=nb.int32)
    
    for jj in range(Nj):
        idx[idx_start[jj]: idx_end[jj]] = jj
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


