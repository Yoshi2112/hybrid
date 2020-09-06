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
def load_run(drive, series, run_num, lmissing_t0_offset=0, extract_arrays=True, output_param_file=True):
    global missing_t0_offset
    missing_t0_offset = lmissing_t0_offset   # Flag for when I thought having a t=0 save file wasn't needed. I was wrong.
    manage_dirs(drive, series, run_num)
    load_simulation_params()
    load_species_params()
    initialize_simulation_variables()
    
    if extract_arrays == True:
        extract_all_arrays()
        
    if output_param_file == True:
        output_simulation_parameter_file(series, run_num)
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
        
    try:
        sim_repr   = p_data['sim_repr']
    except:
        pass
        
    try:
        n_contr    = density / (cellpart*sim_repr)                              # Species density contribution: Each macroparticle contributes this density to a cell
    except:
        n_contr    = density / nsp_ppc                                          # Species density contribution: Each macroparticle contributes this density to a cell

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
           particle_shape, part_save_iter, field_save_iter, dt_field, dt_particle, \
           HM_amplitude, HM_frequency, nsp_ppc, q, mp, e0, mu0, kB

    q   = 1.602e-19               # Elementary charge (C)
    mp  = 1.67e-27                # Mass of proton (kg)
    e0  = 8.854e-12               # Epsilon naught - permittivity of free space
    mu0 = (4e-7) * np.pi          # Magnetic Permeability of Free Space (SI units)
    kB  = 1.380649e-23            # Boltzmann's Constant (J/K)


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
        nsp_ppc      = obj['nsp_ppc']
    except:
        HM_amplitude = 0
        HM_frequency = 0
        nsp_ppc      = 0
    
    dt_field        = dt_sim * field_save_iter                         # Time between data slices (seconds)
    dt_particle     = dt_sim * part_save_iter
    return 


def output_simulation_parameter_file(series, run, overwrite=True):
    output_file = run_dir + 'simulation_parameter_file.txt'

    if os.path.exists(output_file) == True and overwrite == False:
        pass
    else:
        xmin = 0.0
        xmax = NX * dx
        
        beta_e   = (2 * mu0 * ne * kB * Te0 ) / B0 ** 2
        beta_par = (2 * mu0 * ne * kB * Tpar) / B0 ** 2
        beta_per = (2 * mu0 * ne * kB * Tper) / B0 ** 2
        
        with open(output_file, 'w') as f:
            print('HYBRID SIMULATION :: PARAMETER FILE', file=f)
            print('', file=f)
            print('Series[run]  :: {}[{}]'.format(series, run), file=f)
            print('Series Desc. :: {}'.format(run_desc), file=f)
            print('Hybrid Type  :: {}'.format(method_type), file=f)
            print('Random Seed  :: {}'.format(seed), file=f)
            print('', file=f)
            print('Temporal Parameters', file=f)
            print('Maximum Sim. Time  :: {} GPeriods'.format(max_rev), file=f)
            print('Simulation cadence :: {} seconds'.format(dt_sim), file=f)
            print('Particle Dump Time :: {} seconds'.format(dt_particle), file=f)
            print('Field Dump Time    :: {} seconds'.format(dt_field), file=f)
            print('Frequency Resol.   :: {} GPeriods'.format(freq_res), file=f)
            print('Gyperiod Resol.    :: {}'.format(orbit_res), file=f)
            print('', file=f)
            print('Simulation Parameters', file=f)
            print('# Spatial Cells :: {}'.format(NX), file=f)
            print('Ion L per cell  :: {}'.format(dxm), file=f)
            print('Cell width      :: {} m'.format(dx), file=f)
            print('Simulation Min  :: {} m'.format(xmin), file=f)
            print('Simulation Max  :: {} m'.format(xmax), file=f)
            print('', file=f)
            print('Background Field Strength :: {} nT'.format(B0*1e9), file=f)
            
            try:
                print('B0 angle to Simulation    :: {} deg'.format(theta * 180. / np.pi), file=f)
            except:
                pass
            
            print('', file=f)
            print('Electron Density     :: {} /cc'.format(ne*1e-6), file=f)
            print('Electron Treatment   :: {}'.format(ie), file=f)
            print('Electron Temperature :: {}K'.format(Te0), file=f)
            print('', file=f)
            print('Particle Parameters', file=f)
            print('Number of Species   :: {}'.format(Nj), file=f)
            print('Particles per Cell  :: {}'.format(cellpart), file=f)
            print('Particle Shape Func :: {}'.format(particle_shape), file=f)
            print('', file=f)
            print('Ion Composition', file=f)
            print('Species Name    :: {}'.format(species_lbl), file=f)
            print('Species Type    :: {}'.format(temp_type), file=f)
            print('Species Dens    :: {} /cc'.format(density*1e-6), file=f)
            print('Species Charge  :: {}'.format(charge), file=f)
            print('Species Mass    :: {} kg'.format(mass), file=f)
            print('Drift Velocity  :: {} m/s'.format(drift_v), file=f)
            print('Perp Temp       :: {} K'.format(Tper), file=f)
            print('Para Temp       :: {} K'.format(Tpar), file=f)
            print('MParticle Contr.:: {} real particles/macroparticle'.format(n_contr), file=f)
            print('', file=f)
            print('Elec Beta       :: {}'.format(beta_e), file=f)
            print('Perp Beta       :: {}'.format(beta_per), file=f)
            print('Para Beta       :: {}'.format(beta_par), file=f)
    return


def initialize_simulation_variables():
    global wpi, gyfreq, gyperiod, time_seconds_field, time_seconds_particle, \
           time_gperiods_field, time_gperiods_particle, time_radperiods_field, time_radperiods_particle, va
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

def extract_all_arrays():
    '''
    Extracts and saves all field arrays separate from the timestep slice files for easy
    access. Note that magnetic field arrays exclude the last value due to periodic
    boundary conditions. This may be changed later.
    '''
    bx_arr,ex_arr,by_arr,ey_arr,bz_arr,ez_arr,vex_arr,jx_arr,vey_arr,jy_arr,vez_arr,jz_arr,te_arr,qdns_arr\
    = [np.zeros((num_field_steps, NX)) for _ in range(14)]

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
            
            B, E, Ve, Te, J, q_dns = load_fields(ii + missing_t0_offset)
            
            bx_arr[ii, :] = B[:-1, 0]
            by_arr[ii, :] = B[:-1, 1]
            bz_arr[ii, :] = B[:-1, 2]
            
            ex_arr[ii, :] = E[:, 0]
            ey_arr[ii, :] = E[:, 1]
            ez_arr[ii, :] = E[:, 2]
            
            try:
                jx_arr[ii, :] = J[:, 0]
                jy_arr[ii, :] = J[:, 1]
                jz_arr[ii, :] = J[:, 2]
            except:
                '''
                Catch for some model runs where I was saving charge density in place of current density
                'Cause I am dumb
                Will just return a zero array instead, no harm (just missing data)
                '''
                pass
            
            vex_arr[ii, :] = Ve[:, 0]
            vey_arr[ii, :] = Ve[:, 1]
            vez_arr[ii, :] = Ve[:, 2]
            
            te_arr[  ii, :] = Te
            qdns_arr[ii, :] = q_dns
        
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
        
        print('Field component arrays saved in {}'.format(temp_dir))
    return


def get_array(component='by', get_all=False):
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


