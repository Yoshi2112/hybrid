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
def load_run(drive, series, run_num, extract_arrays=True, print_summary=True):
    manage_dirs(drive, series, run_num)
    load_simulation_params()
    load_species_params()
    initialize_simulation_variables()
        
    if print_summary == True:
        output_simulation_parameter_file(series, run_num)
    return


def manage_dirs(drive, series, run_num):
    global run_dir, data_dir, anal_dir, temp_dir, base_dir, field_dir, particle_dir, num_field_steps, num_particle_steps
    
    base_dir = '{}/runs/{}/'.format(drive, series)                      # Main series directory, containing runs
    run_dir  = '{}/runs/{}/run_{}/'.format(drive, series, run_num)      # Main run directory
    data_dir = run_dir + 'data/'                                        # Directory containing .npz output files for the simulation run
    anal_dir = run_dir + 'analysis/'                                    # Output directory for all this analysis (each will probably have a subfolder)
    temp_dir = run_dir + 'extracted/'                                   # Saving things like matrices so we only have to do them once

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
           temp_type, temp_color, Tpar, species_lbl, n_contr,  \
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
    global Nj, ne, NX, dxm, seed, dx, Te0, dt_sim, max_rev,   \
           ie, run_desc, orbit_res, freq_res, method_type,    \
           particle_shape, part_save_iter, field_save_iter, dt_field, dt_particle,  \
           ND, NC, N, r_damp, xmax, B_xmax, B_eq, theta_xmax, a, boundary_type,     \
           particle_boundary, rc_hwidth, L, B_nodes, E_nodes, xmin, grid_min, grid_max, grid_mid

    h_name = os.path.join(data_dir, 'simulation_parameters.pckl')       # Load header file
    f      = open(h_name, 'rb')                                         # Open header file
    obj    = pickle.load(f)                                             # Load variables from header file into python object
    f.close()                                                           # Close header file
    
    seed              = obj['seed']
    Nj                = obj['Nj']
    dt_sim            = obj['dt']
    NX                = obj['NX']
    N                 = obj['N']
    dxm               = obj['dxm']
    dx                = obj['dx']
    ne                = obj['ne']
    xmax              = obj['xmax']
    xmin              = obj['xmin']
    B_xmax            = obj['B_xmax']
    B_eq              = obj['B_eq']
    a                 = obj['a']
    L                 = obj['L']
    rc_hwidth         = obj['rc_hwidth']
    theta_xmax        = obj['theta_xmax']
    max_rev           = obj['max_rev']
    orbit_res         = obj['orbit_res']
    run_desc          = obj['run_desc']
    method_type       = obj['method_type'] 
    particle_shape    = obj['particle_shape']
    boundary_type     = obj['boundary_type']
    particle_boundary = obj['particle_boundary']
    
    part_save_iter    = obj['part_save_iter']
    
    dt_particle       = dt_sim * part_save_iter
                
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


def output_simulation_parameter_file(series, run):
    output_file = run_dir + 'simulation_parameter_file.txt'
    
    if os.path.exists(output_file) == True:
        pass
    else:
        with open(output_file, 'a') as f:
            print('HYBRID SIMULATION :: PARAMETER FILE', file=f)
            print('', file=f)
            print('Series[run]  :: {}[{}]'.format(series, run), file=f)
            print('Series Desc. :: {}'.format(run_desc), file=f)
            print('Hybrid Type  :: {}'.format(method_type), file=f)
            print('Random Seed  :: {}'.format(seed), file=f)
            print('', file=f)
            print('Temporal Paramters', file=f)
            print('Maximum Sim. Time : {} GPeriods'.format(max_rev), file=f)
            print('Simulation cadence: {} seconds'.format(dt_sim), file=f)
            print('Particle Dump Time: {} seconds'.format(dt_particle), file=f)
            print('Gyperiod Resol.   : {}'.format(orbit_res), file=f)
            print('', file=f)
            print('Simulation Parameters', file=f)
            print('# Spatial Cells :: {}'.format(NX), file=f)
            print('# Damping Cells :: {}'.format(ND), file=f)
            print('# Cells Total   :: {}'.format(NC), file=f)
            print('Ion L per cell  :: {}'.format(dxm), file=f)
            print('Cell width      :: {} m'.format(dx), file=f)
            print('Simulation Min  :: {} m'.format(xmin), file=f)
            print('Simulation Max  :: {} m'.format(xmax), file=f)
            print('', file=f)
            print('Equatorial Field Strength :: {} nT'.format(B_eq*1e9), file=f)
            print('Boundary   Field Strength :: {} nT'.format(B_xmax*1e9), file=f)
            print('MLAT max/min extent       :: {} deg'.format(theta_xmax), file=f)
            print('McIlwain L value equiv.   :: {}'.format(L), file=f)
            print('', file=f)
            print('Electron Density   : {} /cc'.format(ne*1e-6), file=f)
            print('Electron Treatment : {}'.format(ie), file=f)
            print('Electron Temperature : {}K'.format(Te0), file=f)
            print('', file=f)
            print('Particle Parameters', file=f)
            print('Number of Species   :: {}'.format(Nj), file=f)
            print('Number of Particles :: {}'.format(N), file=f)
            print('Species Per Cell    :: {}'.format(nsp_ppc), file=f)
            print('Species Particles # :: {}'.format(N_species), file=f)
            print('', file=f)
            print('Ion Composition', file=f)
            print('Species Name :: {}'.format(species_lbl), file=f)
            print('Species Type :: {}'.format(temp_type), file=f)
            print('Species Dens :: {} /cc'.format(density*1e-6), file=f)
            print('Species Charge :: {}'.format(charge), file=f)
            print('Species Mass  :: {} kg'.format(mass), file=f)
            print('Drift Velocity :: {} m/s'.format(drift_v), file=f)
            print('Perp Temp :: {}'.format(Tper), file=f)
            print('Para Temp :: {}'.format(Tpar), file=f)
        
       #particle_shape, part_save_iter, field_save_iter, a, boundary_type,     \
        #particle_boundary, rc_hwidth
       # n_contr,  \
       # idx_start, idx_end
    return


def load_particles(ii):  
    part_file  = 'data%05d.npz' % ii             # Define target file
    input_path = particle_dir + part_file        # File location
    data       = np.load(input_path)             # Load file
    
    tx               = data['pos']
    tv               = data['vel']
    tsim_time        = data['sim_time']
    
    try:
        tidx = data['idx']
    except:
        tidx = None
        
    return tx, tv, tidx, tsim_time


