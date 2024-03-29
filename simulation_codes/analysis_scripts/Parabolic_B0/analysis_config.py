import os, sys, pickle, shutil
import numpy as np
'''
Used to initialise values for a run
e.g. directories, simulation/particle parameters, derived quantities, etc.

These are called in the main script by import, and will change each time
load_run() is called

The global calls allow variables to be accessed in the main script without
clogging up its namespace - i.e. run-specific parameters are called by
using e.g. cf.B0
'''
def load_run(drive, series, run_num, extract_arrays=True, print_summary=True, overwrite_summary=False):
    manage_dirs(drive, series, run_num)
    load_simulation_params()
    load_species_params()
    initialize_simulation_variables()
    
    if extract_arrays == True:
        extract_all_arrays()
        
    if print_summary == True:
        output_simulation_parameter_file(series, run_num, overwrite_summary=overwrite_summary)
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
    global species_present, density, dist_type, charge, mass, Tperp,       \
           temp_type, temp_color, Tpar, species_lbl, n_contr,              \
           drift_v, N_species, Bc, nsp_ppc, Te0_arr, idx_start0, idx_end0, \
            vth_par, vth_perp

    p_path = os.path.join(data_dir, 'particle_parameters.npz')                  # File location
    p_data = np.load(p_path, allow_pickle=True )                                # Load file

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
    
    Bc         = p_data['Bc']
    
    idx_start0 = p_data['idx_start']
    idx_end0   = p_data['idx_end']

    Tpar       = p_data['Tpar']
    try:
        Tperp      = p_data['Tperp']
    except:
        Tperp      = p_data['Tper']
    
    vth_par     = p_data['vth_par']
    vth_perp    = p_data['vth_perp']

    try:
        Te0_arr = p_data['Te0']
    except:
        Te0_arr = np.ones(NC, dtype=np.float64) * Te0

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
    global Nj, ne, NX, dxm, seed, dx, Te0, dt_sim, max_wcinv,   \
           ie, run_desc, orbit_res, freq_res, method_type,    \
           particle_shape, part_save_iter, field_save_iter, dt_field, dt_particle,  \
           ND, NC, N, loss_cone, xmax, B_xmax, B_eq, theta_xmax, a, boundary_type,     \
           rc_hwidth, L, B_nodes, E_nodes, xmin, grid_min, grid_max, \
           grid_mid, run_time_str, particle_periodic, particle_reflect, \
           particle_reinit, particle_open, disable_waves, source_smoothing, \
           E_damping, quiet_start, homogenous, field_periodic, damping_multiplier, \
           driven_freq, driven_ampl, pulse_offset, pulse_offset, pulse_width, driven_k,\
           driver_status, num_threads, loop_time, max_inc,\
           x0B, x1B, x0E, x1E, subcycles

    h_name = os.path.join(data_dir, 'simulation_parameters.pckl')       # Load header file
    f      = open(h_name, 'rb')                                         # Open header file
    obj    = pickle.load(f)                                             # Load variables from header file into dict
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
    xmin              = obj['xmin']
    xmax              = obj['xmax']
    B_xmax            = obj['B_xmax']
    B_eq              = obj['B_eq']
    a                 = obj['a']
    L                 = obj['L']
    loss_cone         = obj['loss_cone']
    rc_hwidth         = obj['rc_hwidth']
    theta_xmax        = obj['theta_xmax']
    orbit_res         = obj['orbit_res']
    freq_res          = obj['freq_res']
    run_desc          = obj['run_desc']
    method_type       = obj['method_type'] 
    particle_shape    = obj['particle_shape']
    
    max_inc           = obj['max_inc']
    try:
        max_wcinv     = obj['max_wcinv']
    except:
        max_wcinv     = obj['max_rev'] / (2*np.pi)
    
    try:
        damping_multiplier = obj['damping_multiplier']
    except:
        damping_multiplier = None

    part_save_iter    = obj['part_save_iter']
    field_save_iter   = obj['field_save_iter']
    
    dt_field          = dt_sim * field_save_iter                        # Time between data slices (seconds)
    dt_particle       = dt_sim * part_save_iter
    subcycles         = obj['subcycles']
    
    particle_periodic = obj['particle_periodic']
    particle_reflect  = obj['particle_reflect']
    particle_reinit   = obj['particle_reinit']

    if particle_reinit + particle_reflect + particle_periodic == 0:
        particle_open = 1
    else:
        particle_open = 0
        
    disable_waves    = obj['disable_waves']
    source_smoothing = obj['source_smoothing']
    E_damping        = obj['E_damping']
    quiet_start      = obj['quiet_start']
    homogenous       = obj['homogeneous']
    field_periodic   = obj['field_periodic']
    
    if obj['run_time'] is None:
        run_time = 0
        run_time_str = 'Incomplete'
    else:
        run_time = obj['run_time']
        
        hrs      = int(run_time // 3600)
        rem      = run_time %  3600
        mins     = int(rem // 60)
        sec      = round(rem %  60, 2)
        run_time_str = '{:02}:{:02}:{:02}'.format(hrs, mins, sec)
    
    if obj['loop_time'] is None:
        loop_time = 'N/A'
    else:
        loop_time = round(obj['loop_time'], 3)
    
    try:
        # Test if scalar
        print(Te0[0])
    except:
        # If it is, make it a vector
        Te0 = np.ones(NC, dtype=float) * Te0

    driven_freq   = obj['driven_freq']
    driven_ampl   = obj['driven_ampl']
    pulse_offset  = obj['pulse_offset']
    pulse_offset  = obj['pulse_offset']
    pulse_width   = obj['pulse_width']
    driven_k      = obj['driven_k']
    driver_status = obj['pol_wave'] 
    num_threads   = obj['num_threads']

    # Set spatial boundaries and gridpoints
    x0B, x1B = ND, ND + NX + 1
    x0E, x1E = ND, ND + NX
    
    B_nodes  = (np.arange(NC + 1) - NC // 2)       * dx                 # B grid points position in space
    E_nodes  = (np.arange(NC)     - NC // 2 + 0.5) * dx                 # E grid points position in space    
    return 


def initialize_simulation_variables():
    global wpi, gyfreq, gyfreq_eq, gyfreq_xmax, gyperiod, va
    q   = 1.602e-19               # Elementary charge (C)
    mp  = 1.673e-27               # Mass of proton (kg)
    e0  = 8.854e-12               # Epsilon naught - permittivity of free space
    mu0 = (4e-7) * np.pi          # Magnetic Permeability of Free Space (SI units)
    
    wpi        = np.sqrt(ne * q ** 2 / (mp * e0))            # Ion plasma frequency
    gyfreq     = q * B_eq   / mp                             # Proton gyrofrequency (rad/s) (compatibility)
    gyfreq_eq  = q * B_eq   / mp                             # Proton gyrofrequency (rad/s) (equator)
    gyfreq_xmax= q * B_xmax / mp                             # Proton gyrofrequency (rad/s) (boundary)
    gyperiod   = (mp * 2 * np.pi) / (q * B_eq)               # Proton gyroperiod (s)
    va         = B_eq / np.sqrt(mu0*ne*mp)                   # Alfven speed: Assuming pure proton plasma
    return


def output_simulation_parameter_file(series, run, overwrite_summary=False):
    '''
    To do:
        -- Reformat to make particle parameters clearer
        -- Use thermal velocities, energies, and betas instead of temperature
    '''
    q   = 1.602e-19               # Elementary charge (C)
    mp  = 1.673e-27               # Mass of proton (kg)
    
    output_file = run_dir + 'simulation_parameter_file.txt'

    if particle_open == 1:
        particle_boundary = 'Open'
    elif particle_reinit == 1:
        particle_boundary = 'Reinitialize'
    elif particle_reflect == 1:
        particle_boundary = 'Reflection'
    elif particle_periodic == 1:
        particle_boundary = 'Periodic'
    else:
        particle_boundary = '-'

    if ie == 0:
        electron_treatment = 'Isothermal'
    elif ie == 1:
        electron_treatment = 'Adiabatic'
    else:
        electron_treatment = 'Other'

    echarge  = charge / q
    pmass    = mass   / mp
    va_drift = drift_v / va

    if os.path.exists(output_file) == True and overwrite_summary == False:
        pass
    else:
        with open(output_file, 'w') as f:
            print('HYBRID SIMULATION :: PARAMETER FILE', file=f)
            print('', file=f)
            print('Series[run]   :: {}[{}]'.format(series, run), file=f)
            print('Series Desc.  :: {}'.format(run_desc), file=f)
            print('Hybrid Type   :: {}'.format(method_type), file=f)
            print('Random Seed   :: {}'.format(seed), file=f)
            print('Final runtime :: {}'.format(run_time_str), file=f)
            print('Av. loop time :: {}'.format(loop_time), file=f)
            print('N_loops_start :: {}'.format(max_inc), file=f)
            print('', file=f)
            print('Flags', file=f)
            print('Disable Wave Growth:: {}'.format(disable_waves), file=f)
            print('Source Smoothing   :: {}'.format(source_smoothing), file=f)
            print('E-field Damping    :: {}'.format(E_damping), file=f)
            print('Quiet Start        :: {}'.format(quiet_start), file=f)
            print('Homogenous B0      :: {}'.format(homogenous), file=f)
            print('Field Periodic BCs :: {}'.format(field_periodic), file=f)
            print('', file=f)
            print('Temporal Parameters', file=f)
            print('Maximum Sim. Time  :: {}     wcinv'.format(max_wcinv), file=f)
            print('Maximum Sim. Time  :: {}     seconds'.format(round(max_wcinv/gyfreq, 1)), file=f)
            print('Simulation cadence :: {:.5f} seconds'.format(dt_sim), file=f)
            print('Particle Dump Time :: {:.5f} seconds'.format(dt_particle), file=f)
            print('Field Dump Time    :: {:.5f} seconds'.format(dt_field), file=f)
            print('Frequency Resol.   :: {:.5f} gyroperiods'.format(freq_res), file=f)
            print('Gyro-orbit Resol.  :: {:.5f} gyroperiods'.format(orbit_res), file=f)
            print('Subcycles init.    :: {} '.format(subcycles), file=f)
            print('', file=f)
            print('Simulation Parameters', file=f)
            print('# Spatial Cells    :: {}'.format(NX), file=f)
            print('# Damping Cells    :: {}'.format(ND), file=f)
            print('# Cells Total      :: {}'.format(NC), file=f)
            print('va/pcyc per dx     :: {}'.format(dxm), file=f)
            print('Cell width         :: {:.1f} km'.format(dx*1e-3), file=f)
            print('Simulation Min     :: {:.1f} km'.format(xmin*1e-3), file=f)
            print('Simulation Max     :: {:.1f} km'.format(xmax*1e-3), file=f)
            if damping_multiplier is not None:
                print('Damping Multipl.   :: {:.2f}'.format(damping_multiplier), file=f)
            else:
                print('Damping Multipl.   ::', file=f)
            print('', file=f)
            print('Equatorial B0       :: {:.2f} nT'.format(B_eq*1e9), file=f)
            print('Boundary   B0       :: {:.2f} nT'.format(B_xmax*1e9), file=f)
            print('max MLAT            :: {:.2f} deg'.format(theta_xmax * 180. / np.pi), file=f)
            print('McIlwain L value    :: {:.2f}'.format(L), file=f)
            print('Parabolic s.f. (a)  :: {}'.format(a), file=f)
            print('', file=f)
            print('Electron Density    :: {} /cc'.format(ne*1e-6), file=f)
            print('Electron Treatment  :: {}'.format(electron_treatment), file=f)
            #print('Electron Temperature :: {}K'.format(Te0), file=f)
            #print('Electron Beta        :: {}'.format(beta_e), file=f)
            print('', file=f)
            print('Particle Parameters', file=f)
            print('Number of Species   :: {}'.format(Nj), file=f)
            print('Number of Particles :: {}'.format(N), file=f)
            print('Species Per Cell    :: {}'.format(nsp_ppc), file=f)
            print('Species Particles # :: {}'.format(N_species), file=f)
            print('Particle Shape Func :: {}'.format(particle_shape), file=f)
            print('Particle Bound. Cond:: {}'.format(particle_boundary), file=f)
            print('', file=f)
            
            
            print('Ion Composition', file=f)
            
            ccdens   = density*1e-6
            if vth_par is not None:
                va_para = vth_par  / va
                va_perp = vth_perp / va
            else:
                va_para = np.zeros(echarge.shape)
                va_perp = np.zeros(echarge.shape)
            
            species_str = temp_str = cdens_str = charge_str = va_perp_str = \
            va_para_str = mass_str = drift_str = contr_str = ''
            for ii in range(Nj):
                species_str += '{:>13}'.format(species_lbl[ii])
                temp_str    += '{:>13d}'.format(temp_type[ii])
                cdens_str   += '{:>13.3f}'.format(ccdens[ii])
                charge_str  += '{:>13.1f}'.format(echarge[ii])
                mass_str    += '{:>13.1f}'.format(pmass[ii])
                drift_str   += '{:>13.1f}'.format(va_drift[ii])
                va_perp_str += '{:>13.2f}'.format(va_perp[ii])
                va_para_str += '{:>13.2f}'.format(va_para[ii])
                contr_str   += '{:>13.1f}'.format(n_contr[ii])
    
            print('Species Name    :: {}'.format(species_str), file=f)
            print('Species Type    :: {}'.format(temp_str), file=f)
            print('Species Dens    :: {}  /cc'.format(cdens_str), file=f)
            print('Species Charge  :: {}  elementary units'.format(charge_str), file=f)
            print('Species Mass    :: {}  proton masses'.format(mass_str), file=f)
            print('Drift Velocity  :: {}  vA'.format(drift_str), file=f)
            print('V_thermal Perp  :: {}  vA'.format(va_perp_str), file=f)
            print('V_thermal Para  :: {}  vA'.format(va_para_str), file=f)
            print('MParticle s.f   :: {}  real particles/macroparticle'.format(contr_str), file=f)
    return


def delete_analysis_folders(drive, series, run_num):
    '''
    Used as a blunt tool for when incomplete runs are analysed
    and you want to do a full one later on.
    '''
    print('Deleting analysis and temp folders.')
    run_dir  = '{}/runs/{}/run_{}/'.format(drive, series, run_num)      # Main run directory
    anal_dir = run_dir + 'analysis/'                                    # Output directory for all this analysis (each will probably have a subfolder)
    temp_dir = run_dir + 'extracted/'                                   # Saving things like matrices so we only have to do them once

    # Delete directory and contents
    for directory in [anal_dir, temp_dir]:
        if os.path.exists(directory) == True:
            shutil.rmtree(directory)
    
    # Delete summary file
    param_file = run_dir + 'simulation_parameter_file.txt'
    if os.path.exists(param_file) == True:
        os.remove(param_file)
    return


def load_fields(ii):
    '''
    Note: 'dns' is charge density (usually q_dns)
    '''
    field_file = 'data%05d.npz' % ii             # Define target file
    input_path = field_dir + field_file          # File location
    data       = np.load(input_path)             # Load file

    tB               = data['B']
    tE               = data['E']
    tVe              = data['Ve']
    tTe              = data['Te']
    try:
        tJ               = data['Ji']
    except:
        tJ               = data['J']
    tdns             = data['dns']
    tsim_time        = data['sim_time']
    
    try:
        tdamping_array = data['damping_array']
    except:
        tdamping_array = None
        
    try:
        tB_cent = data['B_cent']
    except:
        tB_cent = None

    return tB, tB_cent, tE, tVe, tTe, tJ, tdns, tsim_time, tdamping_array


def load_particles(ii, shuffled_idx=False, preparticledata=False):  
    '''
    Sort kw for arranging particles by species index since they get jumbled.
    
    Still need to test the jumble fixer more in depth (i.e. multispecies etc.)
    
    Also need to group disabled particles with their active counterparts... but then
    there's a problem with having all 'spare' particles having index -128 for some
    runs - they're not 'deactivated cold particles', they were never active in the
    first place
    
    SOLUTION: For subsequent runs, use idx = -1 for 'spare' particles, since their
    index will be reset when they're turned on.
    
    ACTUALLY, the current iteration of the code doesn't use spare particles. idx_start/end
    should still be valid, maybe put a flag in for it under certain circumstances (For speed)
    
    Flag will be anything that involves spare particles. Load it later (nothing like
    that exists yet).
    '''    
    if preparticledata == True:
        pdir = data_dir + '//equil_particles//'
    else:
        pdir = particle_dir
        
    part_file  = 'data%05d.npz' % ii
    input_path = pdir + part_file
    data       = np.load(input_path)
    
    tx         = data['pos']
    tv         = data['vel']
    tsim_time  = data['sim_time']
    tidx       = data['idx']

    if shuffled_idx == True or particle_open == True or particle_reinit == True:
        order = np.argsort(tidx)                # Retrieve order of elements by index
        tidx  = tidx[order]
        tx    = tx[order]
        tv    = tv[:, order]
    
        idx_start = np.zeros(Nj, dtype=int)
        idx_end   = np.zeros(Nj, dtype=int)
            
        # Get first index of each species. If None found,  
        acc = 0
        for jj in range(Nj):
            found_st = 0; found_en = 0
            for ii in range(acc, tidx.shape[0]):
                if tidx[ii] >= 0:
                    
                    # Find start point (Store, and keep looping through particles)
                    if tidx[ii] == jj and found_st == 0:
                        idx_start[jj] = ii
                        found_st = 1
                        
                    # Find end point (Find first value that doesn't match, if start is found
                    elif tidx[ii] != jj and found_st == 1:
                        idx_end[jj] = ii; found_en = 1; acc = ii
                        break
                    
            # This should only happen with last species in array
            if found_st == 1 and found_en == 0:
                idx_end[jj] = tidx.shape[0]
    else:
        idx_start = idx_start0
        idx_end   = idx_end0
    return tx, tv, tidx, tsim_time, idx_start, idx_end


def unwrap_particle_files():
    '''
    Have option to delete original particle files? Although that will cause
    issues if you want to look at particle quantities per timestep
    '''
    particle_folder = data_dir + '/particles_single/'
    if os.path.exists(particle_folder) == False:
        os.makedirs(particle_folder)
        
    n_times = len(os.listdir(particle_dir))
    particle_array = np.memmap(data_dir + 'all_particles.dat', dtype=np.float64,
                               mode='w+', shape=(n_times, N, 5))
    
    for ii in range(n_times):
        print('Accumulating timestep {} of {}'.format(ii, n_times))
        pos, vel, idx, sim_time, idx_start, idx_end = load_particles(ii)
        
        particle_array[ii, :, 0] = pos[:]
        particle_array[ii, :, 1] = vel[0, :]
        particle_array[ii, :, 2] = vel[1, :]
        particle_array[ii, :, 3] = vel[2, :]
        particle_array[ii, :, 4] = idx[:]
        
        particle_array.flush()
    return


def extract_all_arrays():
    '''
    Extracts and saves all field arrays separate from the timestep slice files for easy
    access. Note that magnetic field arrays exclude the last value due to periodic
    boundary conditions. This may be changed later.
    
    TODO: Have option to delete files once extracted. This probably won't get
    used much, but in the event more storage space is needed, extracted files
    are just duplicating the data and originals aren't needed.
    '''
    # Check if field files exist:
    if len(os.listdir(field_dir)) == 0:
        print('No field files found, skipping extraction.')
        return
    
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
        num_field_steps    = len(os.listdir(field_dir)) 
        
        # Load to specify arrays
        zB, zB_cent, zE, zVe, zTe, zJ, zq_dns, zsim_time, zdamp = load_fields(0)

        bx_arr, by_arr, bz_arr, damping_array = [np.zeros((num_field_steps, zB.shape[0])) for _ in range(4)]
        
        if zB_cent is not None:
            bxc_arr, byc_arr, bzc_arr = [np.zeros((num_field_steps, zB_cent.shape[0])) for _ in range(3)]
        
        ex_arr,ey_arr,ez_arr,vex_arr,jx_arr,vey_arr,jy_arr,vez_arr,jz_arr,te_arr,qdns_arr\
        = [np.zeros((num_field_steps, zE.shape[0])) for _ in range(11)]
    
        field_sim_time = np.zeros(num_field_steps)
    
        print('Extracting fields...')
        for ii in range(num_field_steps):
            sys.stdout.write('\rExtracting field timestep {}'.format(ii))
            sys.stdout.flush()
            
            B, B_cent, E, Ve, Te, J, q_dns, sim_time, damp = load_fields(ii)

            bx_arr[ii, :] = B[:, 0]
            by_arr[ii, :] = B[:, 1]
            bz_arr[ii, :] = B[:, 2]
            
            if B_cent is not None:
                bxc_arr[ii, :] = B_cent[:, 0]
                byc_arr[ii, :] = B_cent[:, 1]
                bzc_arr[ii, :] = B_cent[:, 2]
            
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
        print('\nExtraction Complete.')
        
        np.save(temp_dir + 'bx' +'_array.npy', bx_arr)
        np.save(temp_dir + 'by' +'_array.npy', by_arr)
        np.save(temp_dir + 'bz' +'_array.npy', bz_arr)
        
        if B_cent is not None:
            np.save(temp_dir + 'bxc' +'_array.npy', bxc_arr)
            np.save(temp_dir + 'byc' +'_array.npy', byc_arr)
            np.save(temp_dir + 'bzc' +'_array.npy', bzc_arr)
        
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
        
     Arrays are (time, space)
    '''
    if get_all == False:
        arr_path   = temp_dir + component.lower() + '_array' + '.npy'
        if os.path.exists(arr_path) == True:
            arr = np.load(arr_path)
        else:
            print('File {} does not exist.'.format(component.lower() + '_array' + '.npy'))
            arr = None
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
    For each particle timestep, interpolate field values. Arrays are (time, space)
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