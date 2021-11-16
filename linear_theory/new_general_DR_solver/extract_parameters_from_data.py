# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:42:27 2019

@author: Yoshi
"""
import os, sys, pdb
sys.path.append('D://Google Drive//Uni//PhD 2017//Data//Scripts//')
import numpy as np
from   scipy.optimize          import fsolve
import rbsp_file_readers   as rfr
import rbsp_fields_loader  as rfl
import analysis_scripts    as ascr
import crres_file_readers  as cfr


def HOPE_interpolate_to_time(new_time, HOPE_time, HOPE_dens, HOPE_temp, HOPE_anis):
    '''
    edens_time  :: WAVES electron density time to interpolate data_array to (length M)
    data_time   :: Current HOPE sample times of length N
    data_array  :: Data arrays consisting of ni, Ti, Ai in a 3xN ndarra
    '''
    new_dens = np.zeros((3, new_time.shape[0]), dtype=np.float64)
    new_temp = np.zeros((3, new_time.shape[0]), dtype=np.float64)
    new_anis = np.zeros((3, new_time.shape[0]), dtype=np.float64)
    
    xi = new_time.astype(np.int64)
    xp = HOPE_time.astype(np.int64)
    
    for ii in range(3):
        new_dens[ii, :] = np.interp(xi, xp, HOPE_dens[ii, :])
        new_temp[ii, :] = np.interp(xi, xp, HOPE_temp[ii, :])
        new_anis[ii, :] = np.interp(xi, xp, HOPE_anis[ii, :])

    return new_dens, new_temp, new_anis


def SPICE_interpolate_to_time(new_time, SPICE_times, SPICE_dens, SPICE_temp, SPICE_anis):
    '''
    edens_time  :: WAVES electron density time to interpolate data_array to (length M)
    data_times  :: Current RBSPICE data sample times of length 3xN
    data_array  :: Data arrays consisting of ni, Ti, Ai in a 3xN ndarra
    '''
    new_dens = np.zeros((3, new_time.shape[0]), dtype=np.float64)
    new_temp = np.zeros((3, new_time.shape[0]), dtype=np.float64)
    new_anis = np.zeros((3, new_time.shape[0]), dtype=np.float64)
    
    xi = new_time.astype(np.int64)
    
    for ii in range(3):
        
        # Skip if bad
        if (SPICE_times[ii] is None) or (SPICE_dens[ii] is None):
            continue
        
        xp              = SPICE_times[ii].astype(np.int64)
        new_dens[ii, :] = np.interp(xi, xp, SPICE_dens[ii, :])
        new_temp[ii, :] = np.interp(xi, xp, SPICE_temp[ii, :])
        new_anis[ii, :] = np.interp(xi, xp, SPICE_anis[ii, :])

    return new_dens, new_temp, new_anis


def interpolate_B(new_time, b_time, b_array, dt, LP_filter=True):
    '''
    To do: Add a LP filter. Or does interp already do that?
    '''
    # Filter at Nyquist frequency to prevent aliasing
    if LP_filter == True:
        nyq = 1.0 / (2.0 * dt) 
        for ii in range(3):
            b_array[:, ii] = ascr.clw_low_pass(b_array[:, ii].copy(), nyq, 1./64., filt_order=4)
    
    xp = b_time.astype(np.int64)
    yp = np.sqrt(b_array[:, 0] ** 2 + b_array[:, 1] ** 2 + b_array[:, 2] ** 2)
    
    xi = new_time.astype(np.int64)
    yi = np.interp(xi, xp, yp)
    return yi


def interpolate_ne(new_time, den_time, den_array):    
    return np.interp(new_time.astype(np.int64), den_time.astype(np.int64), den_array)


def load_and_interpolate_plasma_params(time_start, time_end, probe, nsec=None, 
                                       rbsp_path='G://DATA//RBSP//', HM_filter_mhz=None,
                                       HOPE_only=False, time_array=None):
    '''
    Outputs as SI units: B0 in T, densities in /m3, temperatures in eV (pseudo SI)
    
    nsec is cadence of interpolated array in seconds. If None, defaults to using den_times
    as interpolant.
    
    If HOPE or RBSPICE data does not exist, file retrieval returns NoneType - have to deal 
    with that.
    '''
    print('Loading and interpolating satellite data')
    
    # Cold (total?) electron plasma density
    den_times, edens, dens_err = rfr.retrieve_RBSP_electron_density_data(rbsp_path, time_start, time_end,
                                                                         probe, pad=30)
    
    # Magnetic magnitude
    mag_times, raw_mags = rfl.load_magnetic_field(rbsp_path, time_start, time_end, probe, return_raw=True, pad=3600)
    
    # Filter out EMIC waves (background plus HM)
    if HM_filter_mhz is not None:
        filt_mags = np.zeros(raw_mags.shape)
        for ii in range(3):
            filt_mags[:, ii] = ascr.clw_low_pass(raw_mags[:, ii], HM_filter_mhz, 1./64., filt_order=4)
    else:
        filt_mags = raw_mags
    
    # HOPE data
    itime, etime, pdict, perr = rfr.retrieve_RBSP_hope_moment_data(     rbsp_path, time_start, time_end, padding=30, probe=probe)
    hope_dens = np.array([pdict['Dens_p_30'],       pdict['Dens_he_30'],       pdict['Dens_o_30']])
    hope_temp = np.array([pdict['Tperp_p_30'],      pdict['Tperp_he_30'],      pdict['Tperp_o_30']])
    hope_anis = np.array([pdict['Tperp_Tpar_p_30'], pdict['Tperp_Tpar_he_30'], pdict['Tperp_Tpar_o_30']]) - 1
    
    # SPICE data
    spice_dens = [];    spice_temp = [];    spice_anis = [];    spice_times = []
    for product, spec in zip(['TOFxEH', 'TOFxEHe', 'TOFxEO'], ['P', 'He', 'O']):
        spice_epoch , spice_dict  = rfr.retrieve_RBSPICE_data(rbsp_path, time_start, time_end, product , padding=30, probe=probe)
        
        # Collect all times (don't assume every file is good)
        spice_times.append(spice_epoch)
        
        if spice_dict is not None and HOPE_only==False:
            this_dens = spice_dict['F{}DU_Density'.format(spec)]
            this_anis = spice_dict['F{}DU_PerpPressure'.format(spec)] / spice_dict['F{}DU_ParaPressure'.format(spec)] - 1
            
            # Perp Temperature - Calculate as T = P/nk
            kB            = 1.381e-23; q = 1.602e-19
            t_perp_kelvin = 1e-9*spice_dict['F{}DU_PerpPressure'.format(spec)] / (kB*1e6*spice_dict['F{}DU_Density'.format(spec)])
            this_temp     = kB * t_perp_kelvin / q  # Convert from K to eV
        
            spice_dens.append(this_dens)
            spice_temp.append(this_temp)
            spice_anis.append(this_anis)
        else:
            if HOPE_only == True:
                print('Only HOPE data requested, discarding SPICE data...')
            spice_dens.append(None)
            spice_temp.append(None)
            spice_anis.append(None)
    
    # Interpolation step
    if nsec is None:
        # This should let me set my own timebase by feeding in an array
        if time_array is None:
            time_array = den_times.copy()
            iedens     = edens.copy()
        else:
            iedens = interpolate_ne(time_array, den_times, edens)
    else:
        time_array  = np.arange(time_start, time_end, np.timedelta64(nsec, 's'), dtype='datetime64[us]')
        iedens = interpolate_ne(time_array, den_times, edens)
    
    ihope_dens , ihope_temp , ihope_anis  = HOPE_interpolate_to_time(time_array, itime, hope_dens, hope_temp, hope_anis)
    
    ispice_dens, ispice_temp, ispice_anis = SPICE_interpolate_to_time(time_array, spice_times, np.array(spice_dens),
                                                                 np.array(spice_temp), np.array(spice_anis))

    Bi = interpolate_B(time_array, mag_times, filt_mags, nsec, LP_filter=False)
    
    # Subtract energetic components from total electron density (assuming each is singly charged)
    cold_dens = iedens - ihope_dens.sum(axis=0) - ispice_dens.sum(axis=0)
    return time_array, Bi*1e-9, cold_dens*1e6, ihope_dens*1e6, ihope_temp, ihope_anis, ispice_dens*1e6, ispice_temp, ispice_anis


def convert_data_to_hybrid_plasmafile(time_start, time_end, probe, pad, comp=None):
    '''
    Generate plasma_params_***.txt and run_params_***.txt files based on
    data at each point. run_params.txt only needed because it specifies 
    L/B.
    plasma_params_20130725_21247000000_DESCRIPTOR
    
    Don't bother putting too much stuff in the function args: Too much to change.
    
    Also, run_params won't generate unless asked'
    
    # What do we know? 
    Magnetic field is pretty solid
    Warm/Hot particles are pretty accurate (to within relativistic error)
    The biggest uncertainty is the cold composition.
    
    TODO: Check that only HOPE data is used. Also check eV/keV conversion.
          Current code takes temperatures in eV
    '''
    run_dir  = 'C:/Users/iarey/Documents/GitHub/hybrid/simulation_codes//run_inputs/from_data/'
    run_ext  = 'ALL_SPECIES'           
    run_dir +=  run_ext + '/'        
                   
    if os.path.exists(run_dir) == False: os.makedirs(run_dir)
    
    if comp is None:
        comp = [70, 20, 10]
    
    times, B0, cold_dens, hope_dens, hope_temp, hope_anis, spice_dens, spice_temp, spice_anis =\
        load_and_interpolate_plasma_params(time_start, time_end, probe, pad)
    
    cold_dens /= 1e6  ; hope_dens /= 1e6; spice_dens /= 1e6   # Cast densities    from /m to /cm
    cold_temp  = 0.1
    
    Nt    = times.shape[0]
    dens  = np.zeros((9, Nt), dtype=float)
    Tperp = np.zeros((9, Nt), dtype=float)
    A     = np.zeros((9, Nt), dtype=float)
    
    names    = np.array(['cold $H^{+}$', 'cold $He^{+}$', 'cold $O^{+}$',
                         'warm $H^{+}$', 'warm $He^{+}$', 'warm $O^{+}$',
                         'hot $H^{+}$' , 'hot $He^{+}$' , 'hot $O^{+}$'])
    
    colors   = np.array(['b'      , 'm'     , 'g',
                        'r'      , 'gold'  , 'purple',
                        'tomato' , 'orange', 'deeppink'])
    
    temp_flag = np.array([0,     0,    0,   1,   1,    1,   1,   1,    1])
    dist_flag = np.array([0,     0,    0,   0,   0,    0,   0,   0,    0])
    mass      = np.array([1.0, 4.0, 16.0, 1.0, 4.0, 16.0, 1.0, 4.0, 16.0])
    charge    = np.array([1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0])
    drift     = np.array([0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0])
    
    nsp_ppc   = np.array([256,     256,     256,   
                          2048,   2048,    2048,   
                          2048,   2048,    2048])
    
    dens[0] = 1e-2 * cold_dens * comp[0];  dens[3] = hope_dens[0];  dens[6] = spice_dens[0];
    dens[1] = 1e-2 * cold_dens * comp[1];  dens[4] = hope_dens[1];  dens[7] = spice_dens[1];
    dens[2] = 1e-2 * cold_dens * comp[2];  dens[5] = hope_dens[2];  dens[8] = spice_dens[2];

    Tperp[0] = cold_temp; Tperp[3] = hope_temp[0]; Tperp[6] = spice_temp[0]
    Tperp[1] = cold_temp; Tperp[4] = hope_temp[1]; Tperp[7] = spice_temp[1]
    Tperp[2] = cold_temp; Tperp[5] = hope_temp[2]; Tperp[8] = spice_temp[2]

    A[0]  = 0.0; A[3]  = hope_anis[0]; A[6]  = spice_anis[0];
    A[1]  = 0.0; A[4]  = hope_anis[1]; A[7]  = spice_anis[1];
    A[2]  = 0.0; A[5]  = hope_anis[2]; A[8]  = spice_anis[2];

    # Number of rows with species-specific stuff
    N_rows    = 11
    N_species = 9
    
    row_labels = ['LABEL', 'COLOUR', 'TEMP_FLAG', 'DIST_FLAG', 'NSP_PPC', 'MASS_PROTON', 
                  'CHARGE_ELEM', 'DRIFT_VA', 'DENSITY_CM3', 'ANISOTROPY', 'ENERGY_PERP', 
                  'ELECTRON_EV', 'BETA_FLAG', 'L', 'B_EQ']
    
    row_params = [names, colors, temp_flag, dist_flag, nsp_ppc, mass, charge, drift, 
                  dens, A, Tperp]

    electron_ev = cold_temp
    beta_flag   = 0
    L           = 5.35
    b_eq        = '-'
    
    for ii in range(Nt):
        suffix = times[ii].astype(object).strftime('_%Y%m%d_%H%M%S%f_') + run_ext
        
        run_file = run_dir + 'plasma_params' + suffix + '.txt'
        
        with open(run_file, 'w') as f:
            
            # Print the row label for each row, and then entries for each species
            for jj in range(N_rows):
                print('{0: <15}'.format(row_labels[jj]), file=f, end='')
                
                # Loop through species to print that row's stuff (separated by spaces)
                for kk in range(N_species):
                    
                    if dens[kk, ii] != 0.0:
                        if jj > 7: 
                            print('{0: <15}'.format(round(row_params[jj][kk, ii], 9)), file=f, end='')
                        else:
                            print('{0: <15}'.format(row_params[jj][kk]), file=f, end='')
                print('', file=f)
            # Single print specific stuff
            print('ELECTRON_EV    {}'.format(electron_ev), file=f)
            print('BETA_FLAG      {}'.format(beta_flag)  , file=f)
            print('L              {}'.format(L)          , file=f)
            print('B_EQ           {}'.format(b_eq)       , file=f)  
            print('B_XMAX         -'                     , file=f) 
    print('Plasmafiles created.')
    return


def get_pc1_spectra(rbsp_path, time_start, time_end, probe, pc1_res=25.0,
                    overlap=0.99, high_pass_mHz=None):
    '''
    Helper function to load magnetic field and calculate dynamic spectrum for
    overlaying on plots
    '''
    import rbsp_fields_loader as rfl
    import fast_scripts       as fscr
    
    print('Generating magnetic dynamic spectra...')
    
    times, mags, dt, gyfreqs = \
        rfl.load_magnetic_field(rbsp_path, time_start, time_end, probe,
                                pad=3600, LP_B0=1.0, get_gyfreqs=False,
                                return_raw=False, wave_HP=None, wave_LP=None)
    
    if high_pass_mHz is not None:
        mags = ascr.clw_high_pass(mags, high_pass_mHz, dt, filt_order=4)
    
    pc1_xpower, pc1_xtimes, pc1_xfreq = fscr.autopower_spectra(times, mags[:, 0], time_start, 
                                                     time_end, dt, overlap=overlap, df=pc1_res)
    
    pc1_ypower, pc1_ytimes, pc1_yfreq = fscr.autopower_spectra(times, mags[:, 1], time_start, 
                                                     time_end, dt, overlap=overlap, df=pc1_res)
    
    pc1_perp_power = np.log10(pc1_xpower[:, :] + pc1_ypower[:, :])
    
    return pc1_xtimes, pc1_xfreq, pc1_perp_power


def load_CRRES_data(time_start, time_end, crres_path='G://DATA//CRRES//', nsec=None):
    '''
    Since no moment data exists for CRRES, this loads only the cold electron density and
    magnetic field (with option to low-pass filter) and interpolates them to
    the same timebase (linear or cubic? Just do linear for now).
    
    If nsec is none, interpolates B to ne. Else, interpolates both to nsec. 
    CRRES density cadence bounces between 8-9 seconds (terrible for FFT, alright for interp)
    
    den_dict params: ['VTCW', 'YRDOY', 'TIMESTRING', 'FCE_KHZ', 'FUHR_KHZ', 'FPE_KHZ', 'NE_CM3', 'ID', 'M']
    
    TODO: Fix this. B_arr shape and nyq in mHz
    '''
    # Load data
    times, B0, HM, dB, E0, HMe, dE, S, B, E = cfr.get_crres_fields(crres_path,
                  time_start, time_end, pad=1800, E_ratio=5.0, rotation_method='vector', 
                  output_raw_B=True, interpolate_nan=None, B0_LP=1.0,
                  Pc1_LP=5000, Pc1_HP=100, Pc5_LP=30, Pc5_HP=None, dEx_LP=None)
    
    den_times, den_dict = cfr.get_crres_density(crres_path, time_start, time_end, pad=600)
    
    edens = den_dict['NE_CM3']
    
    # Interpolate B only
    if nsec is None:
        
        # Low-pass total field to avoid aliasing (Assume 8.5 second cadence)
        B_dt  = 1.0 / 32.0
        nyq   = 1.0 / (2.0 * 8.5) 
        for ii in range(3):
            B[:, ii] = ascr.clw_low_pass(B[:, ii].copy(), nyq, B_dt, filt_order=4)
            
        # Take magnitude and interpolate            
        B_mag    = np.sqrt(B[:, 0] ** 2 + B[:, 1] ** 2 + B[:, 2] ** 2)
        B_interp = np.interp(den_times.astype(np.int64), times.astype(np.int64), B_mag) 
        
        return den_times, B_interp, edens


    else:
        # Define new time array
        ntime = np.arange(time_start, time_end, np.timedelta64(nsec, 's'), dtype='datetime64[us]')
        
        # Low-pass total field to avoid aliasing (Assume 8.5 second cadence)
        B_dt  = 1.0 / 32.0
        nyq   = 1.0 / (2.0 * nsec) 
        for ii in range(3):
            B[ii] = ascr.clw_low_pass(B[ii].copy(), nyq, B_dt, filt_order=4)
            
        # Take magnitude and interpolate            
        B_mag    = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
        B_interp = np.interp(ntime.astype(np.int64), times.astype(np.int64), B_mag) 
        
        # Also interpolate density
        ne_interp = np.interp(ntime.astype(np.int64), den_times.astype(np.int64), edens) 
        
        return ntime, B_interp, ne_interp
        

def integrate_HOPE_moments(time_start, time_end, probe, pad, rbsp_path='E://DATA//RBSP//'):
    '''
    Testing self-integration of ion moments from pitch angle data :: Compare to actual
    calculated L3-MOM products.
    
    Based on equation from RBSPICE Data Handbook Rev. E (2018) p. 31.
    
    Questions:
        - Do we have to account for gaps between energy channels? As in, fit a smooth
            curve to the fluxes and integrate that way?
        - Should be able to ignore channels below 30eV for the comparison, and then
        include them once the fluxes are corrected for a few things? To get an idea
        of the plasmaspheric composition
    '''
    qp = 1.602e-19
    mp = 1.673e-27
    
    import spacepy.datamodel as dm
    import matplotlib.pyplot as plt
    pa_data  = dm.fromCDF(_test_file_pa)
    mom_data = dm.fromCDF(_test_file_mom)
    
    mom_times = mom_data['Epoch_Ion']
    mom_hdens = mom_data['Dens_p_30']
    
    pa_times  = pa_data['Epoch_Ion']
    
    n_times = pa_times.shape[0]
    vdens     = np.zeros(n_times, dtype=np.float64)
    
    # Organize pitch angle stuff
    pitch  = pa_data['PITCH_ANGLE']             # PA bin centers
    n_pitch= pitch.shape[0]
    da     = np.ones(n_pitch) * 18.             # PA bin widths
    da[0]  = da[-1] = 9.
    da    *= np.pi / 180.
    
    # Do energy stuff
    energy   = pa_data['HOPE_ENERGY_Ion']         # Energy bin centers
    n_energy = energy.shape[1]

    # Energy bin limits
    erange = np.zeros((n_times, n_energy, 2))
    for ii in range(n_energy):
        erange[:, ii, 0] = (pa_data['HOPE_ENERGY_Ion'][:, ii] - pa_data['ENERGY_Ion_DELTA'][:, ii])
        erange[:, ii, 1] = (pa_data['HOPE_ENERGY_Ion'][:, ii] + pa_data['ENERGY_Ion_DELTA'][:, ii])
    
    E_min = 30.0; E_max = None
    for ii in range(n_times):
        E  = energy[ii]
        dE = erange[ii]
     
        # Find energy min/max channels for integration
        if E_min is None:
            min_idx = 0
        else:
            diffs   = np.abs(energy - E_min)
            min_idx = np.where(diffs == diffs.min())[0][0]
            
        if E_max is None:
            max_idx = energy.shape[0] - 1
        else:
            diffs   = np.abs(energy - E_max)
            min_idx = np.where(diffs == diffs.min())[0][0]
        
        #vdens[ii] = vdist[ii, min_idx:max_idx].sum()
        
        # Need:
        # Pitch angle bin list
        # Pitch angle widths
        # Energy bin list
        # Energy bin widths
        
        
    # Filter bad values (fill value -1E-31, and no density should be negative)
    for ii in range(mom_hdens.shape[0]):
        if mom_hdens[ii] < 0.0:
            mom_hdens[ii] = np.nan
    for ii in range(vdens.shape[0]):
        if vdens[ii] < 0.0:
             vdens[ii] = np.nan

    plt.ioff()
    plt.figure()
    plt.plot(mom_times, mom_hdens, label='MOM-L3')
    plt.plot(pa_times , vdens    , label='Manual')
    plt.legend()
    plt.show()
    return


def read_cutoff_file(_filename):
    with open(_filename, 'r') as f:
        ii = 0
        for line in f:
            if ii == 0:
                header_names = line.split()
                cutoff_dict  = {}
                for hname in header_names:
                    cutoff_dict[hname] = []
            else:
                values = line.split()
                for jj in range(len(values)):
                    cutoff_dict[header_names[jj]].append(values[jj])
            ii += 1
    cutoff_dict['CUTOFF_TIME']   = [np.datetime64(this) for this in cutoff_dict['CUTOFF_TIME']]
    cutoff_dict['PACKET_CENTER'] = [np.datetime64(this) for this in cutoff_dict['PACKET_CENTER']]
    cutoff_dict['PACKET_START']  = [np.datetime64(this) for this in cutoff_dict['PACKET_START']]
    cutoff_dict['CUTOFF_HZ']     = [float(this)         for this in cutoff_dict['CUTOFF_HZ']]
    cutoff_dict['CUTOFF_NORM']   = [float(this)         for this in cutoff_dict['CUTOFF_NORM']]
    
    for key in cutoff_dict.keys():
        cutoff_dict[key] = np.asarray(cutoff_dict[key])
    return cutoff_dict


def calculate_o_from_he_and_cutoff(co_freq_norm, he_val):
    '''
    For each possible fractional value of He, calculate the possible values
    of O, for a given cutoff frequency (normalized to the cyclotron frequency).
    
    Solution seems fine for a hydrogen cutoff. Validated with
        Omura (2010) :: co_freqs_norm = 0.3514 (pcyc = 3.7Hz)
        Ohja (2021)  :: co_freqs_norm = 0.3084 (pcyc = 1.3Hz)
    '''
    def cfunc(nO, nHe, w):
        t1 = (1. - nHe - nO) / (1 - w)
        t2 = nHe / (1 - 4*w)
        t3 = nO  / (1 - 16*w)
        return (t1 + t2 + t3 - 1)
    
    o_val = fsolve(cfunc, x0=0.0, args=(he_val, co_freq_norm),
                        xtol=1e-10, maxfev=1000000, full_output=False)
    return o_val[0]


def generate_plasmafile(cutoff_filename, run_dir, run_series_name, he_conc=0.05):
    # Read cutoff textfile
    # Use cutoff data to set cold composition (function)
    # Interpolate cold composition to packet_start time
    # Read data
    # Have variable for He_percent, since this is unknown
    
    # READ CUTOFF FILE AND CALCULATE/INTERPOLATE COMPOSITIONS
    cutoff_dict = read_cutoff_file(cutoff_filename)
    
    n_vals  = cutoff_dict['CUTOFF_NORM'].shape[0]
    o_concs = np.zeros(n_vals)
    for ii in range(n_vals):
        o_concs[ii] = calculate_o_from_he_and_cutoff(cutoff_dict['CUTOFF_NORM'][ii], he_conc)

    o_concs = np.interp(cutoff_dict['PACKET_START'].astype(np.int64),
                        cutoff_dict['CUTOFF_TIME'].astype(np.int64),
                        o_concs)

    # CREATE NEW DIRECTORY FOR RUNFILES
    run_dir +=  run_series_name + '/'        

    if os.path.exists(run_dir) == False: os.makedirs(run_dir)
    
    # Pad times and round to nearest second
    data_start = cutoff_dict['CUTOFF_TIME'][ 0] - np.timedelta64(300, 's')
    data_end   = cutoff_dict['CUTOFF_TIME'][-1] + np.timedelta64(300, 's')
    
    data_start = data_start.astype('datetime64[s]').astype('<M8[us]')
    data_end   = data_end.astype('datetime64[s]').astype('<M8[us]')
    
    # LOAD THE INTERPOLATED DATA
    times, B0, cold_dens, hope_dens, hope_temp, hope_anis, spice_dens, spice_temp, spice_anis =\
        load_and_interpolate_plasma_params(data_start, data_end, 'a',
                                           rbsp_path=_rbsp_path, HM_filter_mhz=30.0, nsec=None,
                                           time_array=cutoff_dict['PACKET_START'])

    # CALCULATE VALUES FOR HYBRID RUNS
    cold_dens /= 1e6  ; hope_dens /= 1e6; spice_dens /= 1e6   # Cast densities from /m to /cm
    cold_temp  = 0.1
    
    Nt    = cutoff_dict['CUTOFF_TIME'].shape[0]
    dens  = np.zeros((6, Nt), dtype=float)
    Tperp = np.zeros((6, Nt), dtype=float)
    A     = np.zeros((6, Nt), dtype=float)
    
    names    = np.array(['cold_$H^{+}$', 'cold_$He^{+}$', 'cold_$O^{+}$',
                         'warm_$H^{+}$', 'warm_$He^{+}$', 'warm_$O^{+}$'])
    
    colors   = np.array(['b'      , 'm'     , 'g',
                        'r'      , 'gold'  , 'purple'])
    
    temp_flag = np.array([0,     0,    0,   1,   1,    1])
    dist_flag = np.array([0,     0,    0,   0,   0,    0])
    mass      = np.array([1.0, 4.0, 16.0, 1.0, 4.0, 16.0])
    charge    = np.array([1.0, 1.0,  1.0, 1.0, 1.0,  1.0])
    drift     = np.array([0.0, 0.0,  0.0, 0.0, 0.0,  0.0])
    
    nsp_ppc   = np.array([512,     512,     512,   
                          4096,   4096,    4096])
    
    # Set cold plasma temperature values
    Tperp[0] = cold_temp; A[0]  = 0.0
    Tperp[1] = cold_temp; A[1]  = 0.0
    Tperp[2] = cold_temp; A[2]  = 0.0
    
    # Set warm plasma values from HOPE
    Tperp[3] = np.round(hope_temp[0], 3)
    Tperp[4] = np.round(hope_temp[1], 3)
    Tperp[5] = np.round(hope_temp[2], 3) 
    
    A[3]  = np.round(hope_anis[0], 3)
    A[4]  = np.round(hope_anis[1], 3)
    A[5]  = np.round(hope_anis[2], 3)
    
    dens[3] = np.round(hope_dens[0], 3)
    dens[4] = np.round(hope_dens[1], 3)
    dens[5] = np.round(hope_dens[2], 3)
    
    # Number of rows with species-specific stuff
    N_rows    = 11
    N_species = 6
    
    row_labels = ['LABEL', 'COLOUR', 'TEMP_FLAG', 'DIST_FLAG', 'NSP_PPC', 'MASS_PROTON', 
                  'CHARGE_ELEM', 'DRIFT_VA', 'DENSITY_CM3', 'ANISOTROPY', 'ENERGY_PERP', 
                  'ELECTRON_EV', 'BETA_FLAG', 'L', 'B_EQ']
    
    row_params = [names, colors, temp_flag, dist_flag, nsp_ppc, mass, charge, drift, 
                  dens, A, Tperp]

    electron_ev = cold_temp
    beta_flag   = 0
    L           = 5.35
    
    
    for ii in range(Nt):
        h_conc = 1. - he_conc - o_concs
        
        # Calculate cold plasma composition for this time
        dens[0] = np.round(cold_dens * h_conc, 3)
        dens[1] = np.round(cold_dens * he_conc , 3)
        dens[2] = np.round(cold_dens * o_concs[ii], 3)
        b_eq    = np.round(B0[ii]*1e9, 2)
        
        suffix = times[ii].astype(object).strftime('_%Y%m%d_%H%M%S_') + run_series_name
        
        run_file = run_dir + 'plasma_params' + suffix + '.plasma'
        
        with open(run_file, 'w') as f:
            
            # Print the row label for each row, and then entries for each species
            for jj in range(N_rows):
                print('{0: <15}'.format(row_labels[jj]), file=f, end='')
                
                # Loop through species to print that row's stuff (separated by spaces)
                for kk in range(N_species):
                    
                    if dens[kk, ii] != 0.0:
                        if jj > 7: 
                            print('{0: <15}'.format(round(row_params[jj][kk, ii], 9)), file=f, end='')
                        else:
                            print('{0: <15}'.format(row_params[jj][kk]), file=f, end='')
                print('', file=f)
            # Single print specific stuff
            print('ELECTRON_EV    {}'.format(electron_ev), file=f)
            print('BETA_FLAG      {}'.format(beta_flag)  , file=f)
            print('L              {}'.format(L)          , file=f)
            print('B_EQ           {}e-9'.format(b_eq)       , file=f)  
            print('B_XMAX         -'                     , file=f) 
    print('Plasmafiles created.')
    return

def generate_script_file(run_dir, run_series, hybrid_method='PREDCORR'):
    run_dir +=  run_series + '/'
    filedir  = f'/batch_runs/{run_series}/'
    
    ii = 0
    for file in os.listdir(run_dir):
        if file[-7:] == '.plasma':
            script_filename = run_dir+f'run{ii}_{run_series}.sh'
            with open(script_filename, 'w') as f:
                # PBS Inputs
                print('#!/bin/bash', file=f)
                print('#PBS -l select=1:ncpus=32:mem=10GB', file=f)
                print('#PBS -l walltime=150:00:00 ', file=f)
                print('#PBS -k oe', file=f)
                print('#PBS -m ae', file=f)
                print('#PBS -M joshua.s.williams@uon.edu.au', file=f)
                print('# Autocreated by Python', file=f)
                
                # Hybrid inputs
                print('source /etc/profile.d/modules.sh', file=f)
                print('module load numba/0.49.1-python.3.6', file=f)
                print('cd $PBS_O_WORKDIR', file=f)
                
                print(f'python /home/c3134027/hybrid/simulation_codes/{hybrid_method}_1D_PARALLEL/main_1D.py -r {filedir}run_params.run -p {filedir}{file} -n {ii}', file=f)
                print('exit 0', file=f)
            ii += 1
    return

def generate_hybrid_files_from_cutoffs(he_conc=0.05):
    run_dir         = 'D://NEW_HYBRID_RUNFILES//'
    
    # July 25 event
    if True:
        cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20130725_RBSP-A//pearl_times.txt'
        run_series_name = 'JUL25_PKTS_{:.0f}HE_CAMCL'.format(he_conc*100)
        generate_plasmafile(cutoff_filename, run_dir, run_series_name, he_conc=he_conc)
        generate_script_file(run_dir, run_series_name, hybrid_method='CAM_CL')
    
    # Jan 16 event
    if True:
        cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20150116_RBSP-A//pearl_times.txt'
        run_series_name = 'JAN16_PKTS_{:.0f}HE_CAMCL'.format(he_conc*100)
        generate_plasmafile(cutoff_filename, run_dir, run_series_name, he_conc=he_conc)
        generate_script_file(run_dir, run_series_name, hybrid_method='CAM_CL')
    return


#%% MAIN FUNCTION :: JUST CHECKING THINGS
if __name__ == '__main__':
    _rbsp_path  = 'E://DATA//RBSP//'
    _crres_path = 'E://DATA//CRRES//'
    _time_start = np.datetime64('2015-01-16T04:05:00')
    _time_end   = np.datetime64('2015-01-16T05:15:00')
    _probe      = 'a'
    _pad        = 0
    _test_file_pa  = 'F://DATA//RBSP//ECT//HOPE//L3//PITCHANGLE//rbspa_rel04_ect-hope-PA-L3_20150116_v7.1.0.cdf'
    _test_file_mom = 'F://DATA//RBSP//ECT//HOPE//L3//MOMENTS//rbspa_rel04_ect-hope-MOM-L3_20150116_v7.1.0.cdf'
    
    #integrate_HOPE_moments(_time_start, _time_end, _probe, _pad, rbsp_path=_rbsp_path)
    generate_hybrid_files_from_cutoffs(he_conc=0.05)
    generate_hybrid_files_from_cutoffs(he_conc=0.10)
    generate_hybrid_files_from_cutoffs(he_conc=0.20)
    
    
    if False:
        _times, _B, _ne = load_CRRES_data(_time_start, _time_end, crres_path=_crres_path, nsec=None)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2)
        ax[0].plot(_times, _B)
        ax[1].plot(_times, _ne)
    
    #convert_data_to_hybrid_plasmafile(_time_start, _time_end, _probe, _pad)
    
    if False:
        _times, _B0, _cold_dens, _hope_dens, _hope_temp, _hope_anis, _spice_dens, _spice_temp, _spice_anis =\
            load_and_interpolate_plasma_params(_time_start, _time_end, _probe, _pad, HM_filter_mhz=50, nsec=None)
        
        ### LOAD RAW VALUES ###
        den_times, edens, dens_err = rfr.retrieve_RBSP_electron_density_data(_rbsp_path, _time_start, _time_end, _probe, pad=_pad)
        mag_times, raw_mags = rfl.load_magnetic_field(_rbsp_path, _time_start, _time_end, _probe, return_raw=True, pad=3600)
        
        itime, etime, pdict, perr = rfr.retrieve_RBSP_hope_moment_data(_rbsp_path, _time_start, _time_end, padding=_pad, probe=_probe)
        hope_dens = np.array([pdict['Dens_p_30'],       pdict['Dens_he_30'],       pdict['Dens_o_30']])
        hope_temp = np.array([pdict['Tperp_p_30'],      pdict['Tperp_he_30'],      pdict['Tperp_o_30']])
        hope_anis = np.array([pdict['Tperp_Tpar_p_30'], pdict['Tperp_Tpar_he_30'], pdict['Tperp_Tpar_o_30']]) - 1
        
        spice_dens = [];    spice_temp = [];    spice_anis = []
        for product, spec in zip(['TOFxEH', 'TOFxEHe', 'TOFxEO'], ['P', 'He', 'O']):
            spice_time , spice_dict  = rfr.retrieve_RBSPICE_data(_rbsp_path, _time_start, _time_end, product , padding=_pad, probe=_probe)
            
            this_dens = spice_dict['F{}DU_Density'.format(spec)]
            this_anis = spice_dict['F{}DU_PerpPressure'.format(spec)] / spice_dict['F{}DU_ParaPressure'.format(spec)] - 1
            
            # Perp Temperature - Calculate as T = P/nk
            kB            = 1.381e-23; q = 1.602e-19
            t_perp_kelvin = 1e-9*spice_dict['F{}DU_PerpPressure'.format(spec)] / (kB*1e6*spice_dict['F{}DU_Density'.format(spec)])
            this_temp     = kB * t_perp_kelvin / q  # Convert from K to eV
            
            spice_dens.append(this_dens)
            spice_temp.append(this_temp)
            spice_anis.append(this_anis)
        

        ### COMPARISON PLOTS FOR CHECKING ###
        import matplotlib.pyplot as plt
        
        # Plot things :: B0HM, cold density, HOPE/RBSPICE hot densities (4 plots) 
        fig, axes = plt.subplots(4)
        
        # B0
        B0       = np.sqrt(raw_mags[:, 0] ** 2 + raw_mags[:, 1] ** 2 + raw_mags[:, 2] ** 2)
        st, en   = ascr.boundary_idx64(mag_times, _time_start, _time_end)
        _st, _en = ascr.boundary_idx64(_times,    _time_start, _time_end)
        
        axes[0].set_title('Magnetic field and Cold e/HOPE/RBSPICE ion densities')
        axes[0].plot(mag_times[ st: en],  1e-9*B0[ st: en], c='k', label='raw')
        axes[0].plot(   _times[_st:_en],      _B0[_st:_en], c='r', label='Filtered + Decimated', marker='o')
        axes[0].legend()
        
        # Cold dens
        axes[1].plot(den_times,  1e6*edens, c='k', label='raw')
        axes[1].plot(   _times, _cold_dens, c='r', label='decimated') 
        
        # HOPE densities
        axes[2].plot(itime, pdict['Dens_p_30']*1e6 , c='r')
        axes[2].plot(itime, pdict['Dens_he_30']*1e6, c='green')  
        axes[2].plot(itime, pdict['Dens_o_30']*1e6 , c='b')
        
        axes[2].plot(_times, _hope_dens[0] , c='r', ls='--')
        axes[2].plot(_times, _hope_dens[1] , c='green', ls='--')  
        axes[2].plot(_times, _hope_dens[2] , c='b', ls='--')
        
        # RBSPICE densities
        axes[3].plot(spice_time, spice_dens[0]*1e6, c='r')
        axes[3].plot(spice_time, spice_dens[1]*1e6, c='green')  
        axes[3].plot(spice_time, spice_dens[2]*1e6, c='b')
        
        axes[3].plot(_times, _spice_dens[0], c='r', ls='--')
        axes[3].plot(_times, _spice_dens[1], c='green', ls='--')  
        axes[3].plot(_times, _spice_dens[2], c='b', ls='--')
        
        for ax in axes:
            ax.set_xlim(_time_start, _time_end)
        
        # Also temp/anis for HOPE/RBSPICE (4 plots)
        fig2, axes2 = plt.subplots(4)
        axes2[0].set_title('HOPE/RBSPICE Temperatures/Anisotropies')
        
        # HOPE Temps
        axes2[0].plot(itime, pdict['Tperp_p_30'] , c='r')
        axes2[0].plot(itime, pdict['Tperp_he_30'], c='green')  
        axes2[0].plot(itime, pdict['Tperp_o_30'] , c='b')
        
        axes2[0].plot(_times, _hope_temp[0], c='r', ls='--')
        axes2[0].plot(_times, _hope_temp[1], c='green', ls='--')  
        axes2[0].plot(_times, _hope_temp[2], c='b', ls='--')
        axes2[0].set_ylabel('HOPE TEMP')
        
        # RBSPICE Temps
        axes2[1].plot(spice_time, spice_temp[0], c='r')
        axes2[1].plot(spice_time, spice_temp[1], c='green')  
        axes2[1].plot(spice_time, spice_temp[2], c='b')
        
        axes2[1].plot(_times, _spice_temp[0], c='r', ls='--')
        axes2[1].plot(_times, _spice_temp[1], c='green', ls='--')  
        axes2[1].plot(_times, _spice_temp[2], c='b', ls='--')
        axes2[1].set_ylabel('RBSPICE TEMP')
        
        # HOPE Anisotropy
        axes2[2].plot(itime, pdict['Tperp_Tpar_p_30'] - 1 , c='r')
        axes2[2].plot(itime, pdict['Tperp_Tpar_he_30'] - 1, c='green')  
        axes2[2].plot(itime, pdict['Tperp_Tpar_o_30'] - 1 , c='b')
        
        axes2[2].plot(_times, _hope_anis[0], c='r', ls='--')
        axes2[2].plot(_times, _hope_anis[1], c='green', ls='--')  
        axes2[2].plot(_times, _hope_anis[2], c='b', ls='--')
        axes2[2].set_ylabel('HOPE A')
        
        # RBSPICE Anisotropy
        axes2[3].plot(spice_time, spice_anis[0], c='r')
        axes2[3].plot(spice_time, spice_anis[1], c='green')  
        axes2[3].plot(spice_time, spice_anis[2], c='b')
        
        axes2[3].plot(_times, _spice_anis[0], c='r', ls='--')
        axes2[3].plot(_times, _spice_anis[1], c='green', ls='--')  
        axes2[3].plot(_times, _spice_anis[2], c='b', ls='--')
        axes2[3].set_ylabel('RBSPICE A')
        
        plt.show()