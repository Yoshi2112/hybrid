# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:42:27 2019

@author: Yoshi
"""
import os
import sys
sys.path.append('F://Google Drive//Uni//PhD 2017//Data//Scripts//')
import numpy as np
import rbsp_file_readers   as rfr
import rbsp_fields_loader  as rfl
import analysis_scripts    as ascr


def interpolate_to_time(new_time, data_time, data_array_dens, data_array_temp, data_array_anis):
    '''
    edens_time  :: WAVES electron density time to interpolate data_array to (length M)
    data_time   :: Current data sample times (HOPE/RBSPICE) of length N
    data_array  :: Data arrays consisting of ni, Ti, Ai in a 3xN ndarra
    '''
    new_data_dens = np.zeros((3, new_time.shape[0]), dtype=np.float64)
    new_data_temp = np.zeros((3, new_time.shape[0]), dtype=np.float64)
    new_data_anis = np.zeros((3, new_time.shape[0]), dtype=np.float64)
    
    xi = new_time.astype(np.int64)
    xp = data_time.astype(np.int64)
    
    for ii in range(3):
        new_data_dens[ii, :] = np.interp(xi, xp, data_array_dens[ii, :])
        new_data_temp[ii, :] = np.interp(xi, xp, data_array_temp[ii, :])
        new_data_anis[ii, :] = np.interp(xi, xp, data_array_anis[ii, :])

    return new_data_dens, new_data_temp, new_data_anis


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


def load_and_interpolate_plasma_params(time_start, time_end, probe, pad, nsec=1, 
                                       rbsp_path='G://DATA//RBSP//', HM_filter_mhz=None):
    '''
    Outputs as SI units: B0 in nT, densities in /m3, temperatures in eV (pseudo SI)
    
    nsec is cadence of interpolated array in seconds
    '''
    print('Loading and interpolating satellite data')
    
    # Cold (total?) electron plasma density
    den_times, edens, dens_err = rfr.retrieve_RBSP_electron_density_data(rbsp_path, time_start, time_end, probe, pad=pad)

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
    itime, etime, pdict, perr = rfr.retrieve_RBSP_hope_moment_data(     rbsp_path, time_start, time_end, padding=pad, probe=probe)
    hope_dens = np.array([pdict['Dens_p_30'],       pdict['Dens_he_30'],       pdict['Dens_o_30']])
    hope_temp = np.array([pdict['Tperp_p_30'],      pdict['Tperp_he_30'],      pdict['Tperp_o_30']])
    hope_anis = np.array([pdict['Tperp_Tpar_p_30'], pdict['Tperp_Tpar_he_30'], pdict['Tperp_Tpar_o_30']]) - 1
    
    # SPICE data
    spice_dens = [];    spice_temp = [];    spice_anis = []
    for product, spec in zip(['TOFxEH', 'TOFxEHe', 'TOFxEO'], ['P', 'He', 'O']):
        spice_time , spice_dict  = rfr.retrieve_RBSPICE_data(rbsp_path, time_start, time_end, product , padding=pad, probe=probe)
        
        this_dens = spice_dict['F{}DU_Density'.format(spec)]
        this_anis = spice_dict['F{}DU_PerpPressure'.format(spec)] / spice_dict['F{}DU_ParaPressure'.format(spec)] - 1
        
        # Perp Temperature - Calculate as T = P/nk
        kB            = 1.381e-23; q = 1.602e-19
        t_perp_kelvin = 1e-9*spice_dict['F{}DU_PerpPressure'.format(spec)] / (kB*1e6*spice_dict['F{}DU_Density'.format(spec)])
        this_temp     = kB * t_perp_kelvin / q  # Convert from K to eV
        
        spice_dens.append(this_dens)
        spice_temp.append(this_temp)
        spice_anis.append(this_anis)
    
    # Interpolation step
    time_array  = np.arange(time_start, time_end, np.timedelta64(nsec, 's'), dtype='datetime64[us]')
    
    
    ihope_dens , ihope_temp , ihope_anis  = interpolate_to_time(time_array, itime, hope_dens, hope_temp, hope_anis)
    ispice_dens, ispice_temp, ispice_anis = interpolate_to_time(time_array, spice_time, np.array(spice_dens),
                                                                 np.array(spice_temp), np.array(spice_anis))
    
    Bi     = interpolate_B(time_array, mag_times, filt_mags, nsec, LP_filter=False)
    iedens = interpolate_ne(time_array, den_times, edens)
    
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
    '''
    run_dir  = 'C:/Users/iarey/Documents/GitHub/hybrid/simulation_codes//run_inputs/from_data/'
    run_ext  = 'H_ONLY'           
    run_dir +=  run_ext + '/'        
                   
    if os.path.exists(run_dir) == False: os.makedirs(run_dir)
    
    if comp is None:
        comp = [100, 0, 0]
    
    times, B0, cold_dens, hope_dens, hope_temp, hope_anis, spice_dens, spice_temp, spice_anis =\
        load_and_interpolate_plasma_params(time_start, time_end, probe, pad)
    
    cold_dens /= 1e6  ; hope_dens /= 1e6; spice_dens /= 1e6   # Cast densities    from /m to /cm
    cold_temp  = 5e-3 ; hope_temp /= 1e3; spice_temp /= 1e3   # Cast temperatures from eV to keV
    
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
            print('BETA_FLAG      {}'.format(beta_flag), file=f)
            print('L              {}'.format(L), file=f)
            print('B_EQ           {}'.format(b_eq), file=f)        
    return


if __name__ == '__main__':
    from spacepy import pycdf
    
    _rbsp_path  = 'G://DATA//RBSP//'
    _time_start = np.datetime64('2013-07-25T21:20:00')
    _time_end   = np.datetime64('2013-07-25T21:50:00')
    _probe      = 'a'
    _pad        = 0
    
    #convert_data_to_hybrid_plasmafile(_time_start, _time_end, _probe, _pad)
    magfold = _rbsp_path + 'EMFISIS//MAG//'
    magname = 'rbsp-a_magnetometer_hires-gse_emfisis-L3_20130725_v1.3.4.cdf'
    magfile = magfold + magname
    if os.path.exists(magfile) == True:
        print('File exists')
    else:
        print('File not found')
        
    cdf_pointer = pycdf.CDF(magfile) 
    print(cdf_pointer.keys())
    
    if False:
        _times, _B0, _cold_dens, _hope_dens, _hope_temp, _hope_anis, _spice_dens, _spice_temp, _spice_anis =\
            load_and_interpolate_plasma_params(_time_start, _time_end, _probe, _pad, HM_filter_mhz=50)
        
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
        
        #%%
        ### COMPARISON PLOTS FOR CHECKING ###
        import matplotlib.pyplot as plt
        
        # Plot things :: B0HM, cold density, HOPE/RBSPICE hot densities (4 plots) 
        fig, axes = plt.subplots(4)
        
        # B0
        B0       = np.sqrt(raw_mags[:, 0] ** 2 + raw_mags[:, 1] ** 2 + raw_mags[:, 2] ** 2)
        st, en   = ascr.boundary_idx64(mag_times, _time_start, _time_end)
        _st, _en = ascr.boundary_idx64(_times,    _time_start, _time_end)
        
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
        