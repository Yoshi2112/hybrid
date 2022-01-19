# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 22:05:39 2022

@author: Yoshi
"""
import os, sys, warnings, pdb, emd
sys.path.append('../../')
import numpy as np
import matplotlib.pyplot   as plt
import matplotlib.colors   as colors
import matplotlib.dates    as mdates
import matplotlib         as mpl
import matplotlib.cm      as cm

sys.path.append('D://Google Drive//Uni//PhD 2017//Data//Scripts//')
import crres_file_readers as cfr
import rbsp_file_readers  as rfr
import rbsp_fields_loader as rfl
import analysis_scripts   as ascr
import fast_scripts       as fscr
import nonlinear_scripts  as nls
import extract_parameters_from_data as epd

sys.path.append('..//new_general_DR_solver//')
from matplotlib.lines              import Line2D
from multiapprox_dispersion_solver import create_species_array, get_dispersion_relation, get_cold_growth_rates


#%% Constants
PCHARGE = 1.602e-19
ECHARGE =-1.602e-19
PMASS   = 1.673e-27
EMASS   = 9.110e-31
EPS0    = 8.854e-12
MU0     = 4e-7*np.pi
RE      = 6.371e6
SPLIGHT = 3e8
KB      = 1.380649e-23
B_SURF  = 3.12e-5



#%% FIELD ANALYSIS
def get_mag_data(time_start, time_end, probe, low_cut=None, high_cut=None):
    '''
    Load and process data
    '''        
    if probe != 'crres':
        ti, pc1_mags, pc1_elec, HM_mags, HM_elec, dt, e_flag, gyfreqs =\
        rfl.load_decomposed_fields(_rbsp_path, time_start, time_end, probe, 
                               pad=600, LP_B0=1.0, LP_HM=30.0, ex_threshold=5.0, 
                               get_gyfreqs=True)
        
    else:
        ti, B0, HM_mags, pc1_mags, \
        E0, HM_elec, pc1_elec, S, B, E = cfr.get_crres_fields(_crres_path, time_start, time_end,
                                        pad=600, output_raw_B=True, Pc5_LP=30.0, B0_LP=1.0,
                                        Pc5_HP=None, dEx_LP=None, interpolate_nan=True)
        dt = 1/32.
    
    # Bandpass selected component and return
    if low_cut is not None:
        dat = ascr.clw_high_pass(pc1_mags, low_cut*1000., dt, filt_order=4)
    if high_cut is not None:
        dat = ascr.clw_low_pass(pc1_mags, high_cut*1000., dt, filt_order=4)
    
    #pc1_res = 5.0
    _xpow, _xtime, _xfreq = fscr.autopower_spectra(ti, pc1_mags[:, 0], time_start, 
                                            time_end, dt, overlap=0.95, df=pc1_res,
                                            window_data=True)
    
    _ypow, _xtime, _xfreq = fscr.autopower_spectra(ti, pc1_mags[:, 1], time_start, 
                                            time_end, dt, overlap=0.95, df=pc1_res,
                                            window_data=True)
    
    _zpow, _xtime, _xfreq = fscr.autopower_spectra(ti, pc1_mags[:, 2], time_start, 
                                            time_end, dt, overlap=0.95, df=pc1_res,
                                            window_data=True)
    
    _pow = np.array([_xpow, _ypow, _zpow])
    return ti, dat, HM_mags, dt, _xtime, _xfreq, _pow


def load_EMIC_IMFs_and_dynspec(imf_start, imf_end, IA_filter=None):
    '''
    Loads IMFs and dynamic spectra
    '''
    _ti, _dat, _HM_mags, _dt, _xtime, _xfreq, _pow = get_mag_data(_time_start, _time_end, 
                                    _probe, low_cut=_band_start, high_cut=_band_end)
    sample_rate = 1./_dt
    
    # Calculate IMFs, get instantaneous phase, frequency, amplitude
    print('Sifting IMFs and performing HHT')
    imfs, IPs, IFs, IAs = [], [], [], []
    for ii, lbl in zip(range(3), ['x', 'y', 'z']):
        imf = emd.sift.sift(_dat[:, ii], sift_thresh=1e-2)
        
        IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert')
        print(f'{imf.shape[1]} IMFs found for B{lbl}')
        
        imfs.append(imf)
        IPs.append(IP)
        IFs.append(IF)
        IAs.append(IA)
        
# =============================================================================
#     if IA_filter is not None:
#         for ii in range(3):
#             for ii in range(IAs[ii].shape[0]):
#                 if 
# =============================================================================
    
    # Snip IMFs to time
    st, en = ascr.boundary_idx64(_ti, imf_start, imf_end)
    for ii in range(3):
        _imf_t  = _ti[st:en]
        IAs[ii] = IAs[ii][st:en]
        IFs[ii] = IFs[ii][st:en]
        IPs[ii] = IPs[ii][st:en]
    return _ti, _dat, _HM_mags, _imf_t, IAs, IFs, IPs, _xtime, _xfreq, _pow


def get_pc1_peaks(sfreq, spower, band_start, band_end, npeaks=None):
    '''
    Returns integrated spower spectrum between band_start and band_end as well
    as index locations of the most prominent npeaks
    '''
    fst, fen = ascr.boundary_idx64(sfreq, band_start, band_end)
    pc1_int_power = spower[fst:fen, :].sum(axis=0)
    peak_idx = ascr.find_peaks(pc1_int_power, npeaks=npeaks, sortby='prom')
    return pc1_int_power, peak_idx



#%% DATA INTERPOLATION AND LOADING
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


def SPICE_interpolate_to_time(new_time, SPICE_times, SPICE_dens, SPICE_PerpPres, SPICE_ParaPres):
    '''
    edens_time  :: WAVES electron density time to interpolate data_array to (length M)
    data_times  :: Current RBSPICE data sample times of length 3xN
    data_array  :: Data arrays consisting of ni, Ti, Ai in a 3xN ndarra
    '''
    new_dens = np.zeros((3, new_time.shape[0]), dtype=np.float64)
    new_Pper = np.zeros((3, new_time.shape[0]), dtype=np.float64)
    new_Ppar = np.zeros((3, new_time.shape[0]), dtype=np.float64)
    
    xi = new_time.astype(np.int64)
    
    for ii in range(3):
        
        # Skip if bad
        if (SPICE_times[ii] is None) or (SPICE_dens[ii] is None):
            continue
        
        xp              = SPICE_times[ii].astype(np.int64)
        new_dens[ii, :] = np.interp(xi, xp, SPICE_dens[ii, :])
        new_Pper[ii, :] = np.interp(xi, xp, SPICE_PerpPres[ii, :])
        new_Ppar[ii, :] = np.interp(xi, xp, SPICE_ParaPres[ii, :])

    return new_dens, new_Pper, new_Ppar


def interpolate_B(new_time, b_time, b_array, dt, LP_filter=True):
    '''
    Does second LP filter based on the Nyquist of the new sample rate
    Different to original filter, which is just to get rid of EMIC signal
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
                                       time_array=None, check_interp=False):
    '''
    Same copy+paste as other versions, without the SPICE stuff
    '''
    print('Loading and interpolating satellite data')
    # Ephemeris data
    # Load ephemeris data
    eph_params  = ['L', 'CD_MLAT', 'CD_MLT']
    eph_times, eph_dict = rfr.retrieve_RBSP_ephemeris_data(rbsp_path, probe, time_start, time_end, 
                            eph_params, padding=[60, 60])
    
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
    hope_dens  = np.array([pdict['Dens_p_30'],       pdict['Dens_he_30'],       pdict['Dens_o_30']])
    hope_tperp = np.array([pdict['Tperp_p_30'],      pdict['Tperp_he_30'],      pdict['Tperp_o_30']])
    hope_tpar  = np.array([pdict['Tpar_p_30'],       pdict['Tpar_he_30'],       pdict['Tpar_o_30']])
    hope_anis  = np.array([pdict['Tperp_Tpar_p_30'], pdict['Tperp_Tpar_he_30'], pdict['Tperp_Tpar_o_30']]) - 1.
    
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
    
    ihope_dens , ihope_tpar , ihope_anis  = HOPE_interpolate_to_time(time_array, itime, hope_dens, hope_tpar, hope_anis)
    ihope_dens , ihope_tperp, ihope_anis  = HOPE_interpolate_to_time(time_array, itime, hope_dens, hope_tperp, hope_anis)

    Bi = interpolate_B(time_array, mag_times, filt_mags, nsec, LP_filter=False)
    
    iL = interpolate_ne(time_array, eph_times, eph_dict['L'])
    if check_interp:
        plt.ioff()
        
        # Cold dens + Magnetic field
        fig1, axes1 = plt.subplots(2)
        
        axes1[0].plot(den_times, edens, c='b')
        axes1[0].plot(time_array, iedens, c='r', lw=0.75)
        axes1[0].set_ylabel('$n_e$')
        
        B_total = np.sqrt(raw_mags[:, 0]**2+raw_mags[:, 1]**2+raw_mags[:, 2]**2)
        axes1[1].plot(mag_times, B_total, c='b')
        axes1[1].plot(time_array, Bi, c='r', lw=0.75)
        axes1[1].set_ylabel('B')
        
        for ax in axes1:
            ax.set_xlim(time_start, time_end)
            
        # HOPE parameters (dens, temp, anis)
        fig2, axes2 = plt.subplots(3)
        
        for xx, clr in zip(range(3), ['r', 'g', 'b']):
            axes2[0].plot(itime, hope_dens[xx], c=clr, ls='-')
            axes2[0].plot(time_array, ihope_dens[xx], c=clr, ls='--')
            axes2[0].set_ylabel('$n_i (cc)$')
            
            axes2[1].plot(itime, hope_tpar[xx], c=clr, ls='-')
            axes2[1].plot(time_array, ihope_tpar[xx], c=clr, ls='--')
            axes2[1].set_ylabel('$T_{\perp, i} (keV)$')
            
            axes2[2].plot(itime, hope_anis[xx], c=clr, ls='-')
            axes2[2].plot(time_array, ihope_anis[xx], c=clr, ls='--')
            axes2[2].set_ylabel('$A_i$')
        
        for ax in axes2:
            ax.set_xlim(time_start, time_end)
        
        plt.show()
    
    # Subtract energetic components from total electron density (assuming each is singly charged)
    cold_dens = iedens - ihope_dens.sum(axis=0)
    
    # Original DTs just for reference
    den_dt   = (  den_times[1] -   den_times[0]) / np.timedelta64(1, 's')
    mag_dt   = (  mag_times[1] -   mag_times[0]) / np.timedelta64(1, 's')
    hope_dt  = (      itime[1] -       itime[0]) / np.timedelta64(1, 's')
    new_dt   = ( time_array[1] -  time_array[0]) / np.timedelta64(1, 's')
    print('Original sample periods:')
    print(f'Cold Plasma Density:   {den_dt} s')
    print(f'Magnetic field:        {mag_dt} s ')
    print(f'HOPE Particle data:    {hope_dt} s')
    print('')
    print(f'New sample period: {new_dt} s')
    return time_array, Bi*1e-9, iedens*1e6, cold_dens*1e6, ihope_dens*1e6, ihope_tpar, ihope_tperp, ihope_anis, iL


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
    

#%% PLOTTING ROUTINES
def add_custom_legend(_ax, _labels, _linestyles, _alpha, _color):
    '''
    TODO: Add some catches for 'if None...' for things like alpha and linestyle
    '''
    legend_elements = []
    for label, style, alpha, clr in zip(_labels, _linestyles, _alpha, _color):
        legend_elements.append(Line2D([0], [0], color=clr, lw=1,
                              label=label, linestyle=style, alpha=alpha))
        
    new_legend = _ax.legend(handles=legend_elements, loc='upper left')
    return new_legend


def plot_velocities_and_energies_single(time_start, time_end, probe='a'):
    # Import cutoff-derived composition information
    cutoff_dict = epd.read_cutoff_file(cutoff_filename)
        
    # Load particle and field information
    if probe.lower() != 'crres':
        time, mag, edens, cold_dens, hope_dens, hope_tpar, hope_tperp, hope_anis, L_vals =\
                                           load_and_interpolate_plasma_params(
                                           time_start, time_end, probe, nsec=None, 
                                           rbsp_path='E://DATA//RBSP//', HM_filter_mhz=50.0,
                                           time_array=None, check_interp=False)
    else:
        load_CRRES_data(time_start, time_end, crres_path='E://DATA//CRRES//', nsec=None)
                  
    mag_time, pc1_mags, HM_mags, imf_time, IA, IF, IP, stime, sfreq, spower = \
            load_EMIC_IMFs_and_dynspec(time_start, time_end)
    spower = spower.sum(axis=0)

    cutoff  = np.interp(time.astype(np.int64),
                        cutoff_dict['CUTOFF_TIME'].astype(np.int64),
                        cutoff_dict['CUTOFF_NORM'])
    o_frac = epd.calculate_o_from_he_and_cutoff(cutoff, he_frac)      
    
    # Get velocity/energy data for time
    plt.ioff()
    for ii in range(time.shape[0]):
        
        # Define time, time index
        this_time   = time[ii]
        maxpwr_tidx = np.where(abs(stime - this_time) == np.min(abs(stime - this_time)))[0][0]
        
        # Find max freq and index
        fst, fen     = ascr.boundary_idx64(sfreq, _band_start, _band_end)
        maxpwr_fidx  = spower[maxpwr_tidx, fst:fen].argmax()
        maxpwr_fidx += fst
        maxpwr_freq  = sfreq[maxpwr_fidx]
        
        #date_string = this_time.astype(object).strftime('%Y%m%d')
        save_string = this_time.astype(object).strftime('%Y%m%d_%H%M%S')
    
        #clr = mapper.to_rgba(time[ii].astype(np.int64))
        print('Doing time:', this_time)
        
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8.0, 0.5*11.00),
                                 gridspec_kw={'width_ratios':[1, 0.01],
                                              'height_ratios':[1, 2, 2, 2]
                                              })
        
        # Spectra/IP
        im0 = axes[0, 0].pcolormesh(stime, sfreq, spower.T, cmap='jet',
                             norm=colors.LogNorm(vmin=1e-4, vmax=1e1))
        axes[0, 0].set_ylim(0, fmax)
        axes[0, 0].set_ylabel('$f$ [Hz]', rotation=90)
        fig.colorbar(im0, cax=axes[0, 1], extend='both').set_label(
                    r'$\frac{nT^2}{Hz}$', fontsize=16, rotation=0, labelpad=15)
        
        axes[0, 0].axvline(this_time, color='white', ls='-' , alpha=0.7)
        axes[0, 0].set_xlim(time_start, time_end)
        axes[0, 0].set_xticklabels([])
        
        axes[0, 0].axvline(this_time,   color='white', ls='-', alpha=0.75)
        axes[0, 0].axhline(maxpwr_freq, color='white', ls='-', alpha=0.75)
        axes[0, 0].set_title(f'Velocities and Energies :: {this_time}')
        
        # Get oxygen concentration from cutoffs
        h_frac = 1. - he_frac - o_frac[ii]

        # Cold plasma params, SI units
        B0      = mag[ii]
        name    = np.array(['H'   , 'He'   , 'O'        ])
        mass    = np.array([1.0   , 4.0    , 16.0       ]) * PMASS
        charge  = np.array([1.0   , 1.0    , 1.0        ]) * PCHARGE
        density = np.array([h_frac, he_frac, o_frac[ii] ]) * edens[ii]
        ani     = np.array([0.0   , 0.0    , 0.0        ])
        tpar    = np.array([0.0   , 0.0    , 0.0        ])
        tper    = (ani + 1) * tpar
        Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
        
        # Frequencies to evaluate, calculate wavenumber (cold approximation)
        f_min  = _band_start
        f_max  = _band_end
        Nf     = 10000
        f_vals = np.linspace(f_min, f_max, Nf)
        w_vals = 2*np.pi*f_vals
        k_vals = nls.get_k_cold(w_vals, Species)
        wv_len = 1e-3 * 2*np.pi / k_vals
        
        fidx = np.where(abs(f_vals - maxpwr_freq) == np.min(abs(f_vals - maxpwr_freq)))[0][0] 
        freq = f_vals[fidx]
        
        axes[1, 0].plot(f_vals, wv_len, c='k')
        axes[1, 0].set_ylabel('$\lambda_{EMIC}$ [km]')
        axes[1, 0].set_ylim(0, 2000)
        
        # Velocities
        Vg, Vp, Vr = nls.get_velocities(w_vals, Species, PP)
    
        axes[2, 0].semilogy(f_vals, Vg*1e-3, c='k', label='$V_g$') 
        axes[2, 0].semilogy(f_vals, Vp*1e-3, c='r', label='$V_p$')
        axes[2, 0].semilogy(f_vals,-Vr*1e-3, c='b', label='$-V_R$')
        axes[2, 0].set_ylim(10, 4500)
        axes[2, 0].set_ylabel('Velocities [km/s]')
        axes[2, 0].legend(bbox_to_anchor=(1.0, 0), loc=3, borderaxespad=0.)
        
        # Energies
        ELAND, ECYCL = nls.get_energies(w_vals, k_vals, PP['pcyc_rad'], PMASS)
        axes[3, 0].semilogy(f_vals, ELAND*1e-3, c='r', label='$E_0$')
        axes[3, 0].semilogy(f_vals, ECYCL*1e-3, c='b', label='$E_1$')
        axes[3, 0].set_ylim(1e-1, 1e3)
        axes[3, 0].set_ylabel('$E_R$ [keV]')
        axes[3, 0].set_xlabel('Freq [Hz]', rotation=0)
        axes[3, 0].legend(bbox_to_anchor=(1.0, 0), loc=3, borderaxespad=0.)
        
        # Work out what the energy is at maxpwr_freq
        num_landau    = ELAND[fidx]*1e-3
        num_cyclotron = ECYCL[fidx]*1e-3
        
        axes[3, 0].text(0.8, 0.9, f'$E_0(f) =$ {num_landau:.1f} keV', horizontalalignment='left',
                        verticalalignment='center', transform=axes[3, 0].transAxes)
        axes[3, 0].text(0.8, 0.8, f'$E_1(f) =$ {num_cyclotron:.1f} keV', horizontalalignment='left',
                        verticalalignment='center', transform=axes[3, 0].transAxes)
        
        axes[0, 0].set_xticklabels([])
        axes[1, 0].set_yticks(np.array([0, 500, 1000, 1500]))
        for ax in axes[1:, 0]:
            ax.set_xlim(_band_start, _band_end)
            ax.axvline(freq, color='k', alpha=0.5, ls='--')
            if ax != axes[-1, 0]:
                ax.set_xticklabels([])
        
        axes[1, 1].set_visible(False)
        axes[2, 1].set_visible(False)
        axes[3, 1].set_visible(False)
        
        #fig.tight_layout(rect=[0, 0, 0.95, 1])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.05, hspace=0)
        fig.align_ylabels()
        
        if save_plot == True:
            print('Saving plot...')
            fig.savefig(_plot_path + 'VELENG_FROMDATA_' + save_string + '.png', dpi=dpi)
            plt.close('all')
        else:
            plt.show()
    return


def plot_HM_and_energy(time_start, time_end, probe):
    '''
    For a specific frequency, calculate what the first order cyclotron resonance
    energy is and plot this as a timeseries. Secondary plots for the Pc1 spectra
    and HM/|B| spectra
    
    Do for all frequencies and select specific frequency at end (or option to
    show as a pcolormesh)
    '''
    # Import cutoff-derived composition information
    cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20130725_RBSP-A//pearl_times.txt'
    cutoff_dict     = epd.read_cutoff_file(cutoff_filename)
        
    time, mag, edens, cold_dens, hope_dens, hope_tpar, hope_tperp, hope_anis, L_vals =\
                                       load_and_interpolate_plasma_params(
                                       time_start, time_end, probe, nsec=None, 
                                       rbsp_path='E://DATA//RBSP//', HM_filter_mhz=50.0,
                                       time_array=None, check_interp=False)
    
    mag_time, pc1_mags, HM_mags, imf_time, IA, IF, IP, stime, sfreq, spower = \
            load_EMIC_IMFs_and_dynspec(time_start, time_end)

    # Frequencies to evaluate, calculate wavenumber (cold approximation)
    f_min  = _band_start
    f_max  = _band_end
    Nf     = 10000
    f_vals = np.linspace(f_min, f_max, Nf)
    w_vals = 2*np.pi*f_vals
    
    all_CR = np.zeros((time.shape[0], Nf), dtype=np.float64)
    all_LR = np.zeros((time.shape[0], Nf), dtype=np.float64)
    all_VP = np.zeros((time.shape[0], Nf), dtype=np.float64)
    all_VG = np.zeros((time.shape[0], Nf), dtype=np.float64)
    all_VR = np.zeros((time.shape[0], Nf), dtype=np.float64)
    
    # Collect info for each time
    for ii in range(time.shape[0]):
        this_time = time[ii]
        print('Doing time:', this_time)
           
        # Get oxygen concentration from cutoffs
        cutoff  = np.interp(this_time.astype(np.int64),
                            cutoff_dict['CUTOFF_TIME'].astype(np.int64),
                            cutoff_dict['CUTOFF_NORM'])
        o_frac = epd.calculate_o_from_he_and_cutoff(cutoff, he_frac)
        h_frac = 1. - he_frac - o_frac
        
        # Cold plasma params, SI units
        B0      = mag[ii]
        name    = np.array(['H'   , 'He'   , 'O'    ])
        mass    = np.array([1.0   , 4.0    , 16.0   ]) * PMASS
        charge  = np.array([1.0   , 1.0    , 1.0    ]) * PCHARGE
        density = np.array([h_frac, he_frac, o_frac ]) * edens[ii]
        ani     = np.array([0.0   , 0.0    , 0.0    ])
        tpar    = np.array([0.0   , 0.0    , 0.0    ])
        tper    = (ani + 1) * tpar
        Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
        
        # Velocities and Energies
        k_vals = nls.get_k_cold(w_vals, Species)
        all_VG[ii], all_VP[ii], all_VR[ii] = nls.get_velocities(w_vals, Species, PP)
        all_LR[ii], all_CR[ii] = nls.get_energies(w_vals, k_vals, PP['pcyc_rad'], PMASS)
        
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8.0, 1.0*11.00),
                                 gridspec_kw={'width_ratios':[1, 0.01]
                                              })

    # To plot: Spectra, velocities (Vg, Vr) and energies (cyclotron, landau resonances)
    # HM/|B| overlay
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axes[0, 0].set_title(f'Velocities and Energies :: {this_time}')
        
        # Spectra
        im0 = axes[0, 0].pcolormesh(stime, sfreq, spower.sum(axis=0).T, cmap='jet',
                             norm=colors.LogNorm(vmin=1e-4, vmax=1e1))
        axes[0, 0].set_ylim(0, fmax)
        axes[0, 0].set_ylabel('$f$ [Hz]', rotation=90)
        fig.colorbar(im0, cax=axes[0, 1], extend='both').set_label(
                    r'$\frac{nT^2}{Hz}$', fontsize=14, rotation=0, labelpad=15)
        
        # E_RES_CYCLOTRON
        im1 = axes[1, 0].pcolormesh(time, f_vals, all_CR.T*1e-3, cmap='jet',
                             norm=colors.Normalize())
        axes[1, 0].set_ylabel('$f$ [Hz]', rotation=90)
        fig.colorbar(im1, cax=axes[1, 1], extend='both').set_label(
                    '$E_R$ Cyclotron\n[keV]', fontsize=14, rotation=0, labelpad=15)
        
        # E_RES_LANDAU
        im2 = axes[2, 0].pcolormesh(time, f_vals, all_LR.T*1e-3, cmap='jet',
                             norm=colors.Normalize())
        axes[2, 0].set_ylabel('$f$ [Hz]', rotation=90)
        fig.colorbar(im2, cax=axes[2, 1], extend='both').set_label(
                    '$E_R$ Landau\n[keV]', fontsize=14, rotation=0, labelpad=15)
        
        # VR
        im3 = axes[3, 0].pcolormesh(time, f_vals, all_VR.T*1e-3, cmap='jet',
                             norm=colors.Normalize())
        axes[3, 0].set_ylabel('$f$ [Hz]', rotation=90)
        fig.colorbar(im3, cax=axes[3, 1], extend='both').set_label(
                    '$V_R$\n[$km/s$]', fontsize=14, rotation=0, labelpad=15)
        
        # VG
        im4 = axes[4, 0].pcolormesh(time, f_vals, all_VG.T*1e-3, cmap='jet',
                             norm=colors.Normalize())
        axes[4, 0].set_ylabel('$f$ [Hz]', rotation=90)
        fig.colorbar(im4, cax=axes[4, 1], extend='both').set_label(
                    '$V_G$\n[$km/s$]', fontsize=14, rotation=0, labelpad=15)
        
        for ax in axes[:, 0]:
            ax.set_xlim(time_start, time_end)
            ax.set_ylim(_band_start, _band_end)
            
            ax2 = ax.twinx()
            ax2.plot(mag_time, HM_mags[:, 2], c='k', lw=0.75)
                
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0)
    fig.align_ylabels()
    
    if save_plot == True:
        save_string = time_start.astype(object).strftime('%Y%m%d_%H%M')
        
        print('Saving plot...')
        fig.savefig(_plot_path + 'VELENG_FROMDATA_PCOLOR_' + save_string + '.png')
        plt.close('all')
    else:
        plt.show()
    return

#%% MAIN
if __name__ == '__main__':
    _rbsp_path  = 'E://DATA//RBSP//'
    _crres_path = 'E://DATA//CRRES//'
    _plot_path  = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//' 
    if not os.path.exists(_plot_path): os.makedirs(_plot_path)
    save_plot   = True
    
    # TODO: Put all important event-specific variables in the switch
    pc1_res = 15.0
    dpi = 200
    if True:
        _probe = 'a'
        _time_start = np.datetime64('2013-07-25T21:25:00')
        _time_end   = np.datetime64('2013-07-25T21:47:00')
        _band_start = 0.20
        _band_end   = 0.80

        _npeaks     = 22
        fmax        = 1.0
        he_frac     = 0.30
        
        L     = 4.70
        B_max = 17.0
        
        #cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20130725_RBSP-A//pearl_times.txt'
        cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20130725_RBSP-A//cutoffs_only.txt'
    else:
        _probe = 'a'
        _time_start = np.datetime64('2015-01-16T04:25:00')
        _time_end   = np.datetime64('2015-01-16T05:15:00')
        _band_start = 0.10
        _band_end   = 0.20
        _npeaks     = 22
        fmax        = 0.5
        he_frac     = 0.30
        
        L     = 5.73
        B_max = 6.00
        
        cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20150116_RBSP-A//cutoffs_only.txt'
        #cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20150116_RBSP-A//cutoffs_only_10mHz.txt'
        #cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20150116_RBSP-A//pearl_times.txt'
    
    if False:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #calculate_all_NL_amplitudes()
            plot_velocities_and_energies_single(_time_start, _time_end, probe=_probe)
            #plot_HM_and_energy(_time_start, _time_end, _probe)
    #sys.exit()
    
    time_start = _time_start
    time_end   = _time_end
    probe      = 'a'
    pad        = 0
    
    plot_start = time_start - np.timedelta64(int(pad), 's')
    plot_end   = time_end   + np.timedelta64(int(pad), 's')
    
    #%% Stuff requiring all the data
    # Import cutoff-derived composition information
    if False:
        cutoff_dict     = epd.read_cutoff_file(cutoff_filename)
            
        time, mag, edens, cold_dens, hope_dens, hope_tpar, hope_tperp, hope_anis, L_vals =\
                                           load_and_interpolate_plasma_params(
                                           time_start, time_end, probe, nsec=None, 
                                           rbsp_path='E://DATA//RBSP//', HM_filter_mhz=50.0,
                                           time_array=None, check_interp=False)
        
        hope_dens[1] *= 0
        hope_dens[2] *= 0
        
        mag_time, pc1_mags, HM_mags, imf_time, IA, IF, IP, stime, sfreq, spower = \
                load_EMIC_IMFs_and_dynspec(time_start, time_end)
        spower = spower.sum(axis=0)
        
        # Frequencies to evaluate, calculate wavenumber (cold approximation)
        f_min  = _band_start
        f_max  = _band_end
        Nf     = 10000
        f_vals = np.linspace(f_min, f_max, Nf)
        w_vals = 2*np.pi*f_vals
        
        all_CR = np.zeros((time.shape[0], Nf), dtype=np.float64)
        all_LR = np.zeros((time.shape[0], Nf), dtype=np.float64)
        all_VP = np.zeros((time.shape[0], Nf), dtype=np.float64)
        all_VG = np.zeros((time.shape[0], Nf), dtype=np.float64)
        all_VR = np.zeros((time.shape[0], Nf), dtype=np.float64)
        
        line_CR = np.zeros(time.shape[0], dtype=np.float64)
        line_LR = np.zeros(time.shape[0], dtype=np.float64)
        line_VP = np.zeros(time.shape[0], dtype=np.float64)
        line_VG = np.zeros(time.shape[0], dtype=np.float64)
        line_VR = np.zeros(time.shape[0], dtype=np.float64)
        
        max_freqs = np.zeros(time.shape[0], dtype=np.float64)
        
        # Get oxygen concentration from cutoffs
        cutoff = np.interp(time.astype(np.int64),
                           cutoff_dict['CUTOFF_TIME'].astype(np.int64),
                           cutoff_dict['CUTOFF_NORM'])
        o_frac = epd.calculate_o_from_he_and_cutoff(cutoff, he_frac)
            
        # Collect info for each time
        for ii in range(time.shape[0]):
            print('Doing time:', time[ii])
            h_frac  = 1. - he_frac - o_frac[ii]
            
            # Define time, time index
            this_time   = time[ii]
            maxpwr_tidx = np.where(abs(stime - this_time) == np.min(abs(stime - this_time)))[0][0]
            
            # Find max freq and index
            fst, fen     = ascr.boundary_idx64(sfreq, _band_start, _band_end)
            maxpwr_fidx  = spower[maxpwr_tidx, fst:fen].argmax()
            maxpwr_fidx += fst
            max_freqs[ii] = sfreq[maxpwr_fidx]
            
            # Cold plasma params, SI units
            B0      = mag[ii]
            name    = np.array(['H'   , 'He'   , 'O'        ])
            mass    = np.array([1.0   , 4.0    , 16.0       ]) * PMASS
            charge  = np.array([1.0   , 1.0    , 1.0        ]) * PCHARGE
            density = np.array([h_frac, he_frac, o_frac[ii] ]) * edens[ii] 
            ani     = np.array([0.0   , 0.0    , 0.0        ])
            tpar    = np.array([0.0   , 0.0    , 0.0        ])
            tper    = (ani + 1) * tpar

            Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
            
            # Velocities and Energies
            k_vals = nls.get_k_cold(w_vals, Species)
            all_VG[ii], all_VP[ii], all_VR[ii] = nls.get_velocities(w_vals, Species, PP)
            all_LR[ii], all_CR[ii] = nls.get_energies(w_vals, k_vals, PP['pcyc_rad'], PMASS)
            
            fidx = np.where(abs(f_vals - max_freqs[ii]) == np.min(abs(f_vals - max_freqs[ii])))[0][0] 
            freq = f_vals[fidx]
        
            # Get values at the frequency of max power
            line_LR[ii] = all_LR[ii, fidx]
            line_CR[ii] = all_CR[ii, fidx]
            
            line_VP[ii] = all_VP[ii, fidx]
            line_VG[ii] = all_VG[ii, fidx]
            line_VR[ii] = all_VR[ii, fidx]
        
           
        #%% PLOT PART        
# =============================================================================
#         fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8.0, 0.5*11.00),
#                                      gridspec_kw={'width_ratios':[1, 0.02]
#                                                   })
#         
#         # To plot: Spectra, velocities (Vg, Vr) and energies (cyclotron, landau resonances)
#         # HM/|B| overlay
#         clpad = 40; cfont=10
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             axes[0, 0].set_title('Velocities and Energies')
#             
#             # Spectra
#             im0 = axes[0, 0].pcolormesh(stime, sfreq, spower.sum(axis=0).T, cmap='jet',
#                                  norm=colors.LogNorm(vmin=1e-4, vmax=1e1))
#             axes[0, 0].set_ylim(0, fmax)
#             axes[0, 0].set_ylabel('$f$ [Hz]', rotation=90)
#             fig.colorbar(im0, cax=axes[0, 1], extend='both').set_label(
#                         r'$\frac{nT^2}{Hz}$', fontsize=cfont, rotation=0, labelpad=clpad)
#             
#             axes[0, 0].axhline(f_vals[fst], c='white', ls='--')
#             axes[0, 0].axhline(f_vals[fen], c='white', ls='--')
#             
#             ax2 = axes[0, 0].twinx()
#             ax2.plot(mag_time, HM_mags[:, 2], c='k', lw=0.75)
#                 
#             
#             # E_RES_CYCLOTRON
#             im1 = axes[1, 0].pcolormesh(time, f_vals, all_CR.T*1e-3, cmap='jet',
#                                  norm=colors.LogNorm(vmin=1e-1, vmax=1e3))
#             axes[1, 0].set_ylabel('$f$ [Hz]', rotation=90)
#             fig.colorbar(im1, cax=axes[1, 1], extend='both').set_label(
#                         '$E_R$ Cyclotron\n[keV]', fontsize=cfont, rotation=0, labelpad=clpad)
#             
#             ax3 = axes[1, 0].twinx()
#             ax3.plot(time, HM_mags[:, 2], c='k', lw=0.75)
#             
#             # VG
#             im4 = axes[2, 0].pcolormesh(time, f_vals, all_VG.T*1e-3, cmap='jet',
#                                  norm=colors.LogNorm(vmin=1e1, vmax=1e3))
#             axes[2, 0].set_ylabel('$f$ [Hz]', rotation=90)
#             fig.colorbar(im4, cax=axes[2, 1], extend='both').set_label(
#                         '$V_G$\n[$km/s$]', fontsize=cfont, rotation=0, labelpad=clpad)
#             
#             for ax in axes[:, 0]:
#                 ax.set_xlim(time_start, time_end)
#                 ax.set_ylim(_band_start, _band_end)
#                 
#                 
#                     
#         fig.tight_layout()
#         fig.subplots_adjust(wspace=0.15, hspace=0)
#         fig.align_ylabels()
#         
#         if save_plot == True:
#             save_string = time_start.astype(object).strftime('%Y%m%d_%H%M')
#             
#             print('Saving plot...')
#             fig.savefig(_plot_path + 'VELENG_FROMDATA_PCOLOR_' + save_string + '.png')
#             plt.close('all')
#         else:
#             plt.show()
#         print('Finished')
# =============================================================================
    
        
        #%% Energy and velocity timeseries
        ## Plot Spectra, resonant energy, 3x velocity
        ## Overlay max frequency as a scatter over spectra
        plt.ioff()
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8.0, 0.5*11.00),
                                      gridspec_kw={'width_ratios':[1, 0.01]
                                                  })
        
        # To plot: Spectra, velocities (Vg, Vr) and energies (cyclotron, landau resonances)
        # HM/|B| overlay
        clpad = 20; cfont=12; fsize=12; lpad=20
        #axes[0, 0].set_title('Velocities and Energies')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Spectra
            im0 = axes[0, 0].pcolormesh(stime, sfreq, spower.T, cmap='jet',
                                 norm=colors.LogNorm(vmin=1e-4, vmax=1e1))
            axes[0, 0].set_ylim(0, fmax)
            axes[0, 0].set_ylabel('$f$\n(Hz)', rotation=0, fontsize=fsize, labelpad=lpad)
            fig.colorbar(im0, cax=axes[0, 1], extend='both').set_label(
                        r'$\frac{nT^2}{Hz}$', fontsize=cfont, rotation=0, labelpad=clpad)
            
            axes[0, 0].axhline(_band_start, c='white', ls='--')
            axes[0, 0].axhline(_band_end  , c='white', ls='--')
            axes[0, 0].scatter(time, max_freqs, c='k', marker='x', s=2)
                
        # Line plot for lower/upper frequency bin (c_res, l_res, vr, vg)
        axes[1, 0].plot(time, line_CR*1e-3, c='k', lw=0.75)
        axes[1, 0].set_ylabel('$E_R$\n(keV)', rotation=0, fontsize=fsize, labelpad=lpad)
        #axes[1, 0].set_ylim(0.0, 25.0)
        axes[1, 0].set_ylim(0.0, 65.0)
        axes[1, 1].set_visible(False)
        
        axes[2, 0].plot(time, line_VG*1e-3, c='k', lw=0.75)
        axes[2, 0].set_ylabel('$V_G$\n(km/s)', rotation=0, fontsize=fsize, labelpad=lpad)
        axes[2, 0].set_ylim(None, None)
        axes[2, 1].set_visible(False)
        
        axes[3, 0].plot(time,-line_VR*1e-3, c='k', lw=0.75)
        axes[3, 0].set_ylabel('$V_R$\n(km/s)', rotation=0, fontsize=fsize, labelpad=lpad)   
        #axes[3, 0].set_ylim(5e2, 2.5e3)
        axes[3, 0].set_ylim(5e2, 4e3)
        axes[3, 1].set_visible(False)
        
        axes[4, 0].plot(time, line_VP*1e-3, c='k', lw=0.75)
        axes[4, 0].set_ylabel('$V_P$\n(km/s)', rotation=0, fontsize=fsize, labelpad=lpad)
        #axes[4, 0].set_ylim(50, 5e2)
        axes[4, 0].set_ylim(50, 4e2)
        axes[4, 1].set_visible(False)
        
        axes[-1, 0].set_xlabel('Time (UT)')
        axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        for ax in axes[:, 0]:
            ax.set_xlim(time_start, time_end)
            
            if ax != axes[-1, 0]:
                ax.set_xticklabels([])
        
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.05, hspace=0)
        fig.align_ylabels()
        
        if save_plot == True:
            save_string = time_start.astype(object).strftime('%Y%m%d_%H%M')
            print('Saving plot...')
            fig.savefig(_plot_path + 'VELENG_TIMESERIES_' + save_string + '.png', dpi=dpi)
            plt.close('all')
        else:
            plt.show()