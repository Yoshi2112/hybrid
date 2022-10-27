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
        ti, pc1_mags, pc1_elec, HM_mags, HM_elec, dt, e_flag, gfreq =\
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
    return ti, dat, HM_mags, dt, _xtime, _xfreq, _pow, gfreq


def load_EMIC_IMFs_and_dynspec(imf_start, imf_end, IA_filter=None):
    '''
    Loads IMFs and dynamic spectra
    '''
    _ti, _dat, _HM_mags, _dt, _xtime, _xfreq, _pow, gfreq = get_mag_data(_time_start, _time_end, 
                                    _probe, low_cut=_band_start, high_cut=_band_end)
    sample_rate = 1./_dt
    
    # Calculate IMFs, get instantaneous phase, frequency, amplitude
    print(f'Sifting IMFs and performing HHT between {imf_start} and {imf_end}')
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
    return _ti, _dat, _HM_mags, _imf_t, IAs, IFs, IPs, _xtime, _xfreq, _pow, gfreq


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


def calculate_all_NL_amplitudes():
    # Import cutoff-derived composition information
    cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20130725_RBSP-A//pearl_times.txt'
    cutoff_dict     = epd.read_cutoff_file(cutoff_filename)
    
    #plot_amplitudes_from_data(_time_start, _time_end, probe=_probe, pad=600)
    time_start = _time_start
    time_end   = _time_end
    probe      = 'a'
    pad        = 0
    
    plot_start = time_start - np.timedelta64(int(pad), 's')
    plot_end   = time_end   + np.timedelta64(int(pad), 's')
    
    time, mag, edens, cold_dens, hope_dens, hope_tpar, hope_tperp, hope_anis, L_vals =\
                                       load_and_interpolate_plasma_params(
                                       plot_start, plot_end, probe, nsec=None, 
                                       rbsp_path='E://DATA//RBSP//', HM_filter_mhz=50.0,
                                       time_array=None, check_interp=False)
                  
    mag_time, pc1_mags, HM_mags, imf_time, IA, IF, IP, stime, sfreq, spower = \
            load_EMIC_IMFs_and_dynspec(plot_start, plot_end)
    
    # Specify color values for time
    time0  = time[ 0].astype(np.int64)
    time1  = time[-1].astype(np.int64)
    norm   = mpl.colors.LogNorm(vmin=time0, vmax=time1, clip=False)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
        
    lpad = 20
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8.27, 11.69),
                                 gridspec_kw={'width_ratios':[1, 0.01],
                                              'height_ratios':[1, 1, 0.5, 2]
                                              })
    
    # Spectra/IP
    im0 = axes[0, 0].pcolormesh(stime, sfreq, spower.sum(axis=0).T, cmap='jet',
                         norm=colors.LogNorm(vmin=1e-4, vmax=1e1))
    axes[0, 0].set_ylim(0, fmax)
    axes[0, 0].set_ylabel('$f$\n(Hz)', rotation=0, labelpad=lpad, fontsize=12)
    fig.colorbar(im0, cax=axes[0, 1], extend='both').set_label(
                r'$\frac{nT^2}{Hz}$', fontsize=16, rotation=0, labelpad=5)
    
    axes[0, 0].plot(imf_time, IF[0][:, 0], c='k', lw=0.75)
    axes[0, 0].plot(imf_time, IF[1][:, 0], c='k', lw=0.75, alpha=0.8)
    #axes[0, 0].plot(imf_time, IF[2][:, 0], c='k', lw=0.75, alpha=0.6)
    #axes[0, 0].axvline(this_time, color='white', ls='-' , alpha=0.7)
    axes[0, 0].set_xlim(plot_start, plot_end)
    axes[0, 0].set_xticklabels([])
    
    axes[0, 0].axhline(_band_start  , color='white', ls='--')
    axes[0, 0].axhline(_band_end    , color='white', ls='--')
    
    # mag_time, pc1_mags, IA, IF, IP, stime, sfreq, spower
    # Timeseries for comparison
    axes[1, 0].plot(mag_time, pc1_mags[:, 0], c='b', label='$\delta B_\\nu$')
    axes[1, 0].plot(mag_time, pc1_mags[:, 1], c='r', label='$\delta B_\phi$', alpha=0.5)
    #axes[1, 0].plot(mag_time, pc1_mags[:, 2], c='k', label='$\delta B_\mu$', alpha=0.25)
    axes[1, 0].set_ylabel('nT', rotation=0, labelpad=lpad)
    axes[1, 0].set_xlim(plot_start, plot_end)
    axes[1, 0].set_xlabel('Time [UT]')
    axes[1, 0].set_xlim(plot_start, plot_end)
    
    # CALCULATE PLASMA PARAMETERS AND AMPLITUDES FOR ALL TIMES
    # Plot all on same graph, use colorbar to discern time
    # Maybe do for cutoff times/packet times
    for ii in range(0, time.shape[0], 4):
        this_time = time[ii]
        clr = mapper.to_rgba(time[ii].astype(np.int64))
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
        
        # Frequencies to evaluate, calculate wavenumber (cold approximation)
        f_min  = 0.07*PP['pcyc_rad'] / (2*np.pi)
        f_max  = 0.24*PP['pcyc_rad'] / (2*np.pi)
        Nf     = 10000
        f_vals = np.linspace(f_min, f_max, Nf)
        w_vals = 2*np.pi*f_vals
        k_vals = nls.get_k_cold(w_vals, Species)
        
        # Define hot proton parameters (velocities normalized c)
        # Remember: temperatures originally in eV
        nh = hope_dens[0][ii]
        wph2 = nh * PCHARGE ** 2 / (PMASS * EPS0) 
        Vth_para = np.sqrt(KB * hope_tpar[0][ii]*(PCHARGE/KB)  / PMASS) / SPLIGHT
        Vth_perp = np.sqrt(KB * hope_tperp[0][ii]*(PCHARGE/KB) / PMASS) / SPLIGHT
        Q = 0.5
        
        # Curvature parameters (this has the most wiggle room)
        L  = 4.7#L_vals[ii]
        a  = 4.5 / (L*RE)**2
        a  = a*(SPLIGHT**2/PP['pcyc_rad']**2)
        
        Vg, Vp, Vr = nls.get_velocities(w_vals, Species, PP, normalize=True)
        s0, s1, s2 = nls.get_inhomogeneity_terms(w_vals, Species, PP, Vth_perp, normalize_vel=True)
        
        # Normalize input parameters
        wph = np.sqrt(wph2) / PP['pcyc_rad']
        w   = w_vals / PP['pcyc_rad']
        
        # DO THE ACTUAL CALCULATION (All hands off from here, using existing code/proforma)
        tau   = 1.00
        B_th  = nls.get_threshold_amplitude(w, wph, Q, s2, a, Vp, Vr, Vth_para, Vth_perp)
        B_opt = nls.get_optimum_amplitude(w, wph, Q, tau, s0, s1, Vg, Vr, Vth_para, Vth_perp)
        T_tr  = nls.get_nonlinear_trapping_period(k_vals, Vth_perp*SPLIGHT, B_opt*PP['B0'])
        T_N   = tau*T_tr*PP['pcyc_rad']
        
        # Filter zeros and infs:
        B_th[B_th == np.inf] = np.nan
        B_th[B_th == 0]      = np.nan
        
        B_opt[B_opt == np.inf] = np.nan
        B_opt[B_opt == 0]      = np.nan
        
        T_N[T_N == np.inf] = np.nan
        T_N[T_N == 0]      = np.nan
        
        ################
        ### PLOTTING ###
        ################
        # Bth, Bopt, Inst. Amplitudes
        axes[3, 0].plot(f_vals, B_th*B0*1e9,  c=clr, ls='--', label=r'$B_{th}$')
        axes[3, 0].plot(f_vals, B_opt*B0*1e9, c=clr, ls='-' , label=r'$B_{opt}$')
        
    axes[3, 0].set_ylabel('$B$ [nT]', rotation=0, labelpad=20, fontsize=16)
    axes[3, 0].set_xlabel('$f$ [Hz]', fontsize=16)
    axes[3, 0].set_ylim(0, 17)
    axes[3, 0].set_xlim(f_vals[0], f_vals[-1])
    axes[3, 0].tick_params(top=True, right=True)
    add_custom_legend(axes[3, 0], [r'$B_{th}$', r'$B_{opt}$'],
                                  ['--', '-'],
                                  [1.0, 1.0], 
                                  ['k', 'k'])

    label_every = 15
    cbar    = fig.colorbar(mapper, cax=axes[3, 1], label='Time', orientation='vertical',
                               ticks=time[::label_every].astype(np.int64))
    for label in cbar.ax.get_yminorticklabels():
        label.set_visible(False)

    cbar.ax.set_yticklabels(time[::label_every].astype('datetime64[m]'))
    
    axes[1, 1].set_visible(False)
    axes[2, 0].set_visible(False)
    axes[2, 1].set_visible(False)
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0)
    fig.align_ylabels()
    
    plt.show()
    return


#%% MAIN
if __name__ == '__main__':
    _rbsp_path  = 'F://DATA//RBSP//'
    _crres_path = 'F://DATA//CRRES//'
    _plot_path  = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//REVISION_PLOTS//' 
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
    
        cases = [0]
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
    
        cases = [4, 5]
        
    time_start = _time_start
    time_end   = _time_end
    probe      = 'a'
    pad        = 0
    
    plot_start = time_start - np.timedelta64(int(pad), 's')
    plot_end   = time_end   + np.timedelta64(int(pad), 's')

    
    #%% Non-linear trace plots
    if True:
        for case in cases:
            if case == 0:
                # Section 1 (Whole)
                parameter_time = np.datetime64('2013-07-25T21:29:40')
                packet_start   = np.datetime64('2013-07-25T21:27:30')
                packet_end     = np.datetime64('2013-07-25T21:32:15')
                cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20130725_RBSP-A//cutoffs_only.txt'

                _band_start = 0.21
                _band_end   = 0.43
            elif case == 1:
                # Single Packet near  end
                parameter_time = np.datetime64('2013-07-25T21:42:12')
                packet_start   = np.datetime64('2013-07-25T21:42:12')
                #packet_end     = np.datetime64('2013-07-25T21:42:45')
                packet_end     = np.datetime64('2013-07-25T21:43:00')
                cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20130725_RBSP-A//cutoffs_only.txt'
                
                _band_start = 0.43
                _band_end   = 0.76
                
            elif case == 2:
                # Single Packet
                parameter_time = np.datetime64('2015-01-16T04:32:04')
                packet_start   = np.datetime64('2015-01-16T04:27:00')
                packet_end     = np.datetime64('2015-01-16T05:08:30')
                cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20150116_RBSP-A//cutoffs_only.txt'
                
                _band_start = 0.12
                _band_end   = 0.35
                
            elif case == 3:
                # Single Packet 1
                packet_start   = np.datetime64('2015-01-16T04:32:17')
                packet_end     = np.datetime64('2015-01-16T04:33:31')
                parameter_time = np.datetime64('2015-01-16T04:32:34')
                cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20150116_RBSP-A//cutoffs_only.txt'
                
                _band_start = 0.12
                _band_end   = 0.18
                
            elif case == 4:
                # Single Packet 2
                packet_start   = np.datetime64('2015-01-16T04:35:03')
                packet_end     = np.datetime64('2015-01-16T04:36:01')
                parameter_time = np.datetime64('2015-01-16T04:35:18')
                cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20150116_RBSP-A//cutoffs_only.txt'
                
                _band_start = 0.12
                _band_end   = 0.18
                
            elif case == 5:
                # Single Packet 2
                packet_start   = np.datetime64('2015-01-16T04:47:24')
                packet_end     = np.datetime64('2015-01-16T04:48:50')
                parameter_time = np.datetime64('2015-01-16T04:47:31')
                cutoff_filename = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Data_Plots//20150116_RBSP-A//cutoffs_only.txt'
                
                _band_start = 0.12
                _band_end   = 0.18
                
                
            # Import cutoff-derived composition information
            cutoff_dict = epd.read_cutoff_file(cutoff_filename)
            
            time, mag, edens, cold_dens, hope_dens, hope_tpar, hope_tperp, hope_anis, L_vals, =\
                                               load_and_interpolate_plasma_params(
                                               plot_start, plot_end, probe, nsec=None, 
                                               rbsp_path=_rbsp_path, HM_filter_mhz=50.0,
                                               time_array=None, check_interp=False)
                
            time_idx       = np.where(abs(time - parameter_time) == np.min(abs(time - parameter_time)))[0][0]
            
            # Get oxygen concentration from cutoffs
            cutoff  = np.interp(parameter_time.astype(np.int64),
                                cutoff_dict['CUTOFF_TIME'].astype(np.int64),
                                cutoff_dict['CUTOFF_NORM'])
            o_frac = epd.calculate_o_from_he_and_cutoff(cutoff, he_frac)
            h_frac = 1. - he_frac - o_frac
            
            # Cold plasma params, SI units
            B0      = mag[time_idx]
            name    = np.array(['H'   , 'He'   , 'O'    ])
            mass    = np.array([1.0   , 4.0    , 16.0   ]) * PMASS
            charge  = np.array([1.0   , 1.0    , 1.0    ]) * PCHARGE
            density = np.array([h_frac, he_frac, o_frac ]) * edens[time_idx]
            ani     = np.array([0.0   , 0.0    , 0.0    ])
            tpar    = np.array([0.0   , 0.0    , 0.0    ])
            tper    = (ani + 1) * tpar
            Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
            
            # Frequencies to evaluate, calculate wavenumber (cold approximation)
            f_min  = 0.07*PP['pcyc_rad'] / (2*np.pi)
            f_max  = 0.24*PP['pcyc_rad'] / (2*np.pi)
            Nf     = 10000
            f_vals = np.linspace(f_min, f_max, Nf)
            w_vals = 2*np.pi*f_vals
            k_vals = nls.get_k_cold(w_vals, Species)
            
            # Define hot proton parameters (velocities normalized c) : vth = sqrt(kT/m)?
            # Remember: temperatures originally in eV
            nh = hope_dens[0][time_idx]
            wph2 = nh * PCHARGE ** 2 / (PMASS * EPS0) 
            Vth_para = np.sqrt(KB * hope_tpar[0][time_idx]*(PCHARGE/KB)  / PMASS) / SPLIGHT
            Vth_perp = np.sqrt(KB * hope_tperp[0][time_idx]*(PCHARGE/KB) / PMASS) / SPLIGHT
            Q = 0.5
            
            # Curvature parameters (this has the most wiggle room)
            a  = 4.5 / (L*RE)**2
            a  = a*(SPLIGHT**2/PP['pcyc_rad']**2)
            
            Vg, Vp, Vr = nls.get_velocities(w_vals, Species, PP, normalize=True)
            s0, s1, s2 = nls.get_inhomogeneity_terms(w_vals, Species, PP, Vth_perp, normalize_vel=True)
            
            # Normalize input parameters
            wph = np.sqrt(wph2) / PP['pcyc_rad']
            w   = w_vals / PP['pcyc_rad']
            
            # DO THE ACTUAL CALCULATION (All hands off from here, using existing code/proforma)
            tau    = 1.00
            B_th   = nls.get_threshold_amplitude(w, wph, Q, s2, a, Vp, Vr, Vth_para, Vth_perp)
            
            # Filter zeros and infs:
            B_th[B_th == np.inf] = np.nan
            B_th[B_th == 0]      = np.nan
            
            # Load EMIC data IMF values
            mag_time, pc1_mags, HM_mags, imf_time, IA, IF, IP, stime, sfreq, spower, gyfreqs = \
                load_EMIC_IMFs_and_dynspec(packet_start, packet_end)
            
            #%% PLOT: NaN's and inf's in arrays: How to filter to plot? Set to all NaN's
            #       Also semilogy doesn't like zero: Set to NaN
            #
            # Plot axes:
            # -- Dynamic spectra for context, /w label for parameter time and packet times. Maybe IF?
            # -- Plot IMF just for check? 
            # -- Large bottom plot: IF vs. IA
            # -- Maybe try a plot for linear/nonlinear growth rates vs. frequency?
            # -- Also should look at velocities/energies so we can see what sort of values are resonant
            #       (Do this separately)
            
            # Filter IF/IA if outside bandpassed frequencies
            if True:
                for ii in range(IF[0].shape[0]):
                    for jj in [0, 1]:
                        if IF[jj][ii, 0] > _band_end or IF[jj][ii, 0] < _band_start:
                            IF[jj][ii, 0] = np.nan
                            
            # Maybe add extra filter for amplitude lower than a certain fraction (e.g. <0.1 nT)
            plt.ioff()
            lpad = 20; fsize=12
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8.00, 0.5*11.00),
                                         gridspec_kw={'width_ratios':[1, 0.01],
                                                      'height_ratios':[1, 0.3, 2]
                                                      })
            
            # Spectra/IP
            im0 = axes[0, 0].pcolormesh(stime, sfreq, spower.sum(axis=0).T, cmap='jet',
                                 norm=colors.LogNorm(vmin=1e-4, vmax=1e1))
            axes[0, 0].set_ylim(0, fmax)
            axes[0, 0].set_ylabel('$f$\n(Hz)', rotation=0, labelpad=lpad, fontsize=fsize)
            fig.colorbar(im0, cax=axes[0, 1], extend='both').set_label(
                        r'$\frac{nT^2}{Hz}$', fontsize=fsize+2, rotation=0, labelpad=20)
                        
            axes[0, 0].plot(mag_time, gyfreqs[1], c='yellow', label='$f_{cHe^+}$')
            axes[0, 0].plot(mag_time, gyfreqs[2], c='r',      label='$f_{cO^+}$')
            axes[0, 0].legend(loc='upper right')
            
            axes[0, 0].plot(imf_time, IF[0][:, 0], c='k', lw=0.75)
            axes[0, 0].plot(imf_time, IF[1][:, 0], c='k', lw=0.75, alpha=0.8)
            #axes[0, 0].plot(imf_time, IF[2][:, 0], c='k', lw=0.75, alpha=0.6)
            axes[0, 0].axvline(parameter_time, color='white', ls='-' , alpha=0.7)
            axes[0, 0].set_xlim(plot_start, plot_end)
            
            axes[0, 0].axhline(_band_start, color='white', ls='--')
            axes[0, 0].axhline(_band_end  , color='white', ls='--')
    
            # Bth, Bopt, Inst. Amplitudes
            axes[2, 0].plot(f_vals, B_th*B0*1e9, c='k', ls='--', label=r'$B_{th}$')
            
            for tau in [0.25, 0.5, 1.0, 2.0]:
                B_opt  = nls.get_optimum_amplitude(w, wph, Q, tau, s0, s1, Vg, Vr, Vth_para, Vth_perp)
                
                B_opt[B_opt == np.inf] = np.nan
                B_opt[B_opt == 0]      = np.nan
                
                tau_lbl = r'$B_{opt}$' if tau==0.25 else None
                
                axes[2, 0].plot(f_vals, B_opt*B0*1e9, c='k', ls='-', label=tau_lbl)
                
                xv = 0.50
                yi = np.where(abs(f_vals - xv) == abs(f_vals - xv).min())[0][0]
                yv = B_opt[yi]*B0*1e9
                
                axes[2, 0].text(xv, yv, f'{tau:.2f}', ha='center', bbox={'facecolor':'white', 'alpha':1.0, 'edgecolor':'white'})
                
            axes[2, 0].set_ylabel('$B$\n(nT)', rotation=0, labelpad=20, fontsize=fsize)
            axes[2, 0].set_xlabel('$f$ (Hz)]', fontsize=fsize)
            axes[2, 0].set_ylim(0, B_max)
            axes[2, 0].set_xlim(f_vals[0], f_vals[-1])
            axes[2, 0].tick_params(top=True, right=True)
               
            m_size = 1
            axes[2, 0].scatter(IF[0][:, 0], IA[0][:, 0], c='b', s=m_size, marker='.', label='$B_\\nu$')
            axes[2, 0].scatter(IF[1][:, 0], IA[1][:, 0], c='r', s=m_size, marker='.', label='$B_\phi$')
            #axes[3, 0].scatter(IF[2][:, 0], IA[2][:, 0], c='k', s=m_size)
            
            # Start/Stop
            axes[2, 0].scatter(IF[0][0, 0], IA[0][0, 0], c='b', s=40, marker='o')
            axes[2, 0].scatter(IF[1][0, 0], IA[1][0, 0], c='r', s=40, marker='o')    
            axes[2, 0].scatter(IF[0][-1, 0], IA[0][-1, 0], c='b', s=40, marker='x')
            axes[2, 0].scatter(IF[1][-1, 0], IA[1][-1, 0], c='r', s=40, marker='x')  
            axes[2, 0].legend(loc='upper right')
            
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
            axes[2, 1].set_visible(False)
            axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.05, hspace=0)
            fig.align_ylabels()
            
            if save_plot == True:
                save_string = parameter_time.astype(object).strftime('%Y%m%d_%H%M%S')
                print('Saving plot...')
                fig.savefig(_plot_path + 'NONLINEAR_TRACE_' + save_string + '.png', dpi=dpi)
                plt.close('all')
            else:
                plt.show() 