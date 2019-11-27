# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:14:56 2019

@author: Yoshi
"""
import analysis_config as cf
import numpy as np
import sys
import pyfftw
from scipy import signal
data_scripts_dir = 'C://Users//iarey//Documents//GitHub//hybrid//linear_theory//'
sys.path.append(data_scripts_dir)


def get_cgr_from_sim(norm_flag=0):
    from convective_growth_rate import calculate_growth_rate
    from analysis_config import species_lbl, density, temp_type, Tper, Tpar, Nj, B0
    
    cold_density = np.zeros(3)
    warm_density = np.zeros(3)
    cgr_ani      = np.zeros(3)
    tempperp     = np.zeros(3)
    anisotropies = Tper / Tpar - 1
    
    for ii in range(Nj):
        if temp_type[ii] == 0:
            if 'H^+'    in species_lbl[ii]:
                cold_density[0] = density[ii] / 1e6
            elif 'He^+' in species_lbl[ii]:
                cold_density[1] = density[ii] / 1e6
            elif 'O^+'  in species_lbl[ii]:
                cold_density[2] = density[ii] / 1e6
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')
                
        if temp_type[ii] == 1:
            if 'H^+'    in species_lbl[ii]:
                warm_density[0] = density[ii] / 1e6
                cgr_ani[0]      = anisotropies[ii]
                tempperp[0]     = Tper[ii] / 11603.
            elif 'He^+' in species_lbl[ii]:
                warm_density[1] = density[ii] / 1e6
                cgr_ani[1]      = anisotropies[ii]
                tempperp[1]     = Tper[ii] / 11603.
            elif 'O^+'  in species_lbl[ii]:
                warm_density[2] = density[ii] / 1e6
                cgr_ani[2]      = anisotropies[ii]
                tempperp[2]     = Tper[ii] / 11603.
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')
    
    freqs, cgr, stop = calculate_growth_rate(B0*1e9, cold_density, warm_density, cgr_ani, temperp=tempperp, norm_freq=norm_flag)
    return freqs, cgr, stop


def get_linear_dispersion_from_sim(k=None, plot=False, save=False):
    '''
    Still not sure how this will work for a H+, O+ mix, but H+-He+ should be fine
    '''
    from chen_warm_dispersion   import get_dispersion_relation
    
    from analysis_config import species_present, NX, dx, Tper, Tpar, temp_type,\
                                species_lbl, density, Nj, B0, anal_dir
                   
    if k is None:
        k         = np.fft.fftfreq(NX, dx)
        k         = k[k>=0]
    
    N_present    = species_present.count(True)
    cold_density = np.zeros(N_present)
    warm_density = np.zeros(N_present)
    cgr_ani      = np.zeros(N_present)
    tempperp     = np.zeros(N_present)
    anisotropies = Tper / Tpar - 1
    
    for ii in range(Nj):
        if temp_type[ii] == 0:
            if 'H^+'    in species_lbl[ii]:
                cold_density[0] = density[ii]
            elif 'He^+' in species_lbl[ii]:
                cold_density[1] = density[ii]
            elif 'O^+'  in species_lbl[ii]:
                cold_density[2] = density[ii]
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')
                
        if temp_type[ii] == 1:
            if 'H^+'    in species_lbl[ii]:
                warm_density[0] = density[ii]
                cgr_ani[0]      = anisotropies[ii]
                tempperp[0]     = Tper[ii] / 11603.
            elif 'He^+' in species_lbl[ii]:
                warm_density[1] = density[ii]
                cgr_ani[1]      = anisotropies[ii]
                tempperp[1]     = Tper[ii] / 11603
            elif 'O^+'  in species_lbl[ii]:
                warm_density[2] = density[ii]
                cgr_ani[2]      = anisotropies[ii]
                tempperp[2]     = Tper[ii] / 11603
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')

    if save == True:
        savepath = anal_dir
    else:
        savepath = None
    
    k_vals, CPDR_solns, warm_solns = get_dispersion_relation(B0, cold_density, warm_density, cgr_ani, tempperp,
               norm_k=False, norm_w=False, kmin=k[0], kmax=k[-1], k_input_norm=0, plot=plot, save=save, savepath=savepath)

    return k_vals, CPDR_solns, warm_solns


def get_wx(component):
    arr = cf.get_array(component)
    
    
    if arr.shape[0]%2 == 0:
        fft_matrix  = np.zeros((arr.shape[0]//2+1, cf.NX), dtype='complex128')
    else:
        fft_matrix  = np.zeros(((arr.shape[0]+1)//2, cf.NX), dtype='complex128')
        
    for ii in range(arr.shape[1]):
        fft_matrix[:, ii] = np.fft.rfft(arr[:, ii] - arr[:, ii].mean())
    
    wx = (fft_matrix * np.conj(fft_matrix)).real
    return wx


def get_kt(component):
    arr = cf.get_array(component)
    
    fft_matrix  = np.zeros(arr.shape, dtype='complex128')
    for ii in range(arr.shape[0]): # Take spatial FFT at each time
        fft_matrix[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())

    kt = (fft_matrix[:, :arr.shape[1] // 2] * np.conj(fft_matrix[:, :arr.shape[1] // 2])).real
    return kt


def get_wk(component):
    arr = cf.get_array(component)
    
    num_times = arr.shape[0]

    df = 1. / (num_times * cf.dt_field)
    dk = 1. / (cf.NX * cf.dx)

    f  = np.arange(0, 1. / (2*cf.dt_field), df)
    k  = np.arange(0, 1. / (2*cf.dx), dk)
    
    fft_matrix  = np.zeros(arr.shape, dtype='complex128')
    fft_matrix2 = np.zeros(arr.shape, dtype='complex128')

    for ii in range(arr.shape[0]): # Take spatial FFT at each time
        fft_matrix[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())

    for ii in range(arr.shape[1]):
        fft_matrix2[:, ii] = np.fft.fft(fft_matrix[:, ii] - fft_matrix[:, ii].mean())

    wk = fft_matrix2[:f.shape[0], :k.shape[0]] * np.conj(fft_matrix2[:f.shape[0], :k.shape[0]])
    return k, f, wk


import pdb
def get_PSD_scipy(component, overlap=0.99, win_idx=None, slide_idx=None, df=50):
    arr = cf.get_array(component)
    
    if win_idx is None:
        t_win   = 1000. / df                                                        # Window length (in seconds) for desired frequency resolution
        win_idx = int(np.ceil(t_win / cf.dt_field))                                 # Window size in indexes
        
    if win_idx%2 == 0:
        win_idx += 1                                                                # Force window length to be odd number (for window-halving in FFT: Center values are always center values)

    if slide_idx is None:
        if overlap > 100:
            overlap /= 100.                                                         # Check for accidental percentage instead of decimal overlap
        slide_idx    = int(win_idx * (1. - overlap))                                # Calculate slide length in index values from overlap percentage
    
    fs = 1 / cf.dt_field
    #pdb.set_trace()
    freqs, times, powers = signal.spectrogram(arr, fs, window=('tukey', 1.0), 
                                              nperseg=win_idx, noverlap=int(overlap*win_idx),
                                              detrend='constant')
    
    return freqs, times, powers

# =============================================================================
# def get_wt_dynamic(component, win_):
#     arr = cf.get_array(component)
#     
#     num_times = arr.shape[0]
# 
#     df = 1. / (num_times * cf.dt_field)
#     dk = 1. / (cf.NX * cf.dx)
# 
#     f  = np.arange(0, 1. / (2*cf.dt_field), df)
#     k  = np.arange(0, 1. / (2*cf.dx), dk)
#     
#     fft_matrix  = np.zeros(arr.shape, dtype='complex128')
#     fft_matrix2 = np.zeros(arr.shape, dtype='complex128')
# 
#     for ii in range(arr.shape[0]): # Take spatial FFT at each time
#         fft_matrix[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())
# 
#     for ii in range(arr.shape[1]):
#         fft_matrix2[:, ii] = np.fft.fft(fft_matrix[:, ii] - fft_matrix[:, ii].mean())
# 
#     wk = fft_matrix2[:f.shape[0], :k.shape[0]] * np.conj(fft_matrix2[:f.shape[0], :k.shape[0]])
#     return k, f, wk
# =============================================================================
# =============================================================================
# def do_stft(dat, win_len, slide, num_slides, positive_only=True):
#     if len(dat.shape) > 1:
#         dat = dat.reshape(dat.shape[0])
# 
#     temp_in    = pyfftw.empty_aligned(win_len, dtype='complex128')              # Set aside some memory for input data
#     temp_out   = pyfftw.empty_aligned(win_len, dtype='complex128')              # And the corresponding output
#     fft_object = pyfftw.FFTW(temp_in, temp_out, flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))                                 # Function to call that will perform FFT
#         
#     window  = np.hanning(win_len)                                               # Hanning window
#     out_mem = np.zeros((num_slides, win_len), dtype=np.complex128)              # Allocate empty memory for STFT output
#         
#     for ii in range(num_slides):
#         try:
#             temp_in[:]     = np.multiply(dat[ii*slide: 
#                                              ii*slide + win_len + 1],               # Do we still need the +1?
#                                              window)                                # Apply FFT windowing function to windowed data and store in FFT input memory address
#             out_mem[ii, :] = fft_object()                                           # Call FFT and store result in output array (involves implicit copy)        
#         except Exception as inst:
#             print(type(inst))     # the exception instance
#             print(inst.args)      # arguments stored in .args
#             print(inst)           # __str__ allows args to be printed directly
#             sys.exit('STFT error')
#             
#     out_mem /=  win_len                                                         # Normalize
#     
#     if positive_only == True:
#         return 2. * out_mem[:, :win_len//2 + 1]                                  # Return only positive frequencies
#     else:
#         return out_mem 
# =============================================================================
# =============================================================================
# def autopower_spectra(times, y_arr, time_start, time_end, overlap=0.5, win_idx=None, slide_idx=None, df=0.05):
#     if times.shape[0] != y_arr.shape[0]:
#         raise IndexError('x, y arrays must have the same length.')
#     
#     if win_idx is None:
#         t_win   = 1000. / df                                                        # Window length (in seconds) for desired frequency resolution
#         win_idx = int(np.ceil(t_win / cf.dt_field))                                 # Window size in indexes
#         
#     if win_idx%2 == 0:
#         win_idx += 1                                                                # Force window length to be odd number (for window-halving in FFT: Center values are always center values)
# 
#     if slide_idx is None:
#         if overlap > 100:
#             overlap /= 100.                                                         # Check for accidental percentage instead of decimal overlap
#         slide_idx    = int(win_idx * (1. - overlap))                                # Calculate slide length in index values from overlap percentage
#     
#     idx_length   = y_arr.shape[0]                                              # Number of indexes between start/end times (exclusive of half-windows on either side)
#     
#     num_slides   = idx_length // slide_idx + 1                                      # Count number of slides required. Unless multiple of idx_length, last value will not explicitly be computed
#     
#     FFT_output   = do_stft(y_arr, win_idx, slide_idx, num_slides, idx_start)        # Do dynamic FFT
#     FFT_times    = np.asarray([times[idx_start + ii*slide_idx] for ii in range(num_slides)])    # Collect times for each FFT slice
# 
#     df           = 1./(win_idx * cf.dt_field)                                        # Frequency increment (in mHz)
#     freq         = np.asarray([df * jj for jj in range(win_idx//2 + 1)])             # Frequency array up to Nyquist
#     power        = np.real(FFT_output * np.conj(FFT_output))
#     return power, FFT_times, freq
# =============================================================================
