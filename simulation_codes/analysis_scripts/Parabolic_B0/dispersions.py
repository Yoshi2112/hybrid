# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:14:56 2019

@author: Yoshi
"""
import pdb
import analysis_config as cf
import numpy as np
import sys

from scipy import signal
data_scripts_dir = 'C://Users//iarey//Documents//GitHub//hybrid//linear_theory//new_general_DR_solver//'
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


def get_linear_dispersion_from_sim(k, plot=False, save=False, zero_cold=True):
    '''
    Still not sure how this will work for a H+, O+ mix, but H+-He+ should be fine
    
    Extracted values units :: 
        Density    -- /m3       (Cold, warm densities)
        Anisotropy -- Number
        Tper       -- eV
    '''
    from dispersion_solver_multispecies import get_dispersion_relations
    from analysis_config                import Tper, Tpar, B_eq
        
    anisotropy = Tper / Tpar - 1
    t_perp     = cf.Tper.copy() / 11603.  
    
    if zero_cold == True:
        for ii in range(t_perp.shape[0]):
            if cf.temp_type[ii] == 0:
                t_perp[ii] = 0.0
    
    # Convert from linear units to angular units for k range to solve over
    kmin = 2*np.pi*k[0]
    kmax = 2*np.pi*k[-1]
    
    k_vals, CPDR_solns, WPDR_solns = get_dispersion_relations(B_eq, cf.species_lbl, cf.mass, cf.charge, \
                                           cf.density, t_perp, anisotropy, norm_k_in=False, norm_w=False,
                                           kmin=kmin, kmax=kmax)
    
    # Convert back from angular units to linear units
    k_vals     /= 2*np.pi
    CPDR_solns /= 2*np.pi
    WPDR_solns /= 2*np.pi

    return k_vals, CPDR_solns, WPDR_solns


def get_wx(component):
    ftime, arr = cf.get_array(component)
    
    if component[0] == 'B':
        ncells = cf.NC + 1
    else:
        ncells = cf.NC
        
    if arr.shape[0]%2 == 0:
        fft_matrix  = np.zeros((arr.shape[0]//2+1, ncells), dtype='complex128')
    else:
        fft_matrix  = np.zeros(((arr.shape[0]+1)//2, ncells), dtype='complex128')
        
    for ii in range(arr.shape[1]):
        fft_matrix[:, ii] = np.fft.rfft(arr[:, ii] - arr[:, ii].mean())
    
    wx = (fft_matrix * np.conj(fft_matrix)).real
    return ftime, wx


def get_kt(component):
    ftime, arr = cf.get_array(component)
    
    # Get first/last indices for FFT range and k-space array
    st = cf.ND
    if component[0].upper() == 'B':
        en = cf.ND + cf.NX + 1
        k  = np.fft.fftfreq(cf.NX + 1, cf.dx)
    else:
        en = cf.ND + cf.NX
        k  = np.fft.fftfreq(cf.NX, cf.dx)
                  
    k   = k[k>=0] * 1e6
    
    fft_matrix  = np.zeros((arr.shape[0], en-st), dtype='complex128')
    for ii in range(arr.shape[0]): # Take spatial FFT at each time, ii
        fft_matrix[ii, :] = np.fft.fft(arr[ii, st:en] - arr[ii, st:en].mean())

    kt = (fft_matrix[:, :k.shape[0]] * np.conj(fft_matrix[:, :k.shape[0]])).real
    
    return k, ftime, kt, st, en


def get_wk(component):
    ftime, arr = cf.get_array(component)
    
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

    
def do_stft(dat, win_len, slide, num_slides):
    '''
    Model version of the STFT that was used for RBSP analysis. Changed so 
    it FFTs the entire run starting from index zero.
    '''
    import pyfftw
    if len(dat.shape) > 1:
        dat = dat.reshape(dat.shape[0])

    temp_in    = pyfftw.empty_aligned(win_len, dtype='complex128')              # Set aside some memory for input data
    temp_out   = pyfftw.empty_aligned(win_len, dtype='complex128')              # And the corresponding output
    fft_object = pyfftw.FFTW(temp_in, temp_out, flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))                                 # Function to call that will perform FFT
        
    window     = np.hanning(win_len)                                            # Hanning window
    out_mem    = np.zeros((num_slides, win_len), dtype=np.complex128)           # Allocate empty memory for STFT output
    
    for ii in range(num_slides):
        try:
            temp_in[:]     = np.multiply(dat[ii*slide: 
                                             ii*slide + win_len],               
                                             window)                            # Apply FFT windowing function to windowed data and store in FFT input memory address
            out_mem[ii, :] = fft_object()                                       # Call FFT and store result in output array (involves implicit copy)        
        except Exception as inst:
            print(type(inst))     # the exception instance
            print(inst.args)      # arguments stored in .args
            print(inst)           # __str__ allows args to be printed directly
            sys.exit('STFT error')
            
    out_mem /=  win_len                                                         # Normalize

    return 2. * out_mem[:, :win_len//2 + 1]                                  # Return only positive frequencies


def autopower_spectra(component='By', overlap=0.5, win_idx=None, slide_idx=None, df=50, cell_idx=None):
    if 'B' in component.upper(): 
        ftime, y_arr = cf.get_array(component)                                      # Field component in nT
        y_arr = y_arr[:, cell_idx]*1e9
    elif 'E' in component.upper():
        ftime, y_arr = cf.get_array(component)                                      # Field component in mV/m
        y_arr = y_arr[:, cell_idx]*1e6
    else:
        sys.exit('Field loading error for kwarg: component={}'.format(component))
   
    dt    = cf.dt_field

    if win_idx is None:
        t_win   = 1000. / df                                                        # Window length (in seconds) for desired frequency resolution
        win_idx = int(np.ceil(t_win / cf.dt_field))                                 # Window size in indexes
        hlen    = (win_idx - 1) // 2                                                # Index of first mid-window, starting from idx 0. After this, displaced by slide_idx.
        
    if win_idx%2 == 0:
        win_idx += 1                                                                # Force window length to be odd number (for window-halving in FFT: Center values are always center values)

    if slide_idx is None:
        if overlap > 100:
            overlap /= 100.                                                         # Check for accidental percentage instead of decimal overlap
        slide_idx = int(win_idx * (1. - overlap))                                   # Calculate slide length in index values from overlap percentage

    num_slides   = (y_arr.shape[0] - win_idx) // slide_idx                          # Count number of slides required. Unless multiple of idx_length, last value will not explicitly be computed

    FFT_output   = do_stft(y_arr, win_idx, slide_idx, num_slides)                   # Do dynamic FFT
    FFT_times    = (np.arange(num_slides) * slide_idx + hlen) * dt                  # Collect times for each FFT slice
    
    df           = 1./(win_idx * cf.dt_field)                                       # Frequency increment (in mHz)
    freq         = np.asarray([df * jj for jj in range(win_idx//2 + 1)])            # Frequency array up to Nyquist
    power        = np.real(FFT_output * np.conj(FFT_output))
    return power, FFT_times, freq
