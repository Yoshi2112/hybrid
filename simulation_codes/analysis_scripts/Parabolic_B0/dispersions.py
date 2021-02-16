# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:14:56 2019

@author: Yoshi
"""
import pdb
import analysis_config as cf
import analysis_backend as bk
import numpy as np
import matplotlib.pyplot as plt
import os, sys

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


def get_linear_dispersion_from_sim(k, zero_cold=True, Nk=1000):
    '''
    Extracted values units :: 
        Density    -- /m3       (Cold, warm densities)
        Anisotropy -- Number
        Tper       -- eV
    '''
    print('Calculating linear dispersion relations...')
    from multiapprox_dispersion_solver  import get_dispersion_relation, create_species_array

    kB = 1.38065e-23     # Boltzmann's Constant (J/K)
    
    # Extract species parameters from run, create Species array (Could simplify T calculation when I'm less lazy)
    t_par      = (cf.mass * cf.vth_par  ** 2 / kB) / 11603.
    t_perp     = (cf.mass * cf.vth_perp ** 2 / kB) / 11603.
    anisotropy = t_perp / t_par - 1
    
    if zero_cold == True:
        for ii in range(t_perp.shape[0]):
            if cf.temp_type[ii] == 0:
                t_perp[ii]     = 0.0
                anisotropy[ii] = 0.0
    
    Species, PP = create_species_array(cf.B_eq, cf.species_lbl, cf.mass, cf.charge,
                                       cf.density, t_perp, anisotropy)
    
    # Convert from linear units to angular units for k range to solve over
    kmin   = 2*np.pi*k[0]
    kmax   = 2*np.pi*k[-1]
    k_vals = np.linspace(kmin, kmax, Nk) 
    
    # Calculate dispersion relations (3 approximations)
    CPDR_solns,  cold_CGR = get_dispersion_relation(Species, k_vals, approx='cold')
    WPDR_solns,  warm_CGR = get_dispersion_relation(Species, k_vals, approx='warm')
    HPDR_solns,  hot_CGR  = get_dispersion_relation(Species, k_vals, approx='hot')

    # Convert back from angular units to linear units
    k_vals     /= 2*np.pi
    CPDR_solns /= 2*np.pi
    WPDR_solns /= 2*np.pi
    HPDR_solns /= 2*np.pi

    return k_vals, CPDR_solns, WPDR_solns, HPDR_solns


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
                  
    k   = k[k>=0]
    
    fft_matrix  = np.zeros((arr.shape[0], en-st), dtype='complex128')
    for ii in range(arr.shape[0]): # Take spatial FFT at each time, ii
        fft_matrix[ii, :] = np.fft.fft(arr[ii, st:en] - arr[ii, st:en].mean())

    kt = (fft_matrix[:, :k.shape[0]] * np.conj(fft_matrix[:, :k.shape[0]])).real
    
    return k, ftime, kt, st, en


def plot_kt_winske(component='by'):
    qi     = 1.602e-19       # Elementary charge (C)
    c      = 3e8             # Speed of light (m/s)
    mp     = 1.67e-27        # Mass of proton (kg)
    e0     = 8.854e-12       # Epsilon naught - permittivity of free space
    
    ftime, arr = cf.get_array(component)
    
    radperiods = ftime * cf.gyfreq
    gperiods   = ftime / cf.gyperiod
    
    ts_folder = cf.anal_dir + '//winske_fourier_modes//'
    if os.path.exists(ts_folder) == False:
        os.makedirs(ts_folder)
    
    # Get first/last indices for FFT range and k-space array
    st = cf.ND
    if component[0].upper() == 'B':
        en = cf.ND + cf.NX
        k  = np.fft.fftfreq(cf.NX, cf.dx)
    else:
        en = cf.ND + cf.NX
        k  = np.fft.fftfreq(cf.NX, cf.dx)
    
    # Normalize to c/wpi
    cwpi = c/np.sqrt(cf.ne * qi ** 2 / (mp * e0))
    
    k   *= cwpi
    k    = k[k>=0]
    kmax = k.shape[0]
    
    fft_matrix  = np.zeros((arr.shape[0], en-st), dtype='complex128')
    for ii in range(arr.shape[0]): # Take spatial FFT at each time, ii
        fft_matrix[ii, :] = np.fft.fft(arr[ii, st:en] - arr[ii, st:en].mean())

    kt = (fft_matrix[:, :k.shape[0]] * np.conj(fft_matrix[:, :k.shape[0]])).real
    
    plt.ioff()

    for ii in range(ftime.shape[0]):
        fig, ax = plt.subplots()
        ax.semilogy(k[1:kmax], kt[ii, 1:kmax], ds='steps-mid')
        ax.set_title('IT={:04d} :: T={:5.2f} :: GP={:5.2f}'.format(ii, radperiods[ii], gperiods[ii]), family='monospace')
        ax.set_xlabel('K')
        ax.set_ylabel('BYK**2')
        ax.set_xlim(k[1], k[kmax-1])
        fig.savefig(ts_folder + 'winske_fourier_{}_{}.png'.format(component, ii), edgecolor='none')
        plt.close('all') 
        
        sys.stdout.write('\rCreating fourier mode plot for timestep {}'.format(ii))
        sys.stdout.flush()

    print('\n')
    return


def plot_fourier_mode_timeseries(it_max=None):
    '''
    Load helical components Bt pos/neg, extract By_pos
    For each snapshot in time, take spatial FFT of By_pos (similar to how helicity is done)
    Save these snapshots in array
    Plot single mode timeseries from this 2D array
    
    Test run: Seems relatively close qualitatively, with a smaller growth rate
                and a few of the modes not quite as large. This could be any 
                number of reasons - from the simulation method to the helicity.
                Will be interesting to compare direct to linear theory via the
                method outlined in Munoz et al. (2018).
    '''
    ftime, By_raw  = cf.get_array('By')
    ftime, Bz_raw  = cf.get_array('Bz')
    radperiods     = ftime * cf.gyfreq
    
    if it_max is None:
        it_max = ftime.shape[0]
    
    ftime, Bt_pos, Bt_neg = bk.get_helical_components()

    By_pos  = Bt_pos.real
    x       = np.linspace(0, cf.NX*cf.dx, cf.NX)
    k_modes = np.fft.rfftfreq(x.shape[0], d=cf.dx)
    Byk_2   = np.zeros((ftime.shape[0], k_modes.shape[0]), dtype=np.float64) 
    
    # Do time loop here. Could also put normalization flag
    for ii in range(ftime.shape[0]):
        Byk          = (1 / k_modes.shape[0]) * np.fft.rfft(By_pos[ii])
        Byk_2[ii, :] = (Byk * np.conj(Byk)).real / cf.B_eq ** 2

    plt.ioff()
    fig, axes = plt.subplots(ncols=2, nrows=3, sharex=True, figsize=(15, 10))
    
    axes[0, 0].semilogy(radperiods, Byk_2[:, 1])
    axes[0, 0].set_title('m = 1')
    axes[0, 0].set_xlim(0, 100)
    axes[0, 0].set_ylim(1e-7, 1e-3) 
    
    axes[1, 0].semilogy(radperiods, Byk_2[:, 2])
    axes[1, 0].set_title('m = 2')
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].set_ylim(1e-6, 1e-1) 
    
    axes[2, 0].semilogy(radperiods, Byk_2[:, 3])
    axes[2, 0].set_title('m = 3')
    axes[2, 0].set_xlim(0, 100)
    axes[2, 0].set_ylim(1e-6, 1e-1) 
    
    axes[0, 1].semilogy(radperiods, Byk_2[:, 4])
    axes[0, 1].set_title('m = 4')
    axes[0, 1].set_xlim(0, 100)
    axes[0, 1].set_ylim(1e-6, 1e-0) 
    
    axes[1, 1].semilogy(radperiods, Byk_2[:, 5])
    axes[1, 1].set_title('m = 5')
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_ylim(1e-6, 1e-0) 
    
    axes[2, 1].semilogy(radperiods, Byk_2[:, 6])
    axes[2, 1].set_title('m = 6')
    axes[2, 1].set_xlim(0, 100)
    axes[2, 1].set_ylim(1e-6, 1e-0) 
    
    fig.savefig(cf.anal_dir + 'fourier_modes.png')
    plt.close('all')
    
    #axes[ii].set_xlim(0, 100)
    #
    #axes[ii].set_xlabel('$\Omega_i t$')
    return


def get_wk(component, linear_only=True):
    '''
    Spatial boundaries start at index
    '''
    ftime, arr = cf.get_array(component)
    
    t0 = 0
    if linear_only == True:
        # Find the point where magnetic energy peaks, FFT only up to 120% of that index
        yftime, by = cf.get_array('By')
        zftime, bz = cf.get_array('Bz')
        bt         = np.sqrt(by ** 2 + bz ** 2)
        mu0        = (4e-7) * np.pi
        U_B        = 0.5 / mu0 * np.square(bt[:, :]).sum(axis=1) * cf.dx
        max_idx    = np.argmax(U_B)
        t1         = int(1.2*max_idx)
        tf         = ftime[t1]
    else:
        t1 = ftime.shape[0]
        tf = ftime[t1-1]
        
    if component.upper()[0] == 'B':
        st = cf.ND; en = cf.ND + cf.NX + 1
    else:
        st = cf.ND; en = cf.ND + cf.NX
    
    num_times = t1-t0

    df = 1. / (num_times * cf.dt_field)
    dk = 1. / (cf.NX     * cf.dx)

    f  = np.arange(0, 1. / (2*cf.dt_field), df)
    k  = np.arange(0, 1. / (2*cf.dx), dk)
    
    fft_matrix  = np.zeros(arr[t0:t1, st:en].shape, dtype='complex128')
    fft_matrix2 = np.zeros(arr[t0:t1, st:en].shape, dtype='complex128')

    # Take spatial FFT at each time
    for ii in range(fft_matrix.shape[0]): 
        fft_matrix[ii, :] = np.fft.fft(arr[t0 + ii, st:en] - arr[t0 + ii, st:en].mean())

    # Take temporal FFT at each position (now k)
    for ii in range(fft_matrix.shape[1]):
        fft_matrix2[:, ii] = np.fft.fft(fft_matrix[:, ii] - fft_matrix[:, ii].mean())

    wk = fft_matrix2[:f.shape[0], :k.shape[0]] * np.conj(fft_matrix2[:f.shape[0], :k.shape[0]])
    return k, f, wk, tf

    
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
