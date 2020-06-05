# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:14:56 2019

@author: Yoshi
"""
import analysis_config as cf
import numpy as np
import os
import sys
import pdb
import pyfftw

linear_theory_dir = 'C://Users//iarey//Documents//GitHub//hybrid//linear_theory//new_general_DR_solver//'
sys.path.append(linear_theory_dir)


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
    from analysis_config                import Tper, Tpar, B0
        
    anisotropy = Tper / Tpar - 1
    t_perp     = cf.Tper.copy() / 11603.  
    
    if zero_cold == True:
        for ii in range(t_perp.shape[0]):
            if cf.temp_type[ii] == 0:
                t_perp[ii] = 0.0
    
    # Convert from linear units to angular units for k range to solve over
    kmin = 2*np.pi*k[0]
    kmax = 2*np.pi*k[-1]
    
    k_vals, CPDR_solns, WPDR_solns = get_dispersion_relations(B0, cf.species_lbl, cf.mass, cf.charge, \
                                           cf.density, t_perp, anisotropy, norm_k_in=False, norm_w=False,
                                           kmin=kmin, kmax=kmax)
    
    # Convert back from angular units to linear units
    k_vals     /= 2*np.pi
    CPDR_solns /= 2*np.pi
    WPDR_solns /= 2*np.pi

    return k_vals, CPDR_solns, WPDR_solns


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
    dk = 1. / (cf.NX     * cf.dx)
    
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
    if len(dat.shape) > 1:
        dat = dat.reshape(dat.shape[0])

    temp_in    = pyfftw.empty_aligned(win_len, dtype='complex128')              # Set aside some memory for input data
    temp_out   = pyfftw.empty_aligned(win_len, dtype='complex128')              # And the corresponding output
    fft_object = pyfftw.FFTW(temp_in, temp_out, flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))                                 # Function to call that will perform FFT
        
    window     = np.hanning(win_len)                                            # Hanning window
    out_mem    = np.zeros((num_slides, win_len), dtype=np.complex128)           # Allocate empty memory for STFT output
    
    import pdb; pdb.set_trace
    
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
        y_arr = cf.get_array(component)[:, cell_idx]*1e9                                # Field component in nT
    elif 'E' in component.upper():
        y_arr = cf.get_array(component)[:, cell_idx]*1e6                                # Field component in mV/m
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


def create_WHAMP_inputs_from_run(series, run_num):
    '''
    Create input scripts by reading in the simulation and particle parameters
    of a hybrid run. This will serve as the first part of the interface.
    
    k_min should always be zero, k_max will be determined by simulation spacing.
    Is k in /m ? Is fstart in Hz? Or a fraction of p_cyc? Or radians/s? Find out.
    
    k_nyq = 1 / (2*dx)   ---> This is the maximum k to plot up to
    '''
    whamp_IO_folder = cf.anal_dir + '/WHAMP/'
    if os.path.exists(whamp_IO_folder) == False:
        os.makedirs(whamp_IO_folder)

    # Calculable k-range from simulation grid spacing (/m)
    Nk     = 500
    k_min  = 0.0
    k_nyq  = 1 / (2*cf.dx)
    k_step = k_nyq / Nk
        
    # SIMULATION/RUN PARAMTERS
    fstart       = 0.15                   # UNITS?? NORMALIZED? I THINK ITS JUST A FIRST GUESS 
    kpar         = [k_min, k_step, k_nyq] # Start/Step/Stop
    kperp        = 0.0                    # Scalar in this case (just to see)
    magfield     = cf.B0 * 1e9            # Background B0 in nT

    # Code quantities (Probably never need to be changed)
    maxiter      = 50                   # Maximum iterations (for solution convergence?)
    uselog       = 0                    # Use logarithmic spacing for k values (0: No, 1: Yes)
    kzfirst      = 0                    # Vary over kz first (Doesn't matter)
    
    
    # SPECIES PARAMETERS :: CONVERT INTO SOMETHING THAT CAN BE PUT INTO WHAMP INPUT FILE
    mass       = cf.mass.copy() / cf.mp         	            # Species ion mass (proton mass units)
    density    = cf.density.copy() * 1e-6                       # Species density in /cc

    E_tot      = (cf.Tpar + cf.Tper)/11603.                     # Total plasma  energy in eV
    E_e        = cf.Te0 / 11603.                                # Electron energy (eV)
    ne         = density.sum()                                  # Electron density
    
    v_therm    = np.sqrt(cf.kB * (cf.Tpar + cf.Tper) / cf.mass) # Array of v_therm for each species
    new_A      = cf.Tper / cf.Tpar                              # Anisotropy (T_per / T_par) (without the minus one as usual - its a WHAMP thing)
    new_drift  = cf.drift_v.copy() / v_therm                    # Normalized drift velocity
    
    param_file = whamp_IO_folder + 'WHAMP_INPUT_PARAMS.txt'
    with open(param_file, 'w', newline="\n") as f:
        f.write('MATLAB-WHAMP INPUT PARAMTER FILE\n')
        f.write('RUN TITLE {}[{}]\n'.format(series.upper(), run_num))
        f.write('__________________________\n')
        f.write('FSTART\t{:10.5e}\n'.format(fstart))
        f.write('KPAR\t{:10.5e}  {:10.5e}  {:10.5e}\n'.format(kpar[0], kpar[1], kpar[2]))
        f.write('KPERP\t{:10.5e}\n'.format(kperp))
        f.write('MAXITER\t{:d}\n'.format(maxiter))
        f.write('USELOG\t{:d}\n'.format(uselog))
        f.write('KZ1ST\t{:d}\n'.format(kzfirst))
        f.write('MAGFIELD_NT\t{:10.5e}'.format(magfield))
    f.close()
        
    species_file = whamp_IO_folder + 'WHAMP_INPUT_SPECIES.txt'
    with open(species_file, 'w', newline="\n") as f:
        f.write('MATLAB-WHAMP INPUT SPECIES FILE\n')
        f.write('RUN TITLE {}[{}]\n'.format(series.upper(), run_num))
        f.write('__________________________\n')
        f.write('SPEC#\tMASS/PMASS\tDENSITY (/CM3)\tTEMPR (EV)\tANIS (TP/TX)\tLCPARAM1(D)\tLCPARAM2(B)\tVDRIFT/VTHERM\n')
    
        # Line for each species
        for ii in range(cf.Nj):
            f.write('{}\t'.format(ii + 1))
            f.write('{:10.6e}\t'.format(mass[ii]))
            f.write('{:10.6e}\t'.format(density[ii]))
            f.write('{:10.6e}\t'.format(E_tot[ii]))
            f.write('{:10.6e}\t'.format(new_A[ii]))
            f.write('{:10.6e}\t'.format(1.0))               # Loss cone param 1
            f.write('{:10.6e}\t'.format(0.0))               # Loss cone param 2
            f.write('{:10.6e}\n'.format(new_drift[ii]))
            
        # Gotta do a line for electrons too   
        f.write('{}\t'.format(cf.Nj + 1))
        f.write('{:10.6e}\t'.format(0.0))
        f.write('{:10.6e}\t'.format(ne))
        f.write('{:10.6e}\t'.format(E_e))
        f.write('{:10.6e}\t'.format(1.0))
        f.write('{:10.6e}\t'.format(1.0))
        f.write('{:10.6e}\t'.format(0.0))
        f.write('{:10.6e}\n'.format(0.0))
    return


def read_WHAMP_CLI_dump(textfile):
    k_perp, k_par, f_real, f_imag = [[] for _ in range(4)]
    
    with open(textfile, 'r') as f:
        for line in f:
            A = line.split()
            
            k_perp.append(float(A[0]))
            k_par.append(float(A[1]))
            f_real.append(float(A[2]))
            f_imag.append(float(A[3]))
            
    return np.array(k_perp), np.array(k_par), np.array(f_real), np.array(f_imag)


def plot_WHAMP_CLI_dumps():
    import matplotlib.pyplot as plt
    
    main_dir  = r'F:\Google Drive\Uni\PhD 2017\Resources\whamp-master\whamp_runs\WHAMP_CLI_DUMP\First'
    file_list = os.listdir(main_dir)
    
    plt.figure()
    for file in file_list:
        textfile = os.path.join(main_dir, file)
        print('Reading {}'.format(textfile))
        k_perp, k_par, f_real, f_imag = read_WHAMP_CLI_dump(textfile)
        plt.plot(k_par, f_real)
        
    plt.ylim(0, 1.0)
    plt.xlim(0, None)
    return


# =============================================================================
# if __name__ == '__main__':
#     plot_WHAMP_CLI_dumps()
# =============================================================================
