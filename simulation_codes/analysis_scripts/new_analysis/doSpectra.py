# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:14:56 2019

@author: Yoshi
"""
import numpy as np
import numba as nb
import os, sys, pdb
from SimulationClass import MAGN_PERMEABILITY, UNIT_CHARGE, PROTON_MASS

data_scripts_dir = os.environ['userprofile'] + '//Documents//GitHub//hybrid//linear_theory//new_general_DR_solver//'
sys.path.append(data_scripts_dir)
'''
Function to do all the backend calculation of things like wx, wk, kt, plots 
from field data.

TODO:
    -- Remove depedence on Sim instances, set functions so they do only calculation
    -- Move some of these decomposition tools into their own script doDecomposition (F/B, helicity, DMD, etc.)
'''
def get_wx(Sim, component, fac=1.0, normalize=True):
    arr = getattr(Sim, component.lower())
    arr *= fac
    
    if component[0].upper() == 'B':
        ncells = Sim.NC + 1
    else:
        ncells = Sim.NC
        
    if arr.shape[0]%2 == 0:
        fft_matrix  = np.zeros((arr.shape[0]//2+1, ncells), dtype='complex128')
    else:
        fft_matrix  = np.zeros(((arr.shape[0]+1)//2, ncells), dtype='complex128')
        
    for ii in range(arr.shape[1]):
        fft_matrix[:, ii] = np.fft.rfft(arr[:, ii] - arr[:, ii].mean())
    
    if normalize:
        fft_matrix *= 2.0 / arr.shape[0]
        
    wx = (fft_matrix * np.conj(fft_matrix)).real
    return wx


def get_kt(Sim, component):
    arr = getattr(Sim, component.lower())
    
    # Get first/last indices for FFT range and k-space array
    if component[0].upper() == 'B':
        st, en = Sim.x0B, Sim.x1B
        k  = np.fft.fftfreq(Sim.NX + 1, Sim.dx)
    else:
        st, en = Sim.x0E, Sim.x1E
        k  = np.fft.fftfreq(Sim.NX, Sim.dx)
                  
    k   = k[k>=0]
    
    fft_matrix  = np.zeros((arr.shape[0], en-st), dtype='complex128')
    for ii in range(arr.shape[0]): # Take spatial FFT at each time, ii
        fft_matrix[ii, :] = np.fft.fft(arr[ii, st:en] - arr[ii, st:en].mean())

    kt = (fft_matrix[:, :k.shape[0]] * np.conj(fft_matrix[:, :k.shape[0]])).real
    
    return k, kt, st, en


def get_wk(Sim, component, linear_only=True, norm_z=False, centre_only=False):
    '''
    Spatial boundaries start at index.
    linear_only is kwarg to either take Boolean type or a number specifying up to
    what time to FFT up to (in units of wcinv)
    
    norm_z normalizes the field to the background magnetic field, if the input component
    is magnetic.
    
    centre_only :: kwarg to only FFT the middle __ % of cells. Calculated by 
    (1.0-centre_only)/2 indices on each side subtracted.
    '''
    arr = getattr(Sim, component.lower())
    if norm_z == True and component.upper()[0] == 'B':
        arr /= Sim.B_eq
    
    t0 = 0; ftime = getattr(Sim, 'field_sim_time')
    if linear_only == True:
        # Find the point where magnetic energy peaks, FFT only up to 120% of that index
        by    = getattr(Sim, 'By')
        bz    = getattr(Sim, 'Bz')
        bt         = np.sqrt(by ** 2 + bz ** 2)
        U_B        = 0.5 / MAGN_PERMEABILITY * np.square(bt[:, :]).sum(axis=1) * Sim.dx
        max_idx    = np.argmax(U_B)
        t1         = int(1.2*max_idx)
        tf         = ftime[t1]
    elif linear_only == False:
        t1 = ftime.shape[0]
        tf = ftime[t1-1]
    else:
        # Assume linear_only is a number
        end_sec  = linear_only * 1./Sim.gyfreq
        diff_sec = np.abs(end_sec - ftime)
        t1       = np.where(diff_sec == diff_sec.min())[0][0]
        tf       = ftime[t1]
    
    if centre_only != False:
        fraction_cut = (1.0 - centre_only) / 2.0
        if component.upper()[0] == 'B':
            cut = int((Sim.x1B - Sim.x0B)*fraction_cut)
        else:
            cut = int((Sim.x1E - Sim.x0E)*fraction_cut)
        print(f'Cutting {cut} indices from each side')
    else:
        cut = 0
    
    if component.upper()[0] == 'B':
        st = Sim.x0B + cut; en = Sim.x1B - cut
    else:
        st = Sim.x0E + cut; en = Sim.x1E - cut

    num_times  = t1-t0
    num_points = en-st

    df = 1. / (num_times  * Sim.dt_field)
    dk = 1. / (num_points * Sim.dx)

    f  = np.arange(0, 1. / (2*Sim.dt_field), df)
    k  = np.arange(0, 1. / (2*Sim.dx), dk) * 2*np.pi
    
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
            temp_in[:]     = np.multiply(dat[ii*slide: 
                                             ii*slide + win_len],               
                                             window)                            # Apply FFT windowing function to windowed data and store in FFT input memory address
            out_mem[ii, :] = fft_object()                                       # Call FFT and store result in output array (involves implicit copy)        
            
    out_mem /=  win_len                                                         # Normalize
    return 2. * out_mem[:, :win_len//2 + 1]                                     # Return only positive frequencies


def autopower_spectra(Sim, component='By', overlap=0.5, win_idx=None, slide_idx=None, df=50, cell_idx=None):
    y_arr = getattr(Sim, component.lower())
    dt    = Sim.dt_field
    
    if 'B' in component.upper():
        y_arr = y_arr[:, cell_idx]*1e9
    else:
        y_arr = y_arr[:, cell_idx]*1e6
   
    if win_idx is None:
        t_win   = 1000. / df                                                        # Window length (in seconds) for desired frequency resolution
        win_idx = int(np.ceil(t_win / dt))                                 # Window size in indexes
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
    
    df           = 1./(win_idx * dt)                                                # Frequency increment (in mHz)
    freq         = np.asarray([df * jj for jj in range(win_idx//2 + 1)])            # Frequency array up to Nyquist
    power        = np.real(FFT_output * np.conj(FFT_output))
    return power, FFT_times, freq


def get_cgr_from_sim(Sim, norm_flag=0):
    from convective_growth_rate import calculate_growth_rate
    
    cold_density = np.zeros(3)
    warm_density = np.zeros(3)
    cgr_ani      = np.zeros(3)
    tempperp     = np.zeros(3)
    anisotropies = Sim.Tper / Sim.Tpar - 1
    
    for ii in range(Sim.Nj):
        if Sim.temp_type[ii] == 0:
            if 'H^+'    in Sim.species_lbl[ii]:
                cold_density[0] = Sim.density[ii] / 1e6
            elif 'He^+' in Sim.species_lbl[ii]:
                cold_density[1] = Sim.density[ii] / 1e6
            elif 'O^+'  in Sim.species_lbl[ii]:
                cold_density[2] = Sim.density[ii] / 1e6
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')
                
        if Sim.temp_type[ii] == 1:
            if 'H^+'    in Sim.species_lbl[ii]:
                warm_density[0] = Sim.density[ii] / 1e6
                cgr_ani[0]      = Sim.anisotropy[ii]
                tempperp[0]     = Sim.Tperp[ii] / 11603.
            elif 'He^+' in Sim.species_lbl[ii]:
                warm_density[1] = Sim.density[ii] / 1e6
                cgr_ani[1]      = Sim.anisotropy[ii]
                tempperp[1]     = Sim.Tperp[ii] / 11603.
            elif 'O^+'  in Sim.species_lbl[ii]:
                warm_density[2] = Sim.density[ii] / 1e6
                cgr_ani[2]      = anisotropies[ii]
                tempperp[2]     = Sim.Tperp[ii] / 11603.
            else:
                print('WARNING: UNKNOWN ION IN DENSITY MIX')
    
    freqs, cgr, stop = calculate_growth_rate(Sim.B_eq*1e9, cold_density, warm_density, cgr_ani, temperp=tempperp, norm_freq=norm_flag)
    return freqs, cgr, stop


def get_linear_dispersion_from_sim(Sim, k, zero_cold=True, Nk=1000):
    '''
    Extracted values units :: 
        Density    -- /m3       (Cold, warm densities)
        Anisotropy -- Number
        Tper       -- eV
    '''
    print('Calculating linear dispersion relations...')
    from multiapprox_dispersion_solver  import get_dispersion_relation, create_species_array
    
    # Extract species parameters from run, create Species array (Could simplify T calculation when I'm less lazy)
    anisotropy = Sim.Tperp / Sim.Tpar - 1
    
    t_perp = Sim.Tperp.copy() / 11603.
    if zero_cold == True:
        for ii in range(t_perp.shape[0]):
            if Sim.temp_type[ii] == 0:
                t_perp[ii]     = 0.0
                anisotropy[ii] = 0.0
    
    Species, PP = create_species_array(Sim.B_eq, Sim.species_lbl, Sim.mass, Sim.charge,
                                       Sim.density, t_perp, Sim.anisotropy)
    
    # Convert from linear units to angular units for k range to solve over
    kmin   = 2*np.pi*k[0]
    kmax   = 2*np.pi*k[-1]
    k_vals = np.linspace(kmin, kmax, Nk) 
    
    # Calculate dispersion relations (3 approximations)
    CPDR_solns,  cold_CGR = get_dispersion_relation(Species, k_vals, approx='cold')
    WPDR_solns,  warm_CGR = get_dispersion_relation(Species, k_vals, approx='warm')
    HPDR_solns,  hot_CGR  = get_dispersion_relation(Species, k_vals, approx='hot')

    # Convert to linear frequency
    CPDR_solns /= 2*np.pi
    WPDR_solns /= 2*np.pi
    HPDR_solns /= 2*np.pi

    return k_vals, CPDR_solns, WPDR_solns, HPDR_solns


def get_helical_components(Sim, overwrite=False, field='B'):
    '''
    TODO: Make this a method
    '''
    temp_dir = Sim.temp_dir  
    ftime = getattr(Sim, 'field_sim_time')
    print('Getting helical components for {} field'.format(field))
    if os.path.exists(temp_dir + '{}_positive_helicity.npy'.format(field)) == False or overwrite == True:
        Fy = getattr(Sim, '{}y'.format(field.lower()))
        Fz = getattr(Sim, '{}z'.format(field.lower()))
        
        Ft_pos = np.zeros((ftime.shape[0], Sim.NX), dtype=np.complex128)
        Ft_neg = np.zeros((ftime.shape[0], Sim.NX), dtype=np.complex128)
        
        for ii in range(Fy.shape[0]):
            print('Calculating helicity for field file', ii)
            Ft_pos[ii, :], Ft_neg[ii, :] = calculate_helicity(Sim, Fy[ii], Fz[ii])
        
        print('Saving {}-helicities to file'.format(field))
        np.save(temp_dir + '{}_positive_helicity.npy'.format(field), Ft_pos)
        np.save(temp_dir + '{}_negative_helicity.npy'.format(field), Ft_neg)
    else:
        print('Loading {}-elicities from file'.format(field))        
        Ft_pos = np.load(temp_dir + '{}_positive_helicity.npy'.format(field))
        Ft_neg = np.load(temp_dir + '{}_negative_helicity.npy'.format(field))
    return ftime, Ft_pos, Ft_neg


def calculate_helicity(Sim, Fy, Fz):
    '''
    For a single snapshot in time, calculate the positive and negative helicity
    components from the y, z components of a field.
    
    This code has been checked by comparing the transverse field magnitude of
    the inputs and outputs, as this should be conserved (and it is).
    
    This can be done much faster by taking the FFT/power spectrum of
    F = Fy + iFz and the k values are explicitly stated. Check this 
    for sure later.
    '''
    x   = np.linspace(0, Sim.NX*Sim.dx, Sim.NX)
    st  = Sim.ND
    en  = Sim.ND + Sim.NX

    k_modes = np.fft.rfftfreq(x.shape[0], d=Sim.dx)
    Fy_fft  = (1 / k_modes.shape[0]) * np.fft.rfft(Fy[st:en])
    Fz_fft  = (1 / k_modes.shape[0]) * np.fft.rfft(Fz[st:en])

    # Four fourier coefficients from FFT (since real inputs give symmetric outputs)
    # If there are any sign issues, it'll be with the sin components, here
    Fy_cos = Fy_fft.real
    Fy_sin = Fy_fft.imag
    Fz_cos = Fz_fft.real
    Fz_sin = Fz_fft.imag
    
    # Construct spiral mode k-coefficients
    Fk_pos = 0.5 * ((Fy_cos + Fz_sin) + 1j * (Fz_cos - Fy_sin ))
    Fk_neg = 0.5 * ((Fy_cos - Fz_sin) + 1j * (Fz_cos + Fy_sin ))
    
    # Construct spiral mode timeseries
    Ft_pos = np.zeros(x.shape[0], dtype=np.complex128)
    Ft_neg = np.zeros(x.shape[0], dtype=np.complex128)
    
    # The sign of the exponential may also be another issue, should check.
    for ii in range(k_modes.shape[0]):
        Ft_pos += Fk_pos[ii] * np.exp(-2j*np.pi*k_modes[ii]*x)
        Ft_neg += Fk_neg[ii] * np.exp( 2j*np.pi*k_modes[ii]*x)
    return Ft_pos, Ft_neg


def get_FB_waves(Sim, overwrite=False, field='B', st=None, en=None):
    '''
    Use method of Shoji et al. (2011) to separate magnetic wave field into
    backwards and forwards components
     -- Should be equivalent to helicity thing
     -- Doesn't generalise well, can't tell between propagation direction and polarisation
     -- Only works for this because EMICs are generated in the L-mode and have single polarisation
     -- Don't cut off damping regions, she'll be right
    Fields are (time, space)
    
    FFT Positions
    # [0]     - Zero frequency term
    # [1:n/2] - Positive k terms
    # [n/2+1:] - Negative k terms
    # n/2 either both pos/neg nyquist (even) or 
    #             (odd) largest positive/negative frequency is on either side
    
    -- Do I have to account for that?
    -- Zeroing the A[0] term will probably just demean the resulting timeseries
    
    STILL NEEDS VALIDATION
     - Conservation: If I add the results, do I get the original waveform back?
     - Compare against existing helicity code, are they equivalent?
     
    The sign in F_perp (I think) just determines which is the +/- wave
    
    TODO: Make this a method
    '''
    print('Calculating positive/negative helicities')
    ftime = getattr(Sim, 'field_sim_time')
    Fy = getattr(Sim, '{}y'.format(field.lower()))
    Fz = getattr(Sim, '{}z'.format(field.lower()))

    F_perp    = Fy[:, st:en] + 1j*Fz[:, st:en]      # Should this be a +ve or -ve? Defines polarisation 
    Nk        = F_perp.shape[1]
    
    F_fwd = np.zeros(F_perp.shape, dtype=np.complex128)
    F_bwd = np.zeros(F_perp.shape, dtype=np.complex128)    
    for ii in range(F_perp.shape[0]):
        Fk      = np.fft.fft(F_perp[ii])
        Fk0     = 0.5*Fk[0]
        
        fwd_fft = Fk.copy()                 # Copy coefficients
        bwd_fft = Fk.copy()

        fwd_fft[Nk//2+1:] *= 0.0            # Zero positive/negative wavenumbers
        bwd_fft[1:Nk//2]  *= 0.0

        fwd_fft[0] = Fk0                    # Not sure what to do with the zero term. Split evenly?
        bwd_fft[0] = Fk0

        F_fwd[ii] = np.fft.ifft(fwd_fft)    # Transform back into spatial-domain data
        F_bwd[ii] = np.fft.ifft(bwd_fft)

    return ftime, F_fwd, F_bwd, F_perp


@nb.njit(parallel=True)
def get_Ev_ratio(vel, Ie, W_elec, ex, ey, ez, ti):
    wE = np.zeros(Ie.shape[0], dtype=nb.float32)
    for pp in nb.prange(vel.shape[1]):   
        Epx = 0.0; Epy = 0.0; Epz = 0.0
        for jj in range(3):        
            Epx += ex[ti, Ie[pp] + jj] * W_elec[jj, pp]
            Epy += ey[ti, Ie[pp] + jj] * W_elec[jj, pp]
            Epz += ez[ti, Ie[pp] + jj] * W_elec[jj, pp]
        Ep  = np.sqrt(Epx**2 + Epy**2 + Epz**2)
        vt  = np.sqrt(vel[0, pp] ** 2 + vel[1, pp] **2 + vel[2, pp] ** 2)
        
        # Redo to prevent race condition
        # Do single variables lead to race conditions? Or is it only arrays?
        wE[pp] = np.abs(Ep / vt)
    return wE.max() * UNIT_CHARGE/PROTON_MASS
    

def get_stability_criteria(Sim, overwrite=False):
    '''
    -- Resolution criteria:
        -- Fastest particle, wL = k*v = pi*v/dx     (Particle data only)
        -- Electric field,   wE = qs*E / ms*v       (Particle and field data)
        -- Gyrofrequency,    wG = qs*B / ms         (Field only)
        -- Dispersion,       wD = k^2*B / mu0*pc    (Field only)
    -- Stability criteria:
        -- Phase speed? Compare this to c or vA? Does it even apply to hybrids? E/H?
        
    For particle criteria, need to fetch fields at particle position, calculate
    wG and wE for each particle, and take the max value. For wL, wD, just take the 
    max value for the particle and field quantities, respectively.
    
    TODO: Make this a method
    '''
    stab_path = Sim.temp_dir + 'stability_criteria.npz'
    if os.path.exists(stab_path) and not overwrite:
        print('Loading stability criteria from file')
        data = np.load(stab_path)
        ftimes = data['ftimes']
        ptimes = data['ptimes']
        wL_max = data['wL_max']
        wE_max = data['wE_max']
        wG_max = data['wG_max']
        wD_max = data['wD_max']
        vdt_max= data['vdt_max']
    else:            
        print('Calculating stability criteria for fields...')
        k = np.pi / Sim.dx
        
        # Load all fields (time, space)
        bx = getattr(Sim, 'bx')
        by = getattr(Sim, 'by')
        bz = getattr(Sim, 'bz')
        bt = np.sqrt(bx ** 2 + by **2 + bz ** 2) + Sim.Bc[:, 0]
        
        ex = getattr(Sim, 'ex')
        ey = getattr(Sim, 'ey')
        ez = getattr(Sim, 'ez')
        
        qdens = getattr(Sim, 'qdens')
        
        # Calculate max dispersion frequency
        wD = k**2 * 0.5*(bt[:, 1:] + bt[:, :-1]) / (MAGN_PERMEABILITY * qdens)
        wD_max = wD.max(axis=1)
        
        # Calculate max gyrofrequency
        wG_max = UNIT_CHARGE * (bt.max(axis=1)) / PROTON_MASS
    
        ptimes = np.zeros(Sim.num_particle_steps)
        wL_max = np.zeros(Sim.num_particle_steps)
        wE_max = np.zeros(Sim.num_particle_steps)
        vdt_max= np.zeros(Sim.num_particle_steps)
    
        print('Calculating stability criteria for particles...')
        for ii in range(Sim.num_particle_steps):
            print('\r {:.0f} percent'.format(ii/Sim.num_particle_steps*100), end="")
            pos, vel, idx, ptimes[ii], idx_start, idx_end = Sim.load_particles(ii)
            max_V = np.abs(vel[0]).max()
            # Calculate fastest particle and corresponding 'frequency'
            wL_max[ii] = np.pi*max_V/Sim.dx
            
            # Also calculate timestep required for dt = 0.5*dx/max_V
            vdt_max[ii] = 0.5*Sim.dx/max_V
            
            # Find field time closest to current particle time
            diff = np.abs(ptimes[ii] - ftimes)
            ti = np.where(diff == diff.min())[0][0]
            
            # Weight fields to particle position as they would have been in the code
            Ie     = np.zeros(    Sim.N,  dtype=int)
            W_elec = np.zeros((3, Sim.N), dtype=float)
            assign_weighting_TSC(Sim, pos, Ie, W_elec, E_nodes=True)
            
            # Calculate largest E/v ratio
            wE_max[ii] = get_Ev_ratio(vel, Ie, W_elec, ex, ey, ez, ti)
            
        print('Saving...')
        np.savez(stab_path, ftimes=ftimes, ptimes=ptimes, wL_max=wL_max,
                           wE_max=wE_max, wG_max=wG_max, wD_max=wD_max,
                           vdt_max=vdt_max)
    return ftimes, ptimes, wL_max, wE_max, wG_max, wD_max, vdt_max



def basic_S(arr, k=5, h=1.0):
    N = arr.shape[0]
    S1 = np.zeros(N)
    S2 = np.zeros(N)
    S3 = np.zeros(N)
    
    for ii in range(N):
        if ii < k:
            left_vals = arr[:ii]
            right_vals = arr[ii + 1:ii + k + 1]
        elif ii >= N - k:
            left_vals  = arr[ii - k: ii]
            right_vals = arr[ii + 1:]
        else:
            left_vals  = arr[ii - k: ii]
            right_vals = arr[ii + 1:ii + k + 1]

        left_dist  = arr[ii] - left_vals
        right_dist = arr[ii] - right_vals
        
        if left_dist.shape[0] == 0:
            left_dist = np.append(left_dist, 0)
        elif right_dist.shape[0] == 0:
            right_dist = np.append(right_dist, 0)
        
        S1[ii] = 0.5 * (left_dist.max()     + right_dist.max()    )
        S2[ii] = 0.5 * (left_dist.sum() / k + right_dist.sum() / k)
        S3[ii] = 0.5 * ((arr[ii] - left_vals.sum() / k) + (arr[ii] - right_vals.sum() / k))
        
    S_ispeak = np.zeros((N, 3))
    
    for S, xx in zip([S1, S2, S3], np.arange(3)):
        for ii in range(N):
            if S[ii] > 0 and (S[ii] - S.mean()) > (h * S.std()):
                S_ispeak[ii, xx] = 1

    for xx in range(3):
        for ii in range(N):
            for jj in range(N):
                if ii != jj and S_ispeak[ii, xx] == 1 and S_ispeak[jj, xx] == 1:
                    if abs(jj - ii) <= k:
                        if arr[ii] < arr[jj]:
                            S_ispeak[ii, xx] = 0
                        else:
                            S_ispeak[jj, xx] = 0
                            
    S1_peaks = np.arange(N)[S_ispeak[:, 0] == 1]
    S2_peaks = np.arange(N)[S_ispeak[:, 1] == 1]
    S3_peaks = np.arange(N)[S_ispeak[:, 2] == 1]
    return S1_peaks, S2_peaks, S3_peaks


#@nb.njit(parallel=True)
def assign_weighting_TSC(Sim, pos, I, W, E_nodes=True):
    '''
    Analysis code version of this since we don't save it. Need to work out
    how to get Sim to play well with parallelization. Maybe will have to
    deal with it as a jitclass? Is that even possible with the I/O required
    to initialize it?
    '''
    Np         = pos.shape[0]
    epsil      = 1e-15
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 0.0
    
    particle_transform = Sim.xmax + (Sim.ND - grid_offset)*Sim.dx  + epsil      # Offset to account for E/B grid and damping nodes
    
    if Sim.field_periodic == 0:
        for ii in nb.prange(Np):
            xp          = (pos[ii] + particle_transform) / Sim.dx       # Shift particle position >= 0
            I[ii]       = int(round(xp) - 1.0)                      # Get leftmost to nearest node (Vectorize?)
            delta_left  = I[ii] - xp                                # Distance from left node in grid units
            
            if abs(pos[ii] - Sim.xmin) < 1e-10:
                I[ii]    = Sim.ND - 1
                W[0, ii] = 0.0
                W[1, ii] = 0.5
                W[2, ii] = 0.0
            elif abs(pos[ii] - Sim.xmax) < 1e-10:
                I[ii]    = Sim.ND + Sim.NX - 1
                W[0, ii] = 0.5
                W[1, ii] = 0.0
                W[2, ii] = 0.0
            else:
                W[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))  # Get weighting factors
                W[1, ii] = 0.75 - np.square(delta_left + 1.)
                W[2, ii] = 1.0  - W[0, ii] - W[1, ii]
    else:
        for ii in nb.prange(Np):
            xp          = (pos[ii] + particle_transform) / Sim.dx   # Shift particle position >= 0
            I[ii]       = int(round(xp) - 1.0)                      # Get leftmost to nearest node (Vectorize?)
            delta_left  = I[ii] - xp                                # Distance from left node in grid units

            W[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))  # Get weighting factors
            W[1, ii] = 0.75 - np.square(delta_left + 1.)
            W[2, ii] = 1.0  - W[0, ii] - W[1, ii]
    return I, W