# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:24:46 2019

@author: Yoshi
"""
#import sys
#data_scripts_dir = 'F://Google Drive//Uni//PhD 2017//Data//Scripts//'
#sys.path.append(data_scripts_dir)
#from analysis_scripts import analytic_signal

import numpy as np
import numba as nb
import os, pdb
import analysis_config as cf
#import pdb
'''
Dump general processing scripts here that don't require global variables: i.e. 
they are completely self-contained or can be easily imported.

These will often be called by plotting scripts so that the main 'analysis'
script is shorter and less painful to work with

If a method requires more than a few functions, it will be split into its own 
module, i.e. get_growth_rates
'''
qi  = 1.602e-19               # Elementary charge (C)
c   = 3e8                     # Speed of light (m/s)
me  = 9.11e-31                # Mass of electron (kg)
mp  = 1.67e-27                # Mass of proton (kg)
e   = -qi                     # Electron charge (C)
mu0 = (4e-7) * np.pi          # Magnetic Permeability of Free Space (SI units)
kB  = 1.38065e-23             # Boltzmann's Constant (J/K)
e0  = 8.854e-12               # Epsilon naught - permittivity of free space

@nb.njit()
def eval_B0x(x):
    return cf.B_eq * (1. + cf.a * x**2)


@nb.njit()
def eval_B0_particle(pos, Bp):
    '''
    Calculates the B0 magnetic field at the position of a particle. B0x is
    non-uniform in space, and B0r (split into y,z components) is the required
    value to keep div(B) = 0
    
    These values are added onto the existing value of B at the particle location,
    Bp. B0x is simply equated since we never expect a non-zero wave field in x.
        
    Could totally vectorise this. Would have to change to give a particle_temp
    array for memory allocation or something
    
    Technically already vectorized but calculating rL would create a new array which
    would kill performance (and require the temp array). Bp would also need to be (3, N)
    '''
    rL     = np.sqrt(pos[1]**2 + pos[2]**2)
    B0_r   = - cf.a * cf.B_eq * pos[0] * rL
    
    Bp[0]  = eval_B0x(pos[0])   
    Bp[1] += B0_r * pos[1] / rL
    Bp[2] += B0_r * pos[2] / rL
    return


def get_B0_particle(x, v, B, sp):
    
    @nb.njit()
    def get_b1(pos, mag):
        xp_mag = np.zeros(3)
        epsil  = 1e-15

        particle_transform = cf.xmax + cf.ND*cf.dx  + epsil   # Offset to account for E/B grid and damping nodes
        
        xp          = (pos + particle_transform) / cf.dx      # Shift particle position >= 0
        Ib          = int(round(xp) - 1.0)                    # Get leftmost to nearest node
        delta_left  = Ib - xp                                 # Distance from left node in grid units
    
        W0 = 0.5  * np.square(1.5 - abs(delta_left))    # Get weighting factors
        W1 = 0.75 - np.square(delta_left + 1.)
        W2 = 1.0  - W0 - W1
        
        for kk in range(3):
            xp_mag[kk] = W0 * mag[Ib, kk] + W1 * mag[Ib + 1, kk] + W2 * mag[Ib + 2, kk]
        
        return xp_mag
    
    b1 = get_b1(x, B)
    
    B0_xp    = np.zeros(3)
    B0_xp[0] = eval_B0x(x)    
    b1t      = np.sqrt(b1[0] ** 2 + b1[1] ** 2 + b1[2] ** 2)
    l_cyc    = (cf.charge[sp] / cf.mass[sp]) * (B0_xp[0] + b1t)
    
    fac      = cf.a * cf.B_eq * x / l_cyc
    B0_xp[1] = v[2] * fac
    B0_xp[2] =-v[1] * fac
    return B0_xp


@nb.njit()
def interpolate_B_to_center(bx, by, bz, zero_boundaries=False):
    ''' 
    Interpolates magnetic field from cell edges to cell centers (where the E
    field is measured). 
    
    bx, by, bz are each (time, space) ndarrays
    
    Also adds on background field in x component (for JxB calculation)
    
    VERIFIED
    '''
    n_times = bx.shape[0]
    NC      = cf.NC                                   # Number of cells
    
    y2x = np.zeros(NC + 1, dtype=nb.float64)          # Second derivatives on B grid
    y2y = np.zeros(NC + 1, dtype=nb.float64)
    y2z = np.zeros(NC + 1, dtype=nb.float64)
    
    bxi = np.zeros((n_times, NC), dtype=nb.float64)   # Interpolation on E grid
    byi = np.zeros((n_times, NC), dtype=nb.float64)
    bzi = np.zeros((n_times, NC), dtype=nb.float64)
    
    # For each time (tt): Calculate second derivative (for each component)
    for tt in range(n_times):
        y2x *= 0
        y2y *= 0
        y2z *= 0
        
        # Interior B-nodes, Centered difference
        for ii in range(1, NC):
            y2x[ii] = bx[tt, ii + 1] - 2*bx[tt, ii] + bx[tt, ii - 1]
            y2y[ii] = by[tt, ii + 1] - 2*by[tt, ii] + by[tt, ii - 1]
            y2z[ii] = bz[tt, ii + 1] - 2*bz[tt, ii] + bz[tt, ii - 1]
                
        # Edge B-nodes, Zero or Forwards/Backwards difference
        if zero_boundaries == True:
            y2x[0 ] = 0.    ;   y2y[0 ] = 0.    ;   y2z[0 ] = 0.
            y2x[NC] = 0.    ;   y2y[NC] = 0.    ;   y2z[NC] = 0.
        else:
            y2x[0]  = 2*bx[tt, 0 ] - 5*bx[tt, 1]      + 4*bx[tt, 2]      - bx[tt, 3]
            y2x[NC] = 2*bx[tt, NC] - 5*bx[tt, NC - 1] + 4*bx[tt, NC - 2] - bx[tt, NC - 3]
            
            y2y[0]  = 2*by[tt, 0 ] - 5*by[tt, 1]      + 4*by[tt, 2]      - by[tt, 3]
            y2y[NC] = 2*by[tt, NC] - 5*by[tt, NC - 1] + 4*by[tt, NC - 2] - by[tt, NC - 3]
            
            y2z[0]  = 2*bz[tt, 0 ] - 5*bz[tt, 1]      + 4*bz[tt, 2]      - bz[tt, 3]
            y2z[NC] = 2*bz[tt, NC] - 5*bz[tt, NC - 1] + 4*bz[tt, NC - 2] - bz[tt, NC - 3]
        
        # Do spline interpolation: E[ii] is bracketed by B[ii], B[ii + 1]
        for ii in range(NC):
            bxi[tt, ii] = 0.5 * (bx[tt, ii] + bx[tt, ii + 1] + (1/6) * (y2x[ii] + y2x[ii + 1]))
            byi[tt, ii] = 0.5 * (by[tt, ii] + by[tt, ii + 1] + (1/6) * (y2y[ii] + y2y[ii + 1]))
            bzi[tt, ii] = 0.5 * (bz[tt, ii] + bz[tt, ii + 1] + (1/6) * (y2z[ii] + y2z[ii + 1]))
                
    return bxi, byi, bzi


def get_electron_temp(qn):
    '''
    Calculate the electron temperature in each cell. Depends on the charge density of each cell
    and the treatment of electrons: i.e. isothermal (ie=0) or adiabatic (ie=1)
    
    qn :: (time, space)
    '''
    Te = np.zeros(qn.shape, dtype=float)
    for ii in range(qn.shape[0]):
        if cf.ie == 0:
            Te[ii, :] = np.ones(qn.shape[0]) * cf.Te0
        elif cf.ie == 1:
            gamma_e = 5./3. - 1.
            Te[ii, :] = cf.Te0 * np.power(qn[ii, :] / (qi * cf.ne), gamma_e)
    return Te


def get_grad_P(qn, te):
    '''
    Returns the electron pressure gradient (in 1D) on the E-field grid using P = nkT and 
    finite difference.
     
    INPUT:
        qn     -- Grid charge density
        te     -- Grid electron temperature
        grad_P -- Output array for electron pressure gradient
        temp   -- intermediary array used to store electron pressure, since both
                  density and temperature may vary (with adiabatic approx.)
        
    Forwards/backwards differencing at the simulation cells at the edge of the
    physical space domain. Guard cells set to zero.
    
    qn, te :: (time, space)
    '''
    Pe     = qn * kB * te / qi       # Store Pe in grad_P array for calculation

    # Central differencing, internal points
    grad_P = np.zeros(qn.shape)
    for ii in nb.prange(1, qn.shape[1] - 1):
        grad_P[:, ii] = (Pe[:, ii + 1] - Pe[:, ii - 1])
    
    # Forwards/Backwards difference at physical boundaries
    grad_P    /= (2*cf.dx)
    
    return grad_P



def get_curl_B(bx, by, bz):
    '''
    Each b component is a (time, space) ndarray. This looks fine.
    '''
    curl_B = np.zeros((bx.shape[0], bx.shape[1] - 1, 3), dtype=np.float64)
    
    for ii in nb.prange(bx.shape[1] - 1):
        curl_B[:, ii, 1] = -(bz[:, ii + 1] - bz[:, ii])
        curl_B[:, ii, 2] =   by[:, ii + 1] - by[:, ii]
    
    curl_B /= (cf.dx * mu0)

    return curl_B



def cross_product(ax, ay, az, bx, by, bz):
    '''
    Vector (cross) product between two vectors, A and B of same dimensions.
    all ai, bi are expected to be (time, space) ndarrays
    '''
    C = np.zeros((az.shape[0], az.shape[1], 3), dtype=np.float64)

    for ii in nb.prange(az.shape[0]):
        C[ii, :, 0] += ay[ii] * bz[ii]
        C[ii, :, 1] += az[ii] * bx[ii]
        C[ii, :, 2] += ax[ii] * by[ii]
        
        C[ii, :, 0] -= az[ii] * by[ii]
        C[ii, :, 1] -= ax[ii] * bz[ii]
        C[ii, :, 2] -= ay[ii] * bx[ii]
    return C



def calculate_E_components(bx, by, bz, jx, jy, jz, q_dens):
    '''
    '''
    # Need to calculate (Fatemi, 2017):
    # Ji x B / qn               Convective Term     LOOKS GOOD
    # del(p) / qn               Ambipolar term      LOOKS GOOD
    # Bx(curl B) / qn*mu0       Hall Term           LOOKS GOOD
    # This version of the code doesn't include an Ohmic term, since eta = 0

    bxi, byi, bzi = interpolate_B_to_center(bx, by, bz)
    B0  = eval_B0x(cf.E_nodes)
    for tt in range(bxi.shape[0]):
        bxi[tt] += B0
        
# =============================================================================
#     import sys
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots(2, figsize=(15, 10))
#     
#     time  = 120
#     space = bx.shape[1] // 2
#     
#     ax[0].plot(cf.B_nodes, by[ time], marker='o', c='b')
#     ax[0].plot(cf.E_nodes, byi[time], marker='x', c='r')
#     
#     ax[1].plot(by[ :, space], marker='o', c='b')
#     ax[1].plot(byi[:, space], marker='x', c='r')
#     
#     sys.exit()
# =============================================================================
    
    # Hall Term
    curl_B = get_curl_B(bx, by, bz)
    BdB    = cross_product(bxi, byi, bzi, curl_B[:, :, 0], curl_B[:, :, 1], curl_B[:, :, 2])
    
    # Ambipolar Term
    Te     = get_electron_temp(q_dens)
    grad_P = get_grad_P(q_dens, Te)                           # temp1D is now del_p term, temp3D2 slice used for computation
    grad_P/= q_dens[:]
    
    # Convective Term
    JxB  = cross_product(jx, jy, jz, bxi, byi, bzi)           # temp3De is now Ve x B term
    
    for ii in range(3):
        BdB[:, :, ii] /= q_dens[:]
        JxB[:, :, ii] /= q_dens[:]
        
    return BdB, grad_P, JxB
    #     hall, amb   , conv





def get_energies(): 
    '''
    Computes and saves field and particle energies at each field/particle timestep.
    
    TO DO: Use particle-interpolated fields rather than raw field files.
    OR    Use these interpolated fields ONLY for the total energy, would be slower.
    '''
    from analysis_config import NX, dx, idx_start, idx_end, Nj, n_contr, mass

    energy_file = cf.temp_dir + 'energies.npz'
    
    if os.path.exists(energy_file) == False:
        mu0 = (4e-7) * np.pi          # Magnetic Permeability of Free Space (SI units)
        kB  = 1.38065e-23             # Boltzmann's Constant (J/K)
        q   = 1.602e-19               # Elementary charge (C)
    
        num_field_steps    = len(os.listdir(cf.field_dir))
        num_particle_steps = len(os.listdir(cf.particle_dir))
    
        mag_energy      = np.zeros( num_field_steps)
        electron_energy = np.zeros( num_field_steps)
        particle_energy = np.zeros((num_particle_steps, Nj, 2))
        
        for ii in range(num_field_steps):
            print('Loading field file {}'.format(ii))
            B, E, Ve, Te, J, q_dns, sim_time, damping_array = cf.load_fields(ii)
            
            mag_energy[ii]      = (0.5 / mu0) * np.square(B[1:-2]).sum() * dx
            electron_energy[ii] = 1.5 * (kB * Te * q_dns / q).sum() * dx
    
        for ii in range(num_particle_steps):
            print('Loading particle file {}'.format(ii))
            pos, vel, psim_time = cf.load_particles(ii)
            for jj in range(Nj):
                '''
                Only works properly for theta = 0 : Fix later
                '''
                vpp2 = vel[0, idx_start[jj]:idx_end[jj]] ** 2
                vpx2 = vel[1, idx_start[jj]:idx_end[jj]] ** 2 + vel[2, idx_start[jj]:idx_end[jj]] ** 2
        
                particle_energy[ii, jj, 0] = 0.5 * mass[jj] * vpp2.sum() * n_contr[jj] * NX * dx
                particle_energy[ii, jj, 1] = 0.5 * mass[jj] * vpx2.sum() * n_contr[jj] * NX * dx
        
        # Calculate total energy
        ptime_sec, pbx, pby, pbz, pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens = \
        cf.interpolate_fields_to_particle_time(num_particle_steps)
        
        pmag_energy      = (0.5 / mu0) * (np.square(pbx) + np.square(pby) + np.square(pbz)).sum(axis=1) * NX * dx
        pelectron_energy = 1.5 * (kB * pte * pqdens / q).sum(axis=1) * NX * dx

        total_energy = np.zeros(num_particle_steps)   # Placeholder until I interpolate this
        for ii in range(num_particle_steps):
            total_energy[ii] = pmag_energy[ii] + pelectron_energy[ii]
            for jj in range(Nj):
                total_energy[ii] += particle_energy[ii, jj, :].sum()
        
        print('Saving energies to file...')
        np.savez(energy_file, mag_energy      = mag_energy,
                              electron_energy = electron_energy,
                              particle_energy = particle_energy,
                              total_energy    = total_energy)
    else:
        print('Loading energies from file...')
        energies        = np.load(energy_file) 
        mag_energy      = energies['mag_energy']
        electron_energy = energies['electron_energy']
        particle_energy = energies['particle_energy']
        total_energy    = energies['total_energy']
    return mag_energy, electron_energy, particle_energy, total_energy


def get_helical_components(overwrite=False, field='B'):
    temp_dir = cf.temp_dir  
    print('Getting helical components for {} field'.format(field))
    if os.path.exists(temp_dir + '{}_positive_helicity.npy'.format(field)) == False or overwrite == True:
        ftime, Fy = cf.get_array('{}y'.format(field))
        ftime, Fz = cf.get_array('{}z'.format(field))
        
        Ft_pos = np.zeros((ftime.shape[0], cf.NX), dtype=np.complex128)
        Ft_neg = np.zeros((ftime.shape[0], cf.NX), dtype=np.complex128)
        
        for ii in range(Fy.shape[0]):
            print('Calculating helicity for field file', ii)
            Ft_pos[ii, :], Ft_neg[ii, :] = calculate_helicity(Fy[ii], Fz[ii])
        
        print('Saving {}-helicities to file'.format(field))
        np.save(temp_dir + '{}_positive_helicity.npy'.format(field), Ft_pos)
        np.save(temp_dir + '{}_negative_helicity.npy'.format(field), Ft_neg)
    else:
        print('Loading {}-elicities from file'.format(field))
        ftime, Fy = cf.get_array('{}y'.format(field))
        
        Ft_pos = np.load(temp_dir + '{}_positive_helicity.npy'.format(field))
        Ft_neg = np.load(temp_dir + '{}_negative_helicity.npy'.format(field))
    return ftime, Ft_pos, Ft_neg


def calculate_helicity(Fy, Fz):
    '''
    For a single snapshot in time, calculate the positive and negative helicity
    components from the y, z components of a field.
    
    This code has been checked by comparing the transverse field magnitude of
    the inputs and outputs, as this should be conserved (and it is).
    
    This can be done much faster by taking the FFT/power spectrum of
    F = Fy + iFz and the k values are explicitly stated. Check this 
    for sure later.
    '''
    x   = np.linspace(0, cf.NX*cf.dx, cf.NX)
    st  = cf.ND
    en  = cf.ND + cf.NX

    k_modes = np.fft.rfftfreq(x.shape[0], d=cf.dx)
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


def get_FB_waves(overwrite=False, field='B', st=None, en=None):
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
    '''
    ftime, Fy = cf.get_array('{}y'.format(field))
    ftime, Fz = cf.get_array('{}z'.format(field))

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


def get_derivative(arr):
    ''' Caculates centered derivative for values in 'arr', with forward and backward differences applied
    for boundary points'''
    
    deriv = np.zeros(arr.shape[0])
    
    deriv[0 ] = (-3*arr[ 0] + 4*arr[ 1] - arr[ 2]) / (2 * cf.dt_field)
    deriv[-1] = ( 3*arr[-1] - 4*arr[-2] + arr[-3]) / (2 * cf.dt_field)
    
    for ii in np.arange(1, arr.shape[0] - 1):
        deriv[ii] = (arr[ii + 1] - arr[ii - 1]) / (2 * cf.dt_field)
    return deriv


# =============================================================================
# def get_envelope(arr):
#     signal_envelope = analytic_signal(arr, dt=cf.dx)
#     return signal_envelope
# =============================================================================


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    f0    = 0.4 # Hz
    t_max = 10000
    dt    = 0.1
    t     = np.arange(0, t_max, dt)
    
    signal = np.sin(2 * np.pi * f0 * t)
    sfft   = 2 / t.shape[0] * np.fft.rfft(signal)
    freqs  = np.fft.rfftfreq(t.shape[0], d=dt)
    
    plt.plot(freqs, sfft.real)
    plt.plot(freqs, sfft.imag)
    #plt.xlim(0, 0.5)
    
    
    