# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:56:57 2021

@author: Yoshi
"""
### IMPORTS
import warnings, sys, pdb, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

### CONSTANTS
qp     = 1.602e-19
qe     =-1.602e-19
mp     = 1.673e-27
me     = 9.110e-31
e0     = 8.854e-12
mu0    = 4e-7*np.pi
RE     = 6.371e6
c      = 3e8
kB     = 1.380649e-23
B_surf = 3.12e-5

'''
Need to recreate trapping plots but for a specific plasma regime. For this:
    -- S must be given a value (worked out via eqns. 40-43 or Omura et al., 2010)
    -- Wave parameters must be set (freq. sweep rate, amplitude, frequency)
    -- Plasma parameters used to calculate velocities and relation between w, k
    -- Gradient of magnetic field set
    
Test:
    -- Try with initial S = 0, equiv. to no sweep and uniform field (i.e. equatorial).
        Should return pendulum phase space.
    -- Use Omura plasma initially, frequency and amplitude of triggering wave (above threshold)
    
For a given frequency, there is unique k, and velocities under the cold approximation
'''
def new_trace_fieldline(L, Npts=1e5):
    '''
    Traces field line position and B-intensity for given L. 
    Traces half, flips and copies the values, and then
    reverses sign of s since the integration is done from the southern hemisphere
    but we normally think of traces from the northern.
        
    Validated: Ionospheric B0 goes to surface B0 for rA = 0.
    B outputted in nT
    '''
    BE     = 0.31*1e5   # Equatorial field strength in nT
    r_A    = 120e3 
    RE     = 6.371e6

    # Force Np odd:
    if Npts%2 == 0:
        Np = int(Npts+1)
    else:
        Np = int(Npts)
        
    iono_lat = np.arccos(np.sqrt((RE + r_A)/(RE*L))) 
    mlat     = np.linspace(-iono_lat, iono_lat, Np, endpoint=True)
    dlat     = mlat[1] - mlat[0]
    mid_idx  = (Np-1) // 2
    
    Bs = np.zeros(Np, dtype=float)
    s  = np.zeros(Np, dtype=float)
    
    # Step through MLAT starting at equator. Calculate B at each point
    current_s = 0.0 
    for ii in range(mid_idx, Np):
        ds     = L*RE*np.cos(mlat[ii])*np.sqrt(4.0 - 3.0*np.cos(mlat[ii]) ** 2) * dlat
        s[ii]  = current_s
        Bs[ii] = (BE/L**3)*np.sqrt(4-3*np.cos(mlat[ii])**2) / np.cos(mlat[ii])**6
        current_s += ds
        
    # Flip around equator for 1st half
    s[ :mid_idx] = -1.0*np.flip( s[mid_idx + 1:])
    Bs[:mid_idx] =      np.flip(Bs[mid_idx + 1:])
    return -s, Bs*1e-9, -mlat*180./np.pi

def get_B_gradient(B, s):
    '''
    Get gradient by finite difference - Central, and F/B for edges
    
    Might need to interpolate onto uniform S grid, but this will do for now
    '''
    grad = np.zeros(B.shape[0])
    for ii in range(1, B.shape[0] - 1):
        dB = B[ii + 1] - B[ii - 1]
        ds = s[ii + 1] - s[ii - 1]
        grad[ii] = dB/ds
        
    # Edges:
    grad[0]  = (-3 * B[0] + 4 * B[1] - B[2]) / (s[2] - s[0])
    grad[-1] = (B[-1] - 4*B[-2] + 3*B[-3]) / (s[-3] - s[-1])
    return grad

def get_S_lims(S_arr, yarr, lim=1.0):
    '''
    Scans through 2D array of time vs position with cvals of S. Find indices
    closest to +/- 1 (or other value
    '''
    pos_lim = np.zeros(S_arr.shape[0])
    neg_lim = np.zeros(S_arr.shape[0])
    for xx in range(S_arr.shape[0]):
        # Filter out nan's, replace with zeros
        Sarr_filt = S_arr[xx].copy()
        Sarr_filt[np.isnan(Sarr_filt) == True] = 1e10
        
        pos_idx   = np.where(abs(Sarr_filt - lim) == abs(Sarr_filt - lim).min())
        neg_idx   = np.where(abs(Sarr_filt + lim) == abs(Sarr_filt + lim).min())
        
        pos_lim[xx] = yarr[pos_idx]
        neg_lim[xx] = yarr[neg_idx]
    return pos_lim, neg_lim

def create_species_array(B0, name, mass, charge, density, tper, ani):
    '''
    For each ion species, total density is collated and an entry for 'electrons' added (treated as cold)
    Also output a PlasmaParameters dict containing things like alfven speed, density, hydrogen gyrofrequency, etc.
    
    Inputs must be in SI units: nT, kg, C, /m3, eV, etc.
    ''' 
    nsp       = name.shape[0]
    ne        = density.sum()
    
    t_par = np.zeros(nsp); alpha_par = np.zeros(nsp)
    for ii in range(nsp):
        t_par[ii] = qp*tper[ii] / (ani[ii] + 1)
        alpha_par[ii] = np.sqrt(2.0 * t_par[ii]  / mass[ii])
    
    # Create initial fields
    Species = np.array([], dtype=[('name', 'U20'),          # Species name
                                  ('mass', 'f8'),           # Mass in kg
                                  ('density', 'f8'),        # Species density in /m3
                                  ('tper', 'f8'),           # Perpendicular temperature in eV
                                  ('anisotropy', 'f8'),     # Anisotropy: T_perp/T_par - 1
                                  ('plasma_freq_sq', 'f8'), # Square of the plasma frequency
                                  ('gyrofreq', 'f8'),       # Cyclotron frequency
                                  ('vth_par', 'f8')])       # Parallel Thermal velocity

    # Insert species values into each
    for ii in range(nsp):
        if density[ii] != 0.0:
            new_species = np.array([(name[ii], mass[ii], density[ii], tper[ii], ani[ii],
                                                    density[ii] * charge[ii] ** 2 / (mass[ii] * e0),
                                                    charge[ii]  * B0 / mass[ii],
                                                    alpha_par[ii])], dtype=Species.dtype)
            Species = np.append(Species, new_species)
    
    # Add cold electrons
    Species = np.append(Species, np.array([('Electrons', me, ne, 0, 0,
                                            ne * qp ** 2 / (me * e0),
                                            -qp  * B0 / me,
                                            0.)], dtype=Species.dtype))
    
    PlasParams = {}
    PlasParams['va']       = B0 / np.sqrt(mu0*(density * mass).sum())  # Alfven speed (m/s)
    PlasParams['n0']       = ne                                        # Electron number density (/m3)
    PlasParams['pcyc_rad'] = qp*B0 / mp                                # Proton cyclotron frequency (rad/s)
    PlasParams['B0']       = B0                                        # Magnetic field value (T)
    return Species, PlasParams

def define_omura2010_parameters(B0):
    '''
    Ambient plasma parameters from Omura et al. (2010) to recreate plots
    '''
    name    = np.array(['H'    , 'He'  , 'O' ])
    mass    = np.array([1.0    , 4.0   , 16.0]) * mp
    charge  = np.array([1.0    , 1.0   , 1.0 ]) * qp
    density = np.array([144.0  , 17.0  , 17.0]) * 1e6
    ani     = np.array([0.0    , 0.0   , 0.0 ])
    tpar    = np.array([0.0    , 0.0   , 0.0 ])
    tper    = (ani + 1) * tpar
    
    Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
    return Species, PP

def get_k_cold(w, Species, mm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cold_sum = 0.0
        for ii in range(Species.shape[0]):
            sum_contr = Species[ii]['plasma_freq_sq'] / (w * (w - Species[ii]['gyrofreq']))
            cold_sum += sum_contr
    
        k = np.sqrt(1 - cold_sum) * w / c
        
    return k

def get_gamma_c(w, Species):
    # Electron bit (gyfreq is signed, so this is a minus)
    cold_sum = Species[-1]['plasma_freq_sq'] / Species[-1]['gyrofreq']
    
    # Sum over ion species
    for ii in range(Species.shape[0] - 1):
        cold_sum += Species[ii]['plasma_freq_sq'] / (Species[ii]['gyrofreq'] - w)
    return cold_sum

def get_group_velocity(w, k, Species):
    gam_c = get_gamma_c(w, Species) 
    
    # Ions only?
    ion_sum = 0.0
    for ii in range(Species.shape[0] - 1):
        ion_sum += Species[ii]['plasma_freq_sq'] / (Species[ii]['gyrofreq'] - w) ** 2
        
    denom = gam_c + w*ion_sum
    Vg    = 2 * c * c * k / denom
    return Vg

def get_resonance_velocity(w, k, Species, PP):
    Vr = (w - PP['pcyc_rad']) / k
    return Vr

def get_phase_velocity(w, k, Species):
    return w / k

def get_velocities(w, k, Species, PP):
    Vg = get_group_velocity(w, k, Species)
    Vr = get_resonance_velocity(w, k, Species, PP)
    Vp = get_phase_velocity(w, k, Species)
    return Vg, Vp, Vr

def get_S_dipole(L_shell, dB=0.0, Bw=0.5e-9, f=1.5, N_field=1e5):
    '''
    Define a Bw and w in a specified plasma regime.
        
    Stanard hc range around 320km with only low amplitude triggering wave
    3700km by the time the non-linear growth has maximized and the wave has grown
    
    Note: Currently not changing density or cold composition along field line,
    could do!
    
    Because plasma doesn't '
    '''
    # Main inputs
    dwdt     = 0*2 * np.pi * (1.2/48)      # Frequency sweep rate (rad/s2) 1.2Hz in 48sec
    vth_perp = 8e5                         #  m/s    
    w        = 2*np.pi*f                   # rad/s
    omw      = qp * Bw / mp                # 'Gyrofrequency' effect of wave field
    
    # Field parameters
    s, Bs, mlat = new_trace_fieldline(L_shell, Npts=N_field)
    Bs         += dB
    B_grad      = get_B_gradient(Bs, s)
    Om_grad     = qp / mp * B_grad
    
    # Output arrays
    Vr_array    = np.zeros(s.shape[0])
    wtr_array   = np.zeros(s.shape[0])
    Vtr_array   = np.zeros(s.shape[0])
    k_array     = np.zeros(s.shape[0])
    S_ratio     = np.zeros(s.shape[0])
    
    # Calculate inhomogeneity ratio at each point along field
    for ii in range(s.shape[0]):
        Species, PP = define_omura2010_parameters(Bs[ii])
        
        # Wave parameters
        k_array[ii]   = get_k_cold(w, Species, ii)            # /m ??
        pcyc          = PP['pcyc_rad']                    # Proton cyclotron frequency
        wtr_array[ii] = np.sqrt(k_array[ii]*vth_perp*omw) # Trapping frequency
        Vtr_array[ii] = 2 * wtr_array[ii] / k_array[ii]   # Trapping velocity
        
        Vg, Vp, Vr_array[ii] = get_velocities(w, k_array[ii], Species, PP)
        
        # Define s-values
        s0 = vth_perp / Vp
        s1 = (1.0 - Vr_array[ii]/Vg) ** 2
        s2 = (vth_perp**2/(2*Vp**2) + Vr_array[ii]**2 / (Vp*Vg) - Vr_array[ii]**2/(2*Vp**2))*(w / pcyc) - Vr_array[ii]/Vp
        
        S_ratio[ii] = 1 / (s0 * w * omw) * (s1*dwdt + Vp*s2*Om_grad[ii])
        
        #if np.isnan(S_ratio[ii]) == True and abs(mlat[ii]) < 10.0:
#            pdb.set_trace()
        
    return s, mlat, S_ratio, Vr_array, wtr_array, Vtr_array, k_array


if __name__ == '__main__':    
    main_folder = 'F://NONLINEAR//'
    
    L = 4.27

    # Check 
    if True:
        S, MLAT, S_RAT, VR, WTR, VTR, K = get_S_dipole(L, dB=-20e-9,
                                                       Bw=0.5e-9, f=1.5,
                                                       N_field=1e3)
        
        plt.figure()
        plt.plot(S/1e3, S_RAT)
        plt.ylabel('S ratio')
        plt.xlabel('s (km)')
        plt.ylim(-1, 1)
        
        plt.figure()
        plt.plot(MLAT, S_RAT)
        plt.ylabel('S ratio')
        plt.xlabel('MLAT')
        plt.ylim(-1, 1)

    if False:
        # Number of points along field line
        nB = 1e3
        
        # Wave params
        pc5_ampl = 20.0e-9  # nT
        pc5_freq = 5.0      # mHz
        
        # Just do one period
        t_max    = 1000./pc5_freq                 # Max time (s)
        Nt       = 100                            # Time cadence (s)
        t        = np.linspace(0.0, t_max, Nt)    # Time array
        
        pc5_wave = pc5_ampl * np.sin(2 * np.pi * pc5_freq/1000. * t)
        
        S_RAT_ALL = np.zeros((t.shape[0], int(nB+1)), dtype=float)
        VR_ALL    = np.zeros((t.shape[0], int(nB+1)), dtype=float)
        
        # Not super accurate: Compression would shorten field line?
        # This assumes same "L/MLAT" in space and just heightens B-field
        for mm in range(t.shape[0]):
            print('Doing t = {}s'.format(t[mm]))
            S, MLAT, S_RAT_ALL[mm], VR_ALL[mm], WTR, VTR, K = \
                get_S_dipole(L, dB=pc5_wave[mm], Bw=0.5e-9, f=1.5, N_field=nB)
        
        pos_lim, neg_lim = get_S_lims(S_RAT_ALL, MLAT, lim=1.0)
        
        VR_ALL[VR_ALL > 3e8] = np.nan
        
        fig, ax = plt.subplots(3, sharex=True)
        
        im1 = ax[0].pcolormesh(t, MLAT, S_RAT_ALL.T,
                      norm=colors.SymLogNorm(linthresh=1.0,
                      vmin=-1e3, vmax=1e3, base=10), cmap='bwr',
                      shading='auto')
        
        ax[0].plot(t, pos_lim, c='k')
        ax[0].plot(t, neg_lim, c='k')
    
        ax[0].set_title('S along field line :: L = {} :: Pc5 {}nT at {}mHz'.format(L, pc5_ampl*1e9, pc5_freq))
        ax[1].set_xlabel('Time (s)')
        ax[0].set_ylabel('MLAT\n(deg)', rotation=0)
        #cbar = plt.colorbar(im1, orientation='vertical', extend='both')
        #cbar.set_label('S', rotation=0)
        
        ax[1].plot(t, pc5_wave*1e9)
        ax[1].set_ylabel('$\delta B_{HM}$\n(nT)', rotation=0)
        
        im1 = ax[2].pcolormesh(t, MLAT, VR_ALL.T, cmap='jet',
                      shading='auto')
        ax[2].set_ylabel('$V_r$', rotation=0)
        
    if False:
        # Plot phase space for each position along the field (only for S < 1.5)
        run_folder = 'standard_dipole//'
        save_dir   = main_folder + run_folder
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        
        # Assume there's +/- S on either side of the equator
        S_lim  = 1.5
        st_idx = np.where(np.abs(S_lim - S_RAT) == np.abs(S_lim - S_RAT).min())[0][0]
        en_idx = np.where(np.abs(S_lim + S_RAT) == np.abs(S_lim + S_RAT).min())[0][0]
        
        # Don't un-normalize the const calculation yet, probably won't change anything
        skip = 5;   counter = 0
        for mm in range(st_idx, en_idx):
            if counter%skip == 0:
                print('Plotting {} of {}'.format(mm - st_idx, en_idx - st_idx))
                zeta   = np.linspace(0, 2*np.pi, 250)
                theta  = np.linspace(-2, 2, 500) 
                ze, th = np.meshgrid(zeta, theta)
                
                # Calculate curves of constancy (njit this?)
                const = np.zeros(th.shape)
                for ii in range(th.shape[0]):
                    for jj in range(th.shape[1]):
                        const[ii, jj] = th[ii, jj]**2 + 0.5 * (np.cos(ze[ii, jj]) - S_RAT[mm] * ze[ii, jj])
                
                # Work out const of separatrix
                ze0         =   np.pi + np.arcsin(np.abs(S_RAT[mm]))
                ze1         = 2*np.pi - np.arcsin(np.abs(S_RAT[mm]))    
                separatrix  = 0.5*(np.cos(ze1) - S_RAT[mm]*ze1)
                
                # Plot curves
                plt.ioff()
                fig, ax = plt.subplots(figsize=(16, 10))
                ax.contour(ze, th, const, levels=75, colors='k', linestyles='-', alpha=0.5)
                ax.contour(ze, th, const, levels=[separatrix], colors='k', linestyles='-')
                
                ax.contourf(ze, th, const, levels=np.array([separatrix, const.max()]), colors='grey')
                ax.axvspan(ze1, 2*np.pi, color='grey')
                
                ax.set_xlabel('$\zeta$', fontsize=20)
                ax.set_ylabel('$\\frac{\\theta}{2 \omega_{tr}}$', rotation=0, labelpad=20, fontsize=20)
                ax.set_title('Dipole Resonant Proton Phase Space :: S = %.2f :: L = %.2f :: MLAT = %.2f$^{\circ}$ :: s = %.0f km'
                             % (S_RAT[mm], L, MLAT[mm], S[mm]/1e3), fontsize=14)
                
                fig.savefig(save_dir + 'phase_space_{:06}.png'.format(mm))
                plt.close('all')
            counter += 1