# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:52:48 2020

@author: Yoshi

Note: This is a super inefficient solve-as-you-go script (e.g. I solve for k(w)
about 10 times in some places). Can always optimize later.
"""
import sys, pdb
import numpy as np
import matplotlib.pyplot as plt
from   scipy.special     import wofz

sys.path.append('..//new_general_DR_solver//')

from dispersion_solver_multispecies import create_species_array, dispersion_relation_solver

def get_k_cold(w, Species):
    '''
    Calculate the k of a specific angular frequency w in a cold multicomponent plasma
    '''
    cold_sum = 0.0
    for ii in range(Species.shape[0]):
        cold_sum += Species[ii]['plasma_freq_sq'] / (w * (w - Species[ii]['gyrofreq']))

    k = np.sqrt(1 - cold_sum) * w / c
    return k


def get_gamma_c(w, Species):
    # Electron bit (gyfreq is signed, so this is a minus)
    cold_sum = Species[-1]['plasma_freq_sq'] / Species[-1]['gyrofreq']
    
    # Sum over ion species
    for ii in range(Species.shape[0] - 1):
        cold_sum += Species[ii]['plasma_freq_sq'] / (Species[ii]['gyrofreq'] - w)
    return cold_sum


def get_cold_dispersion_omura(w, Species):
    '''
    UNFINISHED
    '''
    k     = get_k_cold(w, Species)
    gam_c = get_gamma_c(w, Species)
    return


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


def get_inhomogeneity_terms(w, Species, PP, Vth_perp):
    '''
    Validated about as much as possible. Results are dimensionless since 
    velocities and frequencies cancel out. Hence, normalizations also shouldn't
    matter but this function assumes no normalization (could be set based on 
    whether Vth_perp was, but it doesn't matter.
    '''
    # Third term in bracks: - Vr**2/(2*Vp**2)
    Vg, Vp, Vr = get_velocities(w, Species, PP)     # Each length w
    pcyc       = PP['pcyc_rad']
    
    s0 = Vth_perp / Vp
    s1 = (1 - Vr/Vg) ** 2
    s2 = (Vth_perp**2/(2*Vp**2) + Vr**2 / (Vp*Vg) - Vr**2/(2*Vp**2))*(w / pcyc) - Vr/Vp
    return np.array([s0, s1, s2])


def get_velocities(w, Species, PP, normalize=False):
    k  = get_k_cold(w, Species)
    Vg = get_group_velocity(w, k, Species)
    Vr = get_resonance_velocity(w, k, Species, PP)
    Vp = get_phase_velocity(w, k, Species)
    
    if normalize == False:
        return Vg, Vp, Vr
    else:
        return Vg/c, Vp/c, Vr/c


def nonlinear_growth_rate(w, Species, PP, nh, Q, Vth_para, Vth_perp, Bw):
    '''
    Can return non-linear growth rate as a function of frequency, but each
    velocity then needs to also be a function of frequency.
    Hot proton stuff can be incorporated into the "Species" array later, maybe
    
    Need to check: 
        --- Are these really meant to be the thermal velocities?
        --- How valid is getting k like this? Is there a way to do better but stay consistent?
        --- NOTE: This only assumes energy from hot protons - i.e. considers only one resonant species.
        
    # For the Non-linear growth rate, need:
    # -- w_ph     :: Plasma frequency but with hot proton density only? (equiv in Kozyra)
    # -- Q factor :: Describes depletion of trapping region (arbitrary??)
    # -- Omega_w  :: Cyclotron frequency but with Bw only (specific to event)
    # -- V_phase  :: Known from CPDR? Maybe. Just divide solution (w) at each point by the x-value (k) at that point
    # -- V_group  :: Also known from CPDR and eqn
    # -- V_res    :: Resonant velocity, also from CPDR
    # -- V_perp0  :: Perpendicular velocity of proton around field line (Is this a thermal distro too?)
    # -- V_t\para :: Parallel thermal velocity??
    #
    # How does this work? Bw is an argument of Gamma_NL, but obviously NL is going to increase
    # Bw. Do we step through in time and solve it numerically like that? (i.e. integrate it?)
    # Also, if I just pick an initial Bw, does that mean I just get Bw as a function of frequency?
    # Or if I specify a frequency and get just a single value back (for the integration stuff??)
    #
    # Normalize=True could be used and c terms removed from equation
    '''
    wph2 = nh * qp ** 2 / (mp * e0)                 # Constant
    wc_w = qp * Bw / mp                             # Constant
    
    Vg, Vp, Vr = get_velocities(w, Species, PP)     # Each length w

    t1 = 0.5 * wph2 * Q
    t2 = np.sqrt(Vp / (c*wc_w*w)) * Vg / Vth_para
    t3 = (Vth_perp / (c*np.pi)) ** (3/2) * np.exp(- Vr**2 / (2*Vth_para**2))
    return t1 * t2 * t3


def Z(arg):
    '''Return Plasma Dispersion Function : Normalized Fadeeva function'''
    return 1j*np.sqrt(np.pi)*wofz(arg)


def linear_growth_rates(w, Species):
    '''
    Linear theory bit from Chen. Should be able to support multiple hot and
    cold species in the Species array, but the Omura functions probably can't.
    Might lead to some weirdness, just something to be aware of.
    '''
    # Get k for each frequency to evaluate
    k  = get_k_cold(w, Species)
    
    # Calculate Dr/k_para
    w_der_sum = 0.0
    k_der_sum = 0.0
    Di        = 0.0
    for ii in range(Species.shape[0]):
        sp = Species[ii]
        
        # If cold
        if sp['tper'] == 0:
            w_der_sum += sp['plasma_freq_sq'] * sp['gyrofreq'] / (w - sp['gyrofreq'])**2
            k_der_sum += 0.0
            Di        += 0.0
        
        # If hot
        else:
            zs           = (w - sp['gyrofreq']) / (sp['vth_par']*k)
            Yz           = np.real(Z(zs))
            dYz          = -2*(1 + zs*Yz)
            A_bit        = (sp['anisotropy'] + 1) * w / sp['gyrofreq']
            
            # Calculate frequency derivative of Dr (sums bit)
            w_der_outsd  = sp['plasma_freq_sq']*sp['gyrofreq'] / (w*k*sp['vth_par'])
            w_der_first  = A_bit * Yz
            w_der_second = (A_bit - sp['anisotropy']) * w * dYz / (k * sp['vth_par']) 
            w_der_sum   += w_der_outsd * (w_der_first + w_der_second)
    
            # Calculate Di (sums bit)
            Di_bracket = 1 + (sp['anisotropy'] + 1) * (w - sp['gyrofreq']) / sp['gyrofreq']
            Di_after   = sp['gyrofreq'] / (k * sp['vth_par']) * np.sqrt(np.pi) * np.exp(- zs ** 2)
            Di        += sp['plasma_freq_sq'] * Di_bracket * Di_after
    
            # Calculate wavenumber derivative of Dr (sums bit)
            k_der_outsd  = sp['plasma_freq_sq']*sp['gyrofreq'] / (w*k*k*sp['vth_par'])
            k_der_first  = A_bit - sp['anisotropy']
            k_der_second = Yz + zs * dYz
            k_der_sum   += k_der_outsd * k_der_first * k_der_second
    
    # Get and return ratio
    Dr_wder = 2*w + w_der_sum
    Dr_kder = -2*k*c**2 - k_der_sum

    temporal_growth_rate   = - Di / Dr_wder
    group_velocity         = - Dr_kder / Dr_wder
    convective_growth_rate = - temporal_growth_rate / np.abs(group_velocity)
    return temporal_growth_rate, convective_growth_rate, group_velocity, k


def get_stop_bands(k_cold, pythonic_range=False):
    '''
    Grabs the start and stop indices for each stop band by detecting NaN's in
    k values.
    
    idx_start :: Inclusive (i.e. these indices are in the stop band
    idx_end   :: Exclusive (i.e. these indices are NOT in the stop band) if pythonic_range=False
           else  Inclusive
    
    That way range() and slicing calls work the way Python intended.
    '''
    idx_start = []; idx_end = []; st = -1
    for ii in range(k_cold.shape[0]):
        if np.isnan(k_cold[ii]) == True:
            # Start detected
            if st < 0:
                st = ii
                idx_start.append(st)
        else:
            # End detected
            if st >= 0:
                if pythonic_range == True:
                    idx_end.append(ii)
                else:
                    idx_end.append(ii-1)
                st = -1
    
    # If array ended on nan
    if st > 0:
        if pythonic_range == True:
            idx_end.append(None)
        else:
            idx_end.append(ii)
    return idx_start, idx_end


# Things to recreate (from Omura et al., 2010)
#
# -- Get cut-off frequencies via eq. (65) with k = 0, i.e. solve \Gamma_c = 0
#     >>> Generally the cutoff is read from a spectrum and used to infer potential cold compositions
#
# -- How to get some of those homogeneity graphs?
#     >>> Need to be able to calculate/measure V_R, V_P, V_G, etc. by solving eq. (33) and (23) with frequency
#     >>> How do they get k? CPDR rearranged for k
#
# Important values
# Satellite observation of cutoff frequency: ~1.3Hz
#                       cyclotron frequency: ~3.7Hz
# Densities of helium, oxygen are nearly equal
#
# v_perp :: 800km/s
# v_para :: 600km/s
# Q = 0.5
# nh = 0.05*nH (5% of the cold proton density?)
# 
# Equations seem to assume that what the satellite measures is the equatorial values?
# Also very lousy distinctions between variables and their '0' versions (e.g. OmH vs. OmH0)
#
    
def omega_equation(Om, w, Vth_para, Vth_perp, Vg, Vp, Vr, s, Q, wph2, a):
    '''
    Value of the derivative expressed in eq. 63.  SEEMS OK? HOW TO VERIFY?
    '''
    nob = Q*wph2/(2*Vth_para)                       # No bracket (first "term")
    br1 = (Vth_perp / np.pi) ** (3/2)               # Bracket 1
    br2 = np.sqrt(Vp * Om / w)                      # Bracket 2
    exp = np.exp(-0.5 * Vr ** 2 / Vth_para ** 2)    # Exponential bit
    lat = 5 * Vp * s[2] * a / (s[0] * w)            # Last Term
    return (Vg * (nob * br1 * br2 * exp - lat))


def push_Om(Om_in, w, Vth_para, Vth_perp, Vg, Vp, Vr, s, Q, wph2, a, dt):
    '''
    Solution to eq. 63 via RK4 method. SEEMS OK? HOW TO VERIFY?
    '''
    k1 = omega_equation(Om_in          , w, Vth_para, Vth_perp, Vg, Vp, Vr, s, Q, wph2, a)
    k2 = omega_equation(Om_in + dt*k1/2, w, Vth_para, Vth_perp, Vg, Vp, Vr, s, Q, wph2, a)
    k3 = omega_equation(Om_in + dt*k2/2, w, Vth_para, Vth_perp, Vg, Vp, Vr, s, Q, wph2, a)
    k4 = omega_equation(Om_in + dt*k3  , w, Vth_para, Vth_perp, Vg, Vp, Vr, s, Q, wph2, a)
    
    Om_out = Om_in + (1/6) * dt * (k1 + 2*k2 + 2*k3 + k4)
    return Om_out


def push_w(w_old, Om, s, dt):
    '''
    Solution to eq. 64 via finite difference. SEEMS OK? HOW TO VERIFY?
    '''
    K1 = 0.4 * s[0] / s[1]
    Z  = 0.5 * dt * K1 * Om
    
    w_new = w_old * (1 + Z) / (1 - Z)
    return w_new


def get_threshold_value_normalized(w, wph2, Q, s, a, Vp, Vr, Vth_para, Vth_perp):
    '''
    This is mega sensitive to initial frequency
    '''
    t1 = 100 * (np.pi * Vp) ** 3 / (w * wph2 ** 2 * Vth_perp ** 5)
    t2 = (a * s[2] * Vth_para / Q) ** 2
    t3 = np.exp((Vr / Vth_para)**2)
    return t1 * t2 * t3


def get_threshold_value(w, Species, PP, nh, a, Q, Vth_para, Vth_perp, Bw):
    '''None of these values are normalized'''
    Vg, Vp, Vr = get_velocities(w, Species, PP)
    s          = get_inhomogeneity_terms(w, Species, PP, Vth_perp)
    NL         = nonlinear_growth_rate(w, Species, PP, nh, Q, Vth_para, Vth_perp, Bw)
    
    Om_thresh  = 5*Vp*a*s[2]*PP['pcyc_rad']*Vg / (s[0]*w*NL)
    Bw_thresh  = 1e9*Om_thresh*mp / qp
    return Bw_thresh


def get_threshold_value_UNIO(w, Species, PP, nh, Vth_para, Vth_perp, Q, L):
    '''
    Un-normalized input, converted to un-normalized output
    '''
    # Un-normalized values
    #B0         = PP['B0']
    pcyc       = PP['pcyc_rad']
    Vg, Vp, Vr = get_velocities(w, Species, PP)
    S          = get_inhomogeneity_terms(w, Species, PP, Vth_perp)
    wph        = np.sqrt(nh * qp ** 2/ (mp * e0))
    a          = 4.5 / (L*RE)**2
    
    # Normalizations
    VPN        = Vp/c
    VRN        = Vr/c
    WN         = w/pcyc 
    WPHN       = wph/pcyc
    AN         = a * c**2 / pcyc ** 2
    VTH_PERP_N = Vth_perp/c
    VTH_PARA_N = Vth_para/c

    Bth  = np.zeros(w.shape[0])
    for NN in range(w.shape[0]):
        # Calculate threshold
        t1      = 100 * (np.pi * VPN[NN]) ** 3 / (WN[NN] * WPHN ** 4 * VTH_PERP_N ** 5)
        t2      = (AN * S[2, NN] * VTH_PARA_N / Q) ** 2
        t3      = np.exp((VRN[NN] / VTH_PARA_N)**2)
        OMTH    = t1 * t2 * t3
        Bth[NN] = OMTH
    return Bth


def solve_coupled_equations(Species, PP, init_f, init_Bw, L, nh, Q, Vth_para, Vth_perp):
    '''
    Each time the frequency changes, the velocities will change (changing the s factors).
    Constants will be:
        wph2, a, pcyc
        
    -- w will have to be un-normalized before querying velocities or s-terms. So will Vth_perp.
    -- What's the limit on dt in this case? Well-resolving the frequencies I guess? 
        --- Just set to some really small value for now, fiddle later
        
    Initial frequency in Hz
    Initial wave magnitude in T
    
    Interesting : "Threshold" value not attained by initial Bw? Could be mistake, but not sure where
    --- Could the cyclotron frequency be wrong? Is it meant to be in rad/s ?
    --- Must come down to a problem in a computed quantity: E.g. V, s, etc.
    '''
    # Constants:
    B0        = PP['B0']                                 # Equatorial magnetic field
    pcyc_eq   = PP['pcyc_rad']                           # Equatorial proton cyclotron frequency
    wph2      = nh * qp ** 2 / (mp * e0)                 # Hot proton plasma frequency squared (normalized)
    a         = 4.5 / ((L * RE)**2)                      # Parabolic magnetic field scale factor
    w_init    = 0.4*pcyc_eq #2 * np.pi * init_f

    Bw_thresh = get_threshold_value(w_init, Species, PP, nh, a, Q, Vth_para, Vth_perp, init_Bw)
    print('Threshold |Bw|: {:>6.2f} nT'.format(Bw_thresh))

    # Normalizations
    a         = a * ((c/pcyc_eq) ** 2)                   # Parabolic magnetic field scale factor (normalized)
    Vth_para  = Vth_para / c
    Vth_perp  = Vth_perp / c
    wph2      = wph2/(pcyc_eq**2)

    # Define time variables (in seconds) and intialize arrays with normalized initial values
    # Time units aren't critical, its the saturation values/frequencies (regardless of how "long" they take)
    t_max = 120.                                        # Max time  (s)
    dt    = 0.01                                        # Time step (s)
    t_arr = np.arange(0.0, t_max*pcyc_eq, dt*pcyc_eq)   # Normalized to cyclotron frequency
    
    w_arr  = np.zeros(t_arr.shape[0], dtype=float)
    Om_arr = np.zeros(t_arr.shape[0], dtype=float) 
    
    w_arr[0]  = w_init  / pcyc_eq
    Om_arr[0] = init_Bw / B0
    
    ## TEST STUFF :: CAN DELETE
    Species[0]['plasma_freq_sq'] = (679.0*pcyc_eq) ** 2
    Species[1]['plasma_freq_sq'] = (117.0*pcyc_eq) ** 2
    Species[2]['plasma_freq_sq'] = (58.30*pcyc_eq) ** 2
    ## UP TO HERE
    
    # Get initial parameters, solve for threshold
    Vg, Vp, Vr = get_velocities(w_arr[0]*pcyc_eq, Species, PP, normalize=True)
    s          = get_inhomogeneity_terms(w_arr[0]*pcyc_eq, Species, PP, Vth_perp*c)
    
    # Still can't get this to be right - still comes out at ~1nT (1.087nT) instead of 0.48nT
    Om_thresh_norm = get_threshold_value_normalized(w_arr[0], wph2, Q, s, a, Vp, Vr, Vth_para, Vth_perp)
    Bw_thresh_norm = Om_thresh_norm*B0*1e9
    print('Threshold |Bw|: {:>6.2f} nT'.format(Bw_thresh_norm*B0*1e9))
    #return None, None, None
    pdb.set_trace()
    
    # Retard initial w soln to N - 1/2 (overwriting w[0]):
    w_arr[0]   = push_w(w_arr[0], Om_arr[0], s, -0.5*dt) 
    s          = get_inhomogeneity_terms(w_arr[0]*pcyc_eq, Species, PP, Vth_perp*c)
    
    # Leapfrog LOOP
    for ii in range(1, t_arr.shape[0]):
        # Push w, re-solve functions of w
        w_arr[ii]  = push_w(w_arr[ii - 1], Om_arr[ii - 1], s, dt) 
        s          = get_inhomogeneity_terms(w_arr[ii]*pcyc_eq, Species, PP, Vth_perp*c)
        Vg, Vp, Vr = get_velocities(w_arr[ii]*pcyc_eq, Species, PP, normalize=True)
        
        # Check saturation (THIS PART IS UNCERTAIN. WHY TIME-VARYING LIMIT??)
        # Maybe incorporate check into whether or not to solve as above?
        Om_limit   = (Vp / (4*Vth_perp)) * ((1 - w_arr[ii]) ** 2) / w_arr[ii]
        if Om_arr[ii - 1] > Om_limit:
            Om_arr[ii] = Om_arr[ii - 1]
        else:
            Om_arr[ii] = push_Om(Om_arr[ii - 1], w_arr[ii], Vth_para, Vth_perp, Vg, Vp, Vr, s, Q, wph2, a, dt)
    
    # Un-normalize solutions and return plottable values
    f_arr  = (w_arr * pcyc_eq) / (2 * np.pi)
    Bw_arr = Om_arr * B0
    t_arr  = t_arr / pcyc_eq
    return t_arr, f_arr, Bw_arr


    
if __name__ == '__main__':
    qp = 1.602e-19
    qe =-1.602e-19
    mp = 1.673e-27
    me = 9.110e-31
    e0 = 8.854e-12
    mu0= 4e-7*np.pi
    RE = 6.371e6
    c  = 3e8
    kB     = 1.380649e-23
    B_surf = 3.12e-5                            # Magnetic field strength at Earth surface (equatorial)
    
    
    # Parameters from Omura et al. (2010)
    
    # Plasma and energetic parameters
    pcyc     = 3.7 # Hz
    Th_para  = (mp * (6e5)**2 / kB) / 11603.
    Th_perp  = (mp * (8e5)**2 / kB) / 11603.
    Ah       = Th_perp / Th_para - 1
    apar_h   = np.sqrt(2.0 * qp * Th_para  / mp)
    
    # Parameters in SI units (Note: Added the hot bit here. Is it going to break anything?) nh = 7.2
    _B0      = 2 * np.pi * mp * pcyc / qp
    _name    = np.array(['H'    , 'He'  , 'O' ])
    _mass    = np.array([1.0    , 4.0   , 16.0]) * mp
    _charge  = np.array([1.0    , 1.0   , 1.0 ]) * qp
    _density = np.array([144.0  , 17.0  , 17.0]) * 1e6
    _tpar    = np.array([0.0    , 0.0   , 0.0 ])
    _ani     = np.array([0.0    , 0.0   , 0.0 ])
    _tper    = (_ani + 1) * _tpar
    
    _Species, _PP = create_species_array(_B0, _name, _mass, _charge, _density, _tper, _ani)

    
    # Shoji et al. (2012) recreation of 2D plot
    #   --- Varies with normalized frequency (to pcyc) and ion density (plasma freq. proxy)
    #   --- Need to get Bth first, then NL growth rate calculated with Bw = Bth
    #   --- BTH ALMOST VALIDATED. Min growth region looked a little off, and weird nan's at bottom left.
    #       -- NOPE, H band completely different (way lower in higher density lower frequency section). Error?
    if False:
        Nw   = 500
        _B0  = 243e-9   # T 
        pcyc = 23.2     # rad/s 
        ecyc = 4.27e4   # rad/s
        
        Nr          = 500
        wpe_wce_max = 17    # Ratio max
        
        # Frequency axis
        pcyc_min = 0.375
        pcyc_max = 1.0
        w_axis   = np.linspace(pcyc_min, pcyc_max, Nw)
        _w       = w_axis * pcyc
        
        # Density axis 
        wpe_wce  = np.linspace(0.0, wpe_wce_max, Nr)
        ne       = wpe_wce ** 2 * _B0 ** 2 * e0 / me
        
        # Energetic plasma parameters
        _Q        = 0.5
        _Vth_para = 0.00800*c
        _Vth_perp = 0.01068*c
        
        # Other parameters
        _L        = 4.0
        _Bw_init  = 0.5e-9
        
        # Cold plasma parameters
        _name    = np.array(['H'    , 'He'  , 'O'   ])
        _mass    = np.array([1.0    , 4.0   , 16.0  ]) * mp
        _charge  = np.array([1.0    , 1.0   , 1.0   ]) * qp
        _tpar    = np.array([0.0    , 0.0   , 0.0   ])
        _ani     = np.array([0.0    , 0.0   , 0.0   ])
        _tper    = (_ani + 1) * _tpar
    
        _BTH = np.zeros((Nr, Nw), dtype=float)
        for MM in range(Nr):
            _nh           = 0.0081 * ne[MM]
            _density      = np.array([0.8019 , 0.0950, 0.0950]) * ne[MM]
            _Species, _PP = create_species_array(_B0, _name, _mass, _charge, _density, _tper, _ani)

            _BTH[MM] = get_threshold_value_UNIO(_w, _Species, _PP, _nh, _Vth_para, _Vth_perp, _Q, _L)
        
        
        fig, ax = plt.subplots()
        
        im1 = ax.pcolormesh(w_axis, wpe_wce, _BTH, cmap='jet', vmin=0.0, vmax=0.1)
        fig.colorbar(im1)
        
        
        #NL = nonlinear_growth_rate(_w, _Species, _PP, _nh, _Q, _Vth_para, _Vth_perp, _Bw_init)
    
    
    
    # Cold Plasma Dispersion Plot (Figure 3, validated)
    if False:
        wlength = 4e4
        kmax    = 2 * np.pi / wlength
    
        k_vals, CPDR_solns, warm_solns = dispersion_relation_solver(_Species, _PP, norm_k_in=False, norm_k_out=False, \
                                         norm_w=False, plot=False, kmin=0.0, kmax=kmax, Nk=1000)
        
        # Cast to linear frequency
        CPDR_solns /= 2 * np.pi
        
        fig, ax = plt.subplots()
        for ss in range(CPDR_solns.shape[1]):
            ax.plot(1e3*k_vals / (2*np.pi), CPDR_solns[:, ss], c='k')
            
        ax.set_xlim(0, 0.025)
        ax.set_ylim(0, 3.5)
        
        ax.set_title('Cold Plasma Dispersion Relation')
        ax.set_xlabel('f (Hz)', fontsize=14)
        ax.set_ylabel(r'$\frac{1}{\lambda}$ ($m^{-1}$)', rotation=0, fontsize=14, labelpad=30)
        
        
    
    # Resonance and Group Velocities plot (Figure 4a,b validated)
    if False:
        f_max  = 4.0
        f_vals = np.linspace(0.0, f_max, 10000)
        w_vals = 2 * np.pi * f_vals
        
        V_group, V_phase, V_resonance = get_velocities(w_vals, _Species, _PP)

        fig, ax = plt.subplots()
        ax.plot(f_vals, V_resonance/1e3)
        ax.set_xlabel('f (Hz)', fontsize=14)
        ax.set_ylabel('$V_R$\nkm/s', rotation=0, fontsize=14, labelpad=30)
        ax.set_xlim(0, f_max)
        ax.set_ylim(-1500, 0)

        fig, ax = plt.subplots()
        ax.plot(f_vals, V_group/1e3)
        ax.set_xlabel('f (Hz)', fontsize=14)
        ax.set_ylabel('$V_g$\nkm/s', rotation=0, fontsize=14, labelpad=30)
        ax.set_xlim(0, f_max)
        ax.set_ylim(0, 250)
        
        fig, ax = plt.subplots()
        ax.plot(f_vals, V_phase/1e3)
        ax.set_xlabel('f (Hz)', fontsize=14)
        ax.set_ylabel('$V_p$\nkm/s', rotation=0, fontsize=14, labelpad=30)
        ax.set_xlim(0, f_max)
        #ax.set_ylim(0, 250)
        
    
    # Non-linear growth rate (Works, looks good, but no validation)
    if False:
        _nh               = 0.05*_Species[0]['density']
        vth_parallel      = 6e5
        vth_perpendicular = 8e5
        Q_value           = 0.5
        Bw_init           = 0.5e-9
        
        f_max  = 4.0
        f_vals = np.linspace(0.0, f_max, 10000)
        w_vals = 2 * np.pi * f_vals
        
        Gamma_NL = nonlinear_growth_rate(w_vals, _Species, _PP, _nh, Q_value, vth_parallel, vth_perpendicular, Bw_init)
        
        fig, ax = plt.subplots()
        ax.plot(f_vals, Gamma_NL)
        ax.set_title('Non-linear Growth Rate by Frequency (Omura et al., 2010)')
        ax.set_xlabel('f (Hz)', fontsize=14)
        ax.set_ylabel('$\Gamma_{NL}$\n$(s^{-1})$', rotation=0, fontsize=14, labelpad=30)
        ax.set_xlim(0, f_max)
        #ax.set_ylim(0, 250)
        
        
    # Chen et al. (2013) :: Temporal and Convective Growth rates, group velocity
    # -- Group velocity test (Validated)
    # -- This also validates the derivative expressions in the function
    if False:
        f_max  = 4.0
        f_vals = np.linspace(0.0, f_max, 10000)
        w_vals = 2 * np.pi * f_vals
        
        V_group_omura, V_phase, V_resonance               = get_velocities(w_vals, _Species, _PP)
        temporal_GR, convective_GR, V_group_chen, k_cold  = linear_growth_rates(w_vals, _Species) 
        
        
        fig, ax = plt.subplots()
        ax.plot(f_vals, V_group_omura/1e3, label='Omura')
        ax.plot(f_vals, V_group_chen/1e3, label='Chen')
        ax.set_title('Group velocity: Chen vs. Omura comparison')
        ax.set_xlabel('f (Hz)', fontsize=14)
        ax.set_ylabel('$V_g$\nkm/s', rotation=0, fontsize=14, labelpad=30)
        ax.set_xlim(0, f_max)
        ax.set_ylim(0, 250)
        ax.legend()
        
    
    # Check convective growth rate - Not really any way to validate it, but
    # both Vg and the linear growth rate have been cross-validated
    # Hot proton quantities need to be in Species array (unlike other functions)
    if False:
        f_max  = 4.0
        f_vals = np.linspace(0.0, f_max, 10000)
        w_vals = 2 * np.pi * f_vals

        temporal_GR, convective_GR, V_group, k_cold = linear_growth_rates(w_vals, _Species) 
        
        sb0, sb1 = get_stop_bands(k_cold)
        
        fig, ax = plt.subplots(3, sharex=True)
        ax[0].set_title('Convective Growth Rate and Parameters')
        
        ax[0].plot(f_vals, temporal_GR)
        ax[0].set_ylim(None, None)
        ax[0].set_ylabel('$\gamma$\n$(s^{-1})$', rotation=0, fontsize=14, labelpad=30)
               
        ax[1].plot(f_vals, V_group)
        ax[1].set_ylim(0, 250e3)
        ax[1].set_ylabel('$V_G$\n$(ms^{-1})$', rotation=0, fontsize=14, labelpad=30)
        
        ax[2].plot(f_vals, convective_GR)
        ax[2].set_ylim(None, None)
        ax[2].set_ylabel('$S$\n$(m^{-1})$', rotation=0, fontsize=14, labelpad=30)
        
        ax[2].set_xlabel('f (Hz)', fontsize=14)
        ax[2].set_xlim(0, f_max)
        
        for AX in ax:
            for st, en in zip(sb0, sb1):
                AX.axvspan(f_vals[st], f_vals[en], color='k', alpha=0.5, lw=0)
        
        
    
    # -- Temporal growth rate validation (Validated)
    # NOTE :: Old TGR code calculated with k's from WPDR. May not perfectly match because
    #         these k's come from CPDR. But maybe they will? Who knows. k variation is v. small.
    #
    # Plot temporal growth vs. frequency and vs. wavenumber (Chen) just to see (Validated)
    # Small deviations from independent-k solution, probably because Omura's plasma config.
    # has only a very small energetic component.
    #
    # Also, free-k solver doesn't seem to lock onto the right solution for Nf=1000 (needs higher #)
    # Might want to double check linear growth stuff with Omura's version, since only very small density
    # and marginal growth rate.
    if False:
        Nf      = 10000
        f_max   = 3.5
        f_vals  = np.linspace(0.0, f_max, Nf)
        w_vals  = 2 * np.pi * f_vals
        k_cold  = get_k_cold(w_vals, _Species)
        kmax    = k_cold[np.isnan(k_cold) == False].max()
        
        temporal_GR, convective_GR, V_group_chen, k_cold = linear_growth_rates(w_vals, _Species) 
        
        k_warm, CPDR_solns, WPDR_solns = dispersion_relation_solver(_Species, _PP, norm_k_in=False, norm_k_out=False, \
                                         norm_w=False, plot=False, kmin=0.0, kmax=kmax, Nk=Nf)
        
        fig, ax = plt.subplots(2)
        ax[0].plot(f_vals, temporal_GR, label='Omura')
        ax[0].set_title('Temporal Growth Rate :: Chen et al. (2013)')
        ax[0].set_xlabel('f (Hz)', fontsize=14)
        ax[0].set_ylabel('$\gamma$', rotation=0, fontsize=14, labelpad=30)
        ax[0].set_xlim(0, f_max)
        
        ax[1].plot(k_cold[1:], temporal_GR[1:], label='Chen', c='b')
        ax[1].plot(k_warm[1:], WPDR_solns[1:].imag,  label='Wang', c='r')
        ax[1].set_xlabel('k (/m)', fontsize=14)
        ax[1].set_ylabel('$\gamma$', rotation=0, fontsize=14, labelpad=30)
        ax[1].set_xlim(0, kmax)
        ax[1].legend()
        
        
    # Time integration of equations for wave amplitude and frequency (DOESN"T WORK)
    # Things to check:
    # --- Why doesn't it meet the threshold criteria? Error in eqns or input? NOPE??
    # --- Maybe check the ration of plasma frequencies to cyclotron frequencies, make sure they match
    #           --- ACCURATE TO 3 S.F. (ALMOST, wO IS OFF A LITTLE). Why the difference? Does it matter?
    if False:
        ## QUESTION: WHERE'S THE DISTINCTION BETWEEN TRIGGER FREQUENCY AND 
        ##           FREQUENCY OF EMISSION?
        
        f0   = 1.50                             # Initial frequency (Hz)
        Bw0  = 0.5e-9                           # Initial wave field (T)
        _L   = 4.27                             # L-shell
        _nh  = 0.05*_Species[0]['density']      # Hot proton density (/m3)
        _Q   = 0.5                              # Q-factor (proton hole depletion)
        vpar = 6.00e5                           # Parallel thermal velocity (m/s)
        vper = 8.00e5                           # Perpendicular thermal velocity (m/s)
        
        _t, _f, _Bw = solve_coupled_equations(_Species, _PP, f0, Bw0, _L, _nh, _Q, vpar, vper)
        
        #fig, ax = plt.subplots(2, sharex=True)
        
# =============================================================================
#         ax[0].plot(_t, _Bw*1e9)
#         ax[0].set_xlabel('t (s)', fontsize=14)
#         ax[0].set_ylabel('Bw (nT)', rotation=0, fontsize=14, labelpad=30)
#         #ax[0].set_xlim(0, 30)
#         #ax[0].set_ylim(0, 14)
#         
#         ax[1].plot(_t, _f)
#         ax[1].set_xlabel('t (s)', fontsize=14)
#         ax[1].set_ylabel('f (Hz)', rotation=0, fontsize=14, labelpad=30)
#         #ax[1].set_xlim(0, 30)
#         #ax[1].set_ylim(0, 3.0)
# =============================================================================
