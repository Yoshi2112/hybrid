# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:52:48 2020

@author: Yoshi

Note: This is a super inefficient solve-as-you-go script (e.g. I solve for k(w)
about 10 times in some places). Can always optimize later.
"""
import sys, warnings, pdb
import numpy as np
import matplotlib.cm     as cm
import matplotlib.pyplot as plt
from   scipy.special     import wofz

sys.path.append('..//new_general_DR_solver//')

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


#%% MAIN EQUATIONS
def get_k_cold(w, Species, omura=True):
    '''
    Calculate the k of a specific angular frequency w in a cold
    multicomponent plasma. Assumes a cold plasma (i.e. negates 
    thermal effects)
    
    This will give the cold plasma dispersion relation for the Species array
    specified, since the CPDR is surjective in w (i.e. only one possible k for each w)
    
    Derived directly from the SI form of the CPDR for the L mode, Swanson et al.
    (2003), eqn. 2.15. Identical to the Stix definitition, just in SI.
    
    'omura' calculation is based on eqn. 27 of Omura et al. (2010) in which
    the summation is already done.
    '''
    if not omura:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning) 
            cold_sum = 0.0
            for ii in range(Species.shape[0]):
                cold_sum += Species[ii]['plasma_freq_sq'] / (w * (w - Species[ii]['gyrofreq']))
        
            k = np.sqrt(1 - cold_sum) * w / SPLIGHT
    else:
        gam_c = get_gamma_c(w, Species)
        k = np.sqrt(w*gam_c) / SPLIGHT
    return k

def get_gamma_c(w, Species):
    '''Omura et al. (2010) eqn. 16'''
    # Electron bit (gyfreq is signed, so already minus)
    cold_sum = Species[-1]['plasma_freq_sq'] / Species[-1]['gyrofreq']

    # Sum over ion species
    for ii in range(Species.shape[0] - 1):
        cold_sum += Species[ii]['plasma_freq_sq'] / (Species[ii]['gyrofreq'] - w)
    return cold_sum

def get_group_velocity(w, k, Species):
    '''Omura et al. (2010) eqn. 23'''
    gam_c = get_gamma_c(w, Species) 
    
    ion_sum = 0.0
    for ii in range(Species.shape[0] - 1):
        ion_sum += Species[ii]['plasma_freq_sq'] / (Species[ii]['gyrofreq'] - w) ** 2
    
    denom = gam_c + w*ion_sum
    Vg    = 2 * SPLIGHT * SPLIGHT * k / denom
    return Vg

def get_resonance_velocity(w, k, Species, PP):
    '''Omura et al. (2010) eqn. 33'''
    Vr = (w - PP['pcyc_rad']) / k
    return Vr

def get_phase_velocity(w, k, Species, omura=True):
    '''
    These definitions should be equivalent. Omura variation defined in
    eqn. 28 of Omura et al. (2010)
    '''
    if not omura:
        Vp = w/k
    else:
        gam_c = get_gamma_c(w, Species)
        Vp = SPLIGHT*np.sqrt(w/gam_c)
    return Vp

def get_velocities(w, Species, PP, normalize=False):
    '''
    Convenience function to retrieve all relevant velocities at once
    and normalize them by the speed of light if flagged.
    '''
    k  = get_k_cold(w, Species)
    
    Vg = get_group_velocity(w, k, Species)
    Vr = get_resonance_velocity(w, k, Species, PP)
    Vp = get_phase_velocity(w, k, Species)

    if normalize == False:
        return Vg, Vp, Vr
    else:
        return Vg/SPLIGHT, Vp/SPLIGHT, Vr/SPLIGHT
    
def get_energies(w_vals, k_vals, cyc, mass):
    '''
    Calculate the cyclotron and Landau resonance energies given values
    Equation is just the cyclotron resonance equation from Summers et al. (2005)
    with a relativistic adjustment that makes it suitable for electrons too.
    Note that 'cyclotron' resonance has N = 1, 'landau' resonance is N = 0.
    Also note: Forget the relativistic factor, since v is what we're solving for.
    Also, this only does the energies for a single particle species.
    
    Input:
        w_vals -- Wave angular frequency (rad/s)
        k_vals -- Wave angular wavenumber (/m)
        cyc    -- Particle cyclotron frequency (rad/s)
        mass   -- Particle mass (kg)
        
    Returns:
        E_landau    -- Landau resonant energy (eV) (based on phase velocity)
        E_cyclotron -- Cyclotron resonant energy (eV) (based on resonance velocity)
    '''
    v_para_landau = (w_vals - 0*np.abs(cyc)) / k_vals
    E_landau = 0.5*mass*v_para_landau**2 / PCHARGE
    
    v_para_cyclotron = (w_vals - 1*np.abs(cyc)) / k_vals
    E_cyclotron = 0.5*mass*v_para_cyclotron**2 / PCHARGE
    return E_landau, E_cyclotron


def get_Eres_hu(w_vals, Vp, h_cyc, mass):
    '''
    Literally exactly the same as E_cyclotron from get_energies()
    '''
    Eres = 0.5 * mass * Vp**2 * (1 - h_cyc/w_vals)**2
    return Eres / PCHARGE
    
    
def get_inhomogeneity_terms(w, Species, PP, Vth_perp, normalize_vel=True):
    '''
    Validated about as much as possible. Results are dimensionless since 
    velocities and frequencies cancel out. Only input frequency matters
    as dimensional or not, since it involves the pcyc term (whether value or 1)
    
    kwarg is determined by whether or not Vth_perp is normalized or not, since
    if it is (or is not), the output from get_velocities() also has to be (or not)
    '''
    # Third term in bracks: - Vr**2/(2*Vp**2)
    Vg, Vp, Vr = get_velocities(w, Species, PP, normalize=normalize_vel)
    pcyc       = PP['pcyc_rad']

    s0 = Vth_perp / Vp
    s1 = (1.0 - Vr/Vg) ** 2
    s2 = ((Vth_perp**2)/(2*Vp**2) + Vr**2 / (Vp*Vg) - Vr**2/(2*Vp**2))*(w / pcyc) - Vr/Vp
    return s0, s1, s2


def get_threshold_amplitude(w, wph, Q, s2, a, Vp, Vr, Vth_para, Vth_perp):
    '''
    Input values are their normalized counterparts, as per eqn. (62) of Omura et al. (2010)
    
    This seems validated by Shoji et al. (2013)
    
    Output is normalized by B0_eq
    
    Might need to put some sort of filter in here for low k that requires resonance
    velocities > c, especially for low frequencies
    '''
    t1    = 100. * (np.pi ** 3) * (Vp ** 3) / (w * (wph ** 4) * (Vth_perp ** 5))
    t2    = (a * s2 * Vth_para / Q) ** 2
    
    vrat  = Vr**2 / Vth_para**2
    
    # Suppress overflow by calculating np.exp() term manually
    # Assume anything with a logarithm of more than some value is infinite
    t3 = np.zeros(w.shape[0])
    for ii in range(w.shape[0]):
        if vrat[ii] > 600.0:
            t3[ii] = np.inf
        else:
            t3[ii] = np.exp(vrat[ii])

    om_th = t1 * t2 * t3
    return om_th


def get_optimum_amplitude(w, wph, Q, tau, s0, s1, Vg, Vr, Vth_para, Vth_perp):
    '''
    Optimum EMIC amplitude required for non-linear wave growth as per
    eqn. 22 of Shoji et al. (2013)
    
    Input values are all normalized as per the threshold function
    Output is normalized by B0
    '''
    t1 = 0.81*(Q / tau) / np.sqrt(np.pi**5)
    t2 = s1 * Vg / (s0 * w * Vth_para) 
    t3 = (wph**2)*(Vth_perp**2)*np.exp(-0.5*Vr**2 / Vth_para**2)
    om_opt = t1*t2*t3
    return om_opt


def get_nonlinear_trapping_period(k, Vth_perp, Bw):
    '''
    QUESTION: Is Bw a single value? Or is it a function of frequency/k?
    
    Nonlinear trapping period determines the nonlinear transition time T_N by a factor tau
    T_tr   = nls.get_nonlinear_trapping_period(k_vals, Vth_perp*SPLIGHT, B_opt*PP['B0'])
    T_N    = tau*T_tr*PP['pcyc_rad']
    '''
    bottom  = k * Vth_perp * PCHARGE * Bw
    bracket = np.sqrt(PMASS / bottom) 
    return 2 * np.pi * bracket


def nonlinear_growth_rate(w, wph, Q, Vth_para, Vth_perp, Vg, Vp, Vr, Bw):
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
    
    Bw could be a single value or it could be an array with the same length as w
    '''
    Om_w = PCHARGE * Bw / PMASS
    
    t1 = 0.5 * wph**2 * Q
    t2 = np.sqrt(Vp / (SPLIGHT*Om_w*w)) * Vg / Vth_para
    t3 = (Vth_perp / (SPLIGHT*np.pi)) ** (3/2) * np.exp(- Vr**2 / (2*Vth_para**2))
    return t1 * t2 * t3


#%% HELPER AND/OR TEST FUNCTIONS
def Z(arg):
    '''Return Plasma Dispersion Function : Normalized Fadeeva function'''
    return 1j*np.sqrt(np.pi)*wofz(arg)


def linear_growth_rates_chen(w, Species):
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
    Dr_kder = -2*k*SPLIGHT**2 - k_der_sum

    temporal_growth_rate   = - Di / Dr_wder
    group_velocity         = - Dr_kder / Dr_wder
    
    # There is a minus sign in the paper here that I don't think should be there
    convective_growth_rate = temporal_growth_rate / np.abs(group_velocity)
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


def omega_equation(Om, w, Vth_para, Vth_perp, Vg, Vp, Vr, s0, s2, Q, wph2, a):
    '''
    Value of the derivative expressed in eq. 63.  SEEMS OK? HOW TO VERIFY?
    '''
    nob = Q*wph2/(2*Vth_para)                       # No bracket (first "term")
    br1 = (Vth_perp / np.pi) ** (3/2)               # Bracket 1
    br2 = np.sqrt(Vp * Om / w)                      # Bracket 2
    exp = np.exp(-0.5 * Vr ** 2 / Vth_para ** 2)    # Exponential bit
    lat = 5 * Vp * s2 * a / (s0 * w)            # Last Term
    return (Vg * (nob * br1 * br2 * exp - lat))


def push_Om(Om_in, w, Vth_para, Vth_perp, Vg, Vp, Vr, s0, s2, Q, wph2, a, dt):
    '''
    Solution to eq. 63 via RK4 method. SEEMS OK? HOW TO VERIFY?
    '''
    k1 = omega_equation(Om_in          , w, Vth_para, Vth_perp, Vg, Vp, Vr, s0, s2, Q, wph2, a)
    k2 = omega_equation(Om_in + dt*k1/2, w, Vth_para, Vth_perp, Vg, Vp, Vr, s0, s2, Q, wph2, a)
    k3 = omega_equation(Om_in + dt*k2/2, w, Vth_para, Vth_perp, Vg, Vp, Vr, s0, s2, Q, wph2, a)
    k4 = omega_equation(Om_in + dt*k3  , w, Vth_para, Vth_perp, Vg, Vp, Vr, s0, s2, Q, wph2, a)
    
    Om_out = Om_in + (1/6) * dt * (k1 + 2*k2 + 2*k3 + k4)
    return Om_out


def push_w(w_old, Om, s0, s1, dt):
    '''
    Solution to eq. 64 via finite difference. SEEMS OK? HOW TO VERIFY?
    '''
    K1 = 0.4 * s0 / s1
    Z  = 0.5 * dt * K1 * Om
    
    w_new = w_old * (1 + Z) / (1 - Z)
    return w_new





def leapfrog_coupled_equations(Species, PP, init_f, init_Bw, L, nh, Q, Vth_para, Vth_perp):
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
    wph2      = nh * PCHARGE ** 2 / (PMASS * EPS0)       # Hot proton plasma frequency squared
    a         = 4.5 / ((L * RE)**2)                      # Parabolic magnetic field scale factor
    w_init    = 0.4*pcyc_eq #2 * np.pi * init_f

    Bw_thresh = get_threshold_amplitude(w_init, Species, PP, nh, a, Q, Vth_para, Vth_perp, init_Bw)
    print('Threshold |Bw|: {:>6.2f} nT'.format(Bw_thresh))

    # Normalizations
    a         = a * ((SPLIGHT/pcyc_eq) ** 2)                   # Parabolic magnetic field scale factor (normalized)
    Vth_para  = Vth_para / SPLIGHT
    Vth_perp  = Vth_perp / SPLIGHT
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
    
    # Get initial parameters, solve for threshold
    Vg, Vp, Vr = get_velocities(w_arr[0]*pcyc_eq, Species, PP, normalize=True)
    s0, s1, s2 = get_inhomogeneity_terms(w_arr[0]*pcyc_eq, Species, PP, Vth_perp*SPLIGHT)
    
    
    # Retard initial w soln to N - 1/2 (overwriting w[0]):
    w_arr[0]   = push_w(w_arr[0], Om_arr[0], s0, s1, -0.5*dt) 
    s0, s1, s2 = get_inhomogeneity_terms(w_arr[0]*pcyc_eq, Species, PP, Vth_perp*SPLIGHT)
    
    # Leapfrog LOOP
    for ii in range(1, t_arr.shape[0]):
        # Push w, re-solve functions of w
        w_arr[ii]  = push_w(w_arr[ii - 1], Om_arr[ii - 1], s0, s1, dt) 
        s0, s1, s2 = get_inhomogeneity_terms(w_arr[ii]*pcyc_eq, Species, PP, Vth_perp*SPLIGHT)
        Vg, Vp, Vr = get_velocities(w_arr[ii]*pcyc_eq, Species, PP, normalize=True)
        
        # Check saturation (THIS PART IS UNCERTAIN. WHY TIME-VARYING LIMIT??)
        # Maybe incorporate check into whether or not to solve as above?
        Om_limit   = (Vp / (4*Vth_perp)) * ((1 - w_arr[ii]) ** 2) / w_arr[ii]
        if Om_arr[ii - 1] > Om_limit:
            Om_arr[ii] = Om_arr[ii - 1]
        else:
            Om_arr[ii] = push_Om(Om_arr[ii - 1], w_arr[ii], Vth_para, Vth_perp, Vg, Vp, Vr, s0, s2, Q, wph2, a, dt)
    
    # Un-normalize solutions and return plottable values
    f_arr  = (w_arr * pcyc_eq) / (2 * np.pi)
    Bw_arr = Om_arr * B0
    t_arr  = t_arr / pcyc_eq
    return t_arr, f_arr, Bw_arr


#%% TEST PARAMETERS FROM LITERATURE

def define_omura2010_parameters(include_energetic=False):
    '''
    Ambient plasma parameters from Omura et al. (2010) to recreate plots
    
    Note: Does n_H in equations count hot plus cold? Temperature doesn't matter
    for some things. Need to check this.
    '''
    # Parameters in SI units (Note: Added the hot bit here. Is it going to break anything?) nh = 7.2
    pcyc    = 3.7 # Hz
    B0      = 2 * np.pi * PMASS * pcyc / PCHARGE
    
    if include_energetic == True:
        Th_para  = (PMASS * (6e5)**2 / KB) / 11603.
        Th_perp  = (PMASS * (8e5)**2 / KB) / 11603.
        Ah       = Th_perp / Th_para - 1
        #apar_h   = np.sqrt(2.0 * qp * Th_para  / mp)
        
        name    = np.array(['H'    , 'He'  , 'O'  , 'Hot H'])
        mass    = np.array([1.0    , 4.0   , 16.0 , 1.0    ]) * PMASS
        charge  = np.array([1.0    , 1.0   , 1.0  , 1.0    ]) * PCHARGE
        density = np.array([136.8  , 17.0  , 17.0 , 7.2    ]) * 1e6
        ani     = np.array([0.0    , 0.0   , 0.0  , Ah])
        tpar    = np.array([0.0    , 0.0   , 0.0  , Th_para])
        tper    = (ani + 1) * tpar
    else:
        name    = np.array(['H'    , 'He'  , 'O' ])
        mass    = np.array([1.0    , 4.0   , 16.0]) * PMASS
        charge  = np.array([1.0    , 1.0   , 1.0 ]) * PCHARGE
        density = np.array([144.0  , 17.0  , 17.0]) * 1e6
        ani     = np.array([0.0    , 0.0   , 0.0 ])
        tpar    = np.array([0.0    , 0.0   , 0.0 ])
        tper    = (ani + 1) * tpar
    
    Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
    return Species, PP


def define_ohja2021_parameters(ne, include_energetic=False):
    '''
    Ambient plasma parameters from Ohja et al. (2021) to recreate plots
    ne in /cm3, cast to /m3 in function
    '''
    # Parameters in SI units
    pcyc    = 1.3 # Hz
    B0      = 2 * np.pi * PMASS * pcyc / PCHARGE
    #B0      = 85e-9
    
    if include_energetic == True:
        Th_para  = (PMASS * (4.2e5)**2 / KB) / 11603.
        Th_perp  = (PMASS * (5.4e5)**2 / KB) / 11603.
        Ah       = Th_perp / Th_para - 1
        
        name    = np.array(['H'       , 'He'  , 'O'  , 'Hot H'])
        mass    = np.array([1.0       , 4.0   , 16.0 , 1.0    ]) * PMASS
        charge  = np.array([1.0       , 1.0   , 1.0  , 1.0    ]) * PCHARGE
        density = np.array([0.88*0.999, 0.06  , 0.06 , 0.001  ]) * ne*1e6
        ani     = np.array([0.0       , 0.0   , 0.0  , Ah])
        tpar    = np.array([0.0       , 0.0   , 0.0  , Th_para])
        tper    = (ani + 1) * tpar
    else:
        name    = np.array(['H' , 'He'  , 'O' ])
        mass    = np.array([1.0 , 4.0   , 16.0]) * PMASS
        charge  = np.array([1.0 , 1.0   , 1.0 ]) * PCHARGE
        density = np.array([0.88, 0.06  , 0.06]) * ne*1e6
        ani     = np.array([0.0 , 0.0   , 0.0 ])
        tpar    = np.array([0.0 , 0.0   , 0.0 ])
        tper    = (ani + 1) * tpar
    
    Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
    return Species, PP


def define_shoji2013_parameters():
    '''
    Ambient plasma parameters from Ohja et al. (2021) to recreate plots
    ne in /cm3, cast to /m3 in function
    '''
    # Parameters in SI units
    ne      = 178 # cc
    pcyc    = 3.7 # Hz
    B0      = 2 * np.pi * PMASS * pcyc / PCHARGE

    name    = np.array(['H' , 'He'  , 'O'  ])
    mass    = np.array([1.0 , 4.0   , 16.0 ]) * PMASS
    charge  = np.array([1.0 , 1.0   , 1.0  ]) * PCHARGE
    density = np.array([0.81, 0.095 , 0.095]) * ne*1e6
    ani     = np.array([0.0 , 0.0   , 0.0  ])
    tpar    = np.array([0.0 , 0.0   , 0.0  ])
    tper    = (ani + 1) * tpar
    
    Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
    return Species, PP


def define_nakamura2016_parameters(case=0, CH_rat=50):
    '''
    Parameters for Night/Pre-Noon/Post-Noon sectors from dipole model (IGRF?)
    Taken from Nakamura et al. (2016)
    
    Select case:
        0: Nightside
        1: Postnoon
        2: Prenoon
    
    Returns hot plasma parameters as well as cold Species, PP arrays (all in SI)
    L = 8 for all parameters, but not needed for calculation. Equations a bit
    crazy because they're all normalized in the paper.
    
    Questions:
    -- Is N_H+ all hydrogen? Or just cold component? Assume all, since that's the 
        standard
    -- Why is N_H+ = N_He+ on the nightside? What's the precedent? Does
       this inform my parameter values for the Jul25/Jan16 events?
    -- Are these only H/He plasmas? I guess so
    '''
    w_pH     = np.array([50, 200, 150]) * 2*np.pi
    V_perp   = np.array([60, 20, 90])*1e3
    V_para   = np.array([40, 15, 60])*1e3
    Nh_frac  = np.array([0.01, 0.01, 0.01])
    NHe_frac = np.array([1.0, 0.25, 0.11])
    a_value  = np.array([5e-15, 5e-16, 5e-16])
    
    # Parameters in SI units
    ne     = w_pH**2 * PMASS * EPS0 / (PCHARGE**2) # /m3
    w_cH   = w_pH / CH_rat                         # rad/s
    B0     = PMASS * w_cH / PCHARGE                # Tesla

    #N_cold  = ne / (Nh_frac   + 1)          # Cold plasma number density, combination of H, He
    N_hot   = Nh_frac *ne / (Nh_frac  + 1)   # Hot  plasma number density, consists of Hydrogen only
    NHe     = NHe_frac*ne / (NHe_frac + 1)   # Helium number density, assuming cold only
    NH      = ne -  NHe                      # Total Hydrogen number density (hot and cold)

    name    = np.array(['H'     , 'He'      ])
    mass    = np.array([1.0     , 4.0       ]) * PMASS
    charge  = np.array([1.0     , 1.0       ]) * PCHARGE
    density = np.array([NH[case], NHe[case] ])
    ani     = np.array([0.0     , 0.0       ])
    tpar    = np.array([0.0     , 0.0       ])
    tper    = (ani + 1) * tpar
    
    Species, PP = create_species_array(B0[case], name, mass, charge, density, tper, ani)
    
    # Energetic plasma parameters (SI)
    Vth_para = V_para[case]
    Vth_perp = V_perp[case]
    return Species, PP, N_hot[case], Vth_para, Vth_perp, a_value[case]


def define_kim2016_parameters(Nc=5, include_energetic=False):
    '''
    Input: 
        Nc -- Cold plasma density in /cm3
        include_energetic -- Put hot protons in Species array or not (for linear
                                                        theory calculations)
                          -- Otherwise, return variables separately (nh, Ah, Th, a)
    True used for linear analysis (since hot protons add to total density)
    False used for non-linear analysis, since equations require values to be
        separate and use thermal velocities rather than temperatures.
    '''
    L  = 8
    B0 = 100e-9
    cc = np.array([90., 9., 1.])
    a  = 4.5 / (L*RE)**2
    
    Ah     = 0.7
    Eh     = 33e3 # eV
    
    # Transform to (T_perp, T_para) from (T, A)
    E_para = Eh / (Ah + 2)
    E_perp = Eh * (Ah + 1) / (Ah + 2)
    
    T_perp = E_perp * 11603. # K
    
    Nc *= 1e6
    nh  = cc[0]*Nc*0.045
    
    Vth_para = np.sqrt(2*PCHARGE*E_para / PMASS)
    Vth_perp = np.sqrt(2*PCHARGE*E_perp / PMASS)

    if include_energetic == True:
        name    = np.array(['H'     , 'He'    , 'O'     , 'Hot H'])
        mass    = np.array([1.0     , 4.0     , 16.0    , 1.0    ]) * PMASS
        charge  = np.array([1.0     , 1.0     , 1.0     , 1.0    ]) * PCHARGE
        density = np.array([cc[0]*Nc, cc[1]*Nc, cc[2]*Nc, nh     ])
        ani     = np.array([0.0     , 0.0     , 0.0     , Ah     ])
        tper    = np.array([0.0     , 0.0     , 0.0     , T_perp ])
        
        Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
        return Species, PP
    else:        
        name    = np.array(['H'     , 'He'    , 'O'     ])
        mass    = np.array([1.0     , 4.0     , 16.0    ]) * PMASS
        charge  = np.array([1.0     , 1.0     , 1.0     ]) * PCHARGE
        density = np.array([cc[0]*Nc, cc[1]*Nc, cc[2]*Nc])
        ani     = np.array([0.0     , 0.0     , 0.0     ])
        tper    = np.array([0.0     , 0.0     , 0.0     ])
    
        Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
        
        return Species, PP, nh, Vth_para, Vth_perp, a


#%% CREATE PLOTS AS PER LITERATURE (or CHECK THINGS)
def plot_omura2010_figs34():
    '''
    Looks good
    '''
    Species, PP = define_omura2010_parameters()
    pdb.set_trace()
    # Cold dispersion from get_k_cold() (Fig. 3, validated):
    f_max  = 4.0
    Nf     = 10000
    f_vals = np.linspace(0.0, f_max, Nf)
    w_vals = 2*np.pi*f_vals
    k_vals = get_k_cold(w_vals, Species)
    wlen   = 1e-3 * 2*np.pi / k_vals
    
    if False:
        ## DIAGNOSTIC ##
        plt.ioff()
        k_omura = get_k_cold(w_vals, Species, omura=True)
        plt.figure()
        plt.plot(w_vals, k_vals, label='Normal')
        plt.plot(w_vals, k_omura, label='Omura')
        plt.show()
        plt.legend()
        sys.exit()
    
    fig, ax = plt.subplots()
    ax.plot(1/wlen, f_vals, c='k')
    ax.set_xlabel('$1/\lambda$ (/km)')
    ax.set_ylabel('f', rotation=0)
    ax.set_ylim(0, 3.5)
    ax.set_xlim(0, 0.025)
    
    # Resonance and Group Velocities plot (Figure 4a,b validated)
    V_group, V_phase, V_resonance = get_velocities(w_vals, Species, PP)

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
    return


def plot_omura2010_freqamp():
    '''
    Plots Frequency/Amplitude timeseries using parameters from Omura et al. (2010)
    
    Could also use this to plot Figs 5d, 6, 7 of Nakamura et al. (2015)
    '''
    Species, PP = define_omura2010_parameters()
    
    f0   = 1.50                             # Initial frequency (Hz)
    Bw0  = 0.5e-9                           # Initial wave field (T)
    L    = 4.27                             # L-shell
    nh   = 0.05*Species[0]['density']       # Hot proton density (/m3)
    Q    = 0.5                              # Q-factor (proton hole depletion)
    vpar = 6.00e5                           # Parallel thermal velocity (m/s)
    vper = 8.00e5                           # Perpendicular thermal velocity (m/s)
    
    t, f, Bw = leapfrog_coupled_equations(Species, PP, f0, Bw0, L, nh, Q, vpar, vper)
    
    fig, ax = plt.subplots(2, sharex=True)
    
    ax[0].plot(t, Bw*1e9)
    ax[0].set_xlabel('t (s)', fontsize=14)
    ax[0].set_ylabel('Bw (nT)', rotation=0, fontsize=14, labelpad=30)
    #ax[0].set_xlim(0, 30)
    #ax[0].set_ylim(0, 14)
    
    ax[1].plot(t, f)
    ax[1].set_xlabel('t (s)', fontsize=14)
    ax[1].set_ylabel('f (Hz)', rotation=0, fontsize=14, labelpad=30)
    #ax[1].set_xlim(0, 30)
    #ax[1].set_ylim(0, 3.0)
    return


def plot_check_group_velocity_chen():
    '''
    This plot checks the group velocities as derived by Chen et al. (2011) and
    the cold approx as written in Omura et al. (2010). The single (Species, PP)
    call could be doubled to include hot_protons=True, False argument, since
    Chen equation accepts both hot and cold species in its equations.
    '''
    f_max  = 4.0
    f_vals = np.linspace(0.0, f_max, 10000)
    w_vals = 2 * np.pi * f_vals
    
    # Including (w)arm species, (c)old species only
    wSpecies, wPP = define_omura2010_parameters(include_energetic=True)
    cSpecies, cPP = define_omura2010_parameters(include_energetic=False)
    
    V_group_omura, V_phase, V_resonance               = get_velocities(w_vals, cSpecies, cPP)
    temporal_GR, convective_GR, V_group_chen, k_cold  = linear_growth_rates_chen(w_vals, wSpecies) 
    
    fig, ax = plt.subplots()
    ax.plot(f_vals, V_group_omura/1e3, label='Omura')
    ax.plot(f_vals, V_group_chen/1e3, label='Chen')
    ax.set_title('Group velocity: Chen vs. Omura comparison')
    ax.set_xlabel('f (Hz)', fontsize=14)
    ax.set_ylabel('$V_g$\nkm/s', rotation=0, fontsize=14, labelpad=30)
    ax.set_xlim(0, f_max)
    ax.set_ylim(0, 250)
    ax.legend()
    return


def plot_check_temporal_growth_rate_chen():
    '''
    This plot checks the temporal growth rates of the linear equations in Chen
    et al. (2011) using k derived from the cold approximation.... (Double check
    this whole Wang/Chen thing. Weren't they from the same equations?
    '''
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
    Species, PP = define_omura2010_parameters(include_energetic=True)
    
    Nf      = 10000
    f_max   = 3.5
    f_vals  = np.linspace(0.0, f_max, Nf)
    w_vals  = 2 * np.pi * f_vals
    k_cold  = get_k_cold(w_vals, Species)
    kmax    = k_cold[np.isnan(k_cold) == False].max()
    
    temporal_GR, convective_GR, V_group_chen, k_cold = linear_growth_rates_chen(w_vals, Species) 
    
    k_warm, CPDR_solns, WPDR_solns = get_dispersion_relation(Species, PP, norm_k_in=False, norm_k_out=False, \
                                     norm_w=False, plot=False, kmin=0.0, kmax=kmax, Nk=Nf)
    
    fig, ax = plt.subplots(2)
    ax[0].plot(f_vals, temporal_GR, label='Chen')
    ax[0].set_title('Temporal Growth Rate :: Chen et al. (2013)')
    ax[0].set_xlabel('f (Hz)', fontsize=14)
    ax[0].set_ylabel('$\gamma$', rotation=0, fontsize=14, labelpad=30)
    ax[0].set_xlim(0, f_max)
    
    sb0, sb1 = get_stop_bands(k_cold)
    for st, en in zip(sb0, sb1):
        ax[0].axvspan(f_vals[st], f_vals[en], color='k', alpha=0.5, lw=0)
    
    ax[1].plot(k_cold[1:], temporal_GR[1:], label='Chen', c='b')
    ax[1].plot(k_warm[1:], WPDR_solns[1:].imag,  label='Wang', c='r')
    ax[1].set_xlabel('k (/m)', fontsize=14)
    ax[1].set_ylabel('$\gamma$', rotation=0, fontsize=14, labelpad=30)
    ax[1].set_xlim(0, kmax)
    ax[1].legend()
    return


def plot_check_CPDR():
    '''
    Compares solution of CPDR using some mix of ions (e.g. omura2101_parameters)
    using the algebraic cold plasma dispersion relation (algebraic) vs. the 
    root finding method that underpins the warm and kinetic solutions (used
    for Wang et al., Chen et al., etc.)
    
    Note: The CPDR solver retrieves the same frequency multiples times because
    get_k_cold() doesn't return unique values - Different frequencies may return
    the same k (in fact, for a 3 component plasma, 3 frequencies below p_cyc
    will return the same k).
    
    Will get a better result if I code k_min to k_max and get the 3-component
    solution for that.
    '''
    Species, PP = define_omura2010_parameters(include_energetic=False)
    
    # Cold dispersion from get_k_cold()
    f_max  = 4.0
    Nf     = 10000
    f_vals = np.linspace(0.0, f_max, Nf)
    w_vals = 2*np.pi*f_vals
    k_vals = get_k_cold(w_vals, Species)
    wlen   = 1e-3 * 2*np.pi / k_vals
        
    # Use this k range to get new k array
    k_min = k_vals[np.isnan(k_vals) == False].min()
    k_max = k_vals[np.isnan(k_vals) == False].max()
    k_vals_CPDR = np.linspace(k_min, k_max, Nf*3)
    wlen_CPDR = 1e-3 * 2*np.pi / k_vals_CPDR
    
    CPDR, cCGR = get_dispersion_relation(Species, k_vals_CPDR, approx='cold')
    f_vals_CPDR= CPDR.real / (2*np.pi)
        
    fig, ax = plt.subplots()
    ax.plot(1/wlen, f_vals                  , c='k', label='Algebraic')
    ax.plot(1/wlen_CPDR, f_vals_CPDR[:, 0], c='b', label='Solver-H')
    ax.plot(1/wlen_CPDR, f_vals_CPDR[:, 1], c='b', label='Solver-He')
    ax.plot(1/wlen_CPDR, f_vals_CPDR[:, 2], c='b', label='Solver-O')
    ax.set_xlabel('$1/\lambda$ (/km)')
    ax.set_ylabel('f', rotation=0)
    ax.set_ylim(0, 3.5)
    ax.set_xlim(0, 0.025)
    ax.legend()
    plt.show()
    return


def plot_convective_growth_rate_chen():
    '''
    Check this with Fraser (1989) parameters
    '''
    f_max  = 4.0
    f_vals = np.linspace(0.0, f_max, 10000)
    w_vals = 2 * np.pi * f_vals

    Species, PP = define_omura2010_parameters(include_energetic=True)

    temporal_GR, convective_GR, V_group, k_cold = linear_growth_rates_chen(w_vals, Species) 
    
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
    
    sb0, sb1 = get_stop_bands(k_cold)
    for AX in ax:
        for st, en in zip(sb0, sb1):
            AX.axvspan(f_vals[st], f_vals[en], color='k', alpha=0.5, lw=0)
    return


def plot_convective_growth_rate_fraser():
    '''
    Check this with Fraser (1989) parameters
    '''
    f_max  = 3.5
    f_vals = np.linspace(0.0, f_max, 5000)
    w_vals = 2 * np.pi * f_vals

    mp         = 1.673e-27                        # Proton mass (kg)
    qi         = 1.602e-19                        # Elementary charge (C)
    _B0        = 300e-9                           # Background magnetic field in T
    #A_style    = [':', '-', '--']                 # Line style for each anisotropy
    Ah         = 1.5                              # Anisotropy values
    
    _name      = np.array(['Cold H', 'Cold He', 'Cold O', 'Warm H', 'Warm He', 'Warm O'])         # Species label
    _mass      = np.array([1.0     , 4.0      , 16.0    , 1.0     , 4.0      , 16.0    ]) * mp    # Mass   in proton masses (kg)
    _charge    = np.array([1.0     , 1.0      , 1.0     , 1.0     , 1.0      , 1.0     ]) * qi    # Change in elementary units (C)
    _density   = np.array([196.0   , 22.0     , 2.0     , 5.1     , 0.05     , 0.13    ]) * 1e6   # Density in cm3 (/m3)
    _tper      = np.array([0.0     , 0.0      , 0.0     , 30.0    , 10.0     , 10.0    ]) * 1e3   # Parallel temperature in keV (eV)
    _ani       = np.array([0.0     , 0.0      , 0.0     , Ah      , 1.0      , 1.0     ])         # Temperature anisotropy
    
    Species, PP = create_species_array(_B0, _name, _mass, _charge, _density, _tper, _ani)
    
    temporal_GR, convective_GR, V_group, k_cold = linear_growth_rates_chen(w_vals, Species) 
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('Convective Growth Rate :: Fraser (1989) Comparison')
    
    ax.plot(f_vals, convective_GR*1e7)
    ax.set_ylim(0, None)
    ax.set_ylabel('$S$\n$(m^{-1})$', rotation=0, fontsize=14, labelpad=30)
    
    ax.set_xlabel('f (Hz)', fontsize=14)
    ax.set_xlim(0, f_max)
    
    sb0, sb1 = get_stop_bands(k_cold)
    for st, en in zip(sb0, sb1):
        ax.axvspan(f_vals[st], f_vals[en], color='k', alpha=0.5, lw=0)
        
    plt.show()
    return


def plot_shoji2012_2D():
    '''
    Shoji et al. (2012) recreation of 2D plots in Figures 5 and 6
     -- Varies with normalized frequency (to pcyc) and ion density (plasma freq. proxy)
     -- Assume Bth validated via plots from Shoji et al. (2013)
     -- Can re-validate afterwards (but assume this isn't a problem for now)
     -- Non-linear growth rates are a function of density, assume constant B0
     
     Awkward because threshold amplitude requires normalized parameters, but the
     non-linear growth rate function doesn't.
     
    Threshold Amplitude is off, much more for the higher frequency H band than
    the He band (but the threshold amplitude is still lower than it should be)
    '''
    # Magnetospheric parameters
    B0 = 243e-9 
    L  = 4.0
    a  = 4.5/(L*RE)**2
    
    # Frequency axis (rad/s)
    Nw       = 500
    pcyc     = 23.2
    ecyc     = np.abs(ECHARGE) * B0 / EMASS
    pcyc_min = 0.126
    pcyc_max = 0.249
    w_axis   = np.linspace(pcyc_min, pcyc_max, Nw)
    w_vals   = w_axis * pcyc
    
    # Density axis (/m3)
    Nr          = 500
    wpe_wce_max = 18
    wpe_wce     = np.linspace(0.0, wpe_wce_max, Nr)
    ne          = wpe_wce ** 2 * B0 ** 2 * EPS0 / EMASS
    
    # Energetic plasma parameters (normalized)
    Q        = 0.5
    Vth_para = 0.00800
    Vth_perp = 0.01068    
    
    # Cold plasma parameters (constant)
    name   = np.array(['H'    , 'He'  , 'O'   ])
    mass   = np.array([1.0    , 4.0   , 16.0  ]) * PMASS
    charge = np.array([1.0    , 1.0   , 1.0   ]) * PCHARGE
    tpar   = np.array([0.0    , 0.0   , 0.0   ])
    ani    = np.array([0.0    , 0.0   , 0.0   ])
    tper   = (ani + 1) * tpar
    
    # Other parameters
    white_line = np.sqrt(50e6*PCHARGE**2/(EMASS*EPS0))/ecyc
    circ_x = (2*np.pi*0.56)/pcyc
    circ_y = np.sqrt(178e6*PCHARGE**2/(EMASS*EPS0))/ecyc
    
    # Return arrays
    BTH = np.zeros((Nr, Nw), dtype=np.float64)
    NLG = np.zeros((Nr, Nw), dtype=np.float64)
    
    BTH[0] = np.inf
    for MM in range(1, Nr):
        nh           = 0.0081 * ne[MM]
        wph          = np.sqrt(nh*PCHARGE**2/(PMASS*EPS0))
        density      = np.array([0.8019 , 0.0950, 0.0950]) * ne[MM]
        Species, PP  = create_species_array(B0, name, mass, charge, density, tper, ani,
                                            pcyc_rad=pcyc)
        
        # Get (normalized) velocities and inhomogeneity parameters
        Vg, Vp, Vr = get_velocities(w_vals, Species, PP, normalize=True)
        s0, s1, s2 = get_inhomogeneity_terms(w_vals, Species, PP, Vth_perp, normalize_vel=True)
        
        # Normalize the rest of the parameters
        wphn   = wph / pcyc
        a_norm = a*(SPLIGHT**2/PP['pcyc_rad']**2)
        
        # This one uses all normalized parameters
        BTH[MM, :] = get_threshold_amplitude(w_axis, wphn, Q, s2, a_norm, Vp, Vr, Vth_para, Vth_perp)
        
        # This one uses SI units (just to be confusing)
        NLG[MM, :] = nonlinear_growth_rate(w_vals, wph, Q, Vth_para*SPLIGHT,
                                           Vth_perp*SPLIGHT, Vg*SPLIGHT, Vp*SPLIGHT, 
                                           Vr*SPLIGHT, BTH[MM, :]*B0)
    
    NLG = NLG/pcyc
    cmap = cm.get_cmap('jet')
    cmap.set_bad(color=cmap(1.0))

    if True:
        # Plot Bth
        plt.ioff()
        fig, ax = plt.subplots()
        im1 = ax.pcolormesh(w_axis, wpe_wce, BTH, cmap=cmap,
                            vmin=0.0, vmax=0.1, shading='auto')
        ax.axhline(white_line, c='white', ls='--')
        ax.scatter(circ_x, circ_y, marker='o', facecolors='none', edgecolors='white', lw=2.0, s=100)
        fig.colorbar(im1, extend='both')
    
    if False:
        # Plot Non-linear Growth Rate
        plt.ioff()
        fig2, ax2 = plt.subplots()
        im2 = ax2.pcolormesh(w_axis, wpe_wce, NLG, cmap='jet',
                             vmin=0.0, vmax=1.5, shading='auto')
        ax2.axhline(white_line, c='white', ls='--')
        fig2.colorbar(im2, extend='both')
    
    plt.show()
    return


def plot_shoji2013_fig7():
    '''
    Validation plot to generate Figure 7 of Shoji et al. (2013) showing:
    a) Threshold and optimum wave amplitudes for values of tau = 0.25, 0.5, 1.0, 2.0
    b) Nonlinear transition times for the same tau values
    each between a frequency range of ~0.35 - 1.0 normalized proton cyclotron range
    
    Validated against the paper, looks great! Note: The 10^-2 label in the paper looks
    off, but the lines in proportion to the graph look correct.
    '''
    Species, PP = define_shoji2013_parameters()

    # Frequencies to evaluate
    f_min  = 1.3
    f_max  = 3.7
    Nf     = 10000
    f_vals = np.linspace(f_min, f_max, Nf)
    w_vals = 2*np.pi*f_vals
    k_vals = get_k_cold(w_vals, Species)
    
    # Define hot proton parameters
    nh = 0.0405*PP['n0']
    wph2 = nh * PCHARGE ** 2 / (PMASS * EPS0) 
    Vth_para = 0.002
    Vth_perp = 0.00283
    Q = 0.5
    
    # Curvature parameters
    L  = 4.3
    a  = 4.5  / (L*RE)**2
    a  = a*(SPLIGHT**2/PP['pcyc_rad']**2)
    
    Vg, Vp, Vr = get_velocities(w_vals, Species, PP, normalize=True)
    s0, s1, s2 = get_inhomogeneity_terms(w_vals, Species, PP, Vth_perp, normalize_vel=True)
    
    # Normalize input parameters
    wph = np.sqrt(wph2) / PP['pcyc_rad']
    w   = w_vals / PP['pcyc_rad']

    B_th   = get_threshold_amplitude(w, wph, Q, s2, a, Vp, Vr, Vth_para, Vth_perp)
    B_opt1 = get_optimum_amplitude(w, wph, Q, 0.25, s0, s1, Vg, Vr, Vth_para, Vth_perp)
    B_opt2 = get_optimum_amplitude(w, wph, Q, 0.50, s0, s1, Vg, Vr, Vth_para, Vth_perp)
    B_opt3 = get_optimum_amplitude(w, wph, Q, 1.00, s0, s1, Vg, Vr, Vth_para, Vth_perp)
    B_opt4 = get_optimum_amplitude(w, wph, Q, 2.00, s0, s1, Vg, Vr, Vth_para, Vth_perp)
    
    T_tr1  = get_nonlinear_trapping_period(k_vals, Vth_perp*SPLIGHT, B_opt1*PP['B0'])
    T_tr2  = get_nonlinear_trapping_period(k_vals, Vth_perp*SPLIGHT, B_opt2*PP['B0'])
    T_tr3  = get_nonlinear_trapping_period(k_vals, Vth_perp*SPLIGHT, B_opt3*PP['B0'])
    T_tr4  = get_nonlinear_trapping_period(k_vals, Vth_perp*SPLIGHT, B_opt4*PP['B0'])
    
    T_N1   = 0.25*T_tr1*PP['pcyc_rad']
    T_N2   = 0.50*T_tr2*PP['pcyc_rad']
    T_N3   = 1.00*T_tr3*PP['pcyc_rad']
    T_N4   = 2.00*T_tr4*PP['pcyc_rad']
    
    plt.ioff()
    fig, axes = plt.subplots(nrows=2, figsize=(16,9))
    
    axes[0].semilogy(w, B_th, c='k', ls='--')
    axes[0].semilogy(w, B_opt1, c='k', ls='-', label='$\\tau=0.25$')
    axes[0].semilogy(w, B_opt2, c='k', ls='-', label='$\\tau=0.50$')
    axes[0].semilogy(w, B_opt3, c='k', ls='-', label='$\\tau=1.00$')
    axes[0].semilogy(w, B_opt4, c='k', ls='-', label='$\\tau=2.00$')
    axes[0].set_ylim(1e-5, 1e0)
    axes[0].set_ylabel('$B_{opt}/B_{0eq}$')
    axes[0].set_xlabel('$\omega/\Omega_{H0}$')
    axes[0].set_xlim(0.35, 1.0)
    axes[0].tick_params(top=True, right=True)
    
    axes[1].semilogy(w, T_N1, c='k', ls='-', label='$\\tau=0.25$')
    axes[1].semilogy(w, T_N2, c='k', ls='-', label='$\\tau=0.50$')
    axes[1].semilogy(w, T_N3, c='k', ls='-', label='$\\tau=1.00$')
    axes[1].semilogy(w, T_N4, c='k', ls='-', label='$\\tau=2.00$')
    axes[1].set_xlim(0.35, 1.0)
    axes[1].set_ylabel('$T_N \Omega_{H0}$')
    axes[1].set_xlabel('$\omega/\Omega_{H0}$')
    axes[1].set_ylim(1e0, 1e4)
    plt.show()
    return


def plot_nakamura2016_fig11():
    '''
    Recreates three-plot (for three separate cases) from Nakamura et al. (2016)
    Figure 11.
    
    This plot defines three different density/magnetic regimes for three different
    values of a (defined by the position in the magnetosphere). So, nine 
    different calculations take place: Three for each value of a
    
    For each plot (case), feed three different plasma/cyclotron frequency ratios
    into the variable loader.
    
    Do it in loops
    '''
    # Set frequency axis
    w_norm = np.linspace(0.35, 1.0, 1000)
    
    Q = 0.5
    tau = 0.5
    
    plt.ioff()
    fig, axes = plt.subplots(ncols=3)
    clr = ['b', 'c', 'green']
    
    for ii in range(3): 
        for jj, ratio in zip(range(3), [50, 100, 200]):
            Species, PP, nh, vpara, vperp, a = define_nakamura2016_parameters(case=ii, CH_rat=ratio)
            w_vals = w_norm*PP['pcyc_rad']
            
            # Define hot proton plasma frequency and normalize
            wph2     = nh * PCHARGE ** 2 / (PMASS * EPS0)
            wph      = np.sqrt(wph2) / PP['pcyc_rad']
            Vth_perp = vperp / SPLIGHT
            Vth_para = vpara / SPLIGHT
            a_norm   = a*(SPLIGHT**2/PP['pcyc_rad']**2)
            
            # Get plasma parameters
            Vg, Vp, Vr = get_velocities(w_vals, Species, PP, normalize=True)
            s0, s1, s2 = get_inhomogeneity_terms(w_vals, Species, PP, Vth_perp, normalize_vel=True)        
    
            # Calculate amplitudes
            B_th  = get_threshold_amplitude(w_norm, wph, Q, s2, a_norm, Vp, Vr, Vth_para, Vth_perp)
            B_opt = get_optimum_amplitude(w_norm, wph, Q, tau, s0, s1, Vg, Vr, Vth_para, Vth_perp)

            ax = axes[ii]
            ax.semilogy(w_norm, B_opt, c=clr[jj])
            ax.semilogy(w_norm, B_th,  c=clr[jj], ls='--')
            ax.set_xlim(0.35, 1.0)
            ax.set_ylim(1e-10, 1e4)
            
            plt.show()
    return


def plot_kim2016_fig16():
    Species, PP = define_kim2016_parameters(Nc=5, include_energetic=True)
    
    # Cold dispersion from get_k_cold()
    f_min  = 0.1
    f_max  = 0.4
    Nf     = 1000
    f_vals = np.linspace(f_min, f_max, Nf)
    w_vals = 2*np.pi*f_vals
    k_vals = get_k_cold(w_vals, Species)
        
    gamma, S_conv, Vg = get_cold_growth_rates(w_vals, k_vals, Species)

    plt.ioff()
    fig, axes = plt.subplots(nrows=2, sharex=True)
    
    axes[0].plot(f_vals, gamma/PP['pcyc_rad'])
    axes[0].set_xlim(f_min, f_max)
    plt.show()
    return
    
    
def plot_kim2016_fig16_alternative():
    '''
    Two plots:
        1) Linear temporal growth rate with frequency (Normalized to w_pcyc)
        2) Optimum/Threshold amplitudes (Normalized to B0)
    for four different cold plasma densities: 5, 10, 20, 30/cm3
    
    Solve over wider range than needed then xlim() to 0.1-0.4Hz
    
    Resonance at f = 0.38100152 Hz for 5/cm3    (Helium cyclotron frequency)
    '''
    Species, PP = define_kim2016_parameters(Nc=5, include_energetic=True)
        
    knorm_fac   = PP['pcyc_rad'] / PP['va']
    k_vals      = np.linspace(0.0, 3.0, 3000, endpoint=False) * knorm_fac
    
    CPDR_solns,  cold_CGR = get_dispersion_relation(Species, k_vals, approx='cold')
    WPDR_solns,  warm_CGR = get_dispersion_relation(Species, k_vals, approx='warm')
    HPDR_solns,   hot_CGR = get_dispersion_relation(Species, k_vals, approx='hot' )

    CPDR_solns /= PP['pcyc_rad']
    WPDR_solns /= PP['pcyc_rad']
    HPDR_solns /= PP['pcyc_rad']
    k_vals     *= PP['va'] / PP['pcyc_rad']
    
    species_colors = ['r', 'b', 'g']
    
    print('Plotting solutions...')
    plt.ioff()
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    
    for ii in range(CPDR_solns.shape[1]):
        ax1.plot(k_vals[1:], CPDR_solns[1:, ii].real, c=species_colors[ii], linestyle='--', label='Cold')
        ax1.plot(k_vals[1:], WPDR_solns[1:, ii].real, c=species_colors[ii], linestyle=':' , label='Warm')
        ax1.plot(k_vals[1:], HPDR_solns[1:, ii].real, c=species_colors[ii], linestyle='-' , label='Hot')

    ax1.set_title('Dispersion Relation')
    ax1.set_xlabel(r'$kv_A / \Omega_p$')
    ax1.set_ylabel(r'$\omega_r/\Omega_p$')
    ax1.set_xlim(k_vals[0], k_vals[-1])
    
    ax1.set_ylim(0, 1.0)
    ax1.minorticks_on()
    
    for ii in range(CPDR_solns.shape[1]):
        ax2.plot(k_vals[1:], WPDR_solns[1:, ii].imag, c=species_colors[ii], linestyle=':')
        ax2.plot(k_vals[1:], HPDR_solns[1:, ii].imag, c=species_colors[ii], linestyle='-')

    ax2.set_title('Temporal Growth Rate')
    ax2.set_xlabel(r'$kv_A / \Omega_p$')
    ax2.set_ylabel(r'$\gamma/\Omega_p$')
    ax2.set_xlim(k_vals[0], k_vals[-1])
    ax2.set_ylim(-0.05, 0.05)
    
    ax2.minorticks_on()

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized() 
    return
    
def plot_ohja2021_fig8(crosscheck_CPDR=False):
    '''
    3 Panel plot that relies on calculating the dispersion relation, velocities
    (ground, phase, resonance) and resonant energies (cyclotron, Landau)
    
    Panel 1: Dispersion relation of EMIC wavelength (km, y) vs. Frequency (x)
        for total electron densities 4, 6, 8 /cm3
    Panel 2: Velocities (km/s y) vs. Frequency (x) for ne = 6/cm3
    Panel 3: Energies (keV, y) vs. Frequency (x) for ne = 6/cm3
    
    NOTE: Some discrepancies in the calculated wavelengths and velocities.
        Doesn't seem to be a mistake, since my Omura values are spot on.
    '''
    
    
    # Cold dispersion from get_k_cold()
    Spec4, PP4 = define_ohja2021_parameters(4, include_energetic=False)
    Spec6, PP6 = define_ohja2021_parameters(6, include_energetic=False)
    Spec8, PP6 = define_ohja2021_parameters(8, include_energetic=False)
    
    ################
    ## WAVELENGTH ##
    ################
    # Frequencies to evaluate
    f_max  = 1.3
    Nf     = 10000
    f_vals = np.linspace(0.0, f_max, Nf)
    w_vals = 2*np.pi*f_vals
    
    # Calculate k for each
    k_vals4 = get_k_cold(w_vals, Spec4)
    k_vals6 = get_k_cold(w_vals, Spec6)
    k_vals8 = get_k_cold(w_vals, Spec8)
    
    wlen4   = 1e-3 * 2*np.pi / k_vals4
    wlen6   = 1e-3 * 2*np.pi / k_vals6
    wlen8   = 1e-3 * 2*np.pi / k_vals8
    
    # DIAG #
    if True:
        Vg, Vp, Vr = get_velocities(w_vals, Spec6, PP6)
        ELAND, ECYCL = get_energies(w_vals, k_vals6, PP6['pcyc_rad'], PMASS)
        Eres = get_Eres_hu(w_vals, Vp, PP6['pcyc_rad'], PMASS)
        
        fig, ax = plt.subplots()
        ax.semilogy(w_vals, ECYCL, c='k', lw=1.5)
        ax.semilogy(w_vals, Eres, c='r', lw=0.75)
        sys.exit()
    
    ## PLOT ##
    fig, axes = plt.subplots(nrows=3)
    axes[0].plot(f_vals, wlen4, c='lime', label='4 cc') 
    axes[0].plot(f_vals, wlen6, c='k'   , label='6 cc')
    axes[0].plot(f_vals, wlen8, c='r'   , label='8 cc')
    
    if crosscheck_CPDR:
        # Use this k range to get new k array
        k_min = k_vals6[np.isnan(k_vals6) == False].min()
        k_max = k_vals6[np.isnan(k_vals6) == False].max()
        k_vals_CPDR = np.linspace(k_min, k_max, Nf*3)
        wlen_CPDR = 1e-3 * 2*np.pi / k_vals_CPDR
        
        CPDR, cCGR = get_dispersion_relation(Spec6, k_vals_CPDR, approx='cold')
        f_vals_CPDR= CPDR.real / (2*np.pi)
            
        axes[0].plot(f_vals_CPDR[:, 0], wlen_CPDR, c='b', label='Solver-H')
        axes[0].plot(f_vals_CPDR[:, 1], wlen_CPDR, c='b', label='Solver-He')
        axes[0].plot(f_vals_CPDR[:, 2], wlen_CPDR, c='b', label='Solver-O')
        
    axes[0].set_ylabel('$\lambda_{EMIC}$ [km]')
    axes[0].set_ylim(0, 4000)
    axes[0].legend(loc='upper right')
    
    ################
    ## VELOCITIES ##
    ################
    Vg, Vp, Vr = get_velocities(w_vals, Spec6, PP6)
    
    axes[1].semilogy(f_vals, Vg*1e-3, c='k', label='$V_g$') 
    axes[1].semilogy(f_vals, Vp*1e-3, c='r', label='$V_p$')
    axes[1].semilogy(f_vals,-Vr*1e-3, c='b', label='$-V_R$')
    axes[1].set_ylim(10, 2000)
    axes[1].set_ylabel('Velocities [km/s]')
    axes[1].legend(loc='upper right')
    
    for ax in axes:
        ax.set_xlim(0.4, 1.3)
    
    ##############
    ## ENERGIES ##
    ##############
    ELAND, ECYCL = get_energies(w_vals, k_vals6, PP6['pcyc_rad'], PMASS)
    
    axes[2].semilogy(f_vals, ECYCL*1e-3, c='b', label='$E_R$ Cyclotron')
    axes[2].semilogy(f_vals, ELAND*1e-3, c='r', label='$E_R$ Landau')
    axes[2].set_ylim(1e-5, 1e4)
    axes[2].set_ylabel('$E_R$ [keV]')
    axes[2].set_xlabel('Freq [Hz]', rotation=0)
    axes[2].legend(loc='upper right')
    
    plt.show()
    return


def plot_ohja2021_fig9():
    '''
    4 Panel plot showing the non-linear properties of wavepackets from Ohja
    et al. (2021)
    
    Panel 1: Scatterplot with EMD-HHT scatterplot IA/IF from THEMIS data
             Overlaid: Threshold and Optimum amplitudes vs. frequency
    Panel 2: Temporal linear and non-linear (with Bw = thresh, optim.) growth rates
    Panel 3: As above, but convective (no linear)
    Panel 4: Nonlinear transition time T_N = tau * T_tr
    
    Need to calculate:
        Threshold amplitude with eqn. 62 of Omura et al. (2010)
        Optimum amplitude with eqn. 22 of Shoji et al. (2013)
        Non-linear growth rate with eqn. 54 of Omura et al. (2010)
        Non-linear convective growth rate is just Gamma_NL/Vg (Kozyra I guess?)
        Non-linear transition time relying on
            - Nonlinear trapping period (given by a trapping frequency between
                 eqns 39-40 in-text in Omura et al., 2010)
            - What is tau? Some value between 0.25 - 2.0? Weighting factor relating
                the trapping period to the transition time? Tau *is* the ratio,
                and apparent tau = 0.5 is the best
                
    Doesn't work at all
    '''
    Species, PP = define_ohja2021_parameters(6, include_energetic=False)
    
    # Frequencies to evaluate
    f_max  = 1.4
    Nf     = 10000
    f_vals = np.linspace(0.0, f_max, Nf)
    w_vals = 2*np.pi*f_vals
    
    # Define hot proton parameters
    ne = 6e6
    nh = 0.001 * 0.88 * ne  # 0.1% of an 88% Proton plasma (6% each He, O)
    wph2 = nh * PCHARGE ** 2 / (PMASS * EPS0) 
    Vth_para = 420e3 / SPLIGHT
    Vth_perp = 540e3 / SPLIGHT
    
    # Nonlinear hole parameters
    tau = 0.5
    Q = 0.5
    
    # Curvature parameters
    L      = 8
    a      = 4.5  / (L*RE)**2
    a_norm = a*(SPLIGHT**2/PP['pcyc_rad']**2)
    
    Vg, Vp, Vr = get_velocities(w_vals, Species, PP, normalize=True)
    s0, s1, s2 = get_inhomogeneity_terms(w_vals, Species, PP, Vth_perp, normalize_vel=True)
    
    # Normalize a few things
    w_norm  = w_vals / PP['pcyc_rad']
    wph     = np.sqrt(wph2) / PP['pcyc_rad']  # Is this supposed to be squared? I guess if the freq. is...
    
    # Calculate
    B_th = get_threshold_amplitude(w_norm, wph, Q, s2, a_norm, Vp, Vr, Vth_para, Vth_perp)
    B_opt = get_optimum_amplitude(w_norm, wph, Q, tau, s0, s1, Vg, Vr, Vth_para, Vth_perp)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.semilogy(f_vals, B_opt, ls='-' , label='Optimum Amplitude')
    ax.semilogy(f_vals, B_th , ls='--', label='Threshold Amplitude')
    ax.set_xlim(0.0, 1.4)
    ax.set_ylim(10**(-7.5), 1e5)
    plt.show()
    return


def just_random():
    name    = np.array(['H'  ])
    mass    = np.array([1.0  ]) * PMASS
    charge  = np.array([1.0  ]) * PCHARGE
    density = np.array([200.0]) * 1e6
    ani     = np.array([0.0  ])
    tper    = np.array([0.0  ])
    B0      = 200e-9
    
    Species, PP = create_species_array(B0, name, mass, charge, density, tper, ani)
    
    w = 0.3 * PP['pcyc_rad']
    
    k = get_k_cold(w, Species)
    print(k)
    return


if __name__ == '__main__': 
    #plot_omura2010_figs34()
    #plot_shoji2013_fig7()
    #plot_shoji2012_2D()
    
    #plot_check_CPDR()
    plot_ohja2021_fig8()
    #plot_ohja2021_fig9()
    
    #plot_nakamura2016_fig11()
    
    #plot_kim2016_fig16()
    #plot_kim2016_fig16_alternative()
        
# =============================================================================
#     if False:
#         # I get 0.382nT, paper says I should get 0.48
#         # Which bits am I assuming, and what's the uncertainty?
#         #
#         # Phase velocity: Should be w/k
#         #
#         Q    = 0.5
#         wlen = 370e3
#         B0   = 243e-9
#         k    = 2 * np.pi / wlen 
#         fH0  = 3.7
#         omH0 = 2 * np.pi * fH0
#         
#         wn   = 0.4
#         wph  = 679.*np.sqrt(0.05)*omH0
#         wphn = wph / omH0
#         
#         Vth_perpn = 0.00267
#         Vth_paran = 0.002
#         
#         Vpn = (wn * omH0 / k)/c
#         Vrn = (-800e3)/c
#         Vgn = 150e3/c
#         
#         a  = 4.5 / (4.3*RE)**2
#         an = a * c**2/omH0**2
#         
#         s2 = (0.5*(Vth_perpn/Vpn)**2 + Vrn**2/(Vpn*Vgn))*wn - Vrn/Vpn   # - 0.5*(Vrn/Vpn)**2
#         
#         om_th = get_threshold_amplitude()
#         print('Non-linear growth threshold:')
#         print(om_th*B0*1e9, 'nT')
# =============================================================================
