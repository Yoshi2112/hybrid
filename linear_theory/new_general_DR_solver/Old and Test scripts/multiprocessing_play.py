# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:30:20 2020

@author: Yoshi

Testing multiprocessing stuff
"""
import sys, os, warnings, time
import numpy             as np
import multiprocessing
import multiprocessing.sharedctypes
from   scipy.optimize           import fsolve
from   scipy.special            import wofz

sys.path.append('F://Google Drive//Uni//PhD 2017//Data//Scripts//')
import extract_parameters_from_data   as data

c  = 3e8
qp = 1.602e-19
mp = 1.673e-27


#%% DATA MANAGEMENT FUNCTIONS
def nearest_index(items, pivot):
    closest_val = min(items, key=lambda x: abs(x - pivot))
    for ii in range(len(items)):
        if items[ii] == closest_val:
            return ii
    sys.exit('Error: Unable to find index')
    
    
def extract_species_arrays(time_start, time_end, probe, pad, rbsp_path='G://DATA//RBSP//',
                           cmp=[70, 20, 10], return_raw_ne=False, HM_filter_mhz=50, nsec=None):
    '''
    Data module only extracts the 3 component species dictionary from HOPE and RBSPICE 
    energetic measurements. This function creates the single axis arrays required to 
    go into the dispersion solver.
    
    All output values are in SI (/m3, etc.) except temperature, which is in eV
    
    Structure of each array (instr_species, time) where
     -- 0,1,2 are cold             H+, He+, O+
     -- 3,4,5 are HOPE    (warm)   "
     -- 6,7,8 are RBSPICE (warmer) "
    '''
    times, B0, cold_dens, hope_dens, hope_temp, hope_anis, spice_dens, spice_temp, spice_anis\
        = data.load_and_interpolate_plasma_params(time_start, time_end, probe, pad, rbsp_path=rbsp_path,
                                                  HM_filter_mhz=HM_filter_mhz, nsec=nsec)

    Nt       = times.shape[0]
    _density = np.zeros((9, Nt), dtype=float)
    _tper    = np.zeros((9, Nt), dtype=float)
    _ani     = np.zeros((9, Nt), dtype=float)

    _name    = np.array([   'cold $H^{+}$',    'cold $He^{+}$',    'cold $O^{+}$',
                            'HOPE $H^{+}$',    'HOPE $He^{+}$',    'HOPE $O^{+}$',
                         'RBSPICE $H^{+}$', 'RBSPICE $He^{+}$', 'RBSPICE $O^{+}$'])
    
    _mass    = np.array([1.0, 4.0, 16.0, 1.0, 4.0, 16.0, 1.0, 4.0, 16.0]) * mp
    _charge  = np.array([1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0]) * qp
    
    _density[0] = cold_dens * cmp[0]/100.;  _density[3] = hope_dens[0];  _density[6] = spice_dens[0];
    _density[1] = cold_dens * cmp[1]/100.;  _density[4] = hope_dens[1];  _density[7] = spice_dens[1];
    _density[2] = cold_dens * cmp[2]/100.;  _density[5] = hope_dens[2];  _density[8] = spice_dens[2];

    _tper[0] = 0.0; _tper[3] = hope_temp[0]; _tper[6] = spice_temp[0]
    _tper[1] = 0.0; _tper[4] = hope_temp[1]; _tper[7] = spice_temp[1]
    _tper[2] = 0.0; _tper[5] = hope_temp[2]; _tper[8] = spice_temp[2]

    _ani[0]  = 0.0; _ani[3]  = hope_anis[0]; _ani[6]  = spice_anis[0];
    _ani[1]  = 0.0; _ani[4]  = hope_anis[1]; _ani[7]  = spice_anis[1];
    _ani[2]  = 0.0; _ani[5]  = hope_anis[2]; _ani[8]  = spice_anis[2];

    if return_raw_ne == False:
        return times, B0, _name, _mass, _charge, _density, _tper, _ani
    else:
        return times, B0, _name, _mass, _charge, _density, _tper, _ani, cold_dens
    
    
def create_species_array(B0, name, mass, charge, density, tper, ani):
    '''
    For each ion species, total density is collated and an entry for 'electrons' added (treated as cold)
    Also output a PlasmaParameters dict containing things like alfven speed, density, hydrogen gyrofrequency, etc.
    
    Inputs must be in SI units: nT, kg, C, /m3, eV, etc.
    ''' 
    nsp       = name.shape[0]
    e0        = 8.854e-12
    mu0       = 4e-7*np.pi
    q         = 1.602e-19
    me        = 9.101e-31
    mp        = 1.673e-27
    ne        = density.sum()
    
    t_par = np.zeros(nsp); alpha_par = np.zeros(nsp)
    for ii in range(nsp):
        t_par[ii] = q*tper[ii] / (ani[ii] + 1)
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
                                            ne * q ** 2 / (me * e0),
                                            -q  * B0 / me,
                                            0.)], dtype=Species.dtype))
    
    PlasParams = {}
    PlasParams['va']       = B0 / np.sqrt(mu0*(density * mass).sum())  # Alfven speed (m/s)
    PlasParams['n0']       = ne                                        # Electron number density (/m3)
    PlasParams['pcyc_rad'] = q*B0 / mp                                 # Proton cyclotron frequency (rad/s)
    PlasParams['B0']       = B0                                        # Magnetic field value (T)
    return Species, PlasParams


#%% CALCULATION FUNCTIONS    
def Z(arg):
    '''
    Return Plasma Dispersion Function : Normalized Fadeeva function
    Plasma dispersion function related to Fadeeva function
    (Summers & Thorne, 1993) by i*sqrt(pi) factor.
    '''
    return 1j*np.sqrt(np.pi)*wofz(arg)

def Y(arg):
    return np.real(Z(arg))


def hot_dispersion_eqn(w, k, Species):
    '''
    Function used in scipy.fsolve minimizer to find roots of dispersion relation
    for hot plasma approximation.
    Iterates over each k to find values of w that minimize to D(wr, k) = 0
    
    In this case, w is a vector [wr, wi] and fsolve is effectively doing a multivariate
    optimization.
    
    type_out allows purely real or purely imaginary (coefficient only) for root
    finding. Set as anything else for complex output.
    
    FSOLVE OPTIONS :: If bad solution, return np.nan?
    
    Eqns 1, 13 of Chen et al. (2013)
    '''
    wc = w[0] + 1j*w[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hot_sum = 0.0
        for ii in range(Species.shape[0]):
            sp = Species[ii]
            if sp['tper'] == 0:
                hot_sum += sp['plasma_freq_sq'] * wc / (sp['gyrofreq'] - wc)
            else:
                pdisp_arg   = (wc - sp['gyrofreq']) / (sp['vth_par']*k)
                pdisp_func  = Z(pdisp_arg)*sp['gyrofreq'] / (sp['vth_par']*k)
                brackets    = (sp['anisotropy'] + 1) * (wc - sp['gyrofreq'])/sp['gyrofreq'] + 1
                Is          = brackets * pdisp_func + sp['anisotropy']
                hot_sum    += sp['plasma_freq_sq'] * Is

    solution = (wc ** 2) - (c * k) ** 2 + hot_sum
    return np.array([solution.real, solution.imag])


def warm_dispersion_eqn(w, k, Species):
    '''    
    Function used in scipy.fsolve minimizer to find roots of dispersion relation
    for warm plasma approximation.
    Iterates over each k to find values of w that minimize to D(wr, k) = 0
    
    Eqn 14 of Chen et al. (2013)
    '''
    wr = w[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm_sum = 0.0
        for ii in range(Species.shape[0]):
            sp = Species[ii]
            if sp['tper'] == 0:
                warm_sum   += sp['plasma_freq_sq'] * wr / (sp['gyrofreq'] - wr)
            else:
                pdisp_arg   = (wr - sp['gyrofreq']) / (sp['vth_par']*k)
                numer       = ((sp['anisotropy'] + 1)*wr - sp['anisotropy']*sp['gyrofreq'])
                Is          = sp['anisotropy'] + numer * Y(pdisp_arg) / (sp['vth_par']*k)
                warm_sum   += sp['plasma_freq_sq'] * Is
            
    solution = wr ** 2 - (c * k) ** 2 + warm_sum
    return np.array([solution, 0.0])


def cold_dispersion_eqn(w, k, Species):
    '''
    Function used in scipy.fsolve minimizer to find roots of dispersion relation
    for warm plasma approximation.
    Iterates over each k to find values of w that minimize to D(wr, k) = 0
    
    Eqn 19 of Chen et al. (2013)
    '''
    wr = w[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cold_sum = 0.0
        for ii in range(Species.shape[0]):
            cold_sum += Species[ii]['plasma_freq_sq'] * wr / (Species[ii]['gyrofreq'] - wr)
            
    solution = wr ** 2 - (c * k) ** 2 + cold_sum
    return np.array([solution, 0.0])


def get_warm_growth_rates(wr, k, Species):
    '''
    Calculates the temporal and convective linear growth rates for a plasma
    composition contained in Species for each frequency w. Assumes a cold
    dispersion relation is valid for k but uses a warm approximation in the
    solution for D(w, k).
    
    Equations adapted from Chen et al. (2013)
    '''    
    w_der_sum = 0.0
    k_der_sum = 0.0
    Di        = 0.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for ii in range(Species.shape[0]):
            sp = Species[ii]
            
            # If cold
            if sp['tper'] == 0:
                w_der_sum += sp['plasma_freq_sq'] * sp['gyrofreq'] / (wr - sp['gyrofreq'])**2
                k_der_sum += 0.0
                Di        += 0.0
            
            # If hot
            else:
                zs           = (wr - sp['gyrofreq']) / (sp['vth_par']*k)
                Yz           = np.real(Z(zs))
                dYz          = -2*(1 + zs*Yz)
                A_bit        = (sp['anisotropy'] + 1) * wr / sp['gyrofreq']
                
                # Calculate frequency derivative of Dr (sums bit)
                w_der_outsd  = sp['plasma_freq_sq']*sp['gyrofreq'] / (wr*k*sp['vth_par'])
                w_der_first  = A_bit * Yz
                w_der_second = (A_bit - sp['anisotropy']) * wr * dYz / (k * sp['vth_par']) 
                w_der_sum   += w_der_outsd * (w_der_first + w_der_second)
        
                # Calculate Di (sums bit)
                Di_bracket = 1 + (sp['anisotropy'] + 1) * (wr - sp['gyrofreq']) / sp['gyrofreq']
                Di_after   = sp['gyrofreq'] / (k * sp['vth_par']) * np.sqrt(np.pi) * np.exp(- zs ** 2)
                Di        += sp['plasma_freq_sq'] * Di_bracket * Di_after
        
                # Calculate wavenumber derivative of Dr (sums bit)
                k_der_outsd  = sp['plasma_freq_sq']*sp['gyrofreq'] / (wr*k*k*sp['vth_par'])
                k_der_first  = A_bit - sp['anisotropy']
                k_der_second = Yz + zs * dYz
                k_der_sum   += k_der_outsd * k_der_first * k_der_second
    
    # Get and return ratio
    Dr_wder = 2*wr + w_der_sum
    Dr_kder = -2*k*c**2 - k_der_sum

    temporal_growth_rate   = - Di / Dr_wder
    group_velocity         = - Dr_kder / Dr_wder
    convective_growth_rate = - temporal_growth_rate / np.abs(group_velocity)
    return temporal_growth_rate, convective_growth_rate


def get_cold_growth_rates(wr, k, Species):
    '''
    Simplified version of the warm growth rate equation.
    '''
    w_der_sum = 0.0
    Di        = 0.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for ii in range(Species.shape[0]):
            sp         = Species[ii]
            w_der_sum += sp['plasma_freq_sq'] * sp['gyrofreq'] / (wr - sp['gyrofreq'])**2
            
            if sp['vth_par'] != 0.0:
                # Calculate Di (sums bit)
                zs         = (wr - sp['gyrofreq']) / (sp['vth_par']*k)
                Di_bracket = 1 + (sp['anisotropy'] + 1) * (wr - sp['gyrofreq']) / sp['gyrofreq']
                Di_after   = sp['gyrofreq'] / (k * sp['vth_par']) * np.sqrt(np.pi) * np.exp(- zs ** 2)
                Di        += sp['plasma_freq_sq'] * Di_bracket * Di_after
    
    # Get and return ratio
    Dr_wder = 2*wr + w_der_sum
    Dr_kder = -2*k*c**2

    temporal_growth_rate   = - Di / Dr_wder
    group_velocity         = - Dr_kder / Dr_wder
    convective_growth_rate = - temporal_growth_rate / np.abs(group_velocity)
    return temporal_growth_rate, convective_growth_rate


def get_dispersion_relation(Species, k, approx='warm', guesses=None, complex_out=True, print_filtered=True):
    '''
    Given a range of k, returns the real and imaginary parts of the plasma dispersion
    relation specified by the Species present.
    
    Type of dispersion relation controlled by 'approx' kwarg as:
        hot  :: Full dispersion relation for complex w = wr + i*gamma
        warm :: Small growth rate approximation that allows D(wr, k) = 0
        cold :: Dispersion relation used for wr, growth rate calculated as per warm
        
    Gamma is only solved for in the DR with the hot approx
    The other two require calls to another function with a wr arg.
    
    To do: Code a number of 'tries' when a solution fails to converge. Change
            the initial 'guess' for that k by using previous guess multiplied
            by a random number between 0.99-1.01 (i.e. 1% variation)
            
           Could also change to make sure solutions use previous best solution,
            if the previous solution ended up as a np.nan, and implement nan'ing
            solutions as they're solved to prevent completely losing convergence.
    '''
    gyfreqs, counts = np.unique(Species['gyrofreq'], return_counts=True)
    
    # Remove electron count, 
    gyfreqs = gyfreqs[1:]
    N_solns = counts.shape[0] - 1

    # fsolve arguments
    eps    = 1.01           # Offset used to supply initial guess (since right on w_cyc returns an error)
    tol    = 1e-10          # Absolute solution convergence tolerance in rad/s
    fev    = 1000000        # Maximum number of iterations
    Nk     = k.shape[0]     # Number of wavenumbers to solve for
    
    # Solution and error arrays :: Two-soln array for wr, gamma. 
    # PDR_solns init'd as ones because 0.0 returns spurious root
    PDR_solns = np.ones( (Nk, N_solns, 2), dtype=np.float64)*0.01
    OUT_solns = np.zeros((Nk, N_solns   ), dtype=np.complex128)
    ier       = np.zeros((Nk, N_solns   ), dtype=int)
    msg       = np.zeros((Nk, N_solns   ), dtype='<U256')

    # Initial guesses (check this?)
    for ii in range(1, N_solns):
        PDR_solns[0, ii - 1]  = np.array([[gyfreqs[-ii - 1] * eps, 0.0]])

    if approx == 'hot':
        func = hot_dispersion_eqn
    elif approx == 'warm':
        func = warm_dispersion_eqn
    elif approx == 'cold':
        func = cold_dispersion_eqn
    else:
        sys.exit('ABORT :: kwarg approx={} invalid. Must be \'cold\', \'warm\', or \'hot\'.'.format(approx))
    
    # Define function to solve for (all have same arguments)
    if guesses is None or guesses.shape != PDR_solns.shape:
        for jj in range(N_solns):
            for ii in range(1, Nk):
                PDR_solns[ii, jj], infodict, ier[ii, jj], msg[ii, jj] =\
                    fsolve(func, x0=PDR_solns[ii - 1, jj], args=(k[ii], Species), xtol=tol, maxfev=fev, full_output=True)
    
            # Solve for k[0] using initial guess of k[1]
            PDR_solns[0, jj], infodict, ier[0, jj], msg[0, jj] =\
                fsolve(func, x0=PDR_solns[1, jj], args=(k[0], Species), xtol=tol, maxfev=fev, full_output=True)
    else:
        for jj in range(N_solns):
            for ii in range(1, Nk):
                PDR_solns[ii, jj], infodict, ier[ii, jj], msg[ii, jj] =\
                    fsolve(func, x0=guesses[ii, jj], args=(k[ii], Species), xtol=tol, maxfev=fev, full_output=True)

    # Filter out bad solutions
    if True:
        N_bad = 0
        for jj in range(N_solns):
            for ii in range(1, Nk):
                if ier[ii, jj] == 5:
                    PDR_solns[ii, jj] = np.nan
                    N_bad += 1
        if print_filtered == True:
            print('{} solutions filtered for {} approximation.'.format(N_bad, approx))

    # Solve for growth rate/convective growth rate here
    if approx == 'hot':
        conv_growth = None
    elif approx == 'warm':
        for jj in range(N_solns):
            PDR_solns[:, jj, 1], conv_growth = get_warm_growth_rates(PDR_solns[:, jj, 0], k, Species)
    elif approx == 'cold':
        for jj in range(N_solns):
            PDR_solns[:, jj, 1], conv_growth = get_cold_growth_rates(PDR_solns[:, jj, 0], k, Species)
    
    # Convert to complex number if flagged, else return as (Nk, N_solns, 2) for real/imag components
    if complex_out == True:
        for ii in range(Nk):
            for jj in range(N_solns):
                OUT_solns[ii, jj] = PDR_solns[ii, jj, 0] + 1j*PDR_solns[ii, jj, 1]
    else:
        OUT_solns = PDR_solns
    
    return OUT_solns, conv_growth



def get_all_DRs(time_start, time_end, probe, pad, cmp, 
                    kmin=0.0, kmax=1.0, Nk=1000, knorm=True,
                    nsec=None, HM_filter_mhz=50):
    '''
    Calculate dispersion relation and temporal growth rate for all times between
    time_start and time_end. Nk is resolution in k-space for each solution
    (more = slower for each solutions). nsec is time cadence of interpolated
    satellite parameters.
    
    kmin, kmax :: Wavenumber range in units of c/va or /m
    Nk         :: Number of points to calculate wavenumber for
    knorm      :: Flag to specify units of kmin/kmax. True: c/va, False: /m
    
    suff kwarg for parameters varying only by flags.
    
    Filenames are arranged as 'DISP_YYYYMMDD_AAAA_BBBB_cc_JJ_KK_LL.npz'
        for Year, Month, Day
        for A, B    as the start/end UT times
        for J, K, L as the cold plasma composition assumption
        
    Should probably also save things like the filter
    '''
    # Cast as array just in case its a list
    cmp = np.asarray(cmp)
    
    times, B0, name, mass, charge, density, tper, ani, cold_dens = \
    extract_species_arrays(time_start, time_end, probe, pad, cmp=cmp, 
                           return_raw_ne=True, nsec=nsec, HM_filter_mhz=HM_filter_mhz)

    
    # Arrays to dump results in
    Nt         = times.shape[0]
    all_CPDR   = np.zeros((Nt, Nk, 3), dtype=np.complex128)
    all_WPDR   = np.zeros((Nt, Nk, 3), dtype=np.complex128)
    all_HPDR   = np.zeros((Nt, Nk, 3), dtype=np.complex128)
    all_k      = np.zeros((Nt, Nk)   , dtype=np.float64)
    
    start = time.time()
    for ii in range(Nt):
        try:
            print('Calculating dispersion/growth relation for {}'.format(times[ii]))
            
            Species, PP = create_species_array(B0[ii], name, mass, charge, density[:, ii], tper[:, ii], ani[:, ii])
            all_k[ii]   = np.linspace(kmin, kmax, Nk, endpoint=False)
            
            # Convert k-extrema to /m if needed
            if knorm == True:
                all_k[ii] *= PP['pcyc_rad'] / PP['va']

            # Calculate dispersion relation 3 ways
            all_CPDR[ii], cold_CGR = get_dispersion_relation(Species, all_k[ii], approx='cold')
            all_WPDR[ii], warm_CGR = get_dispersion_relation(Species, all_k[ii], approx='warm')
            all_HPDR[ii],  hot_CGR = get_dispersion_relation(Species, all_k[ii], approx='hot' )
        except:
            print('ERROR: Skipping to next time...')
            all_CPDR[ii, :, :] = np.ones((Nk, 3), dtype=np.complex128   ) * np.nan 
            all_WPDR[ii, :, :] = np.ones((Nk, 3), dtype=np.complex128)    * np.nan
            all_k[   ii, :]    = np.ones(Nk     , dtype=np.complex128   ) * np.nan

    print('All processes complete')
    end = time.time()
    print('Total serial time = {}s'.format(str(end-start)))

    return all_k, all_CPDR, all_WPDR, all_HPDR, \
           times, B0, name, mass, charge, density, tper, ani, cold_dens


def get_DRs_chunked(Nk, kmin, kmax, knorm, times, B0, name, mass, charge, density, tper, ani,
                      k_dict, CPDR_dict, WPDR_dict, HPDR_dict,
                      st=0, worker=None):
    '''
    Function designed to be run in parallel. All dispersion inputs as previous. 
    output_PDR arrays are shared memory
    
    Dispersion inputs (B0, times, density, etc.) are chunked so this function calculates
    them all.
    
    st is the first time index to be computed in the main array, gives position in
    output array to place results in
    
    Thus the array index in output_PDRs will be st+ii for ii in range(Nt)
    '''
    k_arr    = np.frombuffer(k_dict['arr']).reshape(k_dict['shape'])
    CPDR_arr = np.frombuffer(CPDR_dict['arr']).reshape(CPDR_dict['shape'])
    WPDR_arr = np.frombuffer(WPDR_dict['arr']).reshape(WPDR_dict['shape'])
    HPDR_arr = np.frombuffer(HPDR_dict['arr']).reshape(HPDR_dict['shape'])
    
    for ii in range(times.shape[0]):
        Species, PP = create_species_array(B0[ii], name, mass, charge, density[:, ii], tper[:, ii], ani[:, ii])
        this_k      = np.linspace(kmin, kmax, Nk, endpoint=False)
        
        # Convert k-extrema to /m if needed
        if knorm == True:
            this_k *= PP['pcyc_rad'] / PP['va']
        
        # Calculate dispersion relations if possible
        print('Worker', worker, '::', times[ii])
        try:
            this_CPDR, cold_CGR = get_dispersion_relation(Species, this_k, approx='cold', complex_out=False, print_filtered=False)
        except:
            print('COLD ERROR: Skipping', times)
            this_CPDR = np.ones((Nk, 3, 2), dtype=np.complex128) * np.nan 
            
        try:            
            this_WPDR, warm_CGR = get_dispersion_relation(Species, this_k, approx='warm', complex_out=False, print_filtered=False)
        except:
            print('WARM ERROR: Skipping', times)
            this_WPDR = np.ones((Nk, 3, 2), dtype=np.complex128) * np.nan
        
        try:
            this_HPDR,  hot_CGR = get_dispersion_relation(Species, this_k, approx='hot' , complex_out=False, print_filtered=False)
        except:
            print('HOT ERROR: Skipping', times)
            this_HPDR = np.ones((Nk, 3, 2), dtype=np.complex128) * np.nan
                 
        k_arr[   st+ii, :]       = this_k[...]
        CPDR_arr[st+ii, :, :, :] = this_CPDR[...]
        WPDR_arr[st+ii, :, :, :] = this_WPDR[...]
        HPDR_arr[st+ii, :, :, :] = this_HPDR[...]
    return


def get_all_DRs_parallel(time_start, time_end, probe, pad, cmp, 
                    kmin=0.0, kmax=1.0, Nk=1000, knorm=True,
                    nsec=None, HM_filter_mhz=50, complex_out=True):

    if nsec is None:
        DR_path = save_dir + 'DISP_{}_cc_{:03}_{:03}_{:03}.npz'.format(save_string, int(cmp[0]), int(cmp[1]), int(cmp[2]))
    else:
        DR_path = save_dir + 'DISP_{}_cc_{:03}_{:03}_{:03}_{}sec.npz'.format(save_string, int(cmp[0]), int(cmp[1]), int(cmp[2]), nsec)
    
    if os.path.exists(DR_path) == False:
        # Load data
        times, B0, name, mass, charge, density, tper, ani, cold_dens = \
        extract_species_arrays(time_start, time_end, probe, pad, cmp=np.asarray(cmp), 
                               return_raw_ne=True, nsec=nsec, HM_filter_mhz=HM_filter_mhz)
    
        Nt      = times.shape[0]
        N_procs = 7
        procs   = []
        
        # Create raw shared memory arrays with shapes. Store in dict to send with each worker
        k_shape      = (Nt, Nk)
        k_shm        = multiprocessing.RawArray('d', Nt * Nk)
        k_dict       = {'arr': k_shm, 'shape': k_shape}
        
        CPDR_shape   = (Nt, Nk, 3, 2)
        CPDR_shm     = multiprocessing.RawArray('d', Nt*Nk*3*2)
        CPDR_dict    = {'arr': CPDR_shm, 'shape': CPDR_shape}
        
        WPDR_shape   = (Nt, Nk, 3, 2)
        WPDR_shm     = multiprocessing.RawArray('d', Nt*Nk*3*2)
        WPDR_dict    = {'arr': WPDR_shm, 'shape': WPDR_shape}
        
        HPDR_shape   = (Nt, Nk, 3, 2)
        HPDR_shm     = multiprocessing.RawArray('d', Nt*Nk*3*2)
        HPDR_dict    = {'arr': HPDR_shm, 'shape': HPDR_shape}
        
        k_np         = np.frombuffer(k_shm).reshape(k_shape)
        CPDR_np      = np.frombuffer(CPDR_shm).reshape(CPDR_shape)
        WPDR_np      = np.frombuffer(WPDR_shm).reshape(WPDR_shape)
        HPDR_np      = np.frombuffer(HPDR_shm).reshape(HPDR_shape)
        
        # Split input data into a list of chunks
        time_chunks    = np.array_split(times,   N_procs)
        field_chunks   = np.array_split(B0,      N_procs)
        density_chunks = np.array_split(density, N_procs, axis=1)
        tper_chunks    = np.array_split(tper,    N_procs, axis=1)
        ani_chunks     = np.array_split(ani,     N_procs, axis=1)
    
        # Instatiate each process with a different chunk
        acc = 0; start = time.time()
        for xx in range(N_procs):
            print('Starting process', xx)
            proc = multiprocessing.Process(target=get_DRs_chunked,
                                        args=(Nk, kmin, kmax, knorm, time_chunks[xx],
                                        field_chunks[xx], name, mass, charge, density_chunks[xx],
                                        tper_chunks[xx], ani_chunks[xx],
                                        k_dict, CPDR_dict, WPDR_dict, HPDR_dict),
                                        kwargs={'st':acc, 'worker':xx})
            procs.append(proc)
            proc.start()
            
            acc += time_chunks[xx].shape[0]
        
        # Complete processes
        for proc in procs:
            proc.join()
                
        print('All processes complete')
        end = time.time()
        print('Total parallel time = {}s'.format(str(end-start)))
        
        # Make output complex or keep as real in [X, 2] array
        if complex_out == True:
            CPDR_out = np.zeros((Nt, Nk, 3), dtype=np.complex128)
            WPDR_out = np.zeros((Nt, Nk, 3), dtype=np.complex128)
            HPDR_out = np.zeros((Nt, Nk, 3), dtype=np.complex128)
        
            for ii in range(Nt):
                for jj in range(Nk):
                    for kk in range(3):
                        CPDR_out[ii, jj, kk] = CPDR_np[ii, jj, kk, 0] + 1j * CPDR_np[ii, jj, kk, 1]
                        WPDR_out[ii, jj, kk] = WPDR_np[ii, jj, kk, 0] + 1j * WPDR_np[ii, jj, kk, 1]
                        HPDR_out[ii, jj, kk] = HPDR_np[ii, jj, kk, 0] + 1j * HPDR_np[ii, jj, kk, 1]
        else:
            CPDR_out = CPDR_np
            WPDR_out = WPDR_np
            HPDR_out = HPDR_np
            
        # Saves data used for DR calculation as well, for future reference (and plotting)
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
                
        print('Saving dispersion history...')
        np.savez(DR_path, all_CPDR=CPDR_out, all_WPDR=WPDR_out, all_HPDR=HPDR_out, all_k=k_np, comp=np.asarray(cmp),
                 times=times, B0=B0, name=name, mass=mass, charge=charge, density=density, tper=tper,
                 ani=ani, cold_dens=cold_dens, HM_filter_mhz=np.array([HM_filter_mhz]))
    else:
        print('Dispersion results already exist, loading from file...')
        DR_file   = np.load(DR_path)
        
        k_np      = DR_file['all_k']
        CPDR_out  = DR_file['all_CPDR']
        WPDR_out  = DR_file['all_WPDR']
        HPDR_out  = DR_file['all_HPDR']
                
        times     = DR_file['times']
        B0        = DR_file['B0']
        name      = DR_file['name']
        mass      = DR_file['mass']
        charge    = DR_file['charge']
        density   = DR_file['density']
        tper      = DR_file['tper']
        ani       = DR_file['ani']
        cold_dens = DR_file['cold_dens']
        
    return k_np, CPDR_out, WPDR_out, HPDR_out, \
           times, B0, name, mass, charge, density, tper, ani, cold_dens


if __name__ == '__main__':
    # To Do:
    # Peaks to line up
    rbsp_path = 'G://DATA//RBSP//'
    save_drive= 'G://'
    
    time_start  = np.datetime64('2013-07-25T21:25:00')
    time_end    = np.datetime64('2013-07-25T21:47:00')
    probe       = 'a'
    pad         = 0
    
    date_string = time_start.astype(object).strftime('%Y%m%d')
    save_string = time_start.astype(object).strftime('%Y%m%d_%H%M_') + time_end.astype(object).strftime('%H%M')
    save_dir    = '{}NEW_LT//EVENT_{}//CHEN_DR_CODE_PARALLEL//'.format(save_drive, date_string)
    
    _Kp, _CPDRp, _WPDRp, _HPDRp, TIMESp, MAGp, NAMEp, MASSp, CHARGEp, DENSp, TPERp, ANIp, COLD_NEp =\
    get_all_DRs_parallel(time_start, time_end, probe, pad, [70, 20, 10], 
                    kmin=0.0, kmax=1.0, Nk=5000, knorm=True,
                    nsec=None, HM_filter_mhz=50, complex_out=True)

# =============================================================================
#     _K, _CPDR, _WPDR, _HPDR, TIMES, MAG, NAME, MASS, CHARGE, DENS, TPER, ANI, COLD_NE =\
#     get_all_DRs(time_start, time_end, probe, pad, [70, 20, 10], 
#                      kmin=0.0, kmax=1.0, Nk=5000, knorm=True,
#                      nsec=None, HM_filter_mhz=50)
# 
# =============================================================================
