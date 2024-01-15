# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:57:08 2023

@author: Yoshi
"""
import warnings, pdb
import numpy as np
from   scipy.special import wofz

SPLIGHT = 3e8
BOLTZ   = 1.380649e-23  # Boltmann's constant
AMU     = 1.660530e-27  # Dalton (atomic mass unit)
PMASS   = 1.672622e-27  # Proton mass
EMASS   = 9.109384e-31  # Electron mass
ECHARGE = 1.602177e-19  # Elementary charge
PERMITT = 8.854e-12     # Permittivity of free space
PERMEAB = np.pi*4e-7    # Permeability of free space

class IonSpecies:
    def __init__(self, density, charge, mass, temp=None, tperp=None, tpar=None, ani=None,
                 name=''):
        '''Defines an ion species present in a plasma. Assumes a (bi)Maxwellian
        distribution definition for temperature parameters.
        
        Parameters
        ----------
            density : float
               Number density of the ion population, in /cm3
            charge : int
               Electric charge of the ion in multiples of the elementary charge
            mass : float
               Mass of ion in amu. Float because some more complicated plasmas
               may have more than just H, He, O. Might change this to a lookup
               when we get to the GUI.
            temp : float, optional
               Temperature, in eV (default is None)
            tperp : float, optional
               Perpendicular temperature relative to the background magnetic
               field, in eV (default is None)
            tpar : float, optional
               Parallel temperature relative to the background magnetic field,
               in eV (default is None)
            ani : float, optional
               Temperature anisotropy, A = tperp/tpar - 1
            name : str, optional
               String identifier label for this particular species. Useful if
               there is a plasma with multiple populations of the same ion.
               Not used in any calculations
            
        Notes
        -----
        If at least two of the temperature parameters are specified, all will
        be calculated. If more than two are specified, only the first two in
        the arglist will be used.
        
        Parameters are input for ease-of-use in /cm3 and eV, and are 
        automatically converted to their SI counterpart in a separate
        attribute as paramSI (e.g. tpar is in eV, tparSI is in K). This is to
        reduce complexity in calling functions by providing both an SI and
        conventional option for values.
        '''
        self.density = density      # Required args
        self.charge = charge
        self.mass = mass
        
        self.temp = temp            # Optional args
        self.tperp = tperp
        self.tpar = tpar
        self.ani = ani
        self.name = name
    
        self.fillTemperatures()     # Calculated args
        self.getSI()
        self.getFrequencies()
        
    
    def fillTemperatures(self):
        '''Calculate temp, tperp, tpar, ani values given any two of these.
        Defaults to using first-in-line if more than two defined:
            temp > tperp > tpar > ani
        Assumes the relationship T = Tper + Tper
        If insufficient values, species is assumed to be 'cold'        
        '''
        if not all([param is None for param in [self.tperp, self.tpar, self.ani]]):
            
            if self.temp is not None:
                if self.tperp is not None:
                    self.tpar = self.temp - self.tperp
                    self.ani = self.tperp / self.tpar - 1
                elif self.tpar is not None:
                    self.tperp = self.temp - self.tpar
                    self.ani = self.tperp / self.tpar - 1
                elif self.ani is not None:
                    self.tperp = self.temp * (self.ani + 1) / (self.ani + 2) 
                    self.tpar = self.temp / (self.ani + 2)
            else:
                # We don't have self.temp
                if self.tperp is not None:
                    if self.tpar is not None:
                        self.temp = self.tperp + self.tpar
                        self.ani = self.tperp/self.tpar - 1
                    elif self.ani is not None:
                        self.temp = self.tperp * (self.ani + 2) / (self.ani + 1) 
                        self.tpar = self.tperp / (self.ani + 1) 
                else:
                    self.temp = self.tpar * (self.ani + 2)
                    self.tperp = self.tpar * (self.ani + 1)
        else:
            # Would it be better to keep these NoneType?
            self.temp = 0.0
            self.ani = 0.0
            self.tpar = 0.0
            self.tperp = 0.0
                      
        
    def getSI(self):
        '''Convert parameters to SI units: density /cm3 -> /m3 and eV -> Kelvin
        '''
        self.densitySI = self.density * 1e-6
        self.tempSI = self.temp * ECHARGE / BOLTZ
        self.tparSI = self.tpar * ECHARGE / BOLTZ
        self.tperpSI = self.tperp * ECHARGE / BOLTZ
        self.massSI = self.mass * AMU
        self.chargeSI = self.charge * ECHARGE
        return
    
    def getFrequencies(self):
        '''Technically thermal velocity isn't a frequency, but meh
        TODO:
            - Check: Apparently old code used tpar in eV? And had a weird factor
            of q in there. Why!?
        '''
        self.plasmaFreq = np.sqrt(self.density * self.charge ** 2 / (PERMITT * self.mass))
        self.cyclotronFreq = None
        self.thermalVelocityPara = np.sqrt(2.0 * self.tpar / self.massSI)
        return
    
    
class LocalPlasma:
    '''
    Local plasma conditions defined by magnetic field
    Requires calculation of cyclotron frequency for each species, and the
    overall Alfven velocity, which will depend on the magnetic field and the 
    mass density from the total species, as will plasma beta.
    
    Parameters
    ----------
    B0 : float
       Magnetic field of plasma in nT
    Species : list
       Ion species present in plasma, instances of IonSpecies class
    '''
    def __init__(self, B0, species):
        self.B0 = B0*1e-9
        self.species = species
        self.numSpecies = len(self.species)
        
        self.calcCyclotron()
        self.uniqueGyrofreq, self.numUniqueSpecies = self.countUnique()
        print(f'Of {self.numSpecies} species, {self.numUniqueSpecies} are unique')
        
    def calcElectronDensity(self):
        self.ne = 0.0
        for ion in self.species:
            self.ne += ion.charge * ion.densitySI
        return
        
    def calcCyclotron(self):
        for ion in self.species:
            ion.cyclotronFreq = ion.chargeSI * self.B0 / ion.massSI
        return
            
    def calcMassDensity(self):
        self.massDensity = 0.0
        for ion in self.species:
            self.massDensity += ion.massSI * ion.densitySI
        return
    
    def calcAlfvenVelocity(self):
        self.vA = self.B0 / np.sqrt(PERMEAB * self.massDensity)
        return
    
    def countUnique(self):
        cyclotronList = np.array([self.species[ii].cyclotronFreq for ii in range(self.numSpecies)])
        gyfreqs, counts = np.unique(cyclotronList, return_counts=True)
        return gyfreqs, counts.shape[0]
    
    def getDispersionRelation(self, approx='warm'):
        '''
        Dispersion relation calculation is a beast, so split into its
        own class
        
        Store dispersion relation within instance? Or output?
        
        Want to generalise this to allow either vs. wavenumber (k) or
        vs frequency (w)
        -- Do wavenumber first since this is easier.
        -- Vs. frequency may require swapping around the arguments in approx
        '''
        dispersionRelation = DispersionHandler(self, approx)
        dispersionRelation.calcDispersion()
        return
       


class DispersionHandler():
    '''
    Class to handle the calculation of dispersion relations, instantiated
    from a LocalPlasma instance.
    
    This class instance will have methods like
    
    dispersionRelation.calcDispersion
    
    which will call (in fsolve)
    
    coldDispersion
    warmDispersion
    hotDispersion
    
    The calcDispersion relies on information contained within the LocalPlasma instance,
    which 
    '''
    def __init__(self, plasma, approx):
        self.approx = approx
        self.plasma = plasma
        
    @staticmethod
    def Z(arg):
        '''
        Plasma Dispersion Function (Normalized Fadeeva function)
        with normalization constant i*sqrt(pi) (Summers & Thorne, 1993)
        '''
        return 1j*np.sqrt(np.pi)*wofz(arg)

    @staticmethod
    def Y(arg):
        '''Real part of plasma dispersion function'''
        return np.real(DispersionHandler.Z(arg))

    @staticmethod
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
                    Is          = sp['anisotropy'] + numer * DispersionHandler.Y(pdisp_arg) / (sp['vth_par']*k)
                    warm_sum   += sp['plasma_freq_sq'] * Is
                
        solution = wr ** 2 - (SPLIGHT * k) ** 2 + warm_sum
        return np.array([solution, 0.0])
     
    @staticmethod
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
        convective_growth_rate =   temporal_growth_rate / np.abs(group_velocity)
        return temporal_growth_rate, convective_growth_rate, group_velocity
    
    
    def calcDispersion(self, approx='warm'):
        '''
        All input values in SI units from the Species array
        
        Remember, species doesn't contain an electron entry
        -- Should it? Calculate as part of LocalPlasma
        
        The number of solutions is equal to the number of unique gyrofrequencies
        '''
# =============================================================================
#         # Solution and error arrays :: Two-soln array for wr, gamma. 
#         # PDR_solns init'd as ones because 0.0 returns spurious root
#         PDR_solns = np.ones( (Nk, N_solns, 2), dtype=np.float64)*0.01
#         CGR_solns = np.zeros((Nk, N_solns   ), dtype=np.float64)
#         VEL_solns = np.zeros((Nk, N_solns   ), dtype=np.float64)
#         ier       = np.zeros((Nk, N_solns   ), dtype=int)
#         msg       = np.zeros((Nk, N_solns   ), dtype='<U256')
# =============================================================================

# =============================================================================
#         # fsolve arguments
#         eps    = 1.01           # Offset used to supply initial guess (since right on w_cyc returns an error)
#         tol    = 1e-10          # Absolute solution convergence tolerance in rad/s
#         fev    = 1000000        # Maximum number of iterations
#         Nk     = k.shape[0]     # Number of wavenumbers to solve for
# =============================================================================
        


# =============================================================================
#         # Initial guesses (check this?)
#         for ii in range(1, N_solns):
#             PDR_solns[0, ii - 1]  = np.array([[gyfreqs[-ii - 1] * eps, 0.0]])
# =============================================================================
        
# =============================================================================
#         if approx == 'hot':
#             func = hot_dispersion_eqn
#         elif approx == 'warm':
#             func = warm_dispersion_eqn
#         elif approx == 'cold':
#             func = cold_dispersion_eqn
#         else:
#             sys.exit('ABORT :: kwarg approx={} invalid. Must be \'cold\', \'warm\', or \'hot\'.'.format(approx))
# =============================================================================
        
# =============================================================================
#         # Define function to solve for (all have same arguments)
#         if guesses is None or guesses.shape != PDR_solns.shape:
#             for jj in range(N_solns):
#                 for ii in range(1, Nk):
#                     #if np.isnan(k[ii]):
#                     #    PDR_solns[ii, jj] = np.nan
#                     #else:
#                         PDR_solns[ii, jj], infodict, ier[ii, jj], msg[ii, jj] =\
#                             fsolve(func, x0=PDR_solns[ii - 1, jj], args=(k[ii], Species), xtol=tol, maxfev=fev, full_output=True)
#                 
#                 if False:
#                     # Solve for k[0] using initial guess of k[1]
#                     PDR_solns[0, jj], infodict, ier[0, jj], msg[0, jj] =\
#                         fsolve(func, x0=PDR_solns[1, jj], args=(k[0], Species), xtol=tol, maxfev=fev, full_output=True)
#                 else:
#                     # Set k[0] as equal to k[1] (better for large Nk)
#                     PDR_solns[0, jj] = PDR_solns[1, jj]
#         else:
#             for jj in range(N_solns):
#                 for ii in range(1, Nk):
#                     PDR_solns[ii, jj], infodict, ier[ii, jj], msg[ii, jj] =\
#                         fsolve(func, x0=guesses[ii, jj], args=(k[ii], Species), xtol=tol, maxfev=fev, full_output=True)
# =============================================================================

# =============================================================================
#         N_bad = remove_bad_solutions(PDR_solns, ier)
#         #N_dup = remove_duplicates(PDR_solns)
#         if print_filtered == True:
#             print(f'{N_bad} solutions filtered for {approx} approximation.')
#             #print(f'{N_dup} duplicates removed.')
# =============================================================================

# =============================================================================
#         # Solve for growth rate/convective growth rate here
#         if approx == 'hot':
#             CGR_solns *= np.nan
#         elif approx == 'warm':
#             for jj in range(N_solns):
#                 PDR_solns[:, jj, 1], CGR_solns[:, jj], VEL_solns[:, jj] = get_warm_growth_rates(PDR_solns[:, jj, 0], k, Species)
#         elif approx == 'cold':
#             for jj in range(N_solns):
#                 PDR_solns[:, jj, 1], CGR_solns[:, jj], VEL_solns[:, jj] = get_cold_growth_rates(PDR_solns[:, jj, 0], k, Species)
# =============================================================================

# =============================================================================
#         # Convert to complex number if flagged, else return as (Nk, N_solns, 2) for real/imag components
#         if complex_out == True:
#             OUT_solns = np.zeros((Nk, N_solns   ), dtype=np.complex128)
#             for ii in range(Nk):
#                 for jj in range(N_solns):
#                     OUT_solns[ii, jj] = PDR_solns[ii, jj, 0] + 1j*PDR_solns[ii, jj, 1]
#         else:
#             OUT_solns = PDR_solns
# =============================================================================

    pass

    
if __name__ == '__main__':
    wH  = IonSpecies(10., 1., 1., temp=None, tperp=None, tpar=50, ani=0.0, name='H+')
    cH  = IonSpecies(200., 1., 1.)
    cHe = IonSpecies(20., 1., 4.)
    
    # Set plasma conditions
    P = LocalPlasma(200., [wH, cH, cHe])
    
    # Calculate plasma dispersion relation, store result in instance
    #P.getDispersionRelation()
        