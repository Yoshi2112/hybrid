# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:57:08 2023

@author: Yoshi
"""
import numpy as np

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
    '''Local plasma conditions defined by magnetic field
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
    def __init__(self, B0, Species):
        self.B0 = B0*1e-9
        self.Species = Species
        
        self.calcCyclotron()
    
    def calcElectronDensity(self):
        self.ne = 0.0
        for ion in self.Species:
            self.ne += ion.charge * ion.densitySI
        return
        
    def calcCyclotron(self):
        for ion in self.Species:
            ion.cyclotronFreq = ion.chargeSI * self.B0 / ion.massSI
        return
            
    def calcMassDensity(self):
        self.massDensity = 0.0
        for ion in self.Species:
            self.massDensity += ion.massSI * ion.densitySI
        return
    
    def calcAlfvenVelocity(self):
        self.vA = self.B0 / np.sqrt(PERMEAB * self.massDensity)
        return
    
    def getDispersionRelation(self):
        '''
        Dispersion relation calculation is a beast, so split into its
        own class
        '''
        dispersionRelation = DispersionHandler(self)
        print(dispersionRelation.numSpecies)
        return
       


class DispersionHandler():
    '''
    Class to handle the calculation of dispersion relations, instantiated
    from a LocalPlasma instance.
    '''
    def __init__(self, plasmaInstance):
        self.plasma = plasmaInstance
        self.numSpecies = len(plasmaInstance.Species)
        #print('The number of species is', numSpecies)
        
    # To Do:
    # Count number of unique species
    # Code solver for dispersion relation
    # Analytic growth rate solver
    pass

    
if __name__ == '__main__':
    wH  = IonSpecies(10., 1., 1., temp=None, tperp=None, tpar=50, ani=0.0, name='H+')
    cH  = IonSpecies(200., 1., 1.)
    cHe = IonSpecies(20., 1., 1.)
    
    P = LocalPlasma(200., [wH, cH, cHe])
    
    P.getDispersionRelation()
        