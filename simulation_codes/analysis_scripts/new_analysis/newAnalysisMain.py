# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:52:23 2022

@author: Yoshi
"""

from SimulationClass import HybridSimulationRun
from standardFieldOuput import plotFieldDiagnostics
from standardEnergyOutput import plotEnergies



if __name__ == '__main__':

    _name = 'energy_conservation_resonant'
    _num  = 0
    
    sim1 = HybridSimulationRun(_name, _num)
    
    #plotFieldDiagnostics(sim1)
    plotEnergies(sim1)