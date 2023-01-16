# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:52:23 2022

@author: Yoshi
"""
from SimulationClass import HybridSimulationRun
from standardFieldOuput import plotFieldDiagnostics
from standardEnergyOutput import plotEnergies
from timesliceOutputs import winskeSummaryPlots, summaryPlots, checkFields
from multiplotEnergy import compareEnergy, plotIonEnergy


if __name__ == '__main__':
    drive = 'F:'
    
    if True:
        name = 'energy_conservation_resonant'
        num  = 1
        
        sim = HybridSimulationRun(name, num, home_dir=f'{drive}/runs/')
        #plotFieldDiagnostics(sim1)
        #plotEnergies(sim1)
        #winskeSummaryPlots(sim1, save=True, skip=1)
        #summaryPlots(sim1, save=True, histogram=False, skip=1, ylim=True)
        checkFields(sim, save=True, ylim=False, skip=1, it_max=100)
    else:
        name = 'energyConservationParticlesOnly'
        num = 0
        simList = []
        while True:
            try:
                simList.append(HybridSimulationRun(name, num, home_dir=f'{drive}/runs/'))
                num += 1
            except OSError:
                break
        #compareEnergy(simList, normalize=False, save2root=True, save_dir=None)
        plotIonEnergy(simList, normalize=False, save2root=True, save_dir=None)