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
    drive = 'D:'
    
    if True:
        name = 'PREDCORR_RK4_test'
        
        for num in [7]:
        
            sim = HybridSimulationRun(name, num, home_dir=f'{drive}/runs/')
        
            plotFieldDiagnostics(sim)
            #plotEnergies(sim)
            #checkFields(sim, save=True, ylim=False, skip=50, it_max=None)
            #winskeSummaryPlots(sim1, save=True, skip=1)
            #summaryPlots(sim, save=True, histogram=False, skip=1, ylim=True)
        
    else:
        
        # Build list of runs to analyse
        names = ['RK4_test']*3
        nums  = [0, 1, 2]
        simList = []
        for name, num in zip(names, nums):
            simList.append(HybridSimulationRun(name, num, home_dir=f'{drive}/runs/'))
        
        # Send list for combined multiplot output
        compareEnergy(simList, normalize=False, save2root=True, save_dir=None)
        plotIonEnergy(simList, normalize=False, save2root=True, save_dir=None)