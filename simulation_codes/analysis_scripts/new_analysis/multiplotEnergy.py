# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:19:01 2023

@author: iarey
"""
import SimulationClass
import matplotlib.pyplot as plt

def compareEnergy(runList, normalize=False, save2root=True, save_dir=None):
    '''
    Plots a timeseries of energies for a list of runs containing SimulationClass instances.
    Different runs denoted by different color. Set somewhere?
    
    runList   -- List of HybridSimulationRun instances for all runs to be plotted
    normalize -- Normalizes each energy to the initial values of the first run in the list
    
    To plot:
        Total energy
        Magnetic field energy
        Electron energy
        Total particle energy
        Perp particle energy
        Para parallel energy
        
    Note: Particle energies of shape (time, species, direction) for 0: Parallel, 1: Perpendicular
          Limited to 4 simultaneous runs due to difficulty in displaying things
          
    Saves file to series directory of initial runif save2root is True, else saves to save_dir
    '''
    run_styles = ['-', '--', ':', '-.']
    if len(runList) > len(run_styles):
        print('Too many runs to compare. Aborting.')
        
    if save2root:
        filepath = runList[0].base_dir + 'compareEnergies.png'
    else:
        if save_dir is None:
            print('save_dir must be specified if save2root is False. Aborting.')
            return
        else:
            filepath = save_dir + 'compareEnergies.png'
    
    fig, axes = plt.subplots(figsize=(16, 9), nrows=6, sharex=True)
    
    for ii, sim in enumerate(runList):
        # Type check
        if not isinstance(sim, SimulationClass.HybridSimulation):
            print('List contains bad runs. Aborting.')
            return
        
        # Retrieve energies
        mag_energy, electron_energy, particle_energy, total_energy = sim.get_energies()
        particle_total = particle_energy.sum(axis=2)
        
        # Do the plotting
        axes[0].plot(sim.particle_sim_time, total_energy, label=f'{sim.series_name[ii]}[{sim.run_num}]')
        axes[1].plot(sim.field_sim_time, mag_energy)
        axes[2].plot(sim.field_sim_time, electron_energy)
        
        for jj in sim.Nj:
            axes[3].plot(sim.particle_sim_time, particle_total, c=sim.temp_color[jj], ls=run_styles[ii], label=sim.species_lbl[jj])
            axes[4].plot(sim.particle_sim_time, particle_energy[:, jj, 0], c=sim.temp_color[jj], ls=run_styles[ii])
            axes[5].plot(sim.particle_sim_time, particle_energy[:, jj, 1], c=sim.temp_color[jj], ls=run_styles[ii])
            
        axes[0].legend()
        axes[2].legend()
        
    fig.savefig(filepath, bbox_inches='tight')
    plt.close('all')
    return