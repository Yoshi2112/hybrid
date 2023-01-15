# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:19:01 2023

@author: iarey
"""
import SimulationClass
import matplotlib.pyplot as plt

def compareEnergy(runList, normalize=False, save2root=True, save_dir=None, 
                  orientation='portrait', time_unts=None):
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
            
    if orientation == 'portrait':
        figdim = (8.27, 11.69)
    elif orientation == 'landscape':
        figdim = (11.69, 8.27)
    else:
        print(f'Orientation kwarg \'{orientation}\' not recognized, defaulting to portrait.')
        figdim = (8.27, 11.69)
    
    plt.ioff()
    fig, axes = plt.subplots(figsize=figdim, nrows=6, sharex=True)
    axes[0].set_title('Energy Comparison (eV) for Hybrid Runs')
    
    for ii, sim in enumerate(runList):
        # Type check
        if not isinstance(sim, SimulationClass.HybridSimulationRun):
            print('List contains bad runs. Aborting.')
            return
        
        # Retrieve energies
        mag_energy, electron_energy, particle_energy, total_energy = sim.get_energies()
        particle_total = particle_energy.sum(axis=2)
        
        # Do the plotting
        axes[0].plot(sim.particle_sim_time, total_energy, label=sim.series_name+f'[{sim.run_num}]')
        axes[0].set_ylabel('Total')
        
        axes[1].plot(sim.field_sim_time, mag_energy)
        axes[1].set_ylabel('Magnetic')
        
        axes[2].plot(sim.field_sim_time, electron_energy)
        axes[2].set_ylabel('Electron Energy')
        
        for jj in range(sim.Nj):
            axes[3].plot(sim.particle_sim_time, particle_total[:, jj], c=sim.temp_color[jj], ls=run_styles[ii], label=sim.species_lbl[jj])
            axes[4].plot(sim.particle_sim_time, particle_energy[:, jj, 0], c=sim.temp_color[jj], ls=run_styles[ii])
            axes[5].plot(sim.particle_sim_time, particle_energy[:, jj, 1], c=sim.temp_color[jj], ls=run_styles[ii])
        axes[3].set_ylabel('Ions')
        axes[4].set_ylabel('Ions $\parallel$')
        axes[5].set_ylabel('Ions $\perp$')
            
    axes[0].legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    axes[3].legend(loc='center left', bbox_to_anchor=(1.04, 0.0))
    
    for ax in axes:
        ax.set_xlim(0.0, sim.particle_sim_time[-1])
    axes[-1].set_xlabel('Time (s)')
    
    fig.savefig(filepath, bbox_inches='tight')
    plt.close('all')
    return


def plotIonEnergy(runList, normalize=False, save2root=True, save_dir=None, 
                  orientation='portrait', time_unts=None):
    '''
    Designed for simulation runs with identical lists of species/particles
    -- Total ion energy
    -- 1 plot per species after that (color based on run?)
    TODO: New plot to show the parallel/perp energy breakdown per species
    '''        
    if save2root:
        filepath = runList[0].base_dir + 'compareIonEnergies.png'
    else:
        if save_dir is None:
            print('save_dir must be specified if save2root is False. Aborting.')
            return
        else:
            filepath = save_dir + 'compareEnergies.png'
            
    if orientation == 'portrait':
        figdim = (8.27, 11.69)
    elif orientation == 'landscape':
        figdim = (11.69, 8.27)
    else:
        print(f'Orientation kwarg \'{orientation}\' not recognized, defaulting to portrait.')
        figdim = (8.27, 11.69)
    
    plt.ioff()
    Nj = runList[0].Nj
    fig, axes = plt.subplots(figsize=figdim, nrows=Nj+1, sharex=True)
    axes[0].set_title('Ion Energy Comparison (eV) for Hybrid Runs')
    
    for ii, sim in enumerate(runList):
        # Type check
        print(ii)
        if not isinstance(sim, SimulationClass.HybridSimulationRun):
            print('List contains bad runs. Aborting.')
            return
        
        # Retrieve energies
        mag_energy, electron_energy, particle_energy, total_energy = sim.get_energies()
        particle_total = particle_energy.sum(axis=2)
        
        axes[0].plot(sim.particle_sim_time, particle_total.sum(axis=1), label=sim.series_name+f'[{sim.run_num}]')
        for jj in range(sim.Nj):
            axes[jj+1].plot(sim.particle_sim_time, particle_total[:, jj], label=sim.species_lbl[jj] if ii==0 else None)
            if ii==0: axes[jj+1].legend(loc='center left', bbox_to_anchor=(1.04, 0.5), labelcolor=sim.temp_color[jj])
    axes[0].legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    
    for ax in axes:
        ax.set_xlim(0.0, sim.particle_sim_time[-1])
        ax.set_ylabel('Energy (J)')
    axes[-1].set_xlabel('Time (s)')
    
    fig.savefig(filepath, bbox_inches='tight')
    plt.close('all')
    return