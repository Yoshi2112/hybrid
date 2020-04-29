## PYTHON MODULES ##
from timeit import default_timer as timer
import numba as nb
import numpy as np

## HYBRID MODULES ##
import init_1D       as init
import particles_1D  as particles
import save_routines as save

from simulation_parameters_1D import save_particles, orbit_res, gyfreq, dx

@nb.njit()
def check_timestep(pos, vel, qq, DT, max_inc, part_save_iter, idx):
    ion_ts          = orbit_res / gyfreq
    vel_ts          = 0.60 * dx / np.abs(vel[0, :]).max()      # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than 'half' a cell in one time step
    DT_part         = min(vel_ts, ion_ts)                      # Smallest of the allowable timesteps
    
    if DT_part < 0.9*DT:

        particles.velocity_update(pos, vel, idx, 0.5*DT)    # Re-sync vel/pos       

        DT         *= 0.5
        max_inc    *= 2
        qq         *= 2
        
        part_save_iter *= 2

        particles.velocity_update(pos, vel, idx, -0.5*DT)   # De-sync vel/pos 
        print('Timestep halved. Syncing particle velocity...')
    return qq, DT, max_inc, part_save_iter

@nb.njit()
def main_loop(pos, vel, idx, qq, DT, max_inc, part_save_iter):

    qq, DT, max_inc, part_save_iter = check_timestep(pos, vel, qq, DT, max_inc, part_save_iter, idx)
    
    particles.velocity_update(pos, vel, idx, DT)
    particles.position_update(pos, vel, idx, DT)  
    
    return qq, DT, max_inc, part_save_iter


if __name__ == '__main__':
    start_time = timer()
    
    pos, vel, idx = init.uniform_gaussian_distribution_quiet()
        
    DT, max_inc, part_save_iter = init.set_timestep(vel)

    if save_particles == 1:
        save.save_particle_data(0, DT, part_save_iter, 0, pos, vel, idx)
            
    particles.velocity_update(pos, vel, idx, -0.5*DT)
    
    qq       = 1;    sim_time = DT
    print('Starting main loop...')
    while qq < max_inc:
        qq, DT, max_inc, part_save_iter = main_loop(pos, vel, idx, qq, DT, max_inc, part_save_iter)

        if qq%part_save_iter == 0 and save_particles == 1:
            save.save_particle_data(sim_time, DT, part_save_iter, qq, pos,
                                    vel, idx)
        
        if qq%100 == 0:
            running_time = int(timer() - start_time)
            hrs          = running_time // 3600
            rem          = running_time %  3600
            
            mins         = rem // 60
            sec          = rem %  60
            print('Step {} of {} :: Current runtime {:02}:{:02}:{:02}'.format(qq, max_inc, hrs, mins, sec))
            
        qq       += 1
        sim_time += DT
        
    print("Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2)))