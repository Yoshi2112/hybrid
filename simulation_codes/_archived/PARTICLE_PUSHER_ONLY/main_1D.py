## PYTHON MODULES ##
from timeit import default_timer as timer

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources
import save_routines as save

from simulation_parameters_1D import save_particles, save_fields


if __name__ == '__main__':
    start_time = timer()
    
    # Initialize simulation: Allocate memory and set time parameters
    pos, vel, Ie, W_elec, idx, Bp, temp_N = init.initialize_particles()
    q_dens, q_dens_adv, Ji, ni, nu, flux                = init.initialize_source_arrays()
    old_particles, old_fields, temp3De, temp3Db, temp1D,\
                                          v_prime, S, T = init.initialize_tertiary_arrays()
    
    # Collect initial moments and save initial state
    sources.collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu)

    DT, max_inc, part_save_iter = init.set_timestep(vel)

    if save_particles == 1:
        save.save_particle_data(0, DT, part_save_iter, 0, pos, vel, idx)

    # Retard velocity
    print('Retarding velocity...')
    particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, v_prime, S, T, temp_N, -0.5*DT)

    qq       = 1;    sim_time = DT
    print('Starting main loop...')
    while qq < max_inc:
        particles.advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, v_prime, S, T,temp_N,\
                                                B, E_int, DT, q_dens_adv, Ji, ni, nu, flux)
        
        q_dens *= 0.5
        q_dens += 0.5 * q_dens_adv

        if qq%part_save_iter == 0 and save_particles == 1:
            save.save_particle_data(sim_time, DT, part_save_iter, qq, pos,
                                    vel, idx)
        
        if qq%50 == 0:            
            running_time = int(timer() - start_time)
            hrs          = running_time // 3600
            rem          = running_time %  3600
            
            mins         = rem // 60
            sec          = rem %  60
            print('Step {} of {} :: Current runtime {:02}:{:02}:{:02}'.format(qq, max_inc, hrs, mins, sec))
            
        qq       += 1
        sim_time += DT

    runtime = round(timer() - start_time,2)
    
    if save_fields == 1 or save_particles == 1:
        save.add_runtime_to_header(runtime)
    print("Time to execute program: {0:.2f} seconds".format(runtime))
